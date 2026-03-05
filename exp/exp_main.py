from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import (
    Informer,
    Autoformer,
    Transformer,
    DLinear,
    Linear,
    NLinear,
    PatchTST,
    DLinear_XYZ,
    ARIMA,
    Prophet,
    ModernTCN,
)
from utils.tools import (
    EarlyStopping,
    StandardScaler,
    adjust_learning_rate,
    visual,
    test_params_flop,
)
from utils.metrics import metric
from utils.my_losses import frequency_loss
from utils.fft_enhancement import FFTEnhancer
from utils.moe_enhancement import MoEEnhancer
from utils.freqMOE_enhancement import FreqMoEEnhancer

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import pandas as pd

import os
import time

import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.fft_enhancer = None
        self.moe_enhancer = None
        self.freq_moe_enhancer = None
        self.freqmoe_stats = None
        self.freqmoe_balance_weight = 0.0
        self.current_freqmoe_balance_loss = None
        if getattr(self.args, "use_fft_enhance_data", False):
            self.fft_enhancer = FFTEnhancer(
                low_freq_ratio=self.args.fft_low_freq_ratio,
                high_freq_ratio=self.args.fft_high_freq_ratio,
                low_freq_boost=self.args.fft_low_freq_boost,
                mid_freq_boost=self.args.fft_mid_freq_boost,
                high_freq_suppress=self.args.fft_high_freq_suppress,
                cutoff_ratio=self.args.fft_cutoff_ratio,
                residual_ratio=self.args.fft_residual_ratio,
                reflection_pad=self.args.fft_reflection_pad,
            )
            print(
                "FFT增强已启用：模型输入将在运行时进行频域加权，评估保持在原始数据空间。"
            )

        if getattr(self.args, "use_moe", False):
            self.moe_enhancer = MoEEnhancer(
                low_freq_ratio=self.args.moe_low_freq_ratio,
                high_freq_ratio=self.args.moe_high_freq_ratio,
                cutoff_ratio=self.args.moe_cutoff_ratio,
                frequency_residual_ratio=self.args.moe_frequency_residual_ratio,
                temperature=self.args.moe_temperature,
                frequency_experts=self.args.moe_frequency_experts,
                frequency_top_k=self.args.moe_frequency_topk,
                frequency_energy_norm=self.args.moe_energy_norm,
            ).to(self.device)
            print("MoE增强已启用：实时融合 FFT/DCT/原始模态。")

        if getattr(self.args, "use_freqmoe", False):
            self.freq_moe_enhancer = FreqMoEEnhancer(
                seq_len=self.args.seq_len,
                num_experts=self.args.freqmoe_experts,
                use_norm=bool(self.args.freqmoe_use_norm),
                use_soft_mask=bool(getattr(self.args, "freqmoe_use_soft_mask", True)),
                sharpness=float(getattr(self.args, "freqmoe_sharpness", 20.0)),
                use_freq_wise_routing=bool(
                    getattr(self.args, "freqmoe_freq_wise_routing", False)
                ),
                use_lowpass_filter=bool(
                    getattr(self.args, "freqmoe_use_lowpass_filter", False)
                ),
                lowpass_cutoff=float(getattr(self.args, "freqmoe_lowpass_cutoff", 0.3)),
                expert_freq_range=float(
                    getattr(self.args, "freqmoe_expert_freq_range", 1.0)
                ),
            ).to(self.device)
            self.freqmoe_balance_weight = float(
                getattr(self.args, "freqmoe_balance_weight", 0.01)
            )
            print(
                f"freqMoE增强已启用：SoftMask={getattr(self.args, 'freqmoe_use_soft_mask', True)}, "
                f"FreqWiseRouting={getattr(self.args, 'freqmoe_freq_wise_routing', False)}, "
                f"LowpassFilter={getattr(self.args, 'freqmoe_use_lowpass_filter', False)}, "
                f"LowpassCutoff={getattr(self.args, 'freqmoe_lowpass_cutoff', 0.3)}, "
                f"ExpertFreqRange={getattr(self.args, 'freqmoe_expert_freq_range', 1.0)}, "
                f"BalanceWeight={self.freqmoe_balance_weight}"
            )
            self._reset_freqmoe_stats()
            self._log_freqmoe_initial_bounds()

    def _build_model(self):
        model_dict = {
            "Autoformer": Autoformer,
            "Transformer": Transformer,
            "Informer": Informer,
            "DLinear": DLinear,
            "NLinear": NLinear,
            "Linear": Linear,
            "PatchTST": PatchTST,
            "DLinear_XYZ": DLinear_XYZ,
            "ARIMA": ARIMA,
            "Prophet": Prophet,
            "ModernTCN": ModernTCN,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _fft_enhance_tensor(self, tensor):
        if self.fft_enhancer is None or tensor is None:
            return tensor
        tensor_np = tensor.detach().cpu().numpy()
        enhanced = np.zeros_like(tensor_np)
        for idx in range(tensor_np.shape[0]):
            enhanced[idx] = self.fft_enhancer.enhance_batch(tensor_np[idx])
        enhanced_tensor = torch.from_numpy(enhanced).to(tensor.device)
        return enhanced_tensor.type_as(tensor)

    def _apply_moe(self, tensor):
        if self.moe_enhancer is None or tensor is None:
            return tensor
        return self.moe_enhancer(tensor)

    def _apply_freq_moe(self, tensor):
        if self.freq_moe_enhancer is None or tensor is None:
            return tensor, None, None
        output, weights, masks = self.freq_moe_enhancer(tensor)

        balance_loss = None
        if (
            self.model is not None
            and self.model.training
            and hasattr(self, "freqmoe_balance_weight")
            and self.freqmoe_balance_weight > 0
        ):
            balance_loss = self.freq_moe_enhancer.get_load_balance_loss(weights)

        if weights is not None and weights.dim() > 2:
            weights = weights.reshape(weights.shape[0], -1)
        if self.model is not None and self.model.training:
            if masks is not None:
                freq_len = self.freq_moe_enhancer.freq_len
                bounds = torch.arange(
                    0, self.freq_moe_enhancer.num_experts + 1, device=tensor.device
                ).float()
                bounds = (bounds / self.freq_moe_enhancer.num_experts * freq_len).long()
                self._accumulate_freqmoe_stats(weights, bounds)
        return output, weights, balance_loss

    def _reset_freqmoe_stats(self):
        if self.freq_moe_enhancer is None:
            self.freqmoe_stats = None
            return
        self.freqmoe_stats = {
            "sum_weights": None,
            "sample_count": 0,
            "bounds": None,
            "freq_len": self.freq_moe_enhancer.freq_len,
        }

    def _accumulate_freqmoe_stats(self, weights, bounds):
        if self.freqmoe_stats is None:
            return
        stats = self.freqmoe_stats
        weights_cpu = weights.detach().to("cpu")
        if weights_cpu.dim() == 1:
            weights_cpu = weights_cpu.unsqueeze(0)
        if stats["sum_weights"] is None:
            stats["sum_weights"] = torch.zeros(weights_cpu.shape[-1])
        stats["sum_weights"] += weights_cpu.sum(dim=0)
        stats["sample_count"] += weights_cpu.shape[0]
        stats["bounds"] = bounds.detach().to("cpu").long()

    def _finalize_freqmoe_stats(self):
        if self.freqmoe_stats is None:
            return
        stats = self.freqmoe_stats
        if (
            stats["sum_weights"] is None
            or stats["sample_count"] == 0
            or stats["bounds"] is None
        ):
            print("freqMoE 未收集到有效的专家统计数据。")
            return
        avg_weights = (stats["sum_weights"] / stats["sample_count"]).tolist()
        bounds = stats["bounds"].tolist()
        freq_len = max(1, int(stats["freq_len"]))
        expert_info = []
        for idx, weight in enumerate(avg_weights):
            start = int(bounds[idx]) if idx < len(bounds) else 0
            end = int(bounds[idx + 1]) if idx + 1 < len(bounds) else freq_len
            coverage = (end - start) / freq_len
            expert_info.append((idx, weight, coverage, start, end))
        expert_info.sort(key=lambda item: item[1], reverse=True)
        print("freqMoE 训练结束专家统计（按平均权重降序）:")
        for idx, weight, coverage, start, end in expert_info:
            print(
                f"  专家{idx:02d}: 平均权重={weight:.4f}, 覆盖比例={coverage:.4f}, 频段=[{start}, {end})"
            )

    def _log_freqmoe_initial_bounds(self):
        if self.freq_moe_enhancer is None:
            return
        segments = self.freq_moe_enhancer.describe_expert_bounds()
        if not segments:
            return
        print("freqMoE 初始专家频段划分:")
        for idx, start, end, coverage in segments:
            print(f"  专家{idx:02d}: 频段=[{start}, {end}), 覆盖比例={coverage:.4f}")

    # 准备模型输入，包括可能的FFT/MoE增强
    def _prepare_model_inputs(self, batch_x, dec_inp):
        self.current_freqmoe_balance_loss = None
        if self.freq_moe_enhancer is not None and getattr(
            self.args, "use_freqmoe", False
        ):
            batch_final, _, balance_loss = self._apply_freq_moe(batch_x)
            self.current_freqmoe_balance_loss = balance_loss
            return batch_final, dec_inp

        if self.moe_enhancer is not None and getattr(self.args, "use_moe", False):
            batch_final = self._apply_moe(batch_x)
            return batch_final, dec_inp

        if self.fft_enhancer is not None:
            batch_x_fft = self._fft_enhance_tensor(batch_x)
            dec_fft = dec_inp.clone()
            label_len = min(self.args.label_len, dec_inp.shape[1])
            if label_len > 0:
                dec_fft[:, :label_len, :] = self._fft_enhance_tensor(
                    dec_inp[:, :label_len, :]
                )
            return batch_x_fft, dec_fft

        return batch_x, dec_inp

    def _select_optimizer(self):  # model.parameters()是什么
        params = list(self.model.parameters())
        if self.moe_enhancer is not None:
            params += list(self.moe_enhancer.parameters())
        if self.freq_moe_enhancer is not None:
            params += list(self.freq_moe_enhancer.parameters())
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _forward_model(
        self, batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y=None
    ):
        simple_models = {"ARIMA", "Prophet"}
        if self.args.model == "ModernTCN":
            return self.model(batch_x, batch_x_mark)
        if (
            "Linear" in self.args.model
            or "TST" in self.args.model
            or self.args.model in simple_models
        ):
            return self.model(batch_x)
        if self.args.output_attention:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
        if batch_y is not None:
            return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
        return self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    # 用验证集评估当前模型性能
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []  # 用于收集每个batch的损失
        self.model.eval()  # 将模型设置为评估模式，关闭 dropout 和 batch normalization 等训练专用层
        moe_prev_mode = None
        freq_prev_mode = None
        if self.moe_enhancer is not None:
            moe_prev_mode = self.moe_enhancer.training
            self.moe_enhancer.train(False)
        if self.freq_moe_enhancer is not None:
            freq_prev_mode = self.freq_moe_enhancer.training
            self.freq_moe_enhancer.train(False)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(
                    self.device
                )  # 输入序列的时间特征编码
                batch_y_mark = batch_y_mark.float().to(
                    self.device
                )  # 目标序列的时间特征编码

                # decoder input 用标签的历史部分+全零的未来部分拼接
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                batch_x_fft, dec_inp_fft = self._prepare_model_inputs(batch_x, dec_inp)
                if self.args.use_amp:
                    # 混合精度指在深度学习训练或推理过程中，同时使用不同数值精度的数据类型（如 float16 和 float32），以提升计算速度、减少显存占用，同时保持模型精度。
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark
                        )
                else:
                    outputs = self._forward_model(
                        batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark
                    )
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        if self.moe_enhancer is not None:
            self.moe_enhancer.train(
                moe_prev_mode if moe_prev_mode is not None else True
            )
        if self.freq_moe_enhancer is not None:
            self.freq_moe_enhancer.train(
                freq_prev_mode if freq_prev_mode is not None else True
            )
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="train")
        vali_data, vali_loader = self._get_data(flag="val")
        test_data, test_loader = self._get_data(flag="test")

        if getattr(self.args, "use_freqmoe", False):
            self._reset_freqmoe_stats()

        if self.args.auxi_lambda != 0:
            print(
                "已使用 frequency_loss，auxi_lambda:"
                + str(self.args.auxi_lambda)
                + "rec_lambda: "
                + str(self.args.rec_lambda)
            )

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(
            train_loader
        )  # 所有样本中一共有多少个batch，每个epoch的训练步数
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=self.args.train_epochs,
            max_lr=self.args.learning_rate,
        )

        # 初始化损失记录列表
        train_loss_history = []
        vali_loss_history = []

        for epoch in range(self.args.train_epochs):
            print("train_epoch: " + str(self.args.train_epochs))
            iter_count = 0
            train_loss = []

            self.model.train()
            if self.moe_enhancer is not None:
                self.moe_enhancer.train()
            epoch_time = time.time()
            # enumerate(train_loader)遍历每个batch
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                # timeenc = 0时两个变量无效
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                # encoder - decoder
                batch_x_fft, dec_inp_fft = self._prepare_model_inputs(batch_x, dec_inp)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark
                        )

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self._forward_model(
                        batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark, batch_y
                    )
                    # print(outputs.shape,batch_y.shape)
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                    # --- 2. 修改训练集的损失计算逻辑 ---
                    # 原始时间域损失
                    loss_rec = criterion(outputs, batch_y)

                    # 辅助的频域损失
                    loss_auxi = frequency_loss(outputs, batch_y)

                    # freqMoE负载均衡损失
                    loss_balance = 0.0
                    if (
                        self.current_freqmoe_balance_loss is not None
                        and self.freqmoe_balance_weight > 0
                    ):
                        loss_balance = self.current_freqmoe_balance_loss

                    # 加权求和得到最终损失
                    loss = (
                        self.args.rec_lambda * loss_rec
                        + self.args.auxi_lambda * loss_auxi
                        + self.freqmoe_balance_weight * loss_balance
                    )
                    # loss = criterion(outputs, batch_y)

                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.args.train_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == "TST":
                    adjust_learning_rate(
                        model_optim, scheduler, epoch + 1, self.args, printout=False
                    )
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 记录当前 epoch 的损失值
            train_loss_history.append(train_loss)
            vali_loss_history.append(vali_loss)

            print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss
                )
            )

            # vali_mae_denorm = self.calculate_denorm_mae(vali_data, vali_loader)
            # test_mae_denorm = self.calculate_denorm_mae(test_data, test_loader)

            # print("Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f}".format(
            #    epoch + 1, train_loss, vali_loss, test_loss))
            # print("反标准化 MAE - Vali: {0:.7f} Test: {1:.7f}".format(vali_mae_denorm, test_mae_denorm))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != "TST":
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print("Updating learning rate to {}".format(scheduler.get_last_lr()[0]))

        best_model_path = path + "/" + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        loss_df = pd.DataFrame(
            {"train_loss": train_loss_history, "validation_loss": vali_loss_history}
        )
        csv_path = os.path.join(path, "loss_history.csv")
        loss_df.to_csv(csv_path, index=False)
        print(f"损失历史已保存至: {csv_path}")

        # 绘制并保存损失曲线图
        plt.figure()
        plt.plot(
            range(1, len(train_loss_history) + 1),
            train_loss_history,
            label="Train Loss",
        )
        plt.plot(
            range(1, len(vali_loss_history) + 1),
            vali_loss_history,
            label="Validation Loss",
        )
        plt.title("Model Loss vs. Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.legend()
        plt.grid(True)

        # 将图片保存到检查点目录下
        loss_curve_path = os.path.join(path, "loss_curve.pdf")
        plt.savefig(loss_curve_path)
        plt.close()  # 释放内存
        print(f"损失曲线图已保存至: {loss_curve_path}")

        if getattr(self.args, "use_freqmoe", False):
            self._finalize_freqmoe_stats()

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        # 因为test数据过少，添加vali数据验证模型的预测效果
        vali_data, vali_loader = self._get_data(flag="val")

        if test:
            print("loading model")
            self.model.load_state_dict(
                torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth"))
            )

        preds = []
        trues = []
        inputx = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        moe_prev_mode = None
        freq_prev_mode = None
        if self.moe_enhancer is not None:
            moe_prev_mode = self.moe_enhancer.training
            self.moe_enhancer.train(False)
        if self.freq_moe_enhancer is not None:
            freq_prev_mode = self.freq_moe_enhancer.training
            self.freq_moe_enhancer.train(False)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                batch_x_fft, dec_inp_fft = self._prepare_model_inputs(batch_x, dec_inp)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark
                        )
                else:
                    outputs = self._forward_model(
                        batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark
                    )

                f_dim = -1 if self.args.features == "MS" else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

                # if i % 5 == 0:   # 每5个batch生成一张图
                # 只处理每个batch的第一个样本 [0]
                #   input_data = batch_x.detach().cpu().numpy()
                #   num_features = pred.shape[-1] # 获取特征数量

                # 遍历所有特征并为每个特征生成一张图
                #   for feature_idx in range(num_features):
                # 拼接历史输入和未来真实值
                #       gt = np.concatenate((input_data[0, :, feature_idx], true[0, :, feature_idx]), axis=0)
                # 拼接历史输入和未来预测值
                #       pd = np.concatenate((input_data[0, :, feature_idx], pred[0, :, feature_idx]), axis=0)

                # 为每个特征生成一个带清晰名称的PDF文件
                #       file_path = os.path.join(folder_path, f'feature_{feature_idx}_comparison_{i}.pdf')
                #       visual(gt, pd, file_path)
                #       print(f"已生成特征 {feature_idx} 的可视化图片: {file_path}")

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if getattr(self.args, "check_self_correlation", False):
            try:
                self._check_self_correlation(preds, inputx, folder_path)
            except Exception as exc:
                print(f"偏相关分析失败: {exc}")

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)

        # 通道维度误差用于逐通道指标输出
        mae_channels = np.mean(np.abs(preds - trues), axis=(0, 1))
        mse_channels = np.mean((preds - trues) ** 2, axis=(0, 1))
        channel_labels = (
            ["x", "y", "z"]
            if preds.shape[-1] == 3
            else [f"channel_{i}" for i in range(preds.shape[-1])]
        )

        print("标准化后 - mse:{}, mae:{}, rse:{}".format(mse, mae, rse))
        for label, ch_mse, ch_mae in zip(channel_labels, mse_channels, mae_channels):
            print(f"标准化后 - {label}: mse={ch_mse:.6f}, mae={ch_mae:.6f}")
        # 新增：反标准化后的指标
        try:
            if hasattr(test_data, "inverse_transform"):
                shape = preds.shape
                preds_2d = preds.reshape(-1, shape[-1])
                trues_2d = trues.reshape(-1, shape[-1])
                preds_denorm = test_data.inverse_transform(preds_2d).reshape(shape)
                trues_denorm = test_data.inverse_transform(trues_2d).reshape(shape)
                print(f"preds_denorm shape: {preds_denorm.shape}")

                (
                    mae_denorm,
                    mse_denorm,
                    rmse_denorm,
                    mape_denorm,
                    mspe_denorm,
                    rse_denorm,
                    corr_denorm,
                ) = metric(preds_denorm, trues_denorm)
                mae_denorm_channels = np.mean(
                    np.abs(preds_denorm - trues_denorm), axis=(0, 1)
                )
                mse_denorm_channels = np.mean(
                    (preds_denorm - trues_denorm) ** 2, axis=(0, 1)
                )

                print(
                    "反标准化后 - mse:{}, mae:{}, rse:{}".format(
                        mse_denorm, mae_denorm, rse_denorm
                    )
                )
                for label, ch_mse, ch_mae in zip(
                    channel_labels, mse_denorm_channels, mae_denorm_channels
                ):
                    print(f"反标准化后 - {label}: mse={ch_mse:.6f}, mae={ch_mae:.6f}")

                # 新增：对反标准化之后的结果画图
                input_shape = inputx.shape
                inputx_denorm = test_data.inverse_transform(
                    inputx.reshape(-1, input_shape[-1])
                ).reshape(input_shape)

                # 2. 循环绘制前5个样本的图像
                num_plots = min(5, preds_denorm.shape[0])  # 最多画5张图
                if num_plots < preds_denorm.shape[0]:
                    indices = np.linspace(
                        0, preds_denorm.shape[0] - 1, num_plots, dtype=int
                    )
                else:
                    indices = range(num_plots)

                for plot_idx, i in enumerate(indices):
                    num_features = preds_denorm.shape[-1]
                    for feature_idx in range(num_features):
                        gt = np.concatenate(
                            (
                                inputx_denorm[i, :, feature_idx],
                                trues_denorm[i, :, feature_idx],
                            ),
                            axis=0,
                        )
                        pd = np.concatenate(
                            (
                                inputx_denorm[i, :, feature_idx],
                                preds_denorm[i, :, feature_idx],
                            ),
                            axis=0,
                        )
                        file_path = os.path.join(
                            folder_path,
                            f"denorm_feature_{feature_idx}_sample_{plot_idx}.pdf",
                        )
                        visual(gt, pd, file_path)
                # for i in range(num_plots):
                #   num_features = preds_denorm.shape[-1]
                #   for feature_idx in range(num_features):
                # 拼接历史输入和未来真实值
                #       gt = np.concatenate((inputx_denorm[i, :, feature_idx], trues_denorm[i, :, feature_idx]), axis=0)
                # 拼接历史输入和未来预测值
                #       pd = np.concatenate((inputx_denorm[i, :, feature_idx], preds_denorm[i, :, feature_idx]), axis=0)

                # 为每个特征生成一个带清晰名称的PDF文件
                #       file_path = os.path.join(folder_path, f'denorm_feature_{feature_idx}_sample_{i}.pdf')
                #       visual(gt, pd, file_path)
                print(
                    f"已生成 {num_plots} 个样本的反标准化后可视化图片。一共有{preds_denorm.shape[0]}个样本。"
                )

                # 同时写入文件
                with open("result.txt", "a") as f:
                    f.write(setting + "  \n")
                    f.write("标准化后 - mse:{}, mae:{}, rse:{}\n".format(mse, mae, rse))
                    for label, ch_mse, ch_mae in zip(
                        channel_labels, mse_channels, mae_channels
                    ):
                        f.write(
                            "标准化后 - {}: mse:{:.6f}, mae:{:.6f}\n".format(
                                label, ch_mse, ch_mae
                            )
                        )
                    f.write(
                        "反标准化后 - mse:{}, mae:{}, rse:{}\n".format(
                            mse_denorm, mae_denorm, rse_denorm
                        )
                    )
                    for label, ch_mse, ch_mae in zip(
                        channel_labels, mse_denorm_channels, mae_denorm_channels
                    ):
                        f.write(
                            "反标准化后 - {}: mse:{:.6f}, mae:{:.6f}\n".format(
                                label, ch_mse, ch_mae
                            )
                        )
                    f.write("\n")
            else:
                print("数据集不支持反标准化")
        except Exception as e:
            print(f"反标准化计算失败: {e}")
            # 保持原有的文件写入逻辑
            with open("result.txt", "a") as f:
                f.write(setting + "  \n")
                f.write("标准化后 - mse:{}, mae:{}, rse:{}\n".format(mse, mae, rse))
                for label, ch_mse, ch_mae in zip(
                    channel_labels, mse_channels, mae_channels
                ):
                    f.write(
                        "标准化后 - {}: mse:{:.6f}, mae:{:.6f}\n".format(
                            label, ch_mse, ch_mae
                        )
                    )
                f.write("\n")

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + "pred.npy", preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)

        # 输出完整拼接数据（输入+预测）
        if getattr(self.args, "output_data_print", False):
            try:
                self._output_concatenated_data(
                    inputx, preds, trues, test_data, folder_path
                )
            except Exception as e:
                print(f"输出拼接数据失败: {e}")

        if self.moe_enhancer is not None:
            self.moe_enhancer.train(
                moe_prev_mode if moe_prev_mode is not None else True
            )
        if self.freq_moe_enhancer is not None:
            self.freq_moe_enhancer.train(
                freq_prev_mode if freq_prev_mode is not None else True
            )
        return

    def _output_concatenated_data(self, inputx, preds, trues, test_data, folder_path):
        """
        输出连续的预测数据到CSV文件。

        由于测试集的采样方式是滑动窗口（步长=1），相邻样本的预测区间大部分重叠。
        为了输出连续的预测序列：
        - 第一个样本：取完整的 seq_len 输入 + pred_len 预测
        - 后续样本：只取预测的最后1个时间步（因为每次窗口只滑动1步）

        这样可以得到一条从测试集起点到终点的连续序列。
        """
        num_samples = preds.shape[0]
        seq_len = inputx.shape[1]
        pred_len = preds.shape[1]
        num_channels = preds.shape[-1]
        channel_labels = (
            ["x", "y", "z"]
            if num_channels == 3
            else [f"channel_{i}" for i in range(num_channels)]
        )

        # 反标准化（如果支持）
        if hasattr(test_data, "inverse_transform"):
            input_shape = inputx.shape
            inputx_denorm = test_data.inverse_transform(
                inputx.reshape(-1, input_shape[-1])
            ).reshape(input_shape)
            preds_denorm = test_data.inverse_transform(
                preds.reshape(-1, num_channels)
            ).reshape(preds.shape)
            trues_denorm = test_data.inverse_transform(
                trues.reshape(-1, num_channels)
            ).reshape(trues.shape)
            use_denorm = True
        else:
            inputx_denorm = inputx
            preds_denorm = preds
            trues_denorm = trues
            use_denorm = False

        # 构建连续序列
        # 第一个样本的输入部分（seq_len 个时间步）
        continuous_input = inputx_denorm[0, :, :]  # [seq_len, channels]

        # 预测部分：第一个样本取完整 pred_len，后续样本只取最后1步
        # 总预测长度 = pred_len + (num_samples - 1) * 1
        continuous_pred = []
        continuous_true = []

        # 第一个样本的完整预测
        continuous_pred.append(preds_denorm[0, :, :])  # [pred_len, channels]
        continuous_true.append(trues_denorm[0, :, :])

        # 后续样本只取最后一个时间步
        for i in range(1, num_samples):
            continuous_pred.append(preds_denorm[i, -1:, :])  # [1, channels]
            continuous_true.append(trues_denorm[i, -1:, :])

        continuous_pred = np.concatenate(
            continuous_pred, axis=0
        )  # [pred_len + num_samples - 1, channels]
        continuous_true = np.concatenate(continuous_true, axis=0)

        # 生成CSV数据
        all_data = []
        total_input_len = continuous_input.shape[0]
        total_pred_len = continuous_pred.shape[0]

        # 输入部分
        for t in range(total_input_len):
            row = {"global_time_step": t, "phase": "input"}
            for ch_idx, ch_label in enumerate(channel_labels):
                row[f"{ch_label}_value"] = continuous_input[t, ch_idx]
                row[f"{ch_label}_true"] = continuous_input[t, ch_idx]
                row[f"{ch_label}_pred"] = np.nan
            all_data.append(row)

        # 预测部分
        for t in range(total_pred_len):
            row = {"global_time_step": total_input_len + t, "phase": "prediction"}
            for ch_idx, ch_label in enumerate(channel_labels):
                row[f"{ch_label}_value"] = continuous_pred[t, ch_idx]
                row[f"{ch_label}_true"] = continuous_true[t, ch_idx]
                row[f"{ch_label}_pred"] = continuous_pred[t, ch_idx]
            all_data.append(row)

        # 保存为CSV
        df = pd.DataFrame(all_data)
        suffix = "_denorm" if use_denorm else "_norm"
        csv_path = os.path.join(folder_path, f"continuous_prediction{suffix}.csv")
        df.to_csv(csv_path, index=False)

        print(f"已输出连续预测数据至: {csv_path}")
        print(
            f"  总长度: {total_input_len} (输入) + {total_pred_len} (预测) = {total_input_len + total_pred_len}"
        )
        print(f"  原始样本数: {num_samples}, seq_len: {seq_len}, pred_len: {pred_len}")

    def _check_self_correlation(self, preds, inputs, folder_path):
        channel_index = int(getattr(self.args, "self_corr_channel", 0))
        if channel_index < 0 or channel_index >= preds.shape[-1]:
            raise ValueError(
                f"self_corr_channel={channel_index} 超出范围, 合法范围为 0 到 {preds.shape[-1] - 1}"
            )

        series_matrix = preds[:, :, channel_index].astype(np.float64)
        inputs = inputs.astype(np.float64)

        if series_matrix.shape[0] < 2:
            raise ValueError("样本数量不足，无法计算 DML 偏相关矩阵")

        if inputs.shape[0] != series_matrix.shape[0]:
            raise ValueError(
                f"预测样本数 {series_matrix.shape[0]} 与输入样本数 {inputs.shape[0]} 不匹配"
            )

        if inputs.ndim != 3:
            raise ValueError("输入张量必须为三维: [samples, seq_len, channels]")
        if channel_index >= inputs.shape[-1]:
            raise ValueError(
                f"输入张量的通道数为 {inputs.shape[-1]}，无法索引 channel {channel_index}"
            )

        base_series = inputs[:, -self.args.seq_len :, channel_index]
        design_matrix = base_series.reshape(base_series.shape[0], -1)

        corr_matrix = self._compute_dml_partial_correlation(
            series_matrix, design_matrix
        )

        matrix_path = os.path.join(
            folder_path, f"self_correlation_channel{channel_index}.csv"
        )
        np.savetxt(matrix_path, corr_matrix, delimiter=",")

        heatmap_path = os.path.join(
            folder_path, f"self_correlation_channel{channel_index}.png"
        )
        self._plot_correlation_heatmap(
            corr_matrix, heatmap_path, channel_index, title_suffix="原始"
        )

        # FFT-based analysis (real & imaginary parts)
        fft_complex = np.fft.fft(series_matrix, axis=1) / np.sqrt(
            series_matrix.shape[1]
        )
        fft_real_full = np.real(fft_complex)
        fft_imag_full = np.imag(fft_complex)

        # 偏相关分析仅保留正频部分，避免共轭对称下的重复信息
        fft_horizon = max(1, series_matrix.shape[1] // 2)
        fft_real = fft_real_full[:, :fft_horizon]
        fft_imag = fft_imag_full[:, :fft_horizon]

        corr_matrix_fft_real = self._compute_dml_partial_correlation(
            fft_real, design_matrix
        )
        corr_matrix_fft_imag = self._compute_dml_partial_correlation(
            fft_imag, design_matrix
        )

        fft_real_matrix_path = os.path.join(
            folder_path, f"self_correlation_fft_real_channel{channel_index}.csv"
        )
        np.savetxt(fft_real_matrix_path, corr_matrix_fft_real, delimiter=",")
        fft_imag_matrix_path = os.path.join(
            folder_path, f"self_correlation_fft_imag_channel{channel_index}.csv"
        )
        np.savetxt(fft_imag_matrix_path, corr_matrix_fft_imag, delimiter=",")

        fft_real_heatmap_path = os.path.join(
            folder_path, f"self_correlation_fft_real_channel{channel_index}.png"
        )
        self._plot_correlation_heatmap(
            corr_matrix_fft_real,
            fft_real_heatmap_path,
            channel_index,
            title_suffix="FFT-Real",
        )
        fft_imag_heatmap_path = os.path.join(
            folder_path, f"self_correlation_fft_imag_channel{channel_index}.png"
        )
        self._plot_correlation_heatmap(
            corr_matrix_fft_imag,
            fft_imag_heatmap_path,
            channel_index,
            title_suffix="FFT-Imag",
        )

        print(f"已保存 DML 偏相关矩阵至: {matrix_path}")
        print(f"已保存 DML 偏相关热力图至: {heatmap_path}")
        print(f"已保存 FFT 实部偏相关矩阵至: {fft_real_matrix_path}")
        print(f"已保存 FFT 实部偏相关热力图至: {fft_real_heatmap_path}")
        print(f"已保存 FFT 虚部偏相关矩阵至: {fft_imag_matrix_path}")
        print(f"已保存 FFT 虚部偏相关热力图至: {fft_imag_heatmap_path}")

    def _compute_dml_partial_correlation(self, series_matrix, design_matrix):
        samples, horizon = series_matrix.shape
        residuals = np.zeros_like(series_matrix, dtype=np.float64)
        design_matrix = np.asarray(design_matrix, dtype=np.float64)

        if design_matrix.shape[0] != samples:
            raise ValueError("Design matrix 行数必须与样本数一致")

        for idx in range(horizon):
            y = series_matrix[:, idx]
            model = LinearRegression()
            model.fit(design_matrix, y)
            residuals[:, idx] = y - model.predict(design_matrix)

        corr_matrix = np.eye(horizon, dtype=np.float64)
        for i in range(horizon):
            for j in range(i, horizon):
                t_res = residuals[:, j]
                y_res = residuals[:, i]
                denom = np.dot(t_res, t_res)

                if denom <= 1e-12:
                    beta = 0.0
                else:
                    beta = np.dot(t_res, y_res) / denom

                if beta > 1.0:
                    beta = 1.0
                elif beta < -1.0:
                    beta = -1.0

                corr_matrix[i, j] = beta

        return corr_matrix

    def _plot_correlation_heatmap(
        self, corr_matrix, file_path, channel_index, title_suffix
    ):
        finite_vals = corr_matrix[np.isfinite(corr_matrix)]
        vmax = np.max(np.abs(finite_vals)) if finite_vals.size > 0 else 1.0
        if vmax <= 1e-6:
            vmax = 1.0

        plt.figure(figsize=(8, 6))
        im = plt.imshow(
            corr_matrix, cmap="RdBu_r", origin="lower", vmin=-vmax, vmax=vmax
        )
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.title(f"DML 偏相关热力图 ({title_suffix}, channel {channel_index})")
        plt.xlabel("时间步 j")
        plt.ylabel("时间步 i")
        plt.tight_layout()
        plt.savefig(file_path, dpi=300)
        plt.close()

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        moe_prev_mode = None
        freq_prev_mode = None
        if self.moe_enhancer is not None:
            moe_prev_mode = self.moe_enhancer.training
            self.moe_enhancer.train(False)
        if self.freq_moe_enhancer is not None:
            freq_prev_mode = self.freq_moe_enhancer.training
            self.freq_moe_enhancer.train(False)
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = (
                    torch.zeros(
                        [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                    )
                    .float()
                    .to(batch_y.device)
                )
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                # encoder - decoder
                batch_x_fft, dec_inp_fft = self._prepare_model_inputs(batch_x, dec_inp)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(
                            batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark
                        )
                else:
                    outputs = self._forward_model(
                        batch_x_fft, batch_x_mark, dec_inp_fft, batch_y_mark
                    )
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)
        if self.moe_enhancer is not None:
            self.moe_enhancer.train(
                moe_prev_mode if moe_prev_mode is not None else True
            )
        if self.freq_moe_enhancer is not None:
            self.freq_moe_enhancer.train(
                freq_prev_mode if freq_prev_mode is not None else True
            )

        return
