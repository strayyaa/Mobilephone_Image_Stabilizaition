import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Autoformer & Transformer family for Time Series Forecasting"
    )

    # random seed
    parser.add_argument("--random_seed", type=int, default=2021, help="random seed")

    # basic config
    parser.add_argument(
        "--is_training", type=int, required=True, default=1, help="status"
    )
    parser.add_argument(
        "--model_id", type=str, required=True, default="test", help="model id"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        default="Autoformer",
        help="model name, options: [Autoformer, Informer, Transformer]",
    )

    # data loader
    parser.add_argument(
        "--data", type=str, required=True, default="ETTm1", help="dataset type"
    )
    parser.add_argument(
        "--root_path",
        type=str,
        default="./data/ETT/",
        help="root path of the data file",
    )
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="data file")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        help="forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate",
    )
    parser.add_argument(
        "--target", type=str, default="OT", help="target feature in S or MS task"
    )
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h",
    )
    # FFT data enhancement
    parser.add_argument(
        "--use_fft_enhance_data",
        action="store_true",
        default=False,
        help="在训练/推理阶段对输入序列执行FFT增强（保持原始数据用于评估）。",
    )
    parser.add_argument(
        "--fft_low_freq_ratio", type=float, default=0.25, help="FFT增强：低频阈值占比"
    )
    parser.add_argument(
        "--fft_high_freq_ratio", type=float, default=0.7, help="FFT增强：高频阈值占比"
    )
    parser.add_argument(
        "--fft_cutoff_ratio", type=float, default=0.9, help="FFT增强：高频截断占比"
    )
    parser.add_argument(
        "--fft_low_freq_boost", type=float, default=1.3, help="FFT增强：低频增益"
    )
    parser.add_argument(
        "--fft_mid_freq_boost", type=float, default=1.1, help="FFT增强：中频增益"
    )
    parser.add_argument(
        "--fft_high_freq_suppress", type=float, default=0.2, help="FFT增强：高频抑制"
    )
    parser.add_argument(
        "--fft_residual_ratio",
        type=float,
        default=0.5,
        help="FFT增强：输出与原始信号的混合比例 (0~1)",
    )
    parser.add_argument(
        "--fft_reflection_pad",
        type=int,
        default=32,
        help="FFT增强：反射填充长度以减轻边界伪影",
    )

    # MoE增强参数
    parser.add_argument(
        "--use_moe",
        action="store_true",
        default=False,
        help="启用基于频段与模态的MoE数据增强",
    )
    parser.add_argument(
        "--moe_low_freq_ratio", type=float, default=0.005, help="MoE频段划分：低频比例"
    )
    parser.add_argument(
        "--moe_high_freq_ratio",
        type=float,
        default=0.01,
        help="MoE频段划分：中频起始比例",
    )
    parser.add_argument(
        "--moe_cutoff_ratio", type=float, default=0.02, help="MoE频段截断比例"
    )
    parser.add_argument(
        "--moe_frequency_residual_ratio",
        type=float,
        default=0.7,
        help="MoE频段残差注入系数",
    )
    parser.add_argument(
        "--moe_modal_residual_ratio",
        type=float,
        default=0.7,
        help="MoE模态融合残差注入系数（当前版本已由直接模态融合替代，仅保留参数以兼容旧脚本）",
    )
    parser.add_argument(
        "--moe_temperature", type=float, default=1.0, help="MoE softmax 温度"
    )
    parser.add_argument(
        "--moe_frequency_experts",
        type=int,
        default=3,
        help="FFT/DCT 频域专家数量 (>=3 时自动细分频段)",
    )
    parser.add_argument(
        "--moe_frequency_topk",
        type=int,
        default=0,
        help="保留能量最大的前K个频域专家 (0 表示使用全部)",
    )
    parser.add_argument(
        "--moe_energy_norm",
        type=str,
        default="density",
        help="频域专家能量归一化策略: sum 或 density",
    )

    # freqMoE增强参数
    parser.add_argument(
        "--use_freqmoe", action="store_true", default=False, help="启用freqMoE频域增强"
    )
    parser.add_argument(
        "--freqmoe_experts", type=int, default=3, help="freqMoE专家数量"
    )
    parser.add_argument(
        "--freqmoe_use_norm",
        type=int,
        default=1,
        help="1表示先归一化再还原，0表示直接处理原始信号",
    )
    parser.add_argument(
        "--freqmoe_use_soft_mask",
        type=int,
        default=1,
        help="使用可微的Soft Frequency Mask",
    )
    parser.add_argument(
        "--freqmoe_sharpness", type=float, default=20.0, help="Soft Mask的锐度参数"
    )
    parser.add_argument(
        "--freqmoe_freq_wise_routing", type=int, default=0, help="使用频点级路由"
    )
    parser.add_argument(
        "--freqmoe_balance_weight", type=float, default=0.01, help="负载均衡损失权重"
    )
    parser.add_argument(
        "--freqmoe_use_lowpass_filter",
        type=int,
        default=0,
        help="启用低通滤波器预处理（先滤高频，再在低频上做MoE）",
    )
    parser.add_argument(
        "--freqmoe_lowpass_cutoff",
        type=float,
        default=0.3,
        help="低通滤波器截止频率占比",
    )
    parser.add_argument(
        "--freqmoe_expert_freq_range",
        type=float,
        default=1.0,
        help="专家频段范围比例 (1.0=全频段, 0.3=仅30%低频区细粒度划分)",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="./checkpoints/",
        help="location of model checkpoints",
    )
    parser.add_argument(
        "--use_augmentation",
        action="store_true",
        help="use data augmentation for training set",
        default=True,
    )
    parser.add_argument(
        "--no_use_augmentation", dest="use_augmentation", action="store_false"
    )
    parser.add_argument(
        "--jitter_sigma",
        type=float,
        default=0.05,
        help="std dev for jittering augmentation",
    )
    parser.add_argument(
        "--scale_alpha", type=float, default=0.1, help="range for scaling augmentation"
    )
    parser.add_argument(
        "--use_smoothing",
        action="store_true",
        help="use data smoothing for training set",
        default=True,
    )
    parser.add_argument(
        "--no_use_smoothing", dest="use_smoothing", action="store_false"
    )
    parser.add_argument(
        "--downsample_rate", type=int, default=2, help="downsample_rate"
    )

    # forecasting task
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument(
        "--pred_len", type=int, default=96, help="prediction sequence length"
    )

    # evaluation extras
    parser.add_argument(
        "--check_self_correlation",
        action="store_true",
        default=False,
        help="compute DML-based self correlation on test predictions",
    )
    parser.add_argument(
        "--self_corr_channel",
        type=int,
        default=0,
        help="output channel index for self-correlation analysis",
    )
    parser.add_argument(
        "--output_data_print",
        action="store_true",
        default=False,
        help="输出完整的预测数据（输入+预测拼接）到CSV文件",
    )

    # DLinear variants
    parser.add_argument(
        "--dlinear_xyz_use_pos_encoding",
        action="store_true",
        help="Enable positional encoding in DLinear_XYZ",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_arima_trend",
        action="store_true",
        help="Use ARIMA trend decomposition in DLinear_XYZ",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_arima_trend_comp",
        action="store_true",
        help="Replace trend head with ARIMA forecast in DLinear_XYZ",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_fft_promote_feature",
        action="store_true",
        help="Fuse FFT-based features into encoder input",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_fft_plus_feature",
        action="store_true",
        help="Add FFT-based projection at output stage",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_ifft_plus",
        action="store_true",
        help="Use learnable inverse FFT branch",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_merge",
        action="store_true",
        help="Learn a merge factor between seasonal and trend components",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_broader_individual",
        action="store_true",
        help="Apply wider shared heads for seasonal/trend",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_mix_linear",
        action="store_true",
        help="Apply channel mixing after prediction",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_autoformer",
        action="store_true",
        help="Cascade AutoCorrelation blocks in the seasonal branch",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_use_full_autoformer",
        action="store_true",
        help="Use encoder-style autocorrelation stacks",
        default=False,
    )
    parser.add_argument(
        "--dlinear_xyz_top_k_fft",
        type=int,
        default=64,
        help="Number of Fourier coefficients to keep",
    )
    parser.add_argument(
        "--dlinear_xyz_top_k_corr",
        type=int,
        default=4,
        help="Top-k lags for autocorrelation selection",
    )
    parser.add_argument(
        "--dlinear_xyz_layers_cnt",
        type=int,
        default=1,
        help="Autocorrelation layer repeats in decoder branch",
    )
    parser.add_argument(
        "--dlinear_xyz_layers_cnt_encoder",
        type=int,
        default=1,
        help="Autocorrelation layer repeats in encoder branch",
    )
    parser.add_argument(
        "--dlinear_xyz_arima_order",
        type=str,
        default="(2,1,0)",
        help="ARIMA order (p,d,q) for trend modelling",
    )
    parser.add_argument(
        "--dlinear_xyz_arima_moving_avg",
        type=int,
        default=5,
        help="Fallback moving-average window for ARIMA failures",
    )
    parser.add_argument(
        "--dlinear_xyz_merge_rate_init",
        type=float,
        default=0.3,
        help="Initial merge rate for frequency fusion",
    )
    parser.add_argument(
        "--dlinear_xyz_merge_between_init",
        type=float,
        default=1.0,
        help="Initial merge rate between seasonal and trend components",
    )

    # PatchTST
    parser.add_argument(
        "--fc_dropout", type=float, default=0.05, help="fully connected dropout"
    )
    parser.add_argument("--head_dropout", type=float, default=0.0, help="head dropout")
    parser.add_argument("--patch_len", type=int, default=16, help="patch length")
    parser.add_argument("--stride", type=int, default=8, help="stride")
    parser.add_argument(
        "--padding_patch", default="end", help="None: None; end: padding on the end"
    )
    parser.add_argument("--revin", type=int, default=1, help="RevIN; True 1 False 0")
    parser.add_argument(
        "--affine", type=int, default=0, help="RevIN-affine; True 1 False 0"
    )
    parser.add_argument(
        "--subtract_last",
        type=int,
        default=0,
        help="0: subtract mean; 1: subtract last",
    )
    parser.add_argument(
        "--decomposition", type=int, default=0, help="decomposition; True 1 False 0"
    )
    parser.add_argument(
        "--kernel_size", type=int, default=25, help="decomposition-kernel"
    )
    parser.add_argument(
        "--individual", type=int, default=0, help="individual head; True 1 False 0"
    )

    # ARIMA / Prophet baselines
    parser.add_argument("--arima_order_p", type=int, default=2, help="ARIMA p order")
    parser.add_argument("--arima_order_d", type=int, default=0, help="ARIMA d order")
    parser.add_argument("--arima_order_q", type=int, default=1, help="ARIMA q order")
    parser.add_argument(
        "--arima_maxiter",
        type=int,
        default=200,
        help="Maximum iterations for ARIMA solver",
    )
    parser.add_argument(
        "--arima_enforce_stationarity",
        action="store_true",
        default=False,
        help="Force ARIMA stationarity constraint",
    )
    parser.add_argument(
        "--arima_enforce_invertibility",
        action="store_true",
        default=False,
        help="Force ARIMA invertibility constraint",
    )
    parser.add_argument(
        "--arima_fallback_window",
        type=int,
        default=5,
        help="Moving-average window when ARIMA fails",
    )

    parser.add_argument(
        "--prophet_changepoint_prior_scale",
        type=float,
        default=0.05,
        help="Prophet changepoint prior scale",
    )
    parser.add_argument(
        "--prophet_seasonality_mode",
        type=str,
        default="additive",
        help="Prophet seasonality mode",
    )
    parser.add_argument(
        "--prophet_growth", type=str, default="linear", help="Prophet growth type"
    )
    parser.add_argument(
        "--prophet_n_changepoints",
        type=int,
        default=25,
        help="Prophet number of changepoints",
    )
    parser.add_argument(
        "--prophet_yearly_seasonality",
        type=str,
        default="auto",
        help="Prophet yearly seasonality setting",
    )
    parser.add_argument(
        "--prophet_weekly_seasonality",
        type=str,
        default="auto",
        help="Prophet weekly seasonality setting",
    )
    parser.add_argument(
        "--prophet_daily_seasonality",
        type=str,
        default="auto",
        help="Prophet daily seasonality setting",
    )
    parser.add_argument(
        "--prophet_fallback_window",
        type=int,
        default=5,
        help="Moving-average window when Prophet fails",
    )

    # Formers
    parser.add_argument(
        "--embed_type",
        type=int,
        default=0,
        help="0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding",
    )
    parser.add_argument(
        "--enc_in", type=int, default=7, help="encoder input size"
    )  # DLinear with --individual, use this hyperparameter as the number of channels
    parser.add_argument("--dec_in", type=int, default=7, help="decoder input size")
    parser.add_argument("--c_out", type=int, default=7, help="output size")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument(
        "--moving_avg", type=int, default=25, help="window size of moving average"
    )
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument(
        "--distil",
        action="store_false",
        help="whether to use distilling in encoder, using this argument means not using distilling",
        default=True,
    )
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout")
    parser.add_argument(
        "--embed",
        type=str,
        default="timeF",
        help="time features encoding, options:[timeF, fixed, learned]",
    )
    parser.add_argument("--activation", type=str, default="gelu", help="activation")
    parser.add_argument(
        "--output_attention",
        action="store_true",
        help="whether to output attention in ecoder",
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="whether to predict unseen future data",
    )

    # ModernTCN settings
    parser.add_argument(
        "--stem_ratio", type=int, default=6, help="ModernTCN stem expansion ratio"
    )
    parser.add_argument(
        "--downsample_ratio",
        type=int,
        default=2,
        help="ModernTCN temporal downsample ratio",
    )
    parser.add_argument(
        "--ffn_ratio",
        type=int,
        default=2,
        help="ModernTCN feed-forward expansion ratio",
    )
    parser.add_argument(
        "--patch_size", type=int, default=16, help="ModernTCN patch size"
    )
    parser.add_argument(
        "--patch_stride", type=int, default=8, help="ModernTCN patch stride"
    )
    parser.add_argument(
        "--num_blocks",
        nargs="+",
        type=int,
        default=[1, 1, 1, 1],
        help="Blocks per ModernTCN stage",
    )
    parser.add_argument(
        "--large_size",
        nargs="+",
        type=int,
        default=[31, 29, 27, 13],
        help="Large kernel sizes per stage",
    )
    parser.add_argument(
        "--small_size",
        nargs="+",
        type=int,
        default=[5, 5, 5, 5],
        help="Small kernel sizes per stage",
    )
    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=[256, 256, 256, 256],
        help="Hidden dims per stage",
    )
    parser.add_argument(
        "--small_kernel_merged",
        action="store_true",
        default=False,
        help="Use merged small kernels in ModernTCN",
    )
    parser.add_argument(
        "--use_multi_scale",
        type=int,
        default=1,
        help="Enable ModernTCN multi-scale fusion (1/0)",
    )

    # optimization
    parser.add_argument(
        "--num_workers", type=int, default=10, help="data loader num workers"
    )
    parser.add_argument("--itr", type=int, default=2, help="experiments times")
    parser.add_argument("--train_epochs", type=int, default=100, help="train epochs")
    parser.add_argument(
        "--batch_size", type=int, default=128, help="batch size of train input data"
    )
    parser.add_argument(
        "--patience", type=int, default=100, help="early stopping patience"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="optimizer learning rate"
    )
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="mse", help="loss function")
    parser.add_argument(
        "--lradj", type=str, default="type3", help="adjust learning rate"
    )
    parser.add_argument("--pct_start", type=float, default=0.3, help="pct_start")
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="use automatic mixed precision training",
        default=False,
    )

    # --- 新增 FreDF 损失权重参数 ---
    parser.add_argument(
        "--rec_lambda",
        type=float,
        default=1.0,
        help="weight for reconstruction loss (time domain)",
    )
    parser.add_argument(
        "--auxi_lambda",
        type=float,
        default=0.0,
        help="weight for auxiliary loss (frequency domain)",
    )
    # --- 新增结束 ---

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--use_multi_gpu", action="store_true", help="use multiple gpus", default=False
    )
    parser.add_argument(
        "--devices", type=str, default="0,1,2,3", help="device ids of multile gpus"
    )
    parser.add_argument(
        "--test_flop",
        action="store_true",
        default=False,
        help="See utils/tools for usage",
    )

    args = parser.parse_args()

    # random seed
    fix_seed = args.random_seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii,
            )

            exp = Exp(args)  # set experiments
            print(
                ">>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting)
            )
            exp.train(setting)

            print(
                ">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting)
            )
            exp.test(setting)

            if args.do_predict:
                print(
                    ">>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(
                        setting
                    )
                )
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}".format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.distil,
            args.des,
            ii,
        )

        exp = Exp(args)  # set experiments
        print(">>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
