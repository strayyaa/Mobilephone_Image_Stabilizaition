import math
import warnings
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "on"}
    return False


def _parse_order(value) -> Tuple[int, int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return int(value[0]), int(value[1]), int(value[2])
    if isinstance(value, str):
        parts = [p for p in value.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",") if p.strip()]
        if len(parts) == 3:
            return int(parts[0]), int(parts[1]), int(parts[2])
    return 2, 1, 0


class MovingAvg(nn.Module):
    def __init__(self, kernel_size: int, stride: int = 1) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_len = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad_len, 1)
        end = x[:, -1:, :].repeat(1, pad_len, 1)
        padded = torch.cat([front, x, end], dim=1)
        out = self.avg(padded.permute(0, 2, 1))
        return out.permute(0, 2, 1)


class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        moving_mean = self.moving_avg(x)
        seasonal = x - moving_mean
        return seasonal, moving_mean


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int) -> None:
        super().__init__()
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / max(d_model, 1)))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer("pe", pe.unsqueeze(0))
        self.learnable_pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.xavier_uniform_(self.learnable_pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :] + self.learnable_pe[:, : x.size(1), :]


class AutoCorrelation(nn.Module):
    def __init__(self, top_k: int = 3) -> None:
        super().__init__()
        self.top_k = max(top_k, 1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = q.shape
        q_fft = torch.fft.rfft(q, dim=1)
        k_fft = torch.fft.rfft(k, dim=1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=length, dim=1)
        corr_mean = corr.mean(dim=-1)
        topk_val, topk_idx = torch.topk(corr_mean, min(self.top_k, corr_mean.size(1)), dim=-1)
        attn = F.softmax(topk_val, dim=-1)
        output = torch.zeros_like(v)
        for i in range(attn.size(1)):
            shift = topk_idx[:, i]
            weight = attn[:, i].unsqueeze(-1).unsqueeze(-1)
            rolled = torch.stack([torch.roll(v[b], shifts=-int(shift[b].item()), dims=0) for b in range(bsz)], dim=0)
            output = output + rolled * weight
        return output


class AutoCorrelationLayer(nn.Module):
    def __init__(self, top_k_corr: int, kernel_size: int) -> None:
        super().__init__()
        self.auto_correlation = AutoCorrelation(top_k=top_k_corr)
        self.decomp = SeriesDecomp(kernel_size)

    def forward(self, seasonal: torch.Tensor, trend: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        temp = self.auto_correlation(seasonal, values, values)
        seasonal, delta = self.decomp(temp + seasonal)
        trend = trend + delta
        return seasonal, trend


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.individual = bool(getattr(configs, "individual", True))

        self.cfg = SimpleNamespace(
            use_pos_encoding=_to_bool(getattr(configs, "dlinear_xyz_use_pos_encoding", False)),
            use_arima_trend=_to_bool(getattr(configs, "dlinear_xyz_use_arima_trend", False)),
            use_arima_trend_comp=_to_bool(getattr(configs, "dlinear_xyz_use_arima_trend_comp", False)),
            use_fft_promote_feature=_to_bool(getattr(configs, "dlinear_xyz_use_fft_promote_feature", False)),
            use_fft_plus_feature=_to_bool(getattr(configs, "dlinear_xyz_use_fft_plus_feature", False)),
            use_ifft_while_using_fft_plus=_to_bool(getattr(configs, "dlinear_xyz_use_ifft_plus", False)),
            use_merge=_to_bool(getattr(configs, "dlinear_xyz_use_merge", False)),
            use_broader_individual=_to_bool(getattr(configs, "dlinear_xyz_use_broader_individual", False)),
            use_mix_linear=_to_bool(getattr(configs, "dlinear_xyz_use_mix_linear", False)),
            use_autoformer=_to_bool(getattr(configs, "dlinear_xyz_use_autoformer", False)),
            use_full_autoformer=_to_bool(getattr(configs, "dlinear_xyz_use_full_autoformer", False)),
            top_k_fft=int(getattr(configs, "dlinear_xyz_top_k_fft", min(self.seq_len // 2 + 1, 64))),
            top_k_corr=int(getattr(configs, "dlinear_xyz_top_k_corr", 4)),
            layers_cnt=int(getattr(configs, "dlinear_xyz_layers_cnt", 1)),
            layers_cnt_encoder=int(getattr(configs, "dlinear_xyz_layers_cnt_encoder", 1)),
            merge_rate_init=float(getattr(configs, "dlinear_xyz_merge_rate_init", 0.3)),
            merge_between_init=float(getattr(configs, "dlinear_xyz_merge_between_init", 1.0)),
            arima_order=_parse_order(getattr(configs, "dlinear_xyz_arima_order", (2, 1, 0))),
            arima_moving_avg=int(getattr(configs, "dlinear_xyz_arima_moving_avg", 5)),
        )

        kernel_size = getattr(configs, "moving_avg", 25)
        self.decomposition = SeriesDecomp(kernel_size)
        self.pos_encoder = PositionalEncoding(self.seq_len, self.channels)

        if self.cfg.use_fft_promote_feature:
            self.freq_projector = nn.Linear(self.channels, self.channels)
            self.freq_gate = nn.Parameter(torch.tensor(0.3))

        if self.cfg.use_fft_plus_feature:
            self.freq_linear = nn.Linear(self.cfg.top_k_fft, self.pred_len)

        if self.cfg.use_ifft_while_using_fft_plus:
            self.freq_linear_real = nn.Linear(self.cfg.top_k_fft, self.cfg.top_k_fft)
            self.freq_linear_imag = nn.Linear(self.cfg.top_k_fft, self.cfg.top_k_fft)

        if self.cfg.use_merge:
            self.merge_between = nn.Parameter(torch.tensor(self.cfg.merge_between_init))
        else:
            self.register_parameter("merge_between", None)

        if self.cfg.use_broader_individual and not self.individual:
            self.broader_individual_seasonal = nn.Linear(self.channels * self.seq_len, self.channels * self.pred_len)
            self.broader_individual_trend = nn.Linear(self.channels * self.seq_len, self.channels * self.pred_len)

        if self.cfg.use_mix_linear:
            self.mix_linear = nn.Linear(self.channels, self.channels)

        if self.cfg.use_autoformer:
            self.auto_correlation_layer = AutoCorrelationLayer(self.cfg.top_k_corr, kernel_size)
        else:
            self.auto_correlation_layer = None

        if self.cfg.use_full_autoformer:
            self.auto_correlation_layer_x = AutoCorrelationLayer(self.cfg.top_k_corr, kernel_size)
            self.auto_correlation_cross = AutoCorrelationLayer(self.cfg.top_k_corr, kernel_size)
        else:
            self.auto_correlation_layer_x = None
            self.auto_correlation_cross = None

        if self.cfg.use_fft_promote_feature:
            self.output = nn.Linear(self.channels, self.channels)
        else:
            self.output = None

        self.merge_rate = nn.Parameter(torch.tensor(self.cfg.merge_rate_init)) if (self.cfg.use_fft_plus_feature or self.cfg.use_ifft_while_using_fft_plus) else None

        if self.individual:
            self.linear_seasonal = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
            self.linear_trend = nn.ModuleList([nn.Linear(self.seq_len, self.pred_len) for _ in range(self.channels)])
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)

        self.register_parameter("dummy_param", nn.Parameter(torch.zeros(1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        freq_plus = None
        freq_ifft = None

        if self.cfg.use_pos_encoding:
            x = self.pos_encoder(x)

        if self.cfg.use_fft_promote_feature:
            freq_promote = self._compute_fft_promote_features(x)
            x = x + self.freq_gate * self.freq_projector(freq_promote)

        if self.cfg.use_fft_plus_feature:
            freq_plus = self._compute_fft_plus_features(x)

        if self.cfg.use_ifft_while_using_fft_plus:
            freq_ifft = self._compute_ifft_features(x)

        arima_trend_seq = None
        arima_trend_pred = None
        if self.cfg.use_arima_trend or self.cfg.use_arima_trend_comp:
            arima_trend_seq, arima_trend_pred = self._compute_arima_trend(x)

        seasonal_init, trend_init = self.decomposition(x)

        if self.cfg.use_arima_trend and arima_trend_seq is not None:
            seasonal_init = x - arima_trend_seq
            trend_init = arima_trend_seq

        if self.cfg.use_full_autoformer and self.auto_correlation_layer_x is not None:
            for _ in range(max(self.cfg.layers_cnt_encoder, 1)):
                seasonal_init, _ = self.auto_correlation_layer_x(seasonal_init, seasonal_init, seasonal_init)

        if self.cfg.use_autoformer and self.auto_correlation_layer is not None:
            for _ in range(max(self.cfg.layers_cnt, 1)):
                seasonal_init, trend_init = self.auto_correlation_layer(seasonal_init, trend_init, seasonal_init)
                if self.cfg.use_full_autoformer and self.auto_correlation_cross is not None:
                    seasonal_init, trend_init = self.auto_correlation_cross(seasonal_init, trend_init, x)

        seasonal_init = seasonal_init.permute(0, 2, 1)
        trend_init = trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros(seasonal_init.size(0), self.channels, self.pred_len, dtype=seasonal_init.dtype, device=seasonal_init.device)
            trend_output = torch.zeros_like(seasonal_output)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.linear_seasonal[i](seasonal_init[:, i, :])
                if self.cfg.use_arima_trend_comp and arima_trend_pred is not None:
                    trend_output[:, i, :] = arima_trend_pred[:, :, i]
                else:
                    trend_output[:, i, :] = self.linear_trend[i](trend_init[:, i, :])
        else:
            if self.cfg.use_broader_individual:
                seasonal_flat = seasonal_init.reshape(seasonal_init.size(0), -1)
                trend_flat = trend_init.reshape(trend_init.size(0), -1)
                seasonal_output = self.broader_individual_seasonal(seasonal_flat).reshape(seasonal_init.size(0), self.channels, self.pred_len)
                if self.cfg.use_arima_trend_comp and arima_trend_pred is not None:
                    trend_output = arima_trend_pred.permute(0, 2, 1)
                else:
                    trend_output = self.broader_individual_trend(trend_flat).reshape(trend_init.size(0), self.channels, self.pred_len)
            else:
                seasonal_output = self.linear_seasonal(seasonal_init)
                if self.cfg.use_arima_trend_comp and arima_trend_pred is not None:
                    trend_output = arima_trend_pred.permute(0, 2, 1)
                else:
                    trend_output = self.linear_trend(trend_init)

        if self.cfg.use_merge and self.merge_between is not None:
            x_out = self.merge_between * seasonal_output + (1.0 - self.merge_between) * trend_output
        else:
            x_out = seasonal_output + trend_output

        if self.cfg.use_fft_plus_feature and freq_plus is not None and self.merge_rate is not None:
            x_out = (2.0 - self.merge_rate) * x_out + self.merge_rate * freq_plus

        if self.cfg.use_ifft_while_using_fft_plus and freq_ifft is not None and self.merge_rate is not None:
            x_out = (2.0 - self.merge_rate) * x_out + self.merge_rate * freq_ifft

        if self.cfg.use_mix_linear:
            x_out = self.mix_linear(x_out.permute(0, 2, 1)).permute(0, 2, 1)

        if self.cfg.use_fft_promote_feature and self.output is not None:
            x_out = self.output(x_out.permute(0, 2, 1)).permute(0, 2, 1)

        return x_out.permute(0, 2, 1)

    def _compute_fft_promote_features(self, x: torch.Tensor) -> torch.Tensor:
        freq_feats = torch.fft.rfft(x, dim=1)
        top_k = min(self.cfg.top_k_fft, freq_feats.size(1))
        mag = torch.abs(freq_feats[:, :top_k, :])
        mag = torch.log1p(mag)
        mag = (mag - mag.mean(dim=1, keepdim=True)) / (mag.std(dim=1, keepdim=True) + 1e-6)
        mag = F.interpolate(mag.permute(0, 2, 1), size=x.size(1), mode="linear", align_corners=False).permute(0, 2, 1)
        return mag

    def _compute_fft_plus_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        freq_feats = torch.fft.rfft(x, dim=1)
        top_k = min(self.cfg.top_k_fft, freq_feats.size(1))
        mag = torch.abs(freq_feats[:, :top_k, :])
        mag = torch.log1p(mag)
        mag = (mag - mag.mean(dim=1, keepdim=True)) / (mag.std(dim=1, keepdim=True) + 1e-6)
        mag = mag.permute(0, 2, 1)
        mag = self._match_last_dim(mag, self.cfg.top_k_fft)
        projected = self.freq_linear(mag)
        return projected.permute(0, 2, 1)

    def _compute_ifft_features(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        freq_feats = torch.fft.rfft(x, dim=1)
        top_k = min(self.cfg.top_k_fft, freq_feats.size(1))
        freq_feats = freq_feats[:, :top_k, :]
        real_part = freq_feats.real.permute(0, 2, 1)
        imag_part = freq_feats.imag.permute(0, 2, 1)
        real_part = self.freq_linear_real(real_part)
        imag_part = self.freq_linear_imag(imag_part)
        target_k = self.pred_len // 2 + 1
        real_part = self._match_last_dim(real_part, target_k)
        imag_part = self._match_last_dim(imag_part, target_k)
        complex_part = torch.complex(real_part, imag_part)
        time_feats = torch.fft.irfft(complex_part, n=self.pred_len, dim=2)
        return time_feats.permute(0, 2, 1)

    def _match_last_dim(self, tensor: torch.Tensor, target: int) -> torch.Tensor:
        current = tensor.size(-1)
        if current == target:
            return tensor
        if current > target:
            return tensor[..., :target]
        pad_shape = list(tensor.shape)
        pad_shape[-1] = target - current
        pad_tensor = tensor.new_zeros(pad_shape)
        return torch.cat([tensor, pad_tensor], dim=-1)

    def _compute_arima_trend(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        try:
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
            from statsmodels.tools.sm_exceptions import ConvergenceWarning
        except ImportError as exc:
            raise ImportError("statsmodels is required for ARIMA-enhanced DLinear, please install it via `pip install statsmodels`." ) from exc

        x_cpu = x.detach().cpu().numpy()
        trend_seq = np.zeros_like(x_cpu)
        trend_pred = np.zeros((x_cpu.shape[0], self.pred_len, x_cpu.shape[2]), dtype=x_cpu.dtype)
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        for i in range(x_cpu.shape[0]):
            for j in range(x_cpu.shape[2]):
                series = x_cpu[i, :, j]
                try:
                    model = StatsARIMA(series, order=self.cfg.arima_order)
                    fitted = model.fit()
                    fitted_values = fitted.fittedvalues[-self.seq_len:]
                    trend_seq[i, :, j] = fitted_values
                    forecast = fitted.forecast(self.pred_len)
                    trend_pred[i, :, j] = forecast
                except Exception:
                    fallback = np.convolve(series, np.ones(self.cfg.arima_moving_avg) / self.cfg.arima_moving_avg, mode="same")
                    trend_seq[i, :, j] = fallback
                    trend_pred[i, :, j] = fallback[-1]
        trend_seq_tensor = torch.from_numpy(trend_seq).to(x.device).type_as(x)
        trend_pred_tensor = torch.from_numpy(trend_pred).to(x.device).type_as(x)
        return trend_seq_tensor, trend_pred_tensor
