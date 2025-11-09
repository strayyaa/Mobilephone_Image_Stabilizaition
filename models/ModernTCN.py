from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from layers.RevIN import RevIN
from .ModernTCN_Layer import SeriesDecomp, FlattenHead


def _get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def _get_bn(channels):
    return nn.BatchNorm1d(channels)


def _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    return nn.Sequential(
        _get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
        _get_bn(out_channels),
    )


def _fuse_bn(conv: nn.Conv1d, bn: nn.BatchNorm1d):
    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    scale = (gamma / std).reshape(-1, 1, 1)
    return kernel * scale, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False):
        super().__init__()
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = _conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups)
            if small_kernel is not None:
                self.small_conv = _conv_bn(
                    in_channels,
                    out_channels,
                    small_kernel,
                    stride,
                    small_kernel // 2,
                    groups,
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "lkb_reparam"):
            return self.lkb_reparam(x)
        out = self.lkb_origin(x)
        if hasattr(self, "small_conv"):
            out = out + self.small_conv(x)
        return out

    def _pad_kernel(self, kernel: torch.Tensor, target: int) -> torch.Tensor:
        pad_total = target - kernel.size(-1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_total <= 0:
            return kernel
        pad_shape = list(kernel.shape)
        pad_shape[-1] = pad_left
        left = kernel.new_zeros(pad_shape)
        pad_shape[-1] = pad_right
        right = kernel.new_zeros(pad_shape)
        return torch.cat([left, kernel, right], dim=-1)

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = _fuse_bn(self.lkb_origin[0], self.lkb_origin[1])
        if hasattr(self, "small_conv"):
            small_k, small_b = _fuse_bn(self.small_conv[0], self.small_conv[1])
            eq_b = eq_b + small_b
            eq_k = eq_k + self._pad_kernel(small_k, eq_k.size(-1))
        return eq_k, eq_b

    def merge_kernel(self):
        if hasattr(self, "lkb_reparam"):
            return
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        conv = self.lkb_origin[0]
        self.lkb_reparam = nn.Conv1d(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size[0],
            stride=conv.stride[0],
            padding=conv.padding[0],
            dilation=conv.dilation[0],
            groups=conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        del self.lkb_origin
        if hasattr(self, "small_conv"):
            del self.small_conv


class Block(nn.Module):
    def __init__(self, large_size, small_size, d_model, d_ff, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * d_model,
            out_channels=nvars * d_model,
            kernel_size=large_size,
            stride=1,
            groups=nvars * d_model,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.ffn1_pw1 = nn.Conv1d(nvars * d_model, nvars * d_ff, 1, groups=nvars)
        self.ffn1_pw2 = nn.Conv1d(nvars * d_ff, nvars * d_model, 1, groups=nvars)
        self.ffn1_drop1 = nn.Dropout(drop)
        self.ffn1_drop2 = nn.Dropout(drop)
        self.ffn1_act = nn.GELU()

        self.ffn2_pw1 = nn.Conv1d(nvars * d_model, nvars * d_ff, 1, groups=d_model)
        self.ffn2_pw2 = nn.Conv1d(nvars * d_ff, nvars * d_model, 1, groups=d_model)
        self.ffn2_drop1 = nn.Dropout(drop)
        self.ffn2_drop2 = nn.Dropout(drop)
        self.ffn2_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        bsz, nvars, d_model, length = x.shape
        x = x.reshape(bsz, nvars * d_model, length)
        x = self.dw(x)
        x = x.reshape(bsz, nvars, d_model, length).reshape(bsz * nvars, d_model, length)
        x = self.norm(x)
        x = x.reshape(bsz, nvars, d_model, length).reshape(bsz, nvars * d_model, length)

        x = self.ffn1_drop1(self.ffn1_pw1(x))
        x = self.ffn1_act(x)
        x = self.ffn1_drop2(self.ffn1_pw2(x))
        x = x.reshape(bsz, nvars, d_model, length)

        x = x.permute(0, 2, 1, 3).reshape(bsz, d_model * nvars, length)
        x = self.ffn2_drop1(self.ffn2_pw1(x))
        x = self.ffn2_act(x)
        x = self.ffn2_drop2(self.ffn2_pw2(x))
        x = x.reshape(bsz, d_model, nvars, length).permute(0, 2, 1, 3)
        return residual + x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, d_model, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        d_ff = d_model * ffn_ratio
        self.blocks = nn.ModuleList(
            [
                Block(large_size, small_size, d_model, d_ff, nvars, small_kernel_merged, drop)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ModernTCNCore(nn.Module):
    def __init__(
        self,
        patch_size,
        patch_stride,
        stem_ratio,
        downsample_ratio,
        ffn_ratio,
        num_blocks: List[int],
        large_size: List[int],
        small_size: List[int],
        dims: List[int],
        nvars: int,
        small_kernel_merged: bool,
        backbone_dropout: float,
        head_dropout: float,
        use_multi_scale: bool,
        revin: bool,
        affine: bool,
        subtract_last: bool,
        freq: str,
        seq_len: int,
        c_in: int,
        individual: bool,
        target_window: int,
    ) -> None:
        super().__init__()
        self.revin = revin
        if revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.num_stage = len(num_blocks)
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stage - 1):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        if freq == "h":
            time_feature_num = 4
        elif freq == "t":
            time_feature_num = 5
        elif freq.endswith("s") or freq.endswith("ms"):
            time_feature_num = 6
        else:
            time_feature_num = 4

        self.te_patch = nn.Sequential(
            nn.Conv1d(time_feature_num, time_feature_num, kernel_size=patch_size, stride=patch_stride, groups=time_feature_num),
            nn.Conv1d(time_feature_num, dims[0], kernel_size=1),
            nn.BatchNorm1d(dims[0]),
        )
        self.spec_k = 64
        self.spec_proj = nn.Conv1d(1, dims[0], 1)
        self.spec_alpha = nn.Parameter(torch.tensor(0.3))

        self.stages = nn.ModuleList(
            [
                Stage(ffn_ratio, num_blocks[i], large_size[i], small_size[i], dims[i], nvars, small_kernel_merged, drop=backbone_dropout)
                for i in range(self.num_stage)
            ]
        )

        self.use_multi_scale = use_multi_scale
        self.up_sample_ratio = downsample_ratio

        self.lat_layer = nn.ModuleList()
        self.smooth_layer = nn.ModuleList()
        self.up_sample_conv = nn.ModuleList()
        align_dim = dims[-1]
        for i in range(self.num_stage):
            lat = nn.Conv1d(dims[i], align_dim, kernel_size=1)
            self.lat_layer.append(lat)
            self.smooth_layer.append(nn.Conv1d(align_dim, align_dim, kernel_size=3, padding=1))
            self.up_sample_conv.append(
                nn.Sequential(
                    nn.ConvTranspose1d(align_dim, align_dim, kernel_size=self.up_sample_ratio, stride=self.up_sample_ratio),
                    nn.BatchNorm1d(align_dim),
                )
            )

        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[-1]
        with torch.no_grad():
            training_state = self.training
            self.train(False)
            dummy_x = torch.zeros(1, c_in, seq_len)
            try:
                inferred_feature = self.forward_feature(dummy_x)
                reduced_patch_num = inferred_feature.shape[-1]
            finally:
                self.train(training_state)
        head_nf = d_model * reduced_patch_num
        self.head = FlattenHead(individual, self.n_vars, head_nf, target_window, head_dropout=head_dropout)

    def structural_reparam(self):
        for module in self.modules():
            if hasattr(module, "merge_kernel"):
                module.merge_kernel()

    def forward_feature(self, x: torch.Tensor, te: torch.Tensor = None) -> torch.Tensor:
        batch, nvars, length = x.shape
        x_raw = x
        x = x.unsqueeze(-2)
        for idx in range(self.num_stage):
            bsz, nvars_, d_model, n_steps = x.shape
            x = x.reshape(bsz * nvars_, d_model, n_steps)
            if idx == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if n_steps % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (n_steps % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[idx](x)
            _, d_model_new, n_steps_new = x.shape
            x = x.reshape(bsz, nvars_, d_model_new, n_steps_new)
            if idx == 0:
                spec = torch.fft.rfft(x_raw, dim=2).abs()
                spec = torch.log1p(spec)
                k = min(self.spec_k, spec.shape[-1])
                spec = spec[..., :k]
                spec = F.interpolate(spec, size=n_steps_new, mode="linear", align_corners=False)
                spec = spec.reshape(batch * nvars, 1, n_steps_new)
                spec = self.spec_proj(spec).reshape(batch, nvars, -1, n_steps_new)
                x = x + self.spec_alpha * spec
            x = self.stages[idx](x)
        return x

    def forward(self, x: torch.Tensor, te: torch.Tensor = None) -> torch.Tensor:
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1)
        x = self.forward_feature(x, te)
        x = self.head(x)
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "denorm")
            x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.stem_ratio = int(getattr(configs, "stem_ratio", 6))
        self.downsample_ratio = int(getattr(configs, "downsample_ratio", 2))
        self.ffn_ratio = int(getattr(configs, "ffn_ratio", 2))
        self.num_blocks = list(getattr(configs, "num_blocks", [1, 1, 1, 1]))
        self.large_size = list(getattr(configs, "large_size", [31, 29, 27, 13]))
        self.small_size = list(getattr(configs, "small_size", [5, 5, 5, 5]))
        self.dims = list(getattr(configs, "dims", [256, 256, 256, 256]))
        self.nvars = int(getattr(configs, "enc_in", 7))
        self.small_kernel_merged = bool(getattr(configs, "small_kernel_merged", False))
        self.drop_backbone = float(getattr(configs, "dropout", 0.1))
        self.drop_head = float(getattr(configs, "head_dropout", 0.0))
        self.use_multi_scale = bool(getattr(configs, "use_multi_scale", True))
        self.revin = bool(getattr(configs, "revin", True))
        self.affine = bool(getattr(configs, "affine", False))
        self.subtract_last = bool(getattr(configs, "subtract_last", False))
        self.freq = getattr(configs, "freq", "h")
        self.seq_len = int(getattr(configs, "seq_len", 96))
        self.individual = bool(getattr(configs, "individual", False))
        self.target_window = int(getattr(configs, "pred_len", 96))
        self.kernel_size = int(getattr(configs, "kernel_size", 25))
        self.patch_size = int(getattr(configs, "patch_size", getattr(configs, "patch_len", 16)))
        self.patch_stride = int(getattr(configs, "patch_stride", getattr(configs, "stride", 8)))
        self.decomposition = bool(getattr(configs, "decomposition", False))

        if self.decomposition:
            self.decomp_module = SeriesDecomp(self.kernel_size)
            self.model_res = ModernTCNCore(
                self.patch_size,
                self.patch_stride,
                self.stem_ratio,
                self.downsample_ratio,
                self.ffn_ratio,
                self.num_blocks,
                self.large_size,
                self.small_size,
                self.dims,
                self.nvars,
                self.small_kernel_merged,
                self.drop_backbone,
                self.drop_head,
                self.use_multi_scale,
                self.revin,
                self.affine,
                self.subtract_last,
                self.freq,
                self.seq_len,
                self.nvars,
                self.individual,
                self.target_window,
            )
            self.model_trend = ModernTCNCore(
                self.patch_size,
                self.patch_stride,
                self.stem_ratio,
                self.downsample_ratio,
                self.ffn_ratio,
                self.num_blocks,
                self.large_size,
                self.small_size,
                self.dims,
                self.nvars,
                self.small_kernel_merged,
                self.drop_backbone,
                self.drop_head,
                self.use_multi_scale,
                self.revin,
                self.affine,
                self.subtract_last,
                self.freq,
                self.seq_len,
                self.nvars,
                self.individual,
                self.target_window,
            )
        else:
            self.model = ModernTCNCore(
                self.patch_size,
                self.patch_stride,
                self.stem_ratio,
                self.downsample_ratio,
                self.ffn_ratio,
                self.num_blocks,
                self.large_size,
                self.small_size,
                self.dims,
                self.nvars,
                self.small_kernel_merged,
                self.drop_backbone,
                self.drop_head,
                self.use_multi_scale,
                self.revin,
                self.affine,
                self.subtract_last,
                self.freq,
                self.seq_len,
                self.nvars,
                self.individual,
                self.target_window,
            )

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor = None) -> torch.Tensor:
        if self.decomposition:
            res, trend = self.decomp_module(x)
            res = res.permute(0, 2, 1)
            trend = trend.permute(0, 2, 1)
            te = x_mark.permute(0, 2, 1) if x_mark is not None else None
            res_out = self.model_res(res, te)
            trend_out = self.model_trend(trend, te)
            out = res_out + trend_out
            return out.permute(0, 2, 1)
        x = x.permute(0, 2, 1)
        te = x_mark.permute(0, 2, 1) if x_mark is not None else None
        out = self.model(x, te)
        return out.permute(0, 2, 1)

    def structural_reparam(self):
        target = self.model if hasattr(self, "model") else [self.model_res, self.model_trend]
        if isinstance(target, list):
            for module in target:
                module.structural_reparam()
        else:
            target.structural_reparam()
