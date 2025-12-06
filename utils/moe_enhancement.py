"""Trainable Mixture-of-Experts utilities for frequency/modal enhancement."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FrequencyBandConfig:
    low_ratio: float
    high_ratio: float
    cutoff_ratio: float
    residual_ratio: float
    temperature: float
    reflection_pad: int = 32
    normalize_variance: bool = True
    use_residual_mix: bool = False
    num_experts: int = 3
    top_k: int = 0
    energy_norm: str = 'density'


class _TrainableSoftmaxGate(nn.Module):
    """Linear-gated softmax whose parameters are trained jointly with the model."""

    def __init__(self, slots: int, temperature: float) -> None:
        super().__init__()
        self.temperature = max(1e-6, float(temperature))
        self.alpha = nn.Parameter(torch.ones(slots))
        self.beta = nn.Parameter(torch.zeros(slots))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        # scores shape [..., slots]
        scores = torch.nan_to_num(scores, nan=0.0, posinf=1e4, neginf=-1e4)
        alpha = self.alpha.view(*([1] * (scores.ndim - 1)), -1)
        beta = self.beta.view(*([1] * (scores.ndim - 1)), -1)
        logits = alpha * scores + beta
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
        logits = logits / self.temperature
        weights = torch.softmax(logits, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=1.0, neginf=0.0)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return weights / denom


class TrainableFrequencyMoE(nn.Module):
    """FFT/DCT frequency experts with learnable softmax gating."""

    def __init__(self, transform: str, config: FrequencyBandConfig) -> None:
        super().__init__()
        if transform not in {"fft", "dct"}:
            raise ValueError('transform 必须是 "fft" 或 "dct"')
        self.transform = transform
        self.config = config
        self.num_experts = max(1, int(config.num_experts))
        self.top_k = max(0, int(config.top_k))
        self.energy_norm = (config.energy_norm or 'density').lower()
        self.gate = _TrainableSoftmaxGate(self.num_experts, config.temperature)
        # start energy scaling at zero so initial logits mirror the prior ratios (e.g., 4:3:1)
        self.energy_scale = nn.Parameter(torch.zeros(self.num_experts))
        self.energy_bias = nn.Parameter(torch.zeros(self.num_experts))
        self._mask_cache: Dict[int, torch.Tensor] = {}
        self._prior_cache: Dict[int, torch.Tensor] = {}
        self._dct_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._neg_fill = -1e4

    def _split_indices(self, length: int) -> Tuple[int, int, int]:
        low_idx = max(1, int(length * self.config.low_ratio))
        high_idx = max(low_idx + 1, int(length * self.config.high_ratio))
        cutoff_idx = max(high_idx + 1, int(length * self.config.cutoff_ratio))
        return min(low_idx, length), min(high_idx, length), min(cutoff_idx, length)

    def _default_ranges(self, length: int) -> List[Tuple[int, int]]:
        low_idx, high_idx, cutoff_idx = self._split_indices(length)
        ranges = [(0, low_idx), (low_idx, high_idx), (high_idx, cutoff_idx)]
        return [(s, e) for s, e in ranges if e > s]

    def _equal_split(self, start: int, end: int, count: int) -> List[Tuple[int, int]]:
        end = max(start + 1, end)
        span = end - start
        ranges: List[Tuple[int, int]] = []
        for idx in range(count):
            seg_start = start + round(span * idx / count)
            seg_end = start + round(span * (idx + 1) / count)
            if seg_end <= seg_start:
                seg_end = min(end, seg_start + 1)
            ranges.append((seg_start, seg_end))
        ranges[-1] = (ranges[-1][0], end)
        return ranges

    def _band_ranges(self, length: int) -> Tuple[List[Tuple[int, int]], List[str]]:
        default_ranges = self._default_ranges(length)
        target = self.num_experts
        if not default_ranges:
            ranges = self._equal_split(0, max(1, length), target)
            return ranges, ['low'] * len(ranges)
        if target <= len(default_ranges):
            ranges = default_ranges[:target]
            tags = ['low', 'mid', 'high'][:len(ranges)]
            return ranges, tags

        lengths = [end - start for start, end in default_ranges]
        total = sum(max(0, l) for l in lengths) or 1
        allocations = [0] * len(default_ranges)
        remaining = target
        for idx, length_val in enumerate(lengths):
            if length_val <= 0:
                continue
            quota = max(1, int(round(target * length_val / total)))
            quota = min(quota, remaining)
            allocations[idx] = quota
            remaining -= quota
            if remaining <= 0:
                break
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
        ptr = 0
        while remaining > 0:
            target_idx = order[ptr % len(order)] if order else 0
            allocations[target_idx] += 1
            remaining -= 1
            ptr += 1

        ranges: List[Tuple[int, int]] = []
        groups: List[str] = []
        tag_pool = ['low', 'mid', 'high']
        for idx_pair, ((start, end), count) in enumerate(zip(default_ranges, allocations)):
            if count <= 0 or end <= start:
                continue
            splits = self._equal_split(start, end, count)
            ranges.extend(splits)
            group_label = tag_pool[idx_pair] if idx_pair < len(tag_pool) else tag_pool[-1]
            groups.extend([group_label] * len(splits))
        if not ranges:
            ranges = [(0, length)]
            groups = ['low']
        return ranges, groups

    def _build_priors(self, groups: List[str]) -> torch.Tensor:
        ratio_map = {'low': 4.0, 'mid': 3.0, 'high': 1.0}
        counts: Dict[str, int] = {}
        for tag in groups:
            counts[tag] = counts.get(tag, 0) + 1
        priors = torch.zeros(len(groups))
        total = 0.0
        for idx, tag in enumerate(groups):
            base = ratio_map.get(tag, 1.0)
            share = base / max(1, counts.get(tag, 1))
            priors[idx] = share
            total += share
        if total <= 0:
            priors.fill_(1.0 / max(1, len(groups)))
        else:
            priors /= total
        return torch.log(priors.clamp_min(1e-6))

    def _get_masks(self, spectrum_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if spectrum_len not in self._mask_cache:
            mask = torch.zeros(self.num_experts, spectrum_len)
            ranges, groups = self._band_ranges(spectrum_len)
            for idx, (start, end) in enumerate(ranges):
                mask[idx, start:end] = 1.0
            self._mask_cache[spectrum_len] = mask
            self._prior_cache[spectrum_len] = self._build_priors(groups)
        return (
            self._mask_cache[spectrum_len].to(device),
            self._prior_cache[spectrum_len].to(device),
        )

    def _get_dct_matrix(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        if key not in self._dct_cache:
            n = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(0)
            k = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
            mat = torch.cos(math.pi / length * (n + 0.5) * k)
            mat = mat * math.sqrt(2.0 / length)
            mat[0] = mat[0] / math.sqrt(2.0)
            self._dct_cache[key] = mat
        return self._dct_cache[key]

    def _apply_dct(self, signal: torch.Tensor) -> torch.Tensor:
        # signal shape [B, C, L]
        n = signal.shape[-1]
        mat = self._get_dct_matrix(n, signal.device)
        flat = signal.reshape(-1, n)
        coeffs = flat @ mat.T
        return coeffs.reshape_as(signal)

    def _apply_idct(self, coeffs: torch.Tensor) -> torch.Tensor:
        n = coeffs.shape[-1]
        mat = self._get_dct_matrix(n, coeffs.device)
        flat = coeffs.reshape(-1, n)
        signal = flat @ mat
        return signal.reshape_as(coeffs)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # inputs shape [B, L, C]
        orig_dtype = inputs.dtype
        x = inputs.permute(0, 2, 1).contiguous().to(torch.float32)  # [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False) + 1e-8
        if not self.config.normalize_variance:
            std = torch.ones_like(std)
        normalized = (x - mean) / std
        normalized = torch.nan_to_num(normalized)
        pad = min(self.config.reflection_pad, normalized.shape[-1] // 2)
        if pad > 0:
            normalized = F.pad(normalized, (pad, pad), mode='reflect')
        normalized = torch.nan_to_num(normalized)
        length = normalized.shape[-1]

        masks, base_logits = self._get_masks(
            spectrum_len=length if self.transform == 'dct' else (length // 2 + 1),
            device=normalized.device,
        )
        masks = masks.unsqueeze(0).unsqueeze(0)  # [1,1,E,F]
        base_logits = base_logits.view(1, 1, -1)

        if self.transform == 'fft':
            coeffs = torch.fft.rfft(normalized, dim=-1)
            masked = coeffs.unsqueeze(2) * masks
            band_signals = torch.fft.irfft(masked, n=length, dim=-1)
            energy = (masked.abs() ** 2).sum(dim=-1)   # gated 可以修改
        else:
            coeffs = self._apply_dct(normalized)
            masked = coeffs.unsqueeze(2) * masks[..., : coeffs.shape[-1]]
            band_signals = self._apply_idct(masked)
            energy = (masked ** 2).sum(dim=-1)

        band_signals = torch.nan_to_num(band_signals)
        energy = torch.nan_to_num(energy, nan=0.0, posinf=1e12, neginf=0.0)
        energy = energy.clamp_min(0.0)

        alpha = self.energy_scale.view(1, 1, -1)
        beta = self.energy_bias.view(1, 1, -1)
        logits = alpha * energy + beta + base_logits
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e12, neginf=-1e12)
        if 0 < self.top_k < self.num_experts:
            top_vals, top_idx = torch.topk(logits, self.top_k, dim=-1)
            mask = torch.zeros_like(logits)
            mask.scatter_(-1, top_idx, 1.0)
            fill = torch.full_like(logits, self._neg_fill)
            logits = torch.where(mask.bool(), logits, fill)
        weights = self.gate(logits)

        mixed = (band_signals * weights.unsqueeze(-1)).sum(dim=2)
        mixed = torch.nan_to_num(mixed)
        if pad > 0:
            mixed = mixed[..., pad:-pad]
        mixed = mixed[..., : x.shape[-1]]
        output = mixed * std + mean
        if self.config.use_residual_mix:
            output = x + self.config.residual_ratio * (output - x)
        output = torch.nan_to_num(output)
        output = output.permute(0, 2, 1).contiguous().to(orig_dtype)
        return output, weights.detach().cpu()


class TrainableModalMoE(nn.Module):
    """Modal fusion with learnable gating."""

    def __init__(self, temperature: float) -> None:
        super().__init__()
        self.gate = _TrainableSoftmaxGate(3, temperature)

    def forward(
        self,
        original: torch.Tensor,
        fft_signal: torch.Tensor,
        dct_signal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # tensors shape [B, L, C]
        orig_dtype = original.dtype
        stack = torch.stack([original, fft_signal, dct_signal], dim=-1).to(torch.float32)  # [B, L, C, 3]
        stack = torch.nan_to_num(stack)
        stats = stack.var(dim=1, unbiased=False)
        weights = self.gate(stats)  # [B, C, 3]
        fused = (stack.permute(0, 2, 1, 3) * weights.unsqueeze(-2)).sum(dim=-1)
        fused = torch.nan_to_num(fused)
        fused = fused.permute(0, 2, 1).contiguous().to(orig_dtype)
        return fused, weights.detach().cpu()


class MoEEnhancer(nn.Module):
    """Trainable MoE pipeline stacked as FFT -> DCT -> Modal fusion."""

    def __init__(
        self,
        low_freq_ratio: float,
        high_freq_ratio: float,
        cutoff_ratio: float,
        frequency_residual_ratio: float,
        temperature: float,
        reflection_pad: int = 32,
        normalize_variance: bool = True,
        frequency_use_residual_mix: bool = False,
        frequency_experts: int = 3,
        frequency_top_k: int = 0,
        frequency_energy_norm: str = 'density',
    ) -> None:
        super().__init__()
        config = FrequencyBandConfig(
            low_ratio=low_freq_ratio,
            high_ratio=high_freq_ratio,
            cutoff_ratio=cutoff_ratio,
            residual_ratio=frequency_residual_ratio,
            temperature=temperature,
            reflection_pad=reflection_pad,
            normalize_variance=normalize_variance,
            use_residual_mix=frequency_use_residual_mix,
            num_experts=frequency_experts,
            top_k=frequency_top_k,
            energy_norm=frequency_energy_norm,
        )
        self.fft_moe = TrainableFrequencyMoE('fft', config)
        self.dct_moe = TrainableFrequencyMoE('dct', config)
        self.modal_moe = TrainableModalMoE(temperature)

    def forward(
        self,
        batch: torch.Tensor,
        collect_stats: bool = False,
    ) -> Tuple[torch.Tensor, List[Dict[str, List[float]]]]:
        fft_out, fft_weights = self.fft_moe(batch)
        dct_out, dct_weights = self.dct_moe(batch)
        fused, modal_weights = self.modal_moe(batch, fft_out, dct_out)

        stats: List[Dict[str, List[float]]] = []
        if collect_stats:
            b = batch.shape[0]
            for idx in range(b):
                stats.append({
                    'fft_weights': fft_weights[idx].tolist(),
                    'dct_weights': dct_weights[idx].tolist(),
                    'modal_weights': modal_weights[idx].tolist(),
                })
        return fused, stats

