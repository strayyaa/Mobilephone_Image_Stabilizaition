"""FFT-based signal enhancement module."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class FFTEnhancer:
    """Enhance 1-D signals in the frequency domain using FFT weights."""

    def __init__(
        self,
        low_freq_ratio: float = 0.005,
        high_freq_ratio: float = 0.01,
        low_freq_boost: float = 1.2,
        mid_freq_boost: float = 1.0,
        high_freq_suppress: float = 0.3,
        cutoff_ratio: float = 0.02,
        residual_ratio: float = 0.7,
        reflection_pad: int = 32,
    ) -> None:
        if not (0.0 < low_freq_ratio < high_freq_ratio < 1.0):
            raise ValueError('低频/高频的分割比例必须满足 0 < low < high < 1')
        if not (0.0 < cutoff_ratio <= 1.0):
            raise ValueError('cutoff_ratio 必须在 (0, 1] 范围内')

        self.low_freq_ratio = low_freq_ratio
        self.high_freq_ratio = high_freq_ratio
        self.low_freq_boost = low_freq_boost
        self.mid_freq_boost = mid_freq_boost
        self.high_freq_suppress = high_freq_suppress
        self.cutoff_ratio = cutoff_ratio
        self.residual_ratio = float(np.clip(residual_ratio, 0.0, 1.0))
        self.reflection_pad = max(0, int(reflection_pad))

    def enhance(self, signal: np.ndarray) -> np.ndarray:
        """Apply FFT weighting and return enhanced signal."""

        if signal.ndim != 1:
            raise ValueError('信号必须是一维向量')

        n = signal.size
        signal_mean = float(np.mean(signal))
        detrended = signal - signal_mean

        pad = min(self.reflection_pad, max(0, n // 2))
        if pad > 0:
            working = np.pad(detrended, pad_width=pad, mode='reflect')
        else:
            working = detrended

        fft_coeffs = np.fft.rfft(working)

        weights = np.ones_like(fft_coeffs, dtype=float)
        spectrum_len = len(fft_coeffs)
        low_idx = max(1, int(spectrum_len * self.low_freq_ratio))
        high_idx = int(spectrum_len * self.high_freq_ratio)
        cutoff_idx = int(spectrum_len * self.cutoff_ratio)

        weights[:low_idx] *= self.low_freq_boost
        weights[low_idx:high_idx] *= self.mid_freq_boost
        weights[high_idx:] *= self.high_freq_suppress
        weights[cutoff_idx:] = 0.0

        enhanced_coeffs = fft_coeffs * weights
        enhanced_working = np.fft.irfft(enhanced_coeffs, n=working.shape[0])

        if pad > 0:
            enhanced_signal = enhanced_working[pad:-pad]
        else:
            enhanced_signal = enhanced_working

        enhanced_signal += signal_mean

        if self.residual_ratio < 1.0:
            return signal + self.residual_ratio * (enhanced_signal - signal)
        return enhanced_signal

    def enhance_batch(self, signals: np.ndarray) -> np.ndarray:
        """Enhance signals shaped [seq_len, channels]."""

        if signals.ndim != 2:
            raise ValueError('signals 必须是二维数组 [seq_len, channels]')
        enhanced = np.zeros_like(signals)
        for idx in range(signals.shape[1]):
            enhanced[:, idx] = self.enhance(signals[:, idx])
        return enhanced


def compare_signals_fft(
    original: np.ndarray,
    enhanced: np.ndarray,
    title: str,
    save_path: str | None = None,
    show_residual: bool = True,
) -> None:
    """Visualize time-domain, spectrum, and residual for FFT enhancement."""

    if original.shape != enhanced.shape:
        raise ValueError('原始与增强信号长度不一致')

    residual = enhanced - original
    rows = 3 if show_residual else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 10 if show_residual else 8))

    axes[0].plot(original, label='原始信号', alpha=0.7)
    axes[0].plot(enhanced, label='FFT增强信号', alpha=0.8)
    axes[0].set_title(f'{title} - 时域对比（含残差混合）')
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('幅度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    orig_fft = np.fft.rfft(original)
    enh_fft = np.fft.rfft(enhanced)
    axes[1].plot(np.abs(orig_fft), label='原始频谱', alpha=0.7)
    axes[1].plot(np.abs(enh_fft), label='增强频谱', alpha=0.8)
    axes[1].set_title(f'{title} - 频谱幅度对比（反射填充后）')
    axes[1].set_xlabel('频率索引')
    axes[1].set_ylabel('|FFT|')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')

    if show_residual:
        axes[2].plot(residual, color='tab:orange', label='增强残差 = 增强 - 原始')
        axes[2].axhline(0, color='k', linewidth=0.8, alpha=0.6)
        axes[2].set_title(f'{title} - 注入的低频残差')
        axes[2].set_xlabel('时间步')
        axes[2].set_ylabel('残差值')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
