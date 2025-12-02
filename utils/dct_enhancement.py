"""
DCT信号增强模块
简洁版 - 基于PSD分析结果的自适应增强
"""
import numpy as np
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt

# Fix Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class DCTEnhancer:
    """DCT信号增强器 - 参考FFT增强器优化版"""
    
    def __init__(
        self, 
        low_freq_boost=1.5, 
        mid_freq_boost=1.0,
        high_freq_suppress=0.15, 
        cutoff_ratio=0.3,
        low_freq_ratio=0.05,
        high_freq_ratio=0.15,
        residual_ratio=0.7,
        reflection_pad=64
    ):
        """
        参数:
            low_freq_boost: 低频增强系数 (>1.0 增强, <1.0 抑制)
            mid_freq_boost: 中频增强系数 (通常为 1.0)
            high_freq_suppress: 高频抑制系数 (0-1之间)
            cutoff_ratio: 保留系数的比例 (0-1之间)
            low_freq_ratio: 低频区域比例 (0-1之间)
            high_freq_ratio: 高频区域起始比例 (0-1之间)
            residual_ratio: 残差混合比例 (0-1之间, 1.0表示完全使用增强信号)
            reflection_pad: 反射填充长度
        """
        self.low_freq_boost = low_freq_boost
        self.mid_freq_boost = mid_freq_boost
        self.high_freq_suppress = high_freq_suppress
        self.cutoff_ratio = cutoff_ratio
        self.low_freq_ratio = low_freq_ratio
        self.high_freq_ratio = high_freq_ratio
        self.residual_ratio = float(np.clip(residual_ratio, 0.0, 1.0))
        self.reflection_pad = max(0, int(reflection_pad))
    
    def enhance(self, signal):
        """
        对信号进行DCT增强
        
        参数:
            signal: 输入信号 (1D数组)
        
        返回:
            enhanced_signal: 增强后的信号
        """
        if signal.ndim != 1:
            raise ValueError('信号必须是一维向量')
        
        n = signal.size
        signal_mean = float(np.mean(signal))
        detrended = signal - signal_mean
        
        # 反射填充以减少边界效应
        pad = min(self.reflection_pad, max(0, n // 2))
        if pad > 0:
            working = np.pad(detrended, pad_width=pad, mode='reflect')
        else:
            working = detrended
        
        # 1. DCT变换
        dct_coeffs = dct(working, type=2, norm='ortho')
        
        # 2. 计算截断点
        n_coeffs = len(dct_coeffs)
        cutoff_idx = int(n_coeffs * self.cutoff_ratio)
        low_idx = int(n_coeffs * self.low_freq_ratio)
        high_idx = int(n_coeffs * self.high_freq_ratio)
        
        # 3. 自适应加权 - 完全匹配 FFT 的逻辑
        weights = np.ones(n_coeffs)
        
        # 低频增强
        weights[:low_idx] *= self.low_freq_boost
        
        # 中频保持或增强
        weights[low_idx:high_idx] *= self.mid_freq_boost
        
        # 高频抑制（从 high_idx 到末尾）
        weights[high_idx:] *= self.high_freq_suppress
        
        # 超过截断点的系数置零（这会覆盖之前的高频抑制）
        weights[cutoff_idx:] = 0.0
        
        # 4. 应用权重
        enhanced_coeffs = dct_coeffs * weights
        
        # 5. 逆DCT变换
        enhanced_working = idct(enhanced_coeffs, type=2, norm='ortho')
        
        # 去除填充
        if pad > 0:
            enhanced_signal = enhanced_working[pad:-pad]
        else:
            enhanced_signal = enhanced_working
        
        # 恢复均值
        enhanced_signal += signal_mean
        
        # 残差混合
        if self.residual_ratio < 1.0:
            return signal + self.residual_ratio * (enhanced_signal - signal)
        return enhanced_signal
    
    def enhance_batch(self, signals):
        """
        批量增强多个信号
        
        参数:
            signals: 输入信号数组 (2D: [n_samples, n_features])
        
        返回:
            enhanced_signals: 增强后的信号数组
        """
        enhanced = np.zeros_like(signals)
        for i in range(signals.shape[1]):
            enhanced[:, i] = self.enhance(signals[:, i])
        return enhanced


def compare_signals(original, enhanced, title='信号对比', save_path=None, show_residual=True):
    """
    对比原始信号和增强信号
    
    参数:
        original: 原始信号
        enhanced: 增强后的信号
        title: 图表标题
        save_path: 保存路径 (可选)
        show_residual: 是否显示残差图
    """
    
    residual = enhanced - original
    rows = 3 if show_residual else 2
    fig, axes = plt.subplots(rows, 1, figsize=(12, 10 if show_residual else 8))
    
    # 时域对比
    axes[0].plot(original, label='原始信号', alpha=0.7, linewidth=1.5)
    axes[0].plot(enhanced, label='DCT增强信号', alpha=0.8, linewidth=1.5)
    axes[0].set_title(f'{title} - 时域对比（含残差混合+反射填充）')
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('幅度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # DCT系数对比
    original_dct = dct(original, type=2, norm='ortho')
    enhanced_dct = dct(enhanced, type=2, norm='ortho')
    
    # 过滤数值误差：将极小值设为最小阈值，避免显示数值噪声
    threshold = 1e-10
    original_dct_clean = np.where(np.abs(original_dct) < threshold, threshold, np.abs(original_dct))
    enhanced_dct_clean = np.where(np.abs(enhanced_dct) < threshold, threshold, np.abs(enhanced_dct))
    
    axes[1].plot(original_dct_clean, label='原始DCT系数', alpha=0.7, linewidth=1.5)
    axes[1].plot(enhanced_dct_clean, label='增强DCT系数', alpha=0.7, linewidth=1.5)
    axes[1].set_title(f'{title} - DCT系数对比（反射填充后）')
    axes[1].set_xlabel('系数索引（0=最低频，越大越高频）')
    axes[1].set_ylabel('系数幅度（对数刻度）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, which='both')
    axes[1].set_yscale('log')
    axes[1].set_ylim([threshold, None])
    
    # 添加频率区域标注
    n = len(original_dct)
    axes[1].axvline(n*0.05, color='green', linestyle='--', alpha=0.5, linewidth=2, label='低频区')
    axes[1].axvline(n*0.15, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='中频区')
    axes[1].axvline(n*0.3, color='red', linestyle='--', alpha=0.5, linewidth=2, label='截断点')
    axes[1].legend()
    
    # 残差图
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


if __name__ == '__main__':
    # 测试
    # 生成测试信号
    t = np.linspace(0, 1, 500)
    signal = np.sin(2*np.pi*5*t) + 0.5*np.sin(2*np.pi*20*t) + 0.1*np.random.randn(500)
    
    # 增强
    enhancer = DCTEnhancer(low_freq_boost=1.5, high_freq_suppress=0.2)
    enhanced = enhancer.enhance(signal)
    
    # 对比
    compare_signals(signal, enhanced, title='DCT增强测试')
