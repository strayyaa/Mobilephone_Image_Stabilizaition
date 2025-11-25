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
    """DCT信号增强器"""
    
    def __init__(self, low_freq_boost=1.2, high_freq_suppress=0.3, cutoff_ratio=0.7):
        """
        参数:
            low_freq_boost: 低频增强系数 (>1.0 增强, <1.0 抑制)
            high_freq_suppress: 高频抑制系数 (0-1之间)
            cutoff_ratio: 保留系数的比例 (0-1之间)
        """
        self.low_freq_boost = low_freq_boost
        self.high_freq_suppress = high_freq_suppress
        self.cutoff_ratio = cutoff_ratio
    
    def enhance(self, signal):
        """
        对信号进行DCT增强
        
        参数:
            signal: 输入信号 (1D数组)
        
        返回:
            enhanced_signal: 增强后的信号
        """
        # 1. DCT变换
        dct_coeffs = dct(signal, type=2, norm='ortho')
        
        # 2. 计算截断点
        n = len(dct_coeffs)
        cutoff_idx = int(n * self.cutoff_ratio)
        
        # 3. 自适应加权
        weights = np.ones(n)
        
        # 低频增强 (前30%系数)
        low_idx = int(n * 0.3)
        weights[:low_idx] = self.low_freq_boost
        
        # 高频抑制 (后30%系数)
        high_idx = int(n * 0.7)
        weights[high_idx:] = self.high_freq_suppress
        
        # 超过截断点的系数置零
        weights[cutoff_idx:] = 0
        
        # 4. 应用权重
        enhanced_coeffs = dct_coeffs * weights
        
        # 5. 逆DCT变换
        enhanced_signal = idct(enhanced_coeffs, type=2, norm='ortho')
        
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


def compare_signals(original, enhanced, title='信号对比', save_path=None):
    """
    对比原始信号和增强信号
    
    参数:
        original: 原始信号
        enhanced: 增强后的信号
        title: 图表标题
        save_path: 保存路径 (可选)
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 时域对比
    axes[0].plot(original, label='原始信号', alpha=0.7)
    axes[0].plot(enhanced, label='增强信号', alpha=0.7)
    axes[0].set_title(f'{title} - 时域对比')
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('幅度')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # DCT系数对比
    original_dct = dct(original, type=2, norm='ortho')
    enhanced_dct = dct(enhanced, type=2, norm='ortho')
    
    axes[1].plot(np.abs(original_dct), label='原始DCT系数', alpha=0.7)
    axes[1].plot(np.abs(enhanced_dct), label='增强DCT系数', alpha=0.7)
    axes[1].set_title(f'{title} - DCT系数对比')
    axes[1].set_xlabel('系数索引')
    axes[1].set_ylabel('幅度')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
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
