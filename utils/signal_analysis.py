"""
功率谱密度(PSD)分析工具
简洁版 - 只保留核心功能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, detrend
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
import os

# Fix Chinese font display
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def compute_psd(signal, fs=500, nperseg=512):
    """
    计算功率谱密度
    
    参数:
        signal: 输入信号 (1D数组)
        fs: 采样频率 (Hz)
        nperseg: 每段长度
    
    返回:
        frequencies: 频率数组
        psd: 功率谱密度数组
    """
    # 去趋势
    signal_detrended = detrend(signal, type='linear')
    
    # 标准化
    signal_normalized = (signal_detrended - signal_detrended.mean()) / signal_detrended.std()
    
    # 计算PSD (Welch方法)
    frequencies, psd = welch(
        signal_normalized,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=nperseg//2,
        nfft=max(nperseg, 512),
        detrend='constant',
        scaling='density'
    )
    
    return frequencies, psd


def analyze_angle_data(data_path, start_idx=6000, end_idx=12000, save_dir='./psd_results'):
    """
    分析陀螺仪角度数据的频谱特性
    
    参数:
        data_path: 数据文件路径
        start_idx: 开始索引
        end_idx: 结束索引
        save_dir: 结果保存目录
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(data_path)
    df_subset = df.iloc[start_idx:end_idx]
    
    # 计算采样频率
    time_diff = df_subset['time'].diff().mean()
    fs = 1 / time_diff
    print(f"采样频率: {fs:.2f} Hz")
    
    # 三个角度通道
    channels = ['x_angle_1', 'y_angle_1', 'z_angle_1']
    
    # 分析每个通道
    results = {}
    plt.figure(figsize=(15, 10))
    
    for idx, channel in enumerate(channels, 1):
        signal = df_subset[channel].values
        
        # 计算PSD
        frequencies, psd = compute_psd(signal, fs=fs)
        
        # 保存结果
        results[channel] = {
            'frequencies': frequencies,
            'psd': psd,
            'dominant_freq': frequencies[np.argmax(psd)],
            'total_power': np.sum(psd)
        }
        
        # 绘图
        plt.subplot(3, 1, idx)
        plt.semilogy(frequencies, psd)
        plt.title(f'{channel} - 功率谱密度')
        plt.xlabel('频率 (Hz)')
        plt.ylabel('功率谱密度')
        plt.grid(True, alpha=0.3)
        plt.xlim([0, 50])  # 只显示0-50Hz
        
        # 标注主导频率
        dominant_freq = results[channel]['dominant_freq']
        plt.axvline(dominant_freq, color='r', linestyle='--', 
                   label=f'主导频率: {dominant_freq:.2f} Hz')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psd_analysis.png'), dpi=150)
    plt.close()
    
    # === 新增: log-log 坐标系分析幂律行为 ===
    plt.figure(figsize=(15, 10))
    
    for idx, channel in enumerate(channels, 1):
        frequencies = results[channel]['frequencies']
        psd = results[channel]['psd']
        
        # 绘制 log-log 图
        plt.subplot(3, 1, idx)
        plt.loglog(frequencies[1:], psd[1:])  # 跳过f=0
        plt.title(f'{channel} - 功率谱密度 (log-log坐标)')
        plt.xlabel('频率 (Hz) - 对数刻度')
        plt.ylabel('功率谱密度 - 对数刻度')
        plt.grid(True, alpha=0.3, which='both')
        plt.xlim([0.5, 50])
        
        # 拟合幂律 (使用低频到中频部分)
        fit_range = (frequencies > 1) & (frequencies < 30)
        log_f = np.log10(frequencies[fit_range])
        log_psd = np.log10(psd[fit_range])
        
        slope, intercept, r_value, _, _ = linregress(log_f, log_psd)
        alpha = -slope  # 幂律指数
        
        # 绘制拟合直线
        fit_f = frequencies[fit_range]
        fit_psd = 10**(intercept) * fit_f**(slope)
        plt.loglog(fit_f, fit_psd, 'r--', linewidth=2, 
                  label=f'幂律拟合: α={alpha:.2f}, R²={r_value**2:.3f}')
        plt.legend()
        
        # 保存幂律指数到结果
        results[channel]['power_law_alpha'] = alpha
        results[channel]['power_law_r2'] = r_value**2
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'psd_loglog.png'), dpi=150)
    plt.close()
    
    # 生成报告
    report_path = os.path.join(save_dir, 'psd_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("功率谱密度分析报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"采样频率: {fs:.2f} Hz\n")
        f.write(f"数据范围: {start_idx} - {end_idx}\n\n")
        
        for channel in channels:
            f.write(f"\n{channel}:\n")
            f.write(f"  主导频率: {results[channel]['dominant_freq']:.2f} Hz\n")
            f.write(f"  总功率: {results[channel]['total_power']:.6f}\n")
            f.write(f"  幂律指数 α: {results[channel]['power_law_alpha']:.2f}\n")
            f.write(f"  幂律拟合度 R²: {results[channel]['power_law_r2']:.3f}\n")
            
            # 频段能量分布
            freq = results[channel]['frequencies']
            psd = results[channel]['psd']
            
            low_freq_power = np.sum(psd[freq < 2])
            mid_freq_power = np.sum(psd[(freq >= 2) & (freq < 20)])
            high_freq_power = np.sum(psd[freq >= 20])
            total = low_freq_power + mid_freq_power + high_freq_power
            
            f.write(f"  低频能量 (0-2Hz): {low_freq_power/total*100:.1f}%\n")
            f.write(f"  中频能量 (2-20Hz): {mid_freq_power/total*100:.1f}%\n")
            f.write(f"  高频能量 (>20Hz): {high_freq_power/total*100:.1f}%\n")
    
    print(f"\n分析完成！结果保存在: {save_dir}")
    print(f"  - 频谱图 (semi-log): psd_analysis.png")
    print(f"  - 频谱图 (log-log): psd_loglog.png")
    print(f"  - 分析报告: psd_report.txt")
    
    return results


if __name__ == '__main__':
    # 测试
    data_path = './dataset/data.csv'
    analyze_angle_data(data_path)
