"""
测试脚本: DCT信号增强
运行这个脚本来测试DCT增强效果
"""
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import os
from utils.dct_enhancement import DCTEnhancer, compare_signals

if __name__ == '__main__':
    print("=" * 50)
    print("开始DCT增强测试...")
    print("=" * 50)
    
    # 读取数据
    df = pd.read_csv('./dataset/data.csv')
    df_subset = df.iloc[6000:6500]  # 取500个点测试
    
    # 创建保存目录
    os.makedirs('./dct_results', exist_ok=True)
    
    # 测试三个通道
    channels = ['x_angle_1', 'y_angle_1', 'z_angle_1']
    
    # 创建增强器 - 参考FFT增强器的激进参数（极致平滑版）
    enhancer = DCTEnhancer(
        low_freq_boost=1.5,       # 低频增强50%
        high_freq_suppress=0.15,  # 高频抑制到15%
        cutoff_ratio=0.05,        # 只保留5%的低频系数（接近FFT的2%）
        low_freq_ratio=0.01,      # 低频区域：前1%
        high_freq_ratio=0.03,     # 高频区域：3%之后
        residual_ratio=0.7,       # 残差混合70%
        reflection_pad=128        # 反射填充128个点（增加平滑度）
    )
    
    for channel in channels:
        print(f"\n处理 {channel}...")
        
        # 获取信号
        signal = df_subset[channel].values
        
        # 增强
        enhanced = enhancer.enhance(signal)
        
        # 计算残差统计
        delta = enhanced - signal
        print(f"  残差均值 {delta.mean():.4e}, 残差标准差 {delta.std():.4e}, 峰值 {delta.max():.4f}/{delta.min():.4f}")
        
        # 对比并保存
        save_path = f'./dct_results/{channel}_comparison.png'
        compare_signals(signal, enhanced, title=channel, save_path=save_path)
        
        print(f"  保存到: {save_path}")
    
    print("\n" + "=" * 50)
    print("DCT增强测试完成！请查看 ./dct_results/ 目录")
    print("=" * 50)
