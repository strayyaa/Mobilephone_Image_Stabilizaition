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
    
    # 创建增强器
    enhancer = DCTEnhancer(
        low_freq_boost=1.3,      # 低频增强30%
        high_freq_suppress=0.2,  # 高频抑制到20%
        cutoff_ratio=0.7         # 保留70%的系数
    )
    
    for channel in channels:
        print(f"\n处理 {channel}...")
        
        # 获取信号
        signal = df_subset[channel].values
        
        # 增强
        enhanced = enhancer.enhance(signal)
        
        # 对比并保存
        save_path = f'./dct_results/{channel}_comparison.png'
        compare_signals(signal, enhanced, title=channel, save_path=save_path)
        
        print(f"  保存到: {save_path}")
    
    print("\n" + "=" * 50)
    print("DCT增强测试完成！请查看 ./dct_results/ 目录")
    print("=" * 50)
