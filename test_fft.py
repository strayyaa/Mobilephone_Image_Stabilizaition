"""测试脚本: FFT 信号增强"""
import sys
sys.path.append('.')

import os
import pandas as pd

from utils.fft_enhancement import FFTEnhancer, compare_signals_fft


if __name__ == '__main__':
    print('=' * 50)
    print('开始 FFT 增强测试...')
    print('=' * 50)

    df = pd.read_csv('./dataset/data.csv')
    df_subset = df.iloc[6000:6500]

    os.makedirs('./fft_results', exist_ok=True)

    channels = ['x_angle_1', 'y_angle_1', 'z_angle_1']
    enhancer = FFTEnhancer(
        low_freq_ratio=0.005,
        high_freq_ratio=0.01,
        low_freq_boost=1.3,
        mid_freq_boost=1.0,
        high_freq_suppress=0.2,
        cutoff_ratio=0.02,
        residual_ratio=0.7,
        reflection_pad=64,
    )

    for channel in channels:
        print(f"\n处理 {channel}...")
        signal = df_subset[channel].values
        enhanced = enhancer.enhance(signal)
        delta = enhanced - signal
        print(f"  残差均值 {delta.mean():.4e}, 残差标准差 {delta.std():.4e}, 峰值 {delta.max():.4f}/{delta.min():.4f}")
        save_path = f'./fft_results/{channel}_comparison.png'
        compare_signals_fft(signal, enhanced, title=f'{channel} (FFT 残差增强+反射填充)', save_path=save_path)
        print(f'  保存到: {save_path}')

    print('\n' + '=' * 50)
    print('FFT 增强测试完成！请查看 ./fft_results/ 目录')
    print('=' * 50)
