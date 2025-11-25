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
        low_freq_ratio=0.25,
        high_freq_ratio=0.7,
        low_freq_boost=1.3,
        mid_freq_boost=1.1,
        high_freq_suppress=0.2,
        cutoff_ratio=0.9,
    )

    for channel in channels:
        print(f"\n处理 {channel}...")
        signal = df_subset[channel].values
        enhanced = enhancer.enhance(signal)
        save_path = f'./fft_results/{channel}_comparison.png'
        compare_signals_fft(signal, enhanced, title=f'{channel} (FFT)', save_path=save_path)
        print(f'  保存到: {save_path}')

    print('\n' + '=' * 50)
    print('FFT 增强测试完成！请查看 ./fft_results/ 目录')
    print('=' * 50)
