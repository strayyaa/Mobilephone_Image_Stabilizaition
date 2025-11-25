"""
测试脚本: PSD分析
运行这个脚本来分析陀螺仪数据的频谱特性
"""
import sys
sys.path.append('.')

from utils.signal_analysis import analyze_angle_data

if __name__ == '__main__':
    print("=" * 50)
    print("开始PSD分析...")
    print("=" * 50)
    
    # 分析数据
    results = analyze_angle_data(
        data_path='./dataset/data.csv',
        start_idx=6000,
        end_idx=12000,
        save_dir='./psd_results'
    )
    
    print("\n分析完成！请查看 ./psd_results/ 目录")
