import pandas as pd

# 读取原始CSV文件
df = pd.read_csv('data.csv')

# 提取指定的三列
new_df = df[['x_angle_1', 'y_angle_1', 'z_angle_1']] * 10000

# 保存到新的CSV文件
new_df.to_csv('new_data.csv', index=False)

print("已成功提取并保存到 new_data.csv")