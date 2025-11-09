import os
import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        # print(cols)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            # print(self.scaler.mean_)
            # exit()
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Angle(Dataset):
    """
    使用x_angle_1, y_angle_1, z_angle_1三列
    数据范围：第6000到12000行
    预测任务：用200条预测200条
    """
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='data.csv',
                 target='z_angle_1', scale=True, timeenc=0, freq='h',
                 start_idx=6000, end_idx=12000,
                 # 新增数据增强相关参数
                 use_augmentation=True, jitter_sigma=0.05, scale_alpha=0.1,
                 use_smoothing=True, downsample_rate=2):
        # size [seq_len, label_len, pred_len]
        # 设置预测任务的参数：用200条预测200条
        if size == None:
            self.seq_len = 200      # 输入序列长度
            self.label_len = 48     # 保持与其他数据集类一致
            self.pred_len = 200     # 预测序列长度
        else:
            self.seq_len = size[0]
            self.label_len = size[1]  
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        # 角度数据特定参数
        self.start_idx = start_idx  # 开始行索引 (6000)
        self.end_idx = end_idx      # 结束行索引 (12000)
        self.angle_columns = ['x_angle_1', 'y_angle_1', 'z_angle_1']

        
        # 初始化数据增强参数
        self.use_augmentation = use_augmentation and flag == 'train'  # 只在训练集上做增强
        self.jitter_sigma = jitter_sigma
        self.scale_alpha = scale_alpha
        
        # --- 初始化平滑参数 (只在训练集上启用) ---
        self.use_smoothing = use_smoothing and flag == 'train'
        self.downsample_rate = downsample_rate
        
        if(self.use_augmentation):
            print(f"已应用use_augmentation")

        if(use_smoothing):
            print(f"已应用use_smoothing")

        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    @staticmethod
    def _plot_data_segment(data_segment, title, save_path):
        """绘制并保存数据段的曲线图"""
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        
        plt.figure(figsize=(20, 8))
        x_axis_col = 'x_angle_1'
        if x_axis_col in data_segment.columns:
            plt.plot(data_segment.index, data_segment[x_axis_col], label=x_axis_col)
        else:
            print(f"警告: 在可视化中找不到列 '{x_axis_col}'")
        
        plt.title(title, fontsize=16)
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(save_path)
        plt.close()
        print(f"已保存可视化图像到: {save_path}")

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        #print(f"原始数据形状: {df_raw.shape}")
        #print(f"使用数据范围: 第{self.start_idx}到第{self.end_idx}行")
        
        # 检查数据范围是否有效
        if self.end_idx > len(df_raw):
            raise ValueError(f"结束索引 {self.end_idx} 超出数据范围 {len(df_raw)}")
        if self.start_idx >= self.end_idx:
            raise ValueError(f"开始索引 {self.start_idx} 必须小于结束索引 {self.end_idx}")
            
        # 提取指定范围的数据
        df_subset = df_raw.iloc[self.start_idx:self.end_idx].copy()
        #print(f"提取后数据形状: {df_subset.shape}")
        
        
        # 检查是否包含所需的角度列
        missing_cols = [col for col in self.angle_columns if col not in df_subset.columns]
        if missing_cols:
            raise ValueError(f"缺少所需的列: {missing_cols}")
            
        # 只使用三个角度列作为特征
        df_data = df_subset[self.angle_columns]
        #print(f"角度数据列: {list(df_data.columns)}")
        #print(f"角度数据形状: {df_data.shape}")
        
        # --- 3. 平滑和可视化逻辑 ---
        if self.set_type == 0 and self.use_smoothing and self.downsample_rate > 1:
            
            vis_len = 200
            
            # a. 绘制平滑前的数据
            data_before_smoothing = df_data.iloc[:vis_len]
            self._plot_data_segment(
                data_segment=data_before_smoothing,
                title=f'Data Segment Before Smoothing (first {vis_len} points)',
                save_path='./smoothing_visualization/before_smoothing.png'
            )

            # b. 创建一个临时变量用于可视化平滑效果，而不是修改 df_data
            original_len = len(data_before_smoothing)
            original_index = np.arange(original_len)
            print(f"降采样rate：{self.downsample_rate}")
            downsampled_values = data_before_smoothing.values[::self.downsample_rate]
            downsampled_index = original_index[::self.downsample_rate]
            
            interpolated_values = np.zeros_like(data_before_smoothing.values)
            for i in range(data_before_smoothing.shape[1]):
                interpolated_values[:, i] = np.interp(original_index, downsampled_index, downsampled_values[:, i])
            # 使用临时变量创建平滑后的 DataFrame 用于绘图
            data_after_smoothing = pd.DataFrame(interpolated_values, columns=data_before_smoothing.columns, index=data_before_smoothing.index)
            
            # c. 绘制平滑后的数据
            self._plot_data_segment(
                data_segment=data_after_smoothing,
                title=f'Data Segment After Smoothing (Downsample Rate: {self.downsample_rate})',
                save_path='./smoothing_visualization/after_smoothing.png'
            )
            
            print(f"已生成平滑前后对比图。平滑处理将在每个训练样本上独立应用。")
        # --- 平滑和可视化逻辑结束 ---

        # 数据分割：70% 训练, 20% 验证, 10% 测试
        total_len = len(df_data)
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.2)
        test_len = total_len - train_len - val_len
        
        #print(f"数据分割 - 训练: {train_len}, 验证: {val_len}, 测试: {test_len}")
        
        # 设置边界索引
        border1s = [0, train_len - self.seq_len, train_len + val_len - self.seq_len]
        border2s = [train_len, train_len + val_len, total_len]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        #print(f"当前数据集类型: {['train', 'val', 'test'][self.set_type]}")
        #print(f"数据边界: [{border1}, {border2})")

        # 数据标准化
        if self.scale:
            # 使用训练数据计算标准化参数
            train_data = df_data.iloc[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            #print("已应用数据标准化")
        else:
            data = df_data.values
            #print("未使用数据标准化")

        time_stamps = np.arange(border2 - border1).reshape(-1, 1)
        
        # 设置数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = time_stamps
        
        #print(f"最终数据形状 - X: {self.data_x.shape}, Y: {self.data_y.shape}, Stamp: {self.data_stamp.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len if self.label_len > 0 else s_end
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]  
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # 1. 平滑处理
        if self.use_smoothing and self.downsample_rate > 1:
            original_len = len(seq_x)
            original_index = np.arange(original_len)
            
            # 降采样
            downsampled_values = seq_x[::self.downsample_rate]
            downsampled_index = original_index[::self.downsample_rate]
            
            # 插值以恢复原始长度
            interpolated_values = np.zeros_like(seq_x)
            for i in range(seq_x.shape[1]): # 对每一列（x, y, z）分别进行插值
                interpolated_values[:, i] = np.interp(original_index, downsampled_index, downsampled_values[:, i])
            
            seq_x = interpolated_values


        # 数据增强
        if self.use_augmentation:
            # 1. Jittering (抖动): 添加高斯噪声
            jitter = np.random.normal(loc=0., scale=self.jitter_sigma, size=seq_x.shape)
            seq_x = seq_x + jitter

            # 2. Scaling (缩放): 乘以一个随机缩放因子
            scaling_factor = np.random.uniform(low=1.0 - self.scale_alpha, high=1.0 + self.scale_alpha,size=(1, seq_x.shape[1]))
            seq_x = seq_x * scaling_factor
            


        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # 确保有足够的数据进行预测
        available_len = len(self.data_x) - self.seq_len - self.pred_len + 1
        return max(0, available_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
