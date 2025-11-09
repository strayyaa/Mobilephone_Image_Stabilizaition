from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred, Dataset_Angle
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'angle': Dataset_Angle,
}


def data_provider(args, flag):
    Data = data_dict[args.data]  # 数据集类在其内部实现了滑动窗口的逻辑？
    # 是否使用时间特征编码 timeenc=1表示使用时间特征编码
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False  # 测试集不进行数据打乱
        drop_last = True  # 丢弃最后一个不完整的batch
        batch_size = args.batch_size
        freq = args.freq  # 测试集的时间频率
    elif flag == 'pred':
        shuffle_flag = False  # 预测时不进行数据打乱
        drop_last = False  # 预测时不丢弃最后一个不完整的batch
        batch_size = 1  # 预测时每次只输入一条数据
        freq = args.freq
        Data = Dataset_Pred  # 预测时使用Dataset_Pred类
    else:
        shuffle_flag = True  # 训练集可以进行样本打乱，模型学习的是一个样本内的时间序列特征
        drop_last = True  # 训练集丢弃最后一个不完整的batch
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,  # 数据集根目录
        data_path=args.data_path,  # 数据集路径
        flag=flag,    # flag=train表示训练集，flag=test表示测试集，flag=pred表示预测集
        size=[args.seq_len, args.label_len, args.pred_len], # 输入/标签/预测长度
        features=args.features, # 特征类型（多变量/单变量/时间序列分类）
        target=args.target, # 目标变量名称
        timeenc=timeenc, # 时间特征编码
        freq=freq,  # 时间频率
        use_augmentation=args.use_augmentation,
        jitter_sigma=args.jitter_sigma,
        scale_alpha=args.scale_alpha,
        use_smoothing=args.use_smoothing,
        downsample_rate=args.downsample_rate
    )
    print(f"当前样本类型: {flag}, 该类型样本数量: {len(data_set)}")
    # 构造pytorch数据加载器
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers, # 使用多线程加载数据
        drop_last=drop_last # 丢弃最后一个不完整的batch
    )  
    return data_set, data_loader
