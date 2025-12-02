import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
from statsmodels.tsa.arima.model import ARIMA
import torch.nn.functional as F

# ======================
# 设置中文字体
# ======================
plt.rcParams['font.sans-serif'] = ['PingFang HK']  
plt.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# ======================
# 数据准备
# ======================
df = pd.read_csv('data.csv').dropna()
df_section = df[(df.index >= 6000) & (df.index <= 12000)].reset_index(drop=True)

angles = df_section[['X角度(°)', 'Y角度', 'Z角度']].values
angles = torch.tensor(angles, dtype=torch.float32)

# ---- 归一化（z-score）: 针对每个维度分别归一化 ----
mean = angles.mean(dim=0)
std = angles.std(dim=0)
angles_norm = (angles - mean) / std

# ---- 数据切分 ----
seq_len = 200
pred_len = 200
train_size = int(0.8 * len(angles_norm))
train_data = angles_norm[:train_size]
test_data = angles_norm[train_size:train_size*2]

#调参：测试不同变体的DLinear
use_arima_trend = False
use_arima_trend_comp = False
use_fft_promote_feature = False
use_fft_plus_feature = False
use_ifft_while_using_fft_plus = False
use_merge = False
use_broader_individual = False
use_mix_Linear = False
use_autoformer = False
use_full_autoformer = False

# ---- 构造样本 ----
def create_dataset(series, seq_len, pred_len):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    return torch.stack(X), torch.stack(y)

X_train, y_train = create_dataset(train_data, seq_len, pred_len)
X_test, y_test = create_dataset(test_data, seq_len, pred_len)

def compute_trend(X, order=(2,1,0)):
    from statsmodels.tsa.arima.model import ARIMA
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    trend = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[2]):
            series = X[i, :, j]
            try:
                model = ARIMA(series, order=order)
                fitted = model.fit()
                # 如果拟合失败或参数异常，则fallback
                if not np.isfinite(fitted.aic):
                    raise ValueError("Invalid fit result")
                trend[i, :, j] = fitted.fittedvalues[-len(series):]
            except Exception:
                trend[i, :, j] = np.convolve(series, np.ones(5)/5, mode='same')  # fallback
    return trend

def fft_feature(x,top_k=10):
    ffted_x = torch.fft.rfft(x, dim=1)
    mag = torch.abs(ffted_x)
    topk_mag = mag[:,:top_k,:]

    return topk_mag

# ======================
# ===== 原始 DLinear 模型 =====
# ======================
class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1)//2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1)//2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)
        self.pre_computed_trend = None

    def forward(self, x):
        if use_arima_trend:
            if self.pre_computed_trend is not None:
                trend = torch.tensor(self.pre_computed_trend, dtype=x.dtype, device=x.device)
                seasonal = x - trend
                return seasonal, trend
            else:
                trend = torch.tensor(compute_trend(x.cpu().numpy()), dtype=x.dtype, device=x.device)
                seasonal = x - trend
                return seasonal, trend
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class PositionalEncoding(nn.Module):
    def __init__(self, seq_len, d_model):
        super(PositionalEncoding, self).__init__()
        # 正弦固定编码
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        # 注册为 buffer（不是参数，不会更新）
        self.register_buffer('pe', pe.unsqueeze(0))  # shape [1, seq_len, d_model]
        # 也可以改为 learnable
        self.learnable_pe = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.xavier_uniform_(self.learnable_pe)

    def forward(self, x):
        # x: [B, seq_len, C]
        # 返回带位置编码的 x
        return x + self.pe[:, :x.size(1), :] + self.learnable_pe[:, :x.size(1), :]

class AutoCorrelation(nn.Module):
    def __init__(self, top_k=3):
        super().__init__()
        self.top_k = top_k

    def forward(self, Q, K, V):
        """
        Q, K, V: [B, L, D]
        输出: [B, L, D]
        """
        B, L, D = Q.shape

        q_fft = torch.fft.rfft(Q, dim=1)
        k_fft = torch.fft.rfft(K, dim=1)
        corr = torch.fft.irfft(q_fft * torch.conj(k_fft), n=L, dim=1)  # [B, L, D]

        corr_mean = corr.mean(dim=-1)  # [B, L]

        topk = torch.topk(corr_mean, self.top_k, dim=-1)  # (values, indices)
        topk_val = topk.values  # [B, K]
        topk_idx = topk.indices  # [B, K]

        attn = F.softmax(topk_val, dim=-1)  # [B, K]

        out = torch.zeros_like(V)
        for i in range(self.top_k):
            tau = topk_idx[:, i]  # [B]
            weight = attn[:, i].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]

            # 对每个 batch 执行 roll
            rolled = torch.stack([
                torch.roll(V[b], shifts=-int(tau[b].item()), dims=0)
                for b in range(B)
            ], dim=0)

            out += rolled * weight  # 加权聚合

        return out

class AutoCorrelationLayer(nn.Module):
    def __init__(self,top_k_corr=4):
        super(AutoCorrelationLayer, self).__init__()
        self.auto_correlation = AutoCorrelation(top_k=top_k_corr)
        self.decomp = series_decomp(kernel_size=25)
    
    def forward(self, S, T, V):
        Temp = self.auto_correlation(S,V,V)
        S,Delta = self.decomp(Temp+S)
        T = T + Delta
        return S,T

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=True, use_pos_encoding=False,top_k_fft=64,top_k_corr=4,layersCnt=1,layersCnt_Encoder=1):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        self.pos_encoder = PositionalEncoding(seq_len, enc_in)
        self.use_pos_encoding = use_pos_encoding
        self.arima_trend = None
        if use_fft_promote_feature:
            self.output = nn.Linear(enc_in,3)
        self.top_k_fft = top_k_fft
        if use_fft_plus_feature:
            self.freq_linear = nn.Linear(top_k_fft, pred_len)
        self.merge_rate = nn.Parameter(torch.tensor(0.3))
        if use_ifft_while_using_fft_plus:
            self.freq_linear_real = nn.Linear(top_k_fft,top_k_fft)
            self.freq_linear_imag = nn.Linear(top_k_fft,top_k_fft)
        if use_merge:
            self.merge_between_season_and_trend = nn.Parameter(torch.tensor(1.0))
        if use_broader_individual:
            self.broader_individual_Seasonal = nn.Linear(self.channels*seq_len, self.channels*pred_len)
            self.broader_individual_Trend = nn.Linear(self.channels*seq_len, self.channels*pred_len)
        if use_mix_Linear:
            self.mix_linear = nn.Linear(self.channels, self.channels)
        if use_autoformer:
            self.auto_correlation_layer = AutoCorrelationLayer(top_k_corr=top_k_corr)
            self.layersCnt = layersCnt
        if use_full_autoformer:
            self.layersCnt_Encoder = layersCnt_Encoder
            self.auto_correlation_layer_x = AutoCorrelationLayer(top_k_corr=top_k_corr)
            self.auto_correlation_cross = AutoCorrelationLayer(top_k_corr=top_k_corr)

        kernel_size = 25
        self.decomposition = series_decomp(kernel_size)

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):

        #位置编码
        if self.use_pos_encoding:
            x = self.pos_encoder(x)
        if use_fft_promote_feature:
            freq_feats = fft_feature(x,top_k=self.top_k_fft)
            freq_feats = torch.log1p(freq_feats)  # 缩放大幅度差异
            freq_feats = (freq_feats - freq_feats.mean(dim=1, keepdim=True)) / (freq_feats.std(dim=1, keepdim=True) + 1e-6)
            freq_feats = torch.nn.functional.interpolate(freq_feats.permute(0, 2, 1), size=x.shape[1]).permute(0, 2, 1)
            # 拼接在时域输入上
            x = torch.cat([x, freq_feats], dim=2)
        if use_fft_plus_feature:
            freq_feats = fft_feature(x,top_k=self.top_k_fft)
            freq_feats = torch.log1p(freq_feats)  # 缩放大幅度差异
            freq_feats = (freq_feats - freq_feats.mean(dim=1, keepdim=True)) / (freq_feats.std(dim=1, keepdim=True) + 1e-6)
            freq_feats = self.freq_linear(freq_feats.permute(0, 2, 1))
        if use_ifft_while_using_fft_plus:
            # 反傅里叶变换还原时域特征
            freq_feats = torch.fft.rfft(x, dim=1)
            freq_feats = freq_feats[:,:self.top_k_fft,:]
            freq_feats_real = freq_feats.real
            freq_feats_imag = freq_feats.imag
            freq_feats_real = self.freq_linear_real(freq_feats_real.permute(0,2,1))
            freq_feats_imag = self.freq_linear_imag(freq_feats_imag.permute(0,2,1))
            freq_feats = torch.complex(freq_feats_real, freq_feats_imag)
            freq_feats = torch.fft.irfft(freq_feats, n=self.pred_len, dim=2)


        seasonal_init, trend_init = self.decomposition(x)
        # x: [Batch, Input length, Channel]
        if use_full_autoformer:
            for _ in range(self.layersCnt_Encoder):
                x,_ = self.auto_correlation_layer_x(x,x,x)

        if use_autoformer:
            for _ in range(self.layersCnt):
                seasonal_init, trend_init = self.auto_correlation_layer(seasonal_init, trend_init,seasonal_init)
                if use_full_autoformer:
                    seasonal_init,trend_init = self.auto_correlation_cross(seasonal_init, trend_init,x)

        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)

        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), self.channels, self.pred_len],
                                          dtype=seasonal_init.dtype,
                                          device=seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), self.channels, self.pred_len],
                                       dtype=trend_init.dtype,
                                       device=trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                if use_arima_trend_comp:
                    if self.arima_trend is not None:
                        trend_output[:, i, :] = torch.tensor(self.arima_trend[:, :, i],
                                                            dtype=trend_init.dtype,
                                                            device=trend_init.device)
                else:
                    trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            if use_broader_individual:
                seasonal_init_reshaped = seasonal_init.reshape(seasonal_init.size(0), -1)
                trend_init_reshaped = trend_init.reshape(trend_init.size(0), -1)
                seasonal_output = self.broader_individual_Seasonal(seasonal_init_reshaped)
                trend_output = self.broader_individual_Trend(trend_init_reshaped)
                seasonal_output = seasonal_output.reshape(seasonal_init.size(0), self.channels, self.pred_len)
                trend_output = trend_output.reshape(trend_init.size(0), self.channels, self.pred_len)
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            if use_arima_trend_comp:
                if self.arima_trend is not None:
                    trend_output = torch.tensor(self.arima_trend,
                                               dtype=trend_init.dtype,
                                               device=trend_init.device)
            else:
                trend_output = self.Linear_Trend(trend_init)

        if use_merge:
            x = self.merge_between_season_and_trend*seasonal_output + \
            (1-self.merge_between_season_and_trend)*trend_output
        else:
            x = seasonal_output + trend_output
        if use_fft_promote_feature:
            x = self.output(x.permute(0,2,1))
            return x
        if use_fft_plus_feature or use_ifft_while_using_fft_plus:
            x = (2-self.merge_rate)*x + self.merge_rate*freq_feats
        if use_mix_Linear:
            x = self.mix_linear(x.permute(0,2,1))
            x = x.permute(0,2,1)
        return x.permute(0, 2, 1)  # [Batch, Pred_len, Channel]

# ======================
# 训练函数
# ======================
def train_model(model, X_train, y_train, num_epochs=1200, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loss_list = []
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        if epoch>300: 
            loss_list.append(loss.item())
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    plt.plot(loss_list)
    plt.xlabel('Epoch(-300)')
    plt.ylabel('MSE Loss')
    return model

# ======================
# 模型训练与预测
# ======================

# 如果使用 ARIMA 计算趋势部分
if use_arima_trend or use_arima_trend_comp:
    print("使用 ARIMA 计算趋势部分")
    trend_train = compute_trend(X_train)
    trend_test = compute_trend(X_test)
    print("趋势部分计算完成")


dlinear_model = DLinear(seq_len, pred_len, enc_in=3, individual=True)
if use_fft_promote_feature:
    print("使用 FFT 频域特征增强输入")
    dlinear_model = DLinear(seq_len, pred_len, enc_in=3*2, individual=True)


if use_arima_trend:
    dlinear_model.decomposition.pre_computed_trend = trend_train
if use_arima_trend_comp:
    dlinear_model.arima_trend = trend_train
dlinear_model = train_model(dlinear_model, X_train, y_train)
if use_fft_plus_feature or use_ifft_while_using_fft_plus:
    print("merge_rate:",dlinear_model.merge_rate.item())
if use_merge:
    print("merge_between_season_and_trend:",dlinear_model.merge_between_season_and_trend.item())

with torch.no_grad():
    if use_arima_trend:
        dlinear_model.decomposition.pre_computed_trend = trend_test
    if use_arima_trend_comp:
        dlinear_model.arima_trend = trend_test
    y_pred = dlinear_model(X_test).detach().numpy()
    y_true = y_test.detach().numpy()

# ---- 反归一化 ----
X_test = X_test * std.numpy() + mean.numpy()
y_pred = y_pred * std.numpy() + mean.numpy()
y_true = y_true * std.numpy() + mean.numpy()

print(X_test.shape)
print(y_pred.shape)
print(y_true.shape)

# ======================
# MAE 计算
# ======================

mae_x = mean_absolute_error(y_true[:, :, 0].flatten(), y_pred[:, :, 0].flatten())
mae_y = mean_absolute_error(y_true[:, :, 1].flatten(), y_pred[:, :, 1].flatten())
mae_z = mean_absolute_error(y_true[:, :, 2].flatten(), y_pred[:, :, 2].flatten())
print(f"MAE(X)={mae_x:.4f}, MAE(Y)={mae_y:.4f}, MAE(Z)={mae_z:.4f}")

# ======================
# 绘图
# ======================
fig, axs = plt.subplots(3, 5, figsize=(12, 10))
time_axis = np.arange(seq_len+pred_len)

for i in range(5):

    axs[0][i].plot(time_axis,np.concatenate([X_test[i*200,:,0],y_true[i*200,:,0]],axis=0), label='真实 X角度', color='blue')
    axs[0][i].plot(time_axis, np.concatenate([X_test[i*200,:,0],y_pred[i*200,:,0]],axis=0), label='预测 X角度', color='orange')
    mae_x = mean_absolute_error(y_true[i, :, 0], y_pred[i, :, 0])
    axs[0][i].set_title(f'X角度预测对比 (MAE={mae_x:.4f})')
    axs[0][i].legend()

    axs[1][i].plot(time_axis, np.concatenate([X_test[i*200,:,1],y_true[i*200,:,1]],axis=0), label='真实 Y角度', color='blue')
    axs[1][i].plot(time_axis, np.concatenate([X_test[i*200,:,1],y_pred[i*200,:,1]],axis=0), label='预测 Y角度', color='orange')
    mae_y = mean_absolute_error(y_true[i, :, 1], y_pred[i, :, 1])
    axs[1][i].set_title(f'Y角度预测对比 (MAE={mae_y:.4f})')
    axs[1][i].legend()

    axs[2][i].plot(time_axis, np.concatenate([X_test[i*200,:,2],y_true[i*200,:,2]],axis=0), label='真实 Z角度', color='blue')
    axs[2][i].plot(time_axis, np.concatenate([X_test[i*200,:,2],y_pred[i*200,:,2]],axis=0), label='预测 Z角度', color='orange')
    mae_z = mean_absolute_error(y_true[i, :, 2], y_pred[i, :, 2])
    axs[2][i].set_title(f'Z角度预测对比 (MAE={mae_z:.4f})')
    axs[2][i].legend()

plt.tight_layout()
plt.show()
