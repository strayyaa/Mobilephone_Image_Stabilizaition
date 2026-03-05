# freqMOE优化方案

## 问题分析

### 原始问题
根据训练结果分析，freqMOE存在严重的**专家塌缩（Expert Collapse）**问题：

```
专家00: 平均权重=0.9539, 覆盖比例=0.4950, 频段=[0, 50)
专家07: 平均权重=0.0256, 覆盖比例=0.3267, 频段=[68, 101)
...
```

低频专家占据了95%以上的权重，其他专家几乎没有被使用。

### 根本原因

1. **θ参数不可微**：使用`long()`和hard mask导致`theta`参数的梯度为0
2. **Gate天然倾向低频**：大多数时间序列的低频能量 >> 高频能量，Gate选择低频专家可以得到更小的loss
3. **缺乏显式反塌缩机制**：MoE结构没有被强制要求多专家工作

## 解决方案

### 方案A：Soft Frequency Mask（已实现）

使用可微的sigmoid窗函数替代hard mask：

```python
# 核心思想：用参数表示专家的"中心位置"和"宽度"
self.centers = nn.Parameter(torch.linspace(0.1, 0.9, num_experts))
self.widths = nn.Parameter(torch.ones(num_experts) * 0.2)

# 创建可微掩码
mask =((freq - left sigmoid_edge) * sharp) - sigmoid((freq - right_edge) * sharp)
```

**优点**：
- θ参与梯度传播
- 频段边界可学习、可移动

### 方案B：Frequency-wise Routing（已实现）

**每个频点一个Gate**：

```python
# 原来：所有频率共享一个gate权重 [B, N]
# 现在：每个频点一个gate权重 [B, F, N]
weights = softmax(gate(|X_f|))
Y_f = Σ_n w_f,n * X_f,n
```

**优点**：
- 无需显式θ划分
- 数据驱动自动学习频段分配

### 方案C：负载均衡Loss（已实现）

使用KL散度强制专家均衡工作：

```python
def get_load_balance_loss(weights):
    avg_weights = weights.mean(dim=0)
    uniform = torch.ones_like(avg_weights) / num_experts
    return F.kl_div(torch.log(avg_weights), uniform)
```

## 代码修改

### 1. freqMOE_enhancement.py

新增参数：
- `use_soft_mask`: 使用可微软掩码
- `sharpness`: 软掩码锐度参数
- `use_freq_wise_routing`: 频点级路由

### 2. run_longExp.py

新增参数：
- `--freqmoe_use_soft_mask`: 1/0
- `--freqmoe_sharpness`: 锐度(默认20.0)
- `--freqmoe_freq_wise_routing`: 1/0
- `--freqmoe_balance_weight`: 负载均衡权重(默认0.01)

### 3. exp_main.py

- 初始化时传入新参数
- 训练时计算负载均衡Loss并加入总损失

## 使用方法

### 推荐配置1（Soft Mask + 负载均衡）
```bash
python run_longExp.py \
  --is_training 1 \
  --model DLinear \
  --data your_data \
  --use_freqmoe \
  --freqmoe_experts 4 \
  --freqmoe_use_soft_mask 1 \
  --freqmoe_sharpness 20.0 \
  --freqmoe_balance_weight 0.01
```

### 推荐配置2（Frequency-wise Routing）
```bash
python run_longExp.py \
  --is_training 1 \
  --model DLinear \
  --data your_data \
  --use_freqmoe \
  --freqmoe_experts 4 \
  --freqmoe_freq_wise_routing 1 \
  --freqmoe_balance_weight 0.01
```

## 预期效果

优化后应该看到：
1. 专家权重分布更均匀（不再是95% vs 5%）
2. 各专家覆盖不同频段
3. MAE指标得到改善

## 参数调优建议

1. **sharpness**: 增大使频段边界更sharp，减小使频段更平滑
2. **balance_weight**: 
   - 过小（<0.001）：专家仍可能塌缩
   - 过大（>0.1）：可能影响重构质量
   - 建议从0.01开始
3. **num_experts**: 3-8个为宜，过多难以训练
