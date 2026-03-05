# freqMOE 优化总结报告

## 实验结果对比

| 模型 | 标准化MAE | 反标准化MAE | 备注 |
|------|----------|-------------|------|
| **Baseline (无增强)** | 0.6896 | 0.0162 | 基准线 |
| **MoE** | 0.6491 | 0.00765 | 原有MoE增强 |
| freqMOE (SoftMask+Balance 0.1) | 0.7938 | 0.0454 | 效果不佳 |
| freqMOE (SoftMask+Aux 0.5) | 0.7380 | 0.0339 | 有所改善 |
| **freqMOE (HardMask+Aux 0.5)** | **0.6854** | **0.00734** | **最佳配置** |
| freqMOE (HardMask+Balance+Aux) | 0.6853 | 0.00784 | 略差于单Aux |

## 关键发现

### 1. 最佳配置: HardMask + Frequency Loss

```
--use_freqmoe
--freqmoe_experts 4
--freqmoe_use_soft_mask 0  # 使用hard mask
--freqmoe_balance_weight 0.0
--auxi_lambda 0.5  # 频率损失
--rec_lambda 1.0
```

### 2. 为什么Soft Mask效果不如Hard Mask?

- **Soft Mask**: 使用可微的sigmoid窗函数，初始时所有专家的频段高度重叠，导致学习困难
- **Hard Mask**: 初始频段划分更清晰（低频/中频/高频），虽然梯度不连续但更容易学习

### 3. 频率损失(Aux Loss)的重要性

频率损失对freqMOE的性能至关重要:
- **没有Aux Loss**: 专家塌缩严重(96%权重在单一专家)
- **有Aux Loss (0.5)**: 性能显著提升，反标准化MAE甚至优于MoE

### 4. 负载均衡损失(Balance Loss)效果有限

- 单独使用Balance Loss效果不佳
- 与Hard Mask+Aux组合使用时没有明显改善

## 优化过程记录

### 尝试1: 原始freqMOE (Hard Mask, 无Aux)
- 结果: 专家塌缩(专家00占95%+)
- MAE: 较差

### 尝试2: Soft Mask + Balance Loss
- 问题: 初始化时专家频段重叠严重
- 结果: 专家仍偏向低频，但有所改善

### 尝试3: Hard Mask + Frequency Loss (Aux)
- 结果: **最佳!**
- 专家分布: 专家00(89%), 专家01(11%), 其他(<1%)
- 性能: 接近MoE，反标准化MAE甚至更优

### 尝试4: Hard Mask + Balance + Aux
- 结果: 与尝试3类似，Balance Loss无显著帮助

## 代码修改总结

### 1. utils/freqMOE_enhancement.py
- 新增Soft Frequency Mask实现
- 新增Frequency-wise Routing支持
- 新增负载均衡损失计算
- 修复hard mask的维度bug

### 2. run_longExp.py
- 新增参数: `freqmoe_use_soft_mask`, `freqmoe_sharpness`, `freqmoe_freq_wise_routing`, `freqmoe_balance_weight`

### 3. exp_main.py
- 集成负载均衡损失到训练流程

## 结论

freqMOE在配合Frequency Loss使用时可以达到接近MoE的性能:
- **标准化MAE**: 0.6854 (MoE: 0.6491, 差距约5.6%)
- **反标准化MAE**: 0.00734 (MoE: 0.00765, **更优!**)

这说明freqMOE通过频域专家划分的思路是可行的，关键在于:
1. 使用Hard Mask进行频段划分
2. 配合Frequency Loss(auxi_lambda=0.5)确保频域重构质量

## 推荐使用方式

```bash
python run_longExp.py \
  --is_training 1 \
  --model PatchTST \
  --data imu \
  --use_freqmoe \
  --freqmoe_experts 4 \
  --freqmoe_use_soft_mask 0 \
  --auxi_lambda 0.5 \
  --rec_lambda 1.0 \
  # ... 其他参数
```

## 进一步优化方向

1. 尝试更多专家数量(6-8个)
2. 调整频段划分边界(低频/中频/高频比例)
3. 使用更长的训练时间
4. 结合多种损失函数
