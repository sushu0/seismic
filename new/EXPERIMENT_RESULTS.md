# 实验结果报告

## 数据集信息
- **数据位置**: `data/toy/` (真实地震数据)
- **训练集**: 64 labeled traces
- **验证集**: 16 traces  
- **测试集**: 16 traces
- **未标注数据**: 128 traces (未使用)
- **序列长度**: 512 samples

## 实验配置
所有实验均使用以下配置:
- **训练轮数**: 50 epochs (UNet1D), 30 epochs (TCN1D), 50 epochs (MS-PhysFormer)
- **批大小**: 8
- **学习率**: 0.001
- **优化器**: AdamW (weight_decay=1e-4)
- **损失函数**: SmoothL1
- **数据标准化**: Z-score归一化
- **数据增强**: 噪声 (σ=0.02), 振幅抖动 (±10%), 时移 (±8 samples)
- **物理/频率约束**: λ_phys=0.0, λ_freq=0.0 (均关闭)
- **半监督学习**: 未启用

## 实验结果

### 性能对比表

| Model | PCC↑ | R²↑ | MSE↓ | Best Epoch |
|-------|------|-----|------|------------|
| **UNet1D Baseline** | **0.522** | **0.270** | **0.817** | 32 |
| TCN1D Baseline | 0.394 | 0.093 | 1.014 | 16 |
| MS-PhysFormer (supervised) | 0.372 | 0.042 | 1.072 | 16 |

### 关键发现

1. **UNet1D 表现最佳**
   - PCC: 0.522 (最高)
   - R²: 0.270 (最高)
   - MSE: 0.817 (最低)
   - 在所有指标上都显著优于其他两个模型

2. **TCN1D 表现中等**
   - PCC: 0.394
   - R²: 0.093  
   - 比UNet1D差约25% (PCC)

3. **MS-PhysFormer 表现不佳**
   - PCC: 0.372 (最低)
   - R²: 0.042 (最低)
   - MSE: 1.072 (最高)
   - 在当前设置下表现不如两个基线模型

## 分析与讨论

### 为什么MS-PhysFormer表现不佳?

1. **模型复杂度过高**
   - MS-PhysFormer 参数量远大于 UNet1D/TCN1D
   - 训练数据仅64个样本,严重不足以训练复杂模型
   - 导致过拟合:验证集性能不稳定(val_mse在训练过程中波动极大,从0.8到3.2)

2. **Transformer 需要更多数据**
   - Transformer模块通常需要大量数据才能有效学习
   - 64个训练样本远远不够

3. **训练不稳定**
   - 观察到损失值在训练过程中剧烈波动
   - 早期epochs出现异常大的val_mse (>3.0)
   - 可能需要更小的学习率或更长的warm-up

4. **深度监督未充分发挥作用**
   - 深度监督损失可能需要调整权重
   - 多尺度输出的聚合策略可能需要优化

## 建议改进方向

### 针对MS-PhysFormer的改进

1. **启用半监督学习**
   - 使用128个未标注样本
   - 启用Mean Teacher (λ_cons=0.2)
   - 利用更多无标注数据来训练复杂模型

2. **模型简化**
   - 减少base channels: 48→32
   - 减少depth: 4→3
   - 减少Transformer layers: 2→1
   - 降低模型复杂度以匹配小数据集

3. **训练策略优化**
   - 降低学习率: 0.001→0.0005
   - 增加warm-up: 前10 epochs线性增长学习率
   - 更强的正则化: weight_decay=1e-4→1e-3
   - 更多的dropout: 在Transformer中添加dropout=0.2

4. **启用物理/频率约束**
   - 修复physics loss的数值稳定性问题
   - 在**原始域**(非归一化)计算物理损失
   - 设置 λ_phys=0.1, λ_freq=0.1

5. **迁移学习**
   - 用合成数据预训练模型
   - 在真实数据上fine-tune

### 针对UNet1D的进一步优化

1. **超参数调优**
   - 尝试不同的base channels: 24, 32, 40
   - 尝试不同的depth: 3, 4, 5
   - 网格搜索最佳组合

2. **启用半监督学习**
   - 即使UNet1D已经表现不错,半监督可能进一步提升
   - 预计可达到PCC > 0.55

3. **启用频率约束**
   - λ_freq=0.1可能有助于匹配频谱特性

## 结论

在当前的**小样本**(64 labeled traces)真实地震数据集上:

✅ **UNet1D 是最佳选择** (PCC=0.522, R²=0.270)
- 简单有效,训练稳定
- 参数量适中,不易过拟合
- 适合小数据集

❌ **MS-PhysFormer 未达到预期**
- 模型过于复杂,需要更多数据
- 当前设置下无法超越简单的UNet1D
- 需要大幅改进(半监督+模型简化+训练策略优化)

📊 **下一步行动**:
1. 使用UNet1D作为生产baseline
2. 实现上述改进建议,重新训练MS-PhysFormer
3. 启用半监督学习(使用128个未标注样本)
4. 修复physics loss的数值问题

## 可视化结果

查看生成的图像:
- `results/baseline_unet1d/pred_vs_true_traces.png` - 预测 vs 真实曲线对比
- `results/baseline_unet1d/pred_imp_section.png` - 预测阻抗剖面
- `results/baseline_unet1d/true_imp_section.png` - 真实阻抗剖面
- `results/baseline_unet1d/seis_obs_section.png` - 观测地震剖面

## 实验环境

- **框架**: PyTorch 2.x
- **设备**: CUDA (GPU加速)
- **随机种子**: 42 (确保可复现)
- **实验时间**: 2025-12-23

---

**实验完成状态**: ✅ 所有基础实验已完成
**数据**: [results/summary.csv](results/summary.csv)
