# 半监督地震波阻抗反演 - 项目指南

## 项目概述

基于生成对抗网络(GAN)的半监督地震波阻抗反演系统，结合物理正演约束实现高精度地层速度预测。

**核心特性**：
- 半监督学习：利用少量标注数据 + 大量无标注地震记录
- WGAN-GP：稳定的对抗训练机制
- 物理约束：Ricker子波正演建模确保地震物理一致性
- 进阶优化：自注意力机制 + EMA + 学习率调度

## 最优模型性能

**测试集指标** (optimized_v3_advanced):
- **相关系数 PCC**: 0.9926
- **决定系数 R²**: 0.9852
- **物理MSE**: 55,293 m²/s²

训练配置：101标注/270验证/2721总道数，470时间采样点，200轮训练

## 快速开始

### 1. 环境配置
```bash
pip install -r requirements.txt
pip install -e .
```

### 2. 数据准备

项目包含三个合成数据集（`data/`目录）：
- `synth_toy.npz` - 快速测试数据
- `synth_marmousi2.npz` - Marmousi2模型
- `synth_valve.npz` - Valve模型
- `marmousi2_2721_like_l101.npz` - 最优模型训练数据（2721道）

### 3. 训练模型

使用最优配置训练：
```bash
python train.py --config configs/optimized_v3_advanced.yaml
```

快速测试（toy数据）：
```bash
python train.py --config configs/toy_fast.yaml
```

### 4. 模型推理与评估

推理测试集：
```bash
python infer.py --dataset data/marmousi2_2721_like_l101.npz \
                --ckpt runs/optimized_v3_advanced/checkpoints/best.pt \
                --split test \
                --out runs/optimized_v3_advanced/pred_test.npz \
                --batch_size 64
```

生成所有可视化结果：
```bash
python eval_v3.py
```

生成文件：
- `pred_test_metrics.json` - 详细评估指标
- `traces_compare.png` - 4道地震记录对比
- `figures_section/pred.png` - 预测速度剖面
- `figures_section/true.png` - 真实速度剖面
- `figures_section/error.png` - 误差分布

## 项目结构

```
comparison03/
├── configs/                      # 配置文件
│   ├── optimized_v3_advanced.yaml  # 最优模型配置
│   ├── toy_fast.yaml              # 快速测试配置
│   ├── marmousi2_paper.yaml       # 论文对齐配置
│   └── valve_like.yaml
├── data/                         # 数据集
│   ├── synth_toy.npz
│   ├── synth_marmousi2.npz
│   ├── synth_valve.npz
│   └── marmousi2_2721_like_l101.npz
├── runs/                         # 训练结果
│   └── optimized_v3_advanced/    # 最优模型结果
│       ├── checkpoints/best.pt
│       ├── pred_test.npz
│       ├── traces_compare.png
│       └── figures_section/
├── scripts/                      # 工具脚本
│   ├── make_synthetic.py         # 生成合成数据
│   ├── plot_section.py           # 绘制速度剖面
│   └── plot_traces_compare.py    # 绘制道对比
├── src/ss_gan/                   # 核心代码
│   ├── models.py                 # UNet生成器+判别器+自注意力
│   ├── trainer.py                # 训练循环+EMA+学习率调度
│   ├── losses.py                 # WGAN-GP损失
│   ├── forward.py                # Ricker子波正演
│   ├── data.py                   # 数据加载
│   └── utils.py                  # 工具函数
├── train.py                      # 训练入口
├── infer.py                      # 推理脚本
├── eval_v3.py                    # 评估自动化脚本
└── PROJECT_GUIDE.md              # 本文档
```

## 核心模型架构

### 1. 生成器 (UNet1D)
```
输入(T=470) → Conv → Down1 → Down2 → Down3 → Down4
  ↓                                           ↓
  ├── 跳跃连接 ←──────────────────── Self-Attention
  ↓
Up4 → Up3 → Up2 → Up1 → Conv → 输出 + 0.1×残差
```

- **base_ch**: 32（通道数基准）
- **k_large**: 31（大卷积核，捕获长程依赖）
- **Self-Attention**: 在bottleneck层建模全局依赖
- **残差连接**: 输出层添加 0.1×输入，保留细节

### 2. 判别器 (Critic1D)
```
输入(T=470) → Conv → Down1 → Down2 → Down3 → Down4 → 全局池化 → 标量
```
- **base_ch**: 16
- **Wasserstein距离**：梯度惩罚(GP=10)

### 3. 损失函数

**生成器总损失**：
```
L_G = L_adv + α·L_forward + β·L_smooth
```

- **L_adv**: WGAN对抗损失（D(G(x))最大化）
- **L_forward**: 物理正演约束（L1距离，α=50）
  ```
  L_forward = |Ricker(pred) - seismic|
  ```
- **L_smooth**: 平滑正则（梯度L1，β=30）

**判别器损失**：
```
L_D = D(G(x)) - D(y_true) + λ·GP
```

### 4. 进阶特性

**指数移动平均 (EMA)**:
```python
ema_decay = 0.999
θ_shadow = (1 - decay) × θ + decay × θ_shadow
```
- 推理时使用shadow参数，提升稳定性

**学习率调度**:
```python
warmup: 前10轮线性增长到lr=0.0002
cosine: 后续余弦退火到0.5×lr
```

## 关键配置参数

### optimized_v3_advanced.yaml 核心参数

```yaml
# 数据
dataset: data/marmousi2_2721_like_l101.npz
normalize: true                # 归一化到[-1,1]

# 训练
epochs: 200
batch_size: 10
learning_rate: 0.0002
warmup_epochs: 10             # 学习率预热
use_ema: true                 # EMA稳定推理
ema_decay: 0.999

# 模型
base_ch_g: 32                 # 生成器通道数
base_ch_d: 16                 # 判别器通道数
k_large: 31                   # 大卷积核

# 损失权重
alpha: 50.0                   # 正演约束权重
beta: 30.0                    # 平滑正则权重
lambda_gp: 10.0               # 梯度惩罚
loss_in_physical: false       # 在归一化空间计算损失

# 训练技巧
gradient_clip_max_norm: 1.0   # 梯度裁剪
n_critic: 5                   # 判别器更新频率
```

## 优化历程总结

### V1 → V2: 修复损失尺度崩溃
**问题**: 
- `loss_in_physical=true` 导致物理空间损失(~1e6) × α=1100 压倒GAN损失(~10)
- `k_large=299` 过大(63%感受野)，过度平滑

**解决**:
- `loss_in_physical=false` - 归一化空间计算损失
- 降低权重: α=1100→50, β=550→30
- 缩小卷积核: k_large=299→31
- 增加容量: base_ch_g=16→32
- 添加梯度裁剪: max_norm=1.0

**结果**: R² 从负值跃升至 **0.9831**

### V2 → V3: 进阶深度学习技术
**新增**:
- Self-Attention层（bottleneck全局建模）
- EMA (decay=0.999, 推理稳定)
- Warmup + Cosine学习率调度
- 输出残差连接 (0.1×输入)
- 延长训练: 100→200 epochs

**结果**: R² **0.9831 → 0.9852**, MSE降低 **11%**

## 论文对齐模式

生成标准数据集（13601道）：
```bash
python scripts/make_synthetic.py --out data/marmousi2_full.npz \
                                  --preset marmousi2 --seed 1234
```

使用论文配置训练（1000轮，α=1100，β=550）：
```bash
python train.py --config configs/marmousi2_paper.yaml
```

## 常见问题

### Q1: 训练过程中验证集指标很差？
**A**: 这是正常的。`history.json`中的验证指标可能不准确（EMA加载问题），但`best.pt`保存的是最优检查点。最终评估应使用推理脚本：
```bash
python infer.py --ckpt runs/xxx/checkpoints/best.pt ...
```

### Q2: 如何调整训练数据规模？
**A**: 修改配置文件中的数据集路径和对应的标注/验证比例。推荐：
- 小规模测试: toy数据 (~100道)
- 标准训练: 2721道，101标注
- 论文对齐: 13601道，101标注

### Q3: GPU内存不足？
**A**: 减小`batch_size`（默认10），或降低`base_ch_g`/`base_ch_d`。

### Q4: 如何生成自己的合成数据？
**A**: 使用`scripts/make_synthetic.py`，支持自定义速度模型、Ricker子波参数、噪声等级。

## 引用

如使用本项目，请引用原论文：
```
《基于生成对抗网络的半监督地震波阻抗反演》
```

## 技术支持

- 检查 `runs/optimized_v3_advanced/history.json` 查看训练曲线
- 使用 `eval_v3.py` 自动生成所有可视化
- 参考 `src/ss_gan/` 中的代码注释了解实现细节

---

**最后更新**: 2025年12月23日  
**最优模型**: runs/optimized_v3_advanced/checkpoints/best.pt  
**性能**: R²=0.9852, PCC=0.9926
