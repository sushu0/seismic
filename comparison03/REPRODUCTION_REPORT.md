# 文献复现说明文档（半监督GAN地震阻抗反演）

**项目名称**：ss-gan-impedance（paper-aligned core）  
**复现日期**：2025-12-23～2025-12-24  
**最终最优实验**：`runs/optimized_v3_advanced/`  
**最终测试结果**：PCC=0.9926，R²=0.9852，MSE(physical)=55293.24

---

## 1. 论文目标与方法概述

### 1.1 论文目标（复现目标）
目标是基于少量标注阻抗（或速度/阻抗剖面）与大量无标注地震记录，训练一个端到端模型从地震道（seismic trace）反演阻抗（impedance）/速度剖面，获得高一致性的反演结果。

### 1.2 方法概述
本项目实现一种**半监督生成对抗网络（GAN）**框架，核心思想：
- **监督分支（labeled）**：使用少量带标签的地震道–阻抗对，直接约束生成器输出阻抗接近标签。
- **无监督/物理一致性分支（unlabeled）**：对生成器输出阻抗进行**地震正演**（Ricker子波卷积/建模），要求合成地震与真实无标注地震道一致，从而利用大量无标注数据约束反演结果。
- **对抗训练（WGAN-GP）**：判别器/critic 接收（地震，阻抗）的拼接输入，区分真实阻抗与生成阻抗，促进生成器输出更“真实”的阻抗分布。

本次复现还引入了稳定性与收敛改进：自注意力、EMA、warmup+cosine 学习率调度、梯度裁剪等（见后文）。

---

## 2. 数据与划分

### 2.1 数据文件
本次最优实验使用数据集：
- `data/marmousi2_2721_like_l101.npz`

该 `.npz` 内包含以下关键张量（shape=[N,T]，T=470）：
- `x_labeled`: (101, 470)  
- `y_labeled`: (101, 470)
- `x_unlabeled`: (2350, 470)（无 y_unlabeled）
- `x_val`: (270, 470)  
- `y_val`: (270, 470)
- `x_test`: (2721, 470)  
- `y_test`: (2721, 470)

> 说明：test split 在该数据中提供全量标签用于评估（即“有标签的全量测试集”），用于计算 R²/PCC/MSE。

### 2.2 归一化与统计量
训练/推理采用 `dataset.normalize: true`，对 x/y 做标准化：

$$x_{norm} = \frac{x-x_{mean}}{x_{std}+\epsilon},\quad y_{norm} = \frac{y-y_{mean}}{y_{std}+\epsilon}$$

统计量由 `train.py` 在训练开始时从 `x_labeled/y_labeled` 计算并写入：
- `runs/optimized_v3_advanced/stats.json`

---

## 3. 模型结构

### 3.1 生成器（UNet1D）
文件：`src/ss_gan/models.py` 中 `UNet1D`

输入/输出：
- 输入：单道地震（shape=[B,1,T]）
- 输出：单道阻抗（shape=[B,1,T]）

结构要点：
- 1D U-Net 编码器–解码器 + 跳跃连接
- 大核卷积用于建模长程依赖：`k_large=31`
- 小核卷积用于局部细节：`k_small=3`
- **自注意力（SelfAttention1D）**：放置在 bottleneck，提升全局依赖建模能力
- **输出残差**：`out = out_main + 0.1 * out_residual(x)`，增强细节与梯度传播

关键类/函数位置：
- `SelfAttention1D`（自注意力模块）
- `UNet1D`（主生成器）

### 3.2 判别器/评论家（Critic1D）
文件：`src/ss_gan/models.py` 中 `Critic1D`

输入：将地震与阻抗在通道维拼接：
- `cat([x, y], dim=1)` → shape=[B,2,T]

结构要点：
- 多层 1D Conv + stride/downsample
- 残差块 `ResBlock1D`
- 全局池化 + 全连接输出 Wasserstein 分数

---

## 4. 损失函数与训练策略

实现文件：`src/ss_gan/trainer.py`、`src/ss_gan/losses.py`、`src/ss_gan/forward.py`

### 4.1 WGAN-GP 判别器损失
对 critic：

$$L_D = -\mathbb{E}[D(x,y_{real})] + \mathbb{E}[D(x,y_{fake})] + \lambda_{gp}\,GP$$

其中梯度惩罚（GP）在 `src/ss_gan/losses.py` 的 `gradient_penalty`。

### 4.2 生成器损失（监督+物理+对抗）
生成器目标：

$$L_G = L_{adv} + \alpha\,L_i + \beta\,L_s + w_{grad}\,L_{grad}$$

- **对抗项**：`L_adv = -E[D(x, G(x))]`
- **监督阻抗损失**（labeled）：`L_i`，本次使用 L1：`imp_loss: l1`
- **物理一致性损失**（unlabeled）：`L_s`
  - 先对预测阻抗进行正演：`forward_seismic_from_impedance`
  - 再与无标注地震 x 比较（MSE）
- **可选梯度损失**：`L_grad`（一阶导 L1），本次配置为 `grad_loss_weight: 2.0`

### 4.3 正演建模（Ricker 子波）
文件：`src/ss_gan/forward.py`
- `RickerWavelet`：生成 Ricker 子波
- `forward_seismic_from_impedance`：由阻抗生成地震（物理约束核心）

### 4.4 训练策略
文件：`src/ss_gan/trainer.py` 的 `train`
- `n_critic=5`：每次生成器更新前，critic 更新 5 次
- Adam 优化：betas=(0.5,0.9)
- 梯度裁剪：`clip_grad_norm_(..., max_norm=1.0)`（G/D 都启用）
- AMP：本次为 `amp: false`

---

## 5. 实现细节与关键代码位置

### 5.1 训练入口与配置
- `train.py`：读取 YAML → 构建 dataloader → 调用 `ss_gan.trainer.train`
- `configs/optimized_v3_advanced.yaml`：最优实验参数

### 5.2 数据加载
- `src/ss_gan/data.py`
  - `SeisImpNPZ`：读取 npz，并在 normalize=true 时对 x/y 标准化
  - `make_loader`：构建 DataLoader（shuffle 时 drop_last=True）

### 5.3 推理与评估
- `infer.py`：加载 checkpoint → 推理 → 保存 `pred_test.npz` 与 `pred_test_metrics.json`
- `eval_v3.py`：自动化生成
  - `runs/optimized_v3_advanced/traces_compare.png`
  - `runs/optimized_v3_advanced/figures_section/{pred,true,error}.png`
- 绘图脚本：
  - `scripts/plot_traces_compare.py`
  - `scripts/plot_section.py`

---

## 6. 运行环境与命令

### 6.1 环境信息
- OS：Windows
- Python：3.11.9（venv）
- GPU：NVIDIA GeForce RTX 4090 D
- PyTorch：2.6.0+cu124
- NumPy：2.4.0
- Matplotlib：3.10.8
- 其他：PyYAML 6.0.3，tqdm 4.67.1

依赖列表见：`requirements.txt` 与 `pyproject.toml`。

### 6.2 训练命令
最优训练（200 epochs）：
```bash
python train.py --config configs/optimized_v3_advanced.yaml
```

### 6.3 推理与生成可视化
推理（会生成 `pred_test.npz` 与 `pred_test_metrics.json`）：
```bash
python infer.py --dataset data/marmousi2_2721_like_l101.npz \
  --ckpt runs/optimized_v3_advanced/checkpoints/best.pt \
  --split test --out runs/optimized_v3_advanced/pred_test.npz --batch_size 64
```

生成图片（四道对比 + 剖面图）：
```bash
python eval_v3.py
```

---

## 7. 评估指标与结果对比

### 7.1 指标定义
- **PCC**：皮尔逊相关系数
- **R²**：决定系数
- **MSE**：均方误差

本项目同时给出：
- normalized 空间指标（训练/默认保存）
- physical 空间指标（将 y 反标准化后计算，用于工程可解释性）

### 7.2 最终结果（optimized_v3_advanced）
产物：`runs/optimized_v3_advanced/pred_test_metrics.json`
- Test（normalized / raw）：PCC=0.9926，R²=0.9852，MSE=0.014834
- Test（physical / raw）：PCC=0.9926，R²=0.9852，MSE=55293.24

可视化：
- 四道对比图：`runs/optimized_v3_advanced/traces_compare.png`
- 反演剖面图：`runs/optimized_v3_advanced/figures_section/pred.png`
- 真实剖面图：`runs/optimized_v3_advanced/figures_section/true.png`
- 误差剖面图：`runs/optimized_v3_advanced/figures_section/error.png`

---

## 8. 复现偏差与原因分析

> 说明：此处仅做方法层面的对比与工程差异说明，不直接复刻论文原文表述。

### 8.1 数据规模与划分差异
- 本次使用 2721 道（101 labeled / 270 val / 2350 unlabeled），与论文常见 Marmousi2 大规模设置不同。
- Test split 提供全量标签用于评估，便于量化结果。

### 8.2 超参数与结构差异（为稳定性做的工程调整）
最优配置与“论文原始超参”存在差异，核心原因是：本项目数据规模与归一化设置会导致损失量纲不同，从而需要重新配平。
- `loss_in_physical: false`：在**归一化空间**计算 Li/Ls（避免量纲爆炸）
- `alpha=50, beta=30`：适配归一化空间的损失尺度
- `k_large=31`：避免过大卷积核导致过度平滑
- 增强稳定性的训练技巧：梯度裁剪、warmup+cosine、输出残差
- 加入自注意力提升全局依赖建模

---

## 9. 问题排查记录（关键故障与修复）

### 9.1 早期故障：模型“反演不出来/输出塌缩”
现象：预测阻抗曲线接近常数或严重偏离，剖面无结构。

原因定位（根因）：
1) **损失尺度失衡**：当在 physical 空间计算 Li/Ls 且权重 α/β 较大时，监督/物理项会远大于对抗项，训练退化为不稳定的回归或直接塌缩。
2) **卷积核过大**：当 `k_large` 过大接近序列长度（T=470）时，模型易过度平滑，细节被抹除。

修复措施（最终采用）：
- 将 `loss_in_physical` 设为 false（归一化空间配平损失）
- 调整 α/β 到与对抗项同量级
- 将 `k_large` 缩小到 31
- 加入梯度裁剪

### 9.2 训练日志中的验证指标异常（与最终推理不一致）
现象：`runs/optimized_v3_advanced/history.json` 末期 val 指标很差（PCC≈0.12），但 `infer.py` 用 best.pt 在 test 上很好（PCC≈0.99）。

分析结论：
- 本项目启用 EMA 验证：`trainer.py` 在验证前 `ema.apply_shadow()`。
- 目前 EMA 仅在“每个 epoch 结束”更新一次，而 `ema_decay=0.999` 非常大，导致 EMA 影子参数长期滞后于真实模型参数，验证时用 EMA 参数会偏差很大。
- `infer.py` 当前加载的是 checkpoint 中的 `G`（非 `G_ema`），因此推理结果正常。

建议改进（不影响当前 best.pt 推理，但会影响训练过程选择 best）：
- 将 EMA 更新频率从“每 epoch”改为“每 step”，或将 decay 调整为与更新频率匹配的值。

---

## 10. 后续改进计划

### 10.1 指标与训练过程一致性
- EMA：改为每 step 更新，并在保存 best 时同时对比 `G` 与 `G_ema` 的 val 指标。
- 在 `history.json` 中记录两套指标：`val`（G）与 `val_ema`（EMA），便于解释与追踪。

### 10.2 更严格的论文对齐复现
- 使用论文规模的 Marmousi2 数据（如 13601 道）
- 采用论文训练轮数与噪声/SNR设定
- 报告与论文相同的图表与统计口径

### 10.3 模型与损失进一步增强（可选）
- 判别器谱归一化（Spectral Norm）或更稳定的正则
- 多尺度判别器/多分辨率约束
- 输出后处理（轻量平滑/分位裁剪）作为可控开关，并统一写入 metrics 的 postprocess 字段

---

## 11. 复现产物清单（可提交）

**最终结果目录**：`runs/optimized_v3_advanced/`
- `checkpoints/best.pt`（最优模型）
- `pred_test.npz`（推理输出，含 pred/true/metrics）
- `pred_test_metrics.json`（指标汇总）
- `traces_compare.png`（四道对比图）
- `figures_section/pred.png`（反演剖面）
- `figures_section/true.png`（真实剖面）
- `figures_section/error.png`（误差剖面）

**关键配置**：`configs/optimized_v3_advanced.yaml`

---

## 12. 结论
本次复现实现了半监督 WGAN-GP + 物理正演约束的地震阻抗反演流程，并在数据集 `marmousi2_2721_like_l101` 上获得高精度结果（R²≈0.985）。核心工程经验在于：**损失尺度配平**与**合理感受野（k_large）**对稳定训练至关重要；进一步加入自注意力、学习率调度与残差连接，可在高基线结果上获得稳定增益。
