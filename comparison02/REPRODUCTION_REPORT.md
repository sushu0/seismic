# 文献复现说明文档（FCRSN-CW 地震波阻抗反演）

文档版本：v1（2025-12-24）  
仓库位置：d:\SEISMIC_CODING\comparison02  

本说明文档用于“可直接提交”的复现记录：覆盖论文目标与方法概述、数据与划分、模型结构、损失与训练策略、实现细节与关键代码位置、运行环境与命令、评估指标与结果、复现偏差与原因分析、问题排查记录与后续改进计划。

---

## 1. 论文目标与方法概述

### 1.1 目标
论文目标为：利用深度网络实现端到端地震波阻抗反演，即输入单道地震记录，直接输出对应的波阻抗序列。

### 1.2 方法主线（论文核心链路）
复现遵循论文的核心流程：

1) 合成数据（正演）：由波阻抗 $Z(t)$ 计算反射系数 $r(t)$，再与 Ricker 子波卷积得到合成地震 $s(t)$。

- 反射系数（逐采样点）：

$$
 r_i = \frac{Z_{i+1}-Z_i}{Z_{i+1}+Z_i}
$$

- 合成地震：

$$
 s(t) = r(t) * w(t)
$$

其中 $w(t)$ 为 Ricker 子波，论文描述使用 30Hz、0°相位。

2) 反演网络（FCRSN-CW）：输入 $s(t)$，输出 $\hat{Z}(t)$。

3) 训练：以均方误差（MSE）作为监督目标，优化器使用 Adam（论文默认 lr=0.001、weight_decay=1e-7、batch=12、epochs=50）。

4) 评估：MSE 与 PCC（浅层/深层），并对输入地震添加不同 SNR 的高斯噪声测试鲁棒性。

---

## 2. 数据与划分

### 2.1 本次复现使用的数据
本次“实际训练/评估”使用的是工程侧提供的 Marmousi2 数据封装文件：

- data.npy（pickled dict）
  - seismic：形状 (2721, 1, 470)
  - acoustic_impedance：形状 (2721, 1, 1880)
  - 数据类型：float64（训练时转为 float32）

说明：该数据的 seismic 与 impedance 采样点数不一致，属于论文未披露的输入格式；本仓库在 paper 模式下允许工程跑通，但会将偏离项写入复现记录（见第 8 节）。

### 2.2 数据预处理与对齐
- 维度处理：将 (N,1,T) squeeze 成 (N,T)
- 采样点数对齐：当 seismic 与 impedance 长度不一致时，在非 strict 模式下允许“整数倍重采样”
  - 本次 run 触发了对 seismic 的上采样，使其长度与 impedance (1880) 对齐

### 2.3 数据划分（train/val/test）
论文固定划分为 10601/1500/1500（总计 13601 道）。

本次数据只有 2721 道，paper_strict 不可用；在 paper（非 strict）模式下，本仓库按比例缩放划分：
- train：2122
- val：299
- test：299

划分索引保存在 runs/marmousi2/split.json。

---

## 3. 模型结构（FCRSN-CW）

### 3.1 网络概述
实现位于 fcrsn_cw/models/fcrsn_cw.py。

核心结构（1D 时间卷积，输入/输出长度保持不变）：
- 输入：seismic (1, T)
- Conv1d(1→16, kernel=k_first) + BN + ReLU
- 通道投影 16→32 + RSBU-CW(32)
- 通道投影 32→64 + RSBU-CW(64) + RSBU-CW(64)
- Conv1d(64→1, kernel=k_last) +（可选输出激活）

RSBU-CW 块：BN → ReLU → Conv(k1) → BN → ReLU → Conv(k2) → Shrinkage → Residual Add → ReLU。

### 3.2 关键超参数
论文设置（本仓库 paper 默认）：
- k_first = 299
- k_res1 = 299
- k_res2 = 3
- k_last = 3
- 输出层 ReLU：默认开启（与论文一致）

本次实际 run（runs/marmousi2/results/repro_record.json）：
- k_first=299, k_res1=299, k_last=3
- output_activation：未显式设置（保持 legacy 行为，由 last_relu 控制）

---

## 4. 损失函数与训练策略

### 4.1 损失函数
论文使用 MSE。

实现：
- 基线 MSE 训练循环：fcrsn_cw/train/engine.py
- 本仓库支持扩展但默认不启用：
  - Huber（loss_type=huber）
  - 时间梯度项（grad_loss_weight>0）

本次实际 run：
- loss_type=mse
- grad_loss_weight=0

### 4.2 优化器与超参
- Adam
- lr=1e-3
- weight_decay=1e-7
- batch_size=12
- epochs=10（为了本次演示与快速迭代；论文为 50）

### 4.3 归一化/缩放
- seismic：z-score（StandardScaler）
- impedance：min-max（MinMaxScaler）

scaler 拟合仅使用训练集，并保存在 runs/<run>/scalers.json。

---

## 5. 实现细节与关键代码位置（代码地图）

### 5.1 数据与预处理
- fcrsn_cw/data/dataset.py
  - SeismicImpedanceDataset：按 split 索引切片 + scaler transform + 输出 torch (1,T)
  - fit_default_scalers：seismic 标准化，impedance 支持 minmax/standard
- fcrsn_cw/data/scaler.py
  - StandardScaler / MinMaxScaler 以及 save/load

### 5.2 正演（论文合成链路）
- fcrsn_cw/physics/forward.py
  - impedance_to_reflectivity
  - reflectivity_to_seismic（Ricker + 相位旋转 + 卷积）
  - make_synthetic_pair_from_impedance
- fcrsn_cw/physics/wavelet.py
  - ricker 与 phase rotation

### 5.3 模型
- fcrsn_cw/models/fcrsn_cw.py
  - FCRSN_CW、RSBU_CW

### 5.4 训练与 checkpoint
- scripts/train.py：训练入口（含 paper/paper_strict、npy/npz/SEG-Y 读取、split、scaler、记录写出）
- fcrsn_cw/train/engine.py：每 epoch 的 train/eval 与可配置 loss

### 5.5 评估与作图
- scripts/evaluate.py：评估入口，输出 metrics/noise_table/pred_section
- scripts/plot_trace_compare_grid.py：指定道号的 2×2 曲线对比图
- scripts/plot_inversion_section.py：True/Pred 剖面图（带 colorbar），可用 true-range 对齐
- fcrsn_cw/utils/metrics.py：MSE、PCC（向量化）、加噪
- fcrsn_cw/utils/plotting.py：pred_section.png 使用的快速绘图函数

---

## 6. 运行环境

以 runs/marmousi2/results/repro_record.json 为准：
- OS：Windows 10 (10.0.19045)
- Python：3.11.9
- NumPy：2.4.0
- SciPy：1.16.3
- PyTorch：2.6.0+cu124（CUDA 12.4 可用）
- GPU：NVIDIA GeForce RTX 4090 D

依赖声明：requirements.txt 与 pyproject.toml。

---

## 7. 复现运行命令（可直接复现本次结果）

### 7.1 安装

- pip install -r requirements.txt
- pip install -e .

### 7.2 训练（本次 runs/marmousi2）

D:/SEISMIC_CODING/comparison02/.venv/Scripts/python.exe scripts/train.py \
  --data_npy data.npy \
  --run_dir runs/marmousi2 \
  --paper \
  --epochs 10 \
  --batch_size 12 \
  --k_first 299 \
  --k_res1 299 \
  --device cuda

训练产物：
- runs/marmousi2/checkpoints/best.pt
- runs/marmousi2/results/loss_curve.json
- runs/marmousi2/results/repro_record.json

### 7.3 评估（输出 paper 风格指标与抗噪表）

D:/SEISMIC_CODING/comparison02/.venv/Scripts/python.exe scripts/evaluate.py \
  --data_npy data.npy \
  --run_dir runs/marmousi2 \
  --paper \
  --device cuda

### 7.4 可视化

1) 四道曲线对比（No.299/599/1699/2299）：

D:/SEISMIC_CODING/comparison02/.venv/Scripts/python.exe scripts/plot_trace_compare_grid.py \
  --run_dir runs/marmousi2 \
  --data_npy data.npy \
  --split all \
  --trace_nos 299 599 1699 2299 \
  --one_based \
  --device cuda

输出：runs/marmousi2/results/trace_compare_grid_all.png

2) 反演剖面图（True/Pred）：

D:/SEISMIC_CODING/comparison02/.venv/Scripts/python.exe scripts/plot_inversion_section.py \
  --run_dir runs/marmousi2 \
  --data_npy data.npy \
  --split all \
  --use_true_range \
  --device cuda

输出：
- runs/marmousi2/results/true_impedance_section_all.png
- runs/marmousi2/results/pred_impedance_section_all.png

---

## 8. 评估指标与结果对比

### 8.1 指标定义
- MSE：在物理尺度阻抗（inverse scaling 后）计算
- PCC shallow/deep：按时间样点划分（shallow=前 1/5，deep=后 4/5）
- 噪声鲁棒性：对 seismic 添加高斯噪声，使 SNR = 35/25/15/5 dB

### 8.2 本次 runs/marmousi2 的结果
结果文件：runs/marmousi2/results/metrics.json
- MSE：37703.8320
- PCC_shallow：0.8868
- PCC_deep：0.9938

噪声鲁棒性（runs/marmousi2/results/noise_table.json）：
- SNR=35：MSE=37724.4648，PCC_shallow=0.8869，PCC_deep=0.9938
- SNR=25：MSE=38194.8516，PCC_shallow=0.8815，PCC_deep=0.9937
- SNR=15：MSE=43529.5742，PCC_shallow=0.8378，PCC_deep=0.9929
- SNR=5 ：MSE=363451.2188，PCC_shallow=0.3643，PCC_deep=0.9712

### 8.3 训练过程（loss 曲线）
见 runs/marmousi2/results/loss_curve.json：
- train loss 从 0.00995 下降到 0.00035
- val loss 最优约 0.000284（epoch 9）

---

## 9. 复现偏差与原因分析

以 runs/marmousi2/results/repro_record.json 为准，本次 run 属于 paper-mode 的工程复现，不是 paper_strict：

1) 数据尺寸偏离论文
- 本次使用 (2721, 1880)，论文为 (13601, 2800)
- 影响：指标不可直接与论文数值一一对照

2) dt 不可用
- data.npy 未提供 dt_s（采样间隔）
- 影响：无法确认是否满足论文 dt=1ms 的物理尺度；时间轴相关对齐不可严格验证

3) 输入输出采样点数不一致导致重采样
- seismic 长度 470，impedance 长度 1880
- 本次允许对 seismic 做整数倍上采样以对齐（非 strict 允许）
- 影响：这属于论文未披露的额外处理，会改变有效带宽与噪声特性

4) 训练轮数不足论文设置
- 本次 epochs=10（论文 50）
- 影响：性能可能未达到论文上限；但足以验证流水线与可视化

---

## 10. 问题排查记录（Troubleshooting Log）

本次复现过程中曾遇到并已解决/规避的典型问题：

- Windows 下 Matplotlib 初始化较慢或卡顿：已在可视化脚本中采用非交互后端（Agg）并使用本地 MPLCONFIGDIR。
- 评估阶段 PCC 计算慢：已在 fcrsn_cw/utils/metrics.py 实现批量向量化 PCC。
- SEG-Y 密度文件异常值导致溢出：训练入口支持 rho_const（常密度）以符合论文假设并避免污染。
- 训练/评估/绘图 kernel size 不一致导致 checkpoint shape mismatch：评估与绘图需要传入与训练一致的 k_first/k_res1（或使用 dt 自适应核时保持一致）。

---

## 11. 后续改进计划（可验证）

按优先级排序：

1) 严格论文复现前置数据补齐
- 获取满足 dt=1ms 且 shape=13601×2800 的 Marmousi2 vp 或 impedance
- 之后启用 --paper_strict，跑满 epochs=50 并输出 paper 风格结果

2) 推理入口一致性
- 为 scripts/infer.py 增加 output_activation / auto_kernel_from_dt / postprocess 等参数，使推理与训练/评估配置完全一致

3) 训练稳定性与视觉质量
- 在不改变论文基线的前提下，允许工程对比：
  - grad_loss_weight（抑制高频抖动）
  - train_snr_db（鲁棒性）
  - 推理期 median 后处理（不改变训练）
- 以上均需在 record 中声明并与基线对照

4) 结果汇总自动化
- 训练结束自动输出简明 summary（数据形状、关键超参、指标、图像路径），减少人工对照成本

---

## 12. 本次复现产物清单（便于验收）

以 runs/marmousi2 为例：
- 训练：runs/marmousi2/checkpoints/best.pt
- 记录：runs/marmousi2/results/repro_record.json
- 曲线：runs/marmousi2/results/loss_curve.json
- 指标：runs/marmousi2/results/metrics.json
- 抗噪：runs/marmousi2/results/noise_table.json
- 四道对比图：runs/marmousi2/results/trace_compare_grid_all.png
- 剖面图：
  - runs/marmousi2/results/true_impedance_section_all.png
  - runs/marmousi2/results/pred_impedance_section_all.png
- 预览：runs/marmousi2/results/pred_section.png
