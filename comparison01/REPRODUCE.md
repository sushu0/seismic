# 复现指南（comparison01 / CNN-BiLSTM 半监督波阻抗反演）

> 目标：尽量按论文流程跑通 **监督训练 + MC Dropout 伪标签 + 半监督微调**，并生成剖面对比图与代表道对比图。

## 1. 环境依赖

建议 Python 3.10+。

最低依赖（按代码导入）：
- numpy
- scipy
- torch
- matplotlib
- segyio（仅用于 `plot_impedance_marmousi2_raw.py` 读 SEG-Y；训练主流程不需要）

可用 `pip` 安装示例：

```bash
pip install numpy scipy matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install segyio
```

> 提示：你的环境里目前工具执行被取消（Copilot 无法代你运行代码），请你在本机终端运行命令。

## 2. 目录与数据

本目录需要：
- `data.npy`：原始字典数据，包含 `seismic` 与 `acoustic_impedance`（你已提供）

训练脚本默认读：
- `seismic.npy`、`impedance.npy`（形状为 `[T, Nx]`）

因此需要先做一次拆分：

```bash
python comparison01/split_marmousi2_from_data_npy.py
```

运行后应生成：
- `comparison01/seismic.npy`
- `comparison01/impedance.npy`

## 3. 训练与半监督

### 3.1 监督 + 半监督（默认开启）

```bash
python comparison01/marmousi_cnn_bilstm.py --data-root D:\SEISMIC_CODING\comparison01 --run-semi
```

脚本会输出：
- `comparison01/marmousi_cnn_bilstm_supervised.pth`
- `comparison01/marmousi_cnn_bilstm_semi.pth`
- `comparison01/norm_params.json`
- 以及一个带时间戳的运行目录：`comparison01/runs/<timestamp>/`（包含 `train.log`、`config.json`、模型副本）

### 3.2 只跑监督（消融）

```bash
python comparison01/marmousi_cnn_bilstm.py --data-root D:\SEISMIC_CODING\comparison01 --no-run-semi
```

> 注意：脚本默认会跑半监督；用 `--no-run-semi` 可严格关闭半监督阶段。

可选：如需半监督阶段不冻结 CNN（全量微调），加 `--no-freeze-cnn`。

## 4. 结果可视化

### 4.1 三联剖面（真值 / 监督 / 半监督）

```bash
python comparison01/plot_impedance_section.py
```

输出：
- `comparison01/impedance_sections.png`

### 4.2 代表道曲线对比

```bash
python comparison01/plot_trace_comparison.py
```

输出：
- `comparison01/impedance_comparison_4traces.png`

### 4.3 真值阻抗剖面（用于核对数据）

```bash
python comparison01/plot_impedance_marmousi2_raw.py
```

该脚本会：
- 优先使用 `Density.segy`/`Vp.segy` 计算 `Z=Vp×rho`
- 若抽样判定 `Density.segy` 异常，会自动回退使用 `data.npy:acoustic_impedance` 或 `impedance.npy`

输出：
- `comparison01/true_impedance_marmousi2_raw.png`

## 5. 与论文对齐检查清单（待你确认论文细节）

已从 `paper_extracted.txt` 确认并在脚本默认值中对齐的关键设置：
- 数据划分：从 2721 道中“均匀选取” 20/5/5（train/val/test）
- 训练：`batch_size=8`，`lr=0.005`，`weight_decay=1e-4`，训练次数 200
- 增广：内插重采样，`N* = 10N`，每道阻抗做 10 道增广
- 伪标签筛选：MC Dropout 评估置信度，阈值 `0.95`

仍需你确认/或需要进一步从论文图表补充的部分：
- 模型结构：卷积层数/通道数/核大小，BiLSTM 隐藏元与层数，是否含残差/池化
- 数据：采样率 dt、子波主频 f0、窗长/输入长度 T
- 归一化：min-max vs z-score；按全局还是按道；仅用训练集统计还是全数据
- 半监督：MC Dropout 次数、置信度的严格定义（论文未给出公式时只能做合理实现）
- 损失：物理一致性项是否也用于伪标签样本、以及权重设置
- 训练：是否使用学习率衰减（论文未明确）
- 指标：PCC、R²、以及是否还有 MSE/MAE/SSIM 等

