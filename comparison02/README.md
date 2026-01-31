# FCRSN-CW 论文复现（核心功能）

对应论文：王康等《基于全卷积残差收缩网络的地震波阻抗反演》（2023）。本项目复现其**核心思路**：
- 通过**正演**从波阻抗生成合成地震：先由阻抗计算反射系数，再与 **Ricker 子波**卷积生成地震记录
- 使用 **FCRSN-CW** 网络（全卷积 + 残差收缩 RSBU-CW）进行端到端回归：输入地震道，直接输出阻抗
- 训练采用 **MSE** 损失与 Adam（lr=0.001, weight_decay=1e-7, batch=12, epochs=50）
- 评估指标：MSE + PCC（浅层/深层，浅层取前 1/5）
- 抗噪：对地震加高斯噪声（SNR=35/25/15/5 dB）再反演

## 0) 提交版复现说明（必读）

- 文献复现说明文档：见 [REPRODUCTION_REPORT.md](REPRODUCTION_REPORT.md)
- 本次复现主 run（使用 data.npy）：`runs/marmousi2/`

  - 指标：`runs/marmousi2/results/metrics.json`
  - 抗噪表：`runs/marmousi2/results/noise_table.json`
  - 复现记录/偏离项：`runs/marmousi2/results/repro_record.json`
  - 训练曲线：`runs/marmousi2/results/loss_curve.json`
  - 四道对比图（299/599/1699/2299）：`runs/marmousi2/results/trace_compare_grid_all.png`
  - 剖面图：`runs/marmousi2/results/true_impedance_section_all.png`、`runs/marmousi2/results/pred_impedance_section_all.png`

> 说明：论文使用 Marmousi2 纵波速度生成阻抗，并假设密度恒定，数据规模为 13601 道 × 2800 采样点（dt=1ms）。
> 本仓库**不内置** Marmousi2 数据，你可以用脚本生成“类 Marmousi”合成数据，或加载自己的阻抗剖面（.npy）。

---

> 性能提示：卷积核 299×1 在 CPU 上训练会非常慢（论文为对齐子波波长而选取该尺度）。如需快速验证流程，可临时使用更小的 `--k_first/--k_res1`（例如 81）或减少数据规模。

## 1) 安装环境

推荐 Python 3.9+。

```bash
pip install -r requirements.txt
# 建议在项目根目录执行（方便脚本导入本地包）
pip install -e .
```

---

## 2) 生成合成数据（可选）

### 2.1) 严格按论文合成（需要 Marmousi2 纵波速度 v）

论文 2.1 节“合成数据”描述的合成流程为：
1) 常数密度假设下计算波阻抗：$Z=\rho v$
2) 计算反射系数：$r_i=\dfrac{Z_{i+1}-Z_i}{Z_{i+1}+Z_i}$
3) 用 30Hz、0° Ricker 子波与 $r$ 卷积得到合成地震

硬性尺寸要求：13601 道 × 2800 采样点，采样间隔 dt=1ms。

本仓库不内置 Marmousi2 速度模型。你需要先准备一个 `vp.npy`（shape 必须为 `[13601, 2800]`，单位自定但需与 `rho_const` 一致）。然后运行：

```bash
python scripts/synthesize_paper_marmousi2.py \
	--vp_npy vp.npy \
	--rho_const 1.0 \
	--dt_s 0.001 \
	--strict \
	--out data/marmousi2_paper_synth.npz
```

会输出：
- `data/marmousi2_paper_synth.npz`（含 `seismic`、`impedance`、`meta`）
- `data/marmousi2_paper_synth.record.json`（完整参数与实现细节，便于复现）

> 注意：论文仅说明“密度为常数”，但未披露其数值；因此脚本要求你显式提供 `--rho_const`。

### 2.2) 按论文正演流程：由阻抗 Z(t) 生成反射系数 r(t) 与合成地震 s(t)

如果你已经有每条道的阻抗序列 `Z(t)`（shape 应为 `13601×2800`，`dt=1ms`），可按论文示意流程生成反射系数与合成地震：

- 反射系数：$r_i=\dfrac{Z_{i+1}-Z_i}{Z_{i+1}+Z_i}$
- 子波：Ricker，主频 30Hz，相位 0°，时间窗 `[-0.4, 1] ms`
- 卷积：$s(t)=r(t)*w(t)$

脚本：

```bash
python scripts/paper_forward_from_impedance.py \
	--impedance_npy impedance.npy \
	--dt_s 0.001 \
	--f0_hz 30 \
	--phase_deg 0 \
	--wavelet_tmin_ms -0.4 \
	--wavelet_tmax_ms 1.0 \
	--strict \
	--out_npz data/paper_forward_output.npz \
	--out_dir runs/paper_forward
```

输出：
- `data/paper_forward_output.npz`：包含 `impedance`、`reflectivity`、`seismic`、`meta`
- `runs/paper_forward/paper_forward_record.json`：记录边界与卷积对齐策略
- `runs/paper_forward/trace_*.png`、`Z_section.png`、`r_section.png`、`s_section.png`：抽样可视化

实现细节（写入 record）：
- **边界策略**：最后一个反射系数点 $r_{T-1}$ 复制自 $r_{T-2}$，保证 `r` 与 `Z` 同长度
- **卷积对齐策略**：对 time window `[-0.4, 1] ms` 的子波，以 $t=0$ 对应的样点作为零时刻索引 `k0`，先做 `full` 卷积，再取 `full[k0 : k0+T]` 作为 `s`，保证输出与输入时间索引一致

### 2.0) 复现前置检查（若使用 SEG-Y）

论文的 Marmousi2 合成实验要求：`dt=1ms`，且剖面尺寸为 `13601×2800`。在开始严格复现前，建议先检查你的 SEG-Y 是否满足这些硬性条件：

```bash
python scripts/check_segy_dataset.py --vp_segy Vp.segy --rho_segy Density.segy --paper --out segy_report.json
```

会输出 `segy_report.json`（含 dt、shape、值域/异常值比例）。

### A. 直接生成“类 Marmousi”数据（默认 13601×2800）

```bash
python scripts/make_synthetic_dataset.py --out data/synth_marmousi_like.npz
```

### B. 用你自己的阻抗剖面生成（.npy: [n_traces, n_samples]）

```bash
python scripts/make_synthetic_dataset.py --impedance_npy /path/to/impedance.npy --out data/custom.npz
```

### C. 仅为可运行性而对现有 data.npy 做格式转换（非严格论文复现）

如果你只有类似 `data.npy`（例如包含 `seismic` 与 `acoustic_impedance` 的 pickled dict），且其尺寸不满足论文的 13601×2800，
任何“补齐/插值/重采样/扩充道数”都属于论文未披露的额外处理，不能计入严格复现。

本仓库提供一个确定性转换脚本，专用于“跑通流程 + 明确记录偏离项”：

```bash
python scripts/convert_data_npy_to_paper_npz.py \
	--input_npy data.npy \
	--out_npz data_paper_format.npz \
	--target_nt 2800 \
	--dt_s 0.001 \
	--seismic_source regenerate
```

它会额外写出 `data_paper_format.conversion.json`，逐条列出偏离论文之处。

---

## 3) 训练

```bash
python scripts/train.py --data data/synth_marmousi_like.npz --run_dir runs/exp1
```

如果你已将 Marmousi2 数据放在项目根目录（`Vp.segy`、`Density.segy`），也可以直接从 SEG-Y 读取并正演生成训练数据：

```bash
python scripts/train.py --vp_segy Vp.segy --rho_segy Density.segy --run_dir runs/exp1
```

> 论文合成实验中假设密度恒定；如需该设定，可用 `--rho_const 1.0`（会忽略 `--rho_segy`）。

> 输出层默认启用 `ReLU`（与论文一致）。如需关闭，可传 `--no-last-relu`。

训练结束后会保存：
- `runs/exp1/checkpoints/best.pt`
- `runs/exp1/scalers.json`
- `runs/exp1/split.json`

---

## 4) 评估与结果输出

```bash
python scripts/evaluate.py --data data/synth_marmousi_like.npz --run_dir runs/exp1
```

输出在 `runs/exp1/results/`：
- `metrics.json`：test MSE（scaled/physical）、浅层/深层 PCC
- `noise_table.json`：SNR=35/25/15/5 dB 的指标表
- `trace_val_650.png`、`trace_val_1250.png`（默认绘制验证集第 650/1250 道，贴合论文描述）
- `pred_section.png`：预测阻抗剖面预览（子集）

### 4.1) 绘制“反演阻抗剖面图”（参考论文/示例图样式）

训练完成后，可用脚本将模型预测的阻抗剖面按“横轴 Trace number、纵轴 Time(ms)、colorbar 显示阻抗范围”的方式绘制：

```bash
python scripts/plot_inversion_section.py \
	--run_dir runs/exp1 \
	--data data/synth_marmousi_like.npz \
	--split test \
	--use_true_range \
	--device cuda
```

会在 `runs/exp1/results/` 输出：
- `true_impedance_section_<split>.png`
- `pred_impedance_section_<split>.png`

---

## 5) 推理（对新地震数据）

输入地震 `.npy`（shape=[n_traces,n_samples]）：

```bash
python scripts/infer.py --seismic_npy /path/to/seismic.npy --run_dir runs/exp1 --out pred_impedance.npy
```

会得到 `pred_impedance.npy` 与对应预览图 `pred_impedance.png`。

---

## 6) 与论文设置对齐的关键点（你可改）

- 卷积核：首层 299×1、末层 3×1，RSBU-CW 内部 299×1 + 3×1；步幅 1、零填充保持尺寸不变 fileciteturn2file0L208-L223  
- 残差块数量：论文对比后选择 3 个残差块 fileciteturn1file0L8-L12  
- 数据划分：10601/1500/1500（train/val/test） fileciteturn2file0L249-L251  
- 浅层/深层 PCC：浅层前 0~0.56ms（约 1/5），深层 0.56~2.80ms（约 4/5） fileciteturn2file0L372-L383  

如你希望进一步对齐论文（例如 Volve 实测数据部分使用序贯高斯模拟扩充数据集 fileciteturn2file0L388-L401），可以在此项目基础上继续扩展数据构建脚本。
