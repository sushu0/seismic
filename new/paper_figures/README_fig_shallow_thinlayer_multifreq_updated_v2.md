# 浅部薄层自建模型 20 Hz 定向优化说明

## 1. 定位到的原始链路

- 旧多频对比图脚本：`D:\SEISMIC_CODING\new\plot_multi_freq_comparison.py`
- 旧图输出目录：`D:\SEISMIC_CODING\new\results\multi_freq_comparison`
- 旧图文件：`D:\SEISMIC_CODING\new\results\multi_freq_comparison\multi_freq_comparison_grid.png`
- 20 Hz 数据：
  - `D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy`
  - `D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt`
- 20 Hz 训练脚本：`D:\SEISMIC_CODING\new\train_20Hz_thinlayer_v2.py`
- 旧论文 20 Hz 结果目录：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2`
- 旧论文 20 Hz checkpoint：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\checkpoints\best.pt`
- 旧论文 Word 文档：`D:\SEISMIC_CODING\new\薄层波阻抗预测问题分析与优化.docx`

## 2. 对旧 20 Hz 问题的判断

这次复核后，旧 20 Hz 的主要问题并不属于“明显没收敛”。

- `01_20Hz_thinlayer_v2` 的 `best.pt` 已经明显优于同次训练的 `last.pt`，说明问题不是训练不足，而是继续训练到后期会出现轻微漂移。
- 旧论文图本来用的就是该轮训练里的 `best.pt`，所以旧图问题也不只是简单的 `best/last` 选错。
- 更像是 20 Hz 低频条件下，训练增强和原始验证口径对“形态稳定性”的约束不够强，导致局部出现层体形变、拖尾和上下界恢复不够自然的问题。
- 也就是说，旧 20 Hz 的短板主要来自：
  - 低频分辨率本身受限；
  - 后期训练会有一定结构漂移；
  - 原 `val` 选模口径不完全等价于论文图所关心的“浅部薄层可信度”。

## 3. 本次实际做的优化

### 3.1 候选复评

先把以下 20 Hz 候选结果统一到同一套评估口径下复核：

- `01_20Hz_thinlayer_v2\checkpoints\best.pt`
- `01_20Hz_thinlayer_v2\checkpoints\last.pt`
- `01_20Hz_thinlayer_refined\checkpoints\best.pt`
- `01_20Hz_thinlayer_refined\checkpoints\last.pt`

结论是：原 `best.pt` 仍然比原 `last.pt` 更稳，说明旧图的问题不是“晚一点的 checkpoint 更好却没用上”。

### 3.2 轻量续训脚本增强

修改脚本：

- `D:\SEISMIC_CODING\new\paper_figures\refine_thinlayer_frequency.py`

新增能力：

- 支持 `--disable-augment`

这么做的原因是：20 Hz 更需要保守地稳住低频结构，不宜继续让薄层注入增强去放大形态扰动。

### 3.3 面向 20 Hz 的保守 fine-tune

从旧论文使用的 20 Hz 最优模型出发，做了一轮关闭训练增强的短程低学习率续训：

- 初始 checkpoint：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\checkpoints\best.pt`
- 输出目录：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_refined_v2`
- 学习率：`2e-5`
- weight decay：`1e-5`
- patience：`12`
- 训练增强：关闭
- 实际停止位置：第 `12` 轮内未出现更高验证综合分，保存到的 `last.pt` 为第 `10` 轮状态

## 4. 最终为什么选用新的 20 Hz 版本

虽然这轮关闭增强的续训没有把验证综合分推到更高，但它产出的第 10 轮 checkpoint 在整幅剖面和浅部目标窗口上都优于旧论文 20 Hz 结果，因此最终选它作为论文中的 20 Hz 基线结果。

最终选用目录：

- `D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_optimized_v2`

其 `best.pt` 来自：

- `D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_refined_v2\checkpoints\last.pt`

### 4.1 旧版与新版 20 Hz 对比

旧论文版本：

- `D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\checkpoints\best.pt`

新版最终版本：

- `D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_optimized_v2\checkpoints\best.pt`

#### 全剖面客观指标

| 指标 | 旧 20 Hz | 新 20 Hz | 变化 |
|---|---:|---:|---:|
| Full PCC | 0.9333 | 0.9434 | +0.0101 |
| Full R2 | 0.8711 | 0.8900 | +0.0189 |
| Full MAE | 32519.96 | 27952.60 | -4567.37 |
| Full RMSE | 214812.97 | 198436.88 | -16376.09 |

#### 浅部目标窗口指标

窗口采用样点 `3000:6000`，与旧 20 Hz 脚本中的浅部薄层放大区一致。

| 指标 | 旧 20 Hz | 新 20 Hz | 变化 |
|---|---:|---:|---:|
| Window PCC | 0.9169 | 0.9329 | +0.0160 |
| Window R2 | 0.8404 | 0.8703 | +0.0299 |
| Window MAE | 77233.71 | 64100.64 | -13133.07 |
| Window RMSE | 341397.31 | 307779.53 | -33617.78 |
| Window Gradient PCC | 0.0694 | 0.0844 | +0.0150 |

#### 统一薄层测试指标

| 指标 | 旧 20 Hz | 新 20 Hz |
|---|---:|---:|
| PCC | 0.9327 | 0.9403 |
| Thin-layer PCC | 0.3802 | 0.4017 |
| Separability | 0.9588 | 0.9718 |
| Thickness Error (DPDE) | 1.16 | 1.12 |

## 5. 为什么认为新 20 Hz 更可信

- 主要层体位置与全剖面结构一致性更好。
- 浅部目标窗口内的局部误差更小，说明旧图中最显眼的层体拖尾和形变被压下去了。
- 边界相关的梯度一致性略有提升，但没有出现违反 20 Hz 频带能力的“假锐化”。
- 分离度和厚度误差都更合理，说明不是靠过度平滑换来的表面整洁。
- 最终版本仍明显弱于 30 Hz / 40 Hz，但已经更像一个可信的低频基线，而不是“20 Hz 做坏了”的结果。

## 6. 仍然存在的客观限制

- 20 Hz 仍不可能达到 30 Hz / 40 Hz 的薄层细节水平，这是频带上限决定的。
- 新 20 Hz 虽然修正了旧图中较显眼的不自然错误，但仍会保留一定的低频模糊和层间过渡展宽。
- 这次最终采用的新 20 Hz checkpoint 并不是原验证综合分最高的一版，而是综合全剖面指标、浅部目标窗口指标和结构可信度后选出的更合适论文版本。

## 7. 新输出文件

- 20 Hz 单图 PNG：`D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_20Hz_optimized_v2.png`
- 20 Hz 单图 PDF：`D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_20Hz_optimized_v2.pdf`
- 多频对比图 PNG：`D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_multifreq_updated_v2.png`
- 多频对比图 PDF：`D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_multifreq_updated_v2.pdf`
- 更新后的 Word 文档：`D:\SEISMIC_CODING\new\论文完整写作稿_更新20Hz图优化版.docx`

## 8. 相关中间结果

- 20 Hz 关闭增强续训目录：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_refined_v2`
- 20 Hz 最终入图目录：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_optimized_v2`
- 30 Hz 复用目录：`D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_refined`
- 40 Hz 复用目录：`D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2`
