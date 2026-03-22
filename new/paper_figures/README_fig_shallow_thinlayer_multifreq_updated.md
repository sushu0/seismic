# 浅部薄层自建模型不同频率反演结果图更新说明

## 1. 任务定位与源文件

- 旧图数据源目录：`D:\SEISMIC_CODING\zmy_data\01\data`
- 旧图绘制脚本：`D:\SEISMIC_CODING\new\plot_multi_freq_comparison.py`
- 旧图候选输出目录：`D:\SEISMIC_CODING\new\results\multi_freq_comparison`
- 旧综合图文件：`D:\SEISMIC_CODING\new\results\multi_freq_comparison\multi_freq_comparison_grid.png`
- 旧 20 Hz 结果目录：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2`
- 旧 30 Hz 结果目录：`D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v3`
- 旧 40 Hz 结果目录：`D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2`
- 论文 Word 源文件：`D:\SEISMIC_CODING\new\薄层波阻抗预测问题分析与优化.docx`

说明：

- `01_20Hz_04.txt`、`01_30Hz_04.txt` 与 `01_40Hz_04.txt` 的参考真值阻抗完全一致，可作为同一自建模型下的统一 reference。
- 仓库内另有 `D:\SEISMIC_CODING\new\results\visualizations\multi_freq_comparison.png` 这条后续报告图链路，但本次图件更新沿用的是与旧论文图更一致的 `ThinLayerNetV2` 多频结果链路。

## 2. 本次对 20 Hz / 30 Hz 做了什么

### 20 Hz

- 以 `D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\checkpoints\best.pt` 为起点做了 60 epoch 低学习率续训尝试。
- 续训参数：
  - 学习率：`6e-5`
  - weight decay：`1e-5`
  - patience：`15`
  - 选择逻辑：不再只看 `val_pcc`，改为综合 `pcc + thin_pcc + separability + r2 - dpde penalty`
- 结果：
  - 续训 15 epoch 后没有出现比原始 best 更好的综合验证分数。
  - 最终保留原 `epoch 287` 对应模型，并在新目录中重新固化与推理。
- 最终 20 Hz 结果目录：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_refined`

### 30 Hz

- 先统一复评了多个 30 Hz 候选目录：
  - `01_30Hz_thinlayer_v2`
  - `01_30Hz_thinlayer_v2_fixed`
  - `01_30Hz_thinlayer_v3`
  - `01_30Hz_verified`
- 发现旧常用结果 `01_30Hz_thinlayer_v3\checkpoints\best.pt` 虽然全局 PCC 较高，但薄层厚度误差 `dpde_mean` 异常大，不适合作为论文图最终版本。
- 选用更平衡的 `D:\SEISMIC_CODING\new\results\01_30Hz_verified\checkpoints\last.pt` 作为续训起点，再做一轮轻量续训。
- 续训参数：
  - 学习率：`5e-5`
  - weight decay：`1e-5`
  - patience：`20`
  - 最终在续训 `epoch 32` 处得到新的最佳综合验证结果
- 最终 30 Hz 结果目录：`D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_refined`

## 3. 最终选用版本与客观指标

统一指标口径：

- `pcc`：整体结构相似性
- `thin_pcc`：薄层区相似性
- `separability_mean`：薄层双峰分离度
- `dpde_mean`：薄层厚度误差，越小越好

### 最终入图版本

| 频率 | 最终结果路径 | pcc | thin_pcc | separability | dpde_mean |
|---|---|---:|---:|---:|---:|
| 20 Hz | `D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_refined\checkpoints\best.pt` | 0.9327 | 0.3802 | 0.9588 | 1.16 |
| 30 Hz | `D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_refined\checkpoints\best.pt` | 0.9582 | 0.4452 | 0.9897 | 1.92 |
| 40 Hz | `D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\checkpoints\best.pt` | 0.9501 | 0.4149 | 0.9793 | 1.48 |

### 30 Hz 新旧对比

| 版本 | pcc | thin_pcc | separability | dpde_mean |
|---|---:|---:|---:|---:|
| 旧候选：`01_30Hz_thinlayer_v3\best.pt` | 0.9431 | 0.3681 | 0.8933 | 42.08 |
| 新候选：`01_30Hz_thinlayer_refined\best.pt` | 0.9582 | 0.4452 | 0.9897 | 1.92 |

结论：

- 30 Hz 新结果不仅整体 PCC 更高，而且薄层区相似性、双峰分离度和厚度误差都明显更合理。
- 旧 30 Hz 结果的问题主要不是“训练完全失败”，而是旧脚本的 `best` 选择逻辑偏向单一 `val_pcc`，把一版结构上不够可信的 checkpoint 选进了图里。

## 4. 为什么认为新图优于旧图

- 20 Hz：本频带本身受分辨率限制，本次续训没有稳定超越旧 best，因此最终保留原 best，并通过统一色标和统一版式重新出图，避免因旧图版式差异放大视觉偏差。
- 30 Hz：新结果的薄层边界更稳定，层间过渡更自然，横向连续性更好，且没有旧版 `dpde_mean` 异常放大的问题。
- 40 Hz：沿用现有最稳妥版本，不引入新的对比偏差。
- 新图统一采用同一 reference、同一色标范围、同一坐标轴和同一面板风格，更适合论文主文直接引用。

## 5. 新输出文件

- 新综合图 PNG：
  - `D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_multifreq_updated.png`
- 新综合图 PDF：
  - `D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_multifreq_updated.pdf`
- 图件来源说明：
  - `D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_multifreq_updated.sources.json`
- 20 Hz 新目录：
  - `D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_refined`
- 30 Hz 新目录：
  - `D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_refined`
- 40 Hz 统一重算指标：
  - `D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\recomputed_metrics_full.json`
- 新文档：
  - `D:\SEISMIC_CODING\new\论文完整写作稿_更新20Hz30Hz图版.docx`

## 6. 论文文档处理说明

- 仓库中的 Word 文档未检测到可稳定定位的原图锚点或内嵌媒体关系，因此未做“原位替换”。
- 已生成新的 `docx` 文件，并在文档后部附入更新图与替换说明：
  - `D:\SEISMIC_CODING\new\论文完整写作稿_更新20Hz30Hz图版.docx`
- 建议在正式排版稿中，将该附图替换到原“浅部薄层自建模型不同频率反演结果图”所在位置，并保留原图号与正文引用。

## 7. 遗留问题与如实说明

- 20 Hz 仍然弱于 30 Hz / 40 Hz，这是频带分辨率限制导致的正常现象；本次没有强行“造锐化”。
- 20 Hz 虽已重新续训和复选，但最终仍以原 best 为最合理结果；因此 20 Hz 的改进更多体现在结果确认和图件规范化，而不是数值指标的大幅提升。
- 由于原 Word 文档缺少可精确替换的图锚点，本次新 `docx` 采用“附加更新图 + 替换说明”的保守方案，避免误改正文结构。
