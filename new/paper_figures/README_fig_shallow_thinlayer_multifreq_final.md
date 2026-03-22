# 浅部薄层自建模型不同频率反演对比图终稿说明

## 1. 最终使用的源文件

### Reference / Ground Truth

- 统一参考阻抗文件：`D:\SEISMIC_CODING\zmy_data\01\data\01_40Hz_04.txt`

说明：

- `01_20Hz_04.txt`、`01_30Hz_04.txt` 与 `01_40Hz_04.txt` 对应的参考阻抗内容一致，可作为同一自建模型下的统一 reference。
- 终稿绘图脚本使用 `01_40Hz_04.txt` 作为统一参考输入。

### 20 Hz

- 最终结果目录：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_optimized_v2`
- 最终 checkpoint：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_optimized_v2\checkpoints\best.pt`
- 预测缓存：`D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_optimized_v2\pred_full.npy`

### 30 Hz

- 最终结果目录：`D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_refined`
- 最终 checkpoint：`D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_refined\checkpoints\best.pt`
- 预测缓存：`D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_refined\pred_full.npy`

### 40 Hz

- 最终结果目录：`D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2`
- 最终 checkpoint：`D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\checkpoints\best.pt`
- 预测缓存：`D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\pred_full.npy`

## 2. 20 Hz、30 Hz、40 Hz 分别对应哪一版结果

- 20 Hz：采用优化后的最终论文版本 `01_20Hz_thinlayer_optimized_v2`
- 30 Hz：采用综合指标和结构质量均更适合论文展示的 `01_30Hz_thinlayer_refined`
- 40 Hz：采用当前稳定、细节表现合理的 `01_40Hz_thinlayer_v2`

## 3. 为什么这样选

### 20 Hz

- 旧论文 20 Hz 结果虽然不是完全失效，但局部存在不自然形变、拖尾偏重和薄层上下界不够自然的问题。
- 优化后的 20 Hz 版本在不做假锐化的前提下，改善了结构稳定性和浅部目标窗口的一致性，更适合作为论文中的低频基线结果。
- 该版本保留了 20 Hz 应有的分辨率受限特征，因此更可信。

### 30 Hz

- 30 Hz 最终版本在薄层边界清晰度、层体分离度和整体稳定性之间取得了更好的平衡。
- 相比旧 30 Hz 候选，当前版本更少出现不合理厚度误差和结构性伪影，更适合作为主文展示结果。

### 40 Hz

- 40 Hz 版本能保留更多局部细节，同时整体结构仍保持合理。
- 终稿图中继续沿用该版本，可保证与 20 Hz / 30 Hz 的对比公平，不人为放大频率优势。

## 4. 相比旧图，新图的主要改进

- 20 Hz 已替换为优化后的最终版本，不再给人“明显做坏了”的印象。
- 30 Hz 使用了更适合论文展示的稳定版本，更能体现其在浅部薄层表征与稳定性之间的平衡优势。
- 40 Hz 继续沿用当前最佳版本，避免重新换图带来新的比较偏差。
- 统一采用同一 reference、同一色标范围、同一坐标范围和同一版式，提升了三频结果间的可比性。
- 图面标题、子图编号、字体、线宽和 colorbar 已统一为论文主文风格，不保留脚本名、版本号或目录名。

## 5. 终稿图件与脚本

- PNG：`D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_multifreq_final.png`
- PDF：`D:\SEISMIC_CODING\new\paper_figures\fig_shallow_thinlayer_multifreq_final.pdf`
- 绘图脚本：`D:\SEISMIC_CODING\new\paper_figures\make_fig_shallow_thinlayer_multifreq_final.py`

图像样式：

- 2 x 2 布局
- 子图为 `(a) Reference Impedance`、`(b) 20 Hz`、`(c) 30 Hz`、`(d) 40 Hz`
- 共享 colorbar，标注为 `Impedance`
- 白色背景
- 统一 `Trace` 横轴与 `Time (ms)` 纵轴
- 输出分辨率为 `600 dpi`

## 6. 建议图题和图注

### 建议图题

图 X 浅部薄层自建模型不同频率参考条件下的波阻抗反演结果对比

### 建议图注

图 X 给出了浅部薄层自建模型在不同频率参考条件下的波阻抗反演结果，其中 (a) 为参考阻抗，(b)–(d) 分别为 20 Hz、30 Hz 和 40 Hz 条件下的反演结果。可以看出，20 Hz 条件下能够恢复主要层体位置和基本形态，但对薄层边界与层间分离的刻画相对有限；30 Hz 条件下，薄层边界清晰度和层体分离度明显改善；40 Hz 条件下进一步保留了更多局部细节。该结果说明，对于浅部薄层目标层段，20–30 Hz 可作为具有实际意义的敏感参考频带，其中 30 Hz 在结构表征与稳定性之间表现出较好的平衡。

## 7. Word 文档处理情况

- 输出的新文档：`D:\SEISMIC_CODING\new\论文完整写作稿_替换多频图终稿版.docx`

处理说明：

- 已尝试基于原论文文档定位旧图对应位置，但未找到可稳定用于“精确原位替换”的图像锚点或明确的段落关系。
- 为避免破坏原文档结构，终稿 docx 采用保守方案：
  - 复制原论文文档；
  - 在文档末尾追加终稿图和替换说明；
  - 不直接修改正文已有结构。

替换建议：

- 将该终稿图替换到原文中“浅部薄层自建模型不同频率反演结果图”所在位置。
- 保留原图号与正文引用关系，仅替换图片内容，并按需要采用本 README 中给出的图题和图注。

## 8. 仍需如实说明的局部限制

- 20 Hz 仍明显弱于 30 Hz / 40 Hz，这属于频带分辨率限制带来的正常现象。
- 40 Hz 保留了更多局部细节，但这种提升并不是简单线性外推到所有部位；局部细节增强并不意味着所有区域都同等改善。
- 当前终稿图已经较好支持论文叙述，但如果后续论文整体版式继续调整，仍建议在最终排版阶段再统一检查与正文邻近图件的字号、宽度和页内位置。
