# Figure 6 浅部目标层段局部放大图说明

## 输出文件

- `D:\SEISMIC_CODING\new\paper_figures\fig6_shallow_target_zoom.png`
- `D:\SEISMIC_CODING\new\paper_figures\fig6_shallow_target_zoom.pdf`
- `D:\SEISMIC_CODING\new\paper_figures\make_fig6_shallow_target_zoom.py`

## 使用的源文件

- `D:\SEISMIC_CODING\new\sgy_inversion_v11_display_99bridge_full\run_config.json`
- `d:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy`
- `D:\SEISMIC_CODING\new\sgy_inversion_v11_display_99bridge_full\impedance_pred_final.npy`
- `D:\SEISMIC_CODING\new\train_sgy_v11.py`

## 选窗原则

- 使用 `v11_edgeplus` 主结果对应的同一批 5000 条均匀采样道
- 在研究时间窗上部 400 个采样点内自动搜索浅部目标层段
- 先按事件强度与时间梯度确定统一的浅部时间窗，再按事件丰富度、边界信息量与横向连续性自动挑选 3 个代表性横向窗口

## 最终窗口范围

### W1
- plotted trace range: `2100 - 2800`
- original SGY trace ids: `217458 - 289841`
- time range (ms): `2760 - 3280`
- score: `0.9423`
- reason: 中部目标段结构最为复杂，能体现边界连续性与细节刻画能力。

### W2
- plotted trace range: `1400 - 2100`
- original SGY trace ids: `144972 - 217355`
- time range (ms): `2760 - 3280`
- score: `0.8335`
- reason: 中部目标段结构最为复杂，能体现边界连续性与细节刻画能力。

### W3
- plotted trace range: `3640 - 4340`
- original SGY trace ids: `376927 - 449310`
- time range (ms): `2760 - 3280`
- score: `0.7366`
- reason: 右侧目标段横向事件较可追踪，适合展示横向连续性与层间过渡。

## 绘图风格

- 横轴为 trace，纵轴为 time (ms)
- 左侧为全剖面概览，上方原始地震、下方主反演阻抗
- 右侧为 3 个局部放大窗口，均采用与主文一致的上下对照排布
- 原始地震与阻抗结果均采用与主文一致的共享显示映射和 `RdBu_r` 配色

## 建议图题

图 6 真实 SGY 浅部目标层段局部放大对比图

## 建议图注

左侧为真实 SGY 主结果对应的全剖面概览，矩形框标出浅部目标层段的 3 个局部放大位置；右侧为对应窗口的原始地震与主反演阻抗结果对照。局部窗口位于研究时间窗的上部目标层段，用于展示浅部目标层段的结构边界、层间过渡和横向连续性表征能力。