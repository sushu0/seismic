# Figure 7 地震风格一致性展示图说明

## 输出文件

- `D:\SEISMIC_CODING\new\paper_figures\fig7_rendered_consistency.png`
- `D:\SEISMIC_CODING\new\paper_figures\fig7_rendered_consistency.pdf`
- `D:\SEISMIC_CODING\new\paper_figures\make_fig7_rendered_consistency.py`

## 使用的源文件

- `D:\SEISMIC_CODING\new\sgy_inversion_v12_rendered_99eval\run_config.json`
- `D:\SEISMIC_CODING\new\sgy_inversion_v12_rendered_99eval\rendered_seismic.npy`
- `D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy`
- `D:\SEISMIC_CODING\new\train_sgy_v11.py`

## 对应结果说明

- 本图使用 `sgy_inversion_v12_rendered_99eval` 中的 `rendered_seismic.npy` 作为 seismic-like rendered view
- 该 rendered view 对应的是基于主反演结果构建的辅助展示层
- 它用于增强观测地震与反演结果之间的展示一致性与解释直观性，不作为阻抗本体真实性的直接证明

## 局部窗口选取原则

- 先在研究时间窗上部 400 个采样点中自动搜索浅部目标层段
- 再按事件丰富度、横向可追踪性与 observed-vs-rendered 局部一致性自动选取 2 个代表性窗口
- 为保证主文版式紧凑，本图采用 2 个局部窗口

## 最终窗口范围

### W1
- plotted trace range: `3680 - 4480`
- original SGY trace ids: `381070 - 463807`
- time range (ms): `2760 - 3280`
- score: `0.7912`
- reason: 右侧目标段横向连续性较强，适合展示 rendered view 对可追踪结构的辅助表达。

### W2
- plotted trace range: `2080 - 2880`
- original SGY trace ids: `215387 - 298125`
- time range (ms): `2760 - 3280`
- score: `0.5742`
- reason: 中部目标段事件密集且层间过渡丰富，最能体现 rendered consistency 对解释直观性的增强。

## 图像用途说明

- 该图属于辅助解释展示图，不属于主反演真实性证明图
- 该图展示的是观测地震与 seismic-like rendered view 的显示一致性增强
- 该图不表示阻抗本体图像与原始地震图像直接等价

## 建议图题

图 7 地震风格一致性展示图

## 建议图注

图 7 给出了观测地震剖面与基于主反演结果构建的 seismic-like rendered view 的对应展示。该结果主要用于增强反演结果的可解释一致性与可视化直观性。需要指出的是，该图反映的是展示层面的一致性增强，而非阻抗本体图像真实性的直接证明。