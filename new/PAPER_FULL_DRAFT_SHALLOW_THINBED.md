# 面向浅部薄层目标层段表征的真实叠后 SGY 无标签深度学习波阻抗反演与可解释一致性展示

## 中文摘要
真实叠后地震资料的波阻抗反演是薄互层/薄层目标识别与层段精细解释的重要手段，但在实际应用中常面临井约束稀缺、标签不足以及子波/相位不确定等问题，使得深度学习方法难以直接迁移并缺乏可靠评估。针对这一问题，本文提出一种面向真实叠后 SGY 数据的无标签深度学习波阻抗反演工作流。该方法首先利用多个监督模型构建先验集成，以中位数先验提供低频锚定、以先验不确定性自适应调节先验约束强度；随后在可微正演闭环下开展自监督适配训练，通过阻抗到反射系数再到合成地震的物理链条，联合局部相关、幅值敏感一致性、时频一致性、低频先验一致性、反射系数精度正则、结构可信度与方差控制等多类损失，实现无真实阻抗标签场景下的稳定反演。

本文所述“浅部”并非近地表意义上的浅层，而是指研究时间窗上部的目标解释层段。针对该层段的频谱统计表明，真实 SGY 全时窗主导能量峰值约为 12.28 Hz，而上部目标层段窗口前 256–400 个采样点的有效能量主要分布于 10–30 Hz，95% 累积能量上限接近 27–28 Hz。结合薄层调谐与分辨率对频带高端敏感的理论认识，本文将 20–30 Hz 视为浅部薄层敏感表征的重要参考频带，而非该数据的主频峰值。

在单条真实 SGY 案例中，本文的主反演结果在无强后处理条件下获得较高的物理闭环一致性与合理阻抗尺度，表现为 `mean_trace_pcc≈0.9985`、`resid_l1_si≈0.0362`、`edge_alignment≈0.4977`，同时保持较为可信的结构与横向特征。进一步地，本文提出“可解释一致性展示”策略，即将反演阻抗经物理正演生成 seismic-like rendered view，并与观测地震在同显示域下进行一致性比较。该辅助结果在保持物理指标基本稳定的前提下实现了大于 99% 的渲染显示一致性，可有效增强浅部薄层目标层段解释的可视化一致性与沟通直观性。本文同时通过反例说明：若直接追求“观测地震图像与阻抗本体显示图像”的极高相似度，虽可获得近乎完美的视觉一致性，却会导致物理闭环失效与阻抗方差发散，从而强调不可作弊一致性定义与多目标评估的重要性。

**关键词**：真实叠后地震资料；无标签波阻抗反演；浅部目标层段；薄层表征；物理闭环自监督；先验集成；可解释一致性展示

## English Abstract
Acoustic impedance inversion from field post-stack seismic data is essential for thin-bed interpretation and interval-scale geological characterization, yet deep-learning approaches are often limited by scarce well labels, uncertain wavelets/phases, and the lack of reliable validation in unlabeled settings. This study proposes a field-ready label-free deep-learning workflow for impedance inversion from a real SGY dataset. A prior ensemble is first constructed from multiple supervised models, where the median prior anchors low-frequency trends and the ensemble uncertainty adaptively modulates prior strength. Self-supervised adaptation is then performed through a differentiable physics loop, i.e., impedance to reflectivity to convolutional forward modeling to synthetic seismic, optimized using local waveform similarity, scale-invariant amplitude consistency, time-frequency consistency, low-frequency prior consistency, reflectivity precision regularization, structural credibility constraints, and variance control.

In this paper, the term "shallow" does not refer to the near-surface, but to the upper target interval within the analysis time window. Spectral analysis of the field SGY indicates that the dominant peak frequency of the full window is about 12.28 Hz, while the upper target interval (first 256–400 samples) preserves effective energy mainly within 10–30 Hz, with the 95% cumulative-power upper bound approaching 27–28 Hz. Considering the frequency dependence of thin-bed tuning and resolution, the 20–30 Hz range is treated as a practical reference band for shallow thin-bed sensitive characterization, rather than the dominant peak frequency of the data.

On the field SGY case, the main inversion result achieves strong physics-loop consistency without aggressive post-processing, with `mean_trace_pcc≈0.9985`, `resid_l1_si≈0.0362`, and `edge_alignment≈0.4977`, while maintaining plausible impedance scale and structural characteristics. In addition, an explainable rendered-consistency display is introduced by forward-modeling the inverted impedance into a seismic-like rendered view and comparing it with the observed seismic in the same display domain. This auxiliary result achieves over 99% rendered-display consistency while keeping the main physics indicators stable, thereby improving interpretability and communication for shallow thin-bed target characterization. A counter-example further shows that directly optimizing for near-perfect similarity between observed seismic images and impedance-display images can catastrophically break physics consistency and cause variance divergence, underscoring the need for non-cheatable consistency definitions and multi-objective evaluation.

**Keywords**: field post-stack seismic; label-free impedance inversion; shallow target interval; thin-bed characterization; physics-loop self-supervision; prior ensemble; explainable rendered-consistency display

## 1 引言
波阻抗反演通过将叠后地震记录映射为阻抗剖面，为岩性识别、储层预测及层段精细解释提供关键物性表征。对于薄互层/薄层地质体而言，层厚小、横向变化快、调谐效应显著，导致传统反演方法与简单监督学习模型都容易出现分辨率不足、结构边界模糊或尺度失真的问题。因此，面向薄层目标层段的高分辨率、可信且可解释的波阻抗反演，一直是地震解释与勘探地球物理中的关键研究方向。

近年来，深度学习为地震波阻抗反演提供了新的技术路径。卷积网络、时序建模网络、残差收缩网络以及半监督/对抗式框架等方法已在合成数据或局部实际数据上展示出较好的拟合性能。然而，真实叠后地震资料场景仍然存在三类突出难题。其一，井约束稀缺，难以获得足够的高质量阻抗标签，导致纯监督方法难以直接用于真实资料。其二，叠后卷积模型天然带限，低频缺失严重，反演问题强不适定，容易出现“合成地震拟合良好但阻抗并不可信”的非唯一解。其三，真实资料中的子波、相位与处理增益并不精确已知，模型误差容易转嫁为阻抗振荡、方差发散或横向粗糙度异常。

这一问题在薄层解释场景下尤为敏感。经典薄层调谐理论指出，可分辨的层厚、调谐厚度以及波形叠置行为与频率成分密切相关，频带高端的有效能量直接影响薄层敏感性与纵向分辨能力。因此，如果方法在真实资料的关键频段上缺乏约束，即便全局波形拟合较好，也未必能够给出可用于薄层解释的阻抗结果。

另一方面，在实际解释流程中，解释人员往往习惯于直接比较“观测地震剖面”和“反演结果剖面”的视觉一致性，希望二者在同一配色方案下“看起来相似”。但本文的系列实验表明，若直接优化“观测地震图像 vs 阻抗本体显示图像”的像素级相似度，会诱导模型走向显示域匹配捷径，导致阻抗方差失控、横向粗糙度爆炸以及物理闭环失效。也就是说，“看起来像”并不等价于“反演可信”。

针对上述问题，本文提出一种面向真实叠后 SGY 的无标签深度学习波阻抗反演框架。该方法以监督先验集成为低频锚点，以可微正演闭环构建无标签自监督训练路径，通过多目标损失联合约束物理一致性、结构可信度与尺度稳定性。在此基础上，本文进一步提出“可解释一致性展示（rendered seismic-like consistency）”，将反演阻抗经物理正演渲染为 seismic-like 视图，并与观测地震在同显示域下比较，以满足解释场景对直观一致性的需求，同时避免“阻抗本体图像直接像地震图像”的方法学误区。

本文的主要贡献包括：

1. 提出一种面向真实叠后 SGY 的无标签深度学习波阻抗反演工作流，将监督先验集成与物理闭环自监督适配统一到同一框架中。
2. 针对真实资料无真值标签的场景，构建了以物理闭环一致性、结构指标、横向连续性和尺度稳定性为核心的多指标评估与候选选择策略。
3. 提出 rendered seismic-like consistency 作为解释友好的辅助展示层，并通过反例证明 naive display matching 的风险，从而给出一种更稳妥的“观测一致性展示”定义。

## 2 相关工作
### 2.1 深度学习波阻抗反演
深度学习波阻抗反演大致可分为三条代表性路线。第一类是端到端卷积回归模型，通过卷积编码器或全卷积框架直接建立地震道到阻抗道的映射关系；第二类是结合时序建模的网络，如 CNN-BiLSTM，将卷积特征提取与双向时序依赖结合，以增强纵向关系表达；第三类是对抗式或半监督式框架，例如基于 GAN/WGAN-GP 的波阻抗反演，通过少量标签与大量无标签地震联合训练提高泛化能力。

在本文工作中，`comparison01`、`comparison02` 和 `comparison03` 分别对 CNN-BiLSTM 半监督方法、FCRSN-CW 方法以及半监督 GAN/WGAN-GP 方法进行了复现，构成本文的代表性对比谱系。这些复现实验的意义，不仅在于比较性能，更在于明确不同方法对物理约束、结构保持与无标签适配能力的侧重差异。

### 2.2 物理约束与闭环反演
为缓解纯数据驱动模型的非唯一性，越来越多研究将物理正演过程嵌入深度学习框架，通过“阻抗—反射系数—合成地震”的可微正演路径，将观测地震重新纳入训练目标。相关工作表明，物理闭环不仅可为无标签数据提供训练信号，还可显著降低“视觉上合理但物理上错误”的退化解风险。本文的自监督主线正建立在这一思路之上。

### 2.3 薄层解释与频带敏感性
薄层目标的识别受调谐效应与频带高端成分控制显著。波形指示反演等传统高分辨率反演工作强调波形结构与频带信息对薄互层识别的重要作用。本文借鉴这一认识，但不直接宣称在无标签真实资料上恢复薄层真值阻抗，而是将“浅部薄层目标层段表征”界定为：在上部目标层段频带适配、物理闭环一致性、结构边界可信度和可解释展示的一致性共同支撑下，对薄层相关结构做出更可信的阻抗表征。

## 3 方法
### 3.1 总体框架
本文方法由三个核心模块构成。第一，监督先验集成模块从多个监督模型生成真实数据上的阻抗预测，并以逐点中位数构建基先验，以逐点标准差表征先验不确定性。第二，主反演模块在真实 SGY 上进行无标签自监督适配训练，通过可微正演闭环建立物理训练目标。第三，渲染一致性展示模块将最终阻抗结果映射为 seismic-like rendered view，用于解释与展示，而不直接参与阻抗真值判定。

### 3.2 监督先验集成
设三个监督模型在真实 SGY 上的 log-impedance 预测分别为 $z_1, z_2, z_3$。本文采用逐点中位数

$$
z_{base}(t,x)=\operatorname{median}\{z_1(t,x),z_2(t,x),z_3(t,x)\}
$$

作为基先验，并以逐点标准差

$$
\sigma_{prior}(t,x)=\operatorname{std}\{z_1(t,x),z_2(t,x),z_3(t,x)\}
$$

表征先验不确定性。该设计可在无标签真实资料上提供低频锚定，并允许模型在高不确定区域相对自由地偏离先验。

### 3.3 有界扰动输出
考虑到无标签场景下直接预测绝对阻抗容易产生尺度漂移，本文预测的是 log-impedance 的有界扰动：

$$
\log Z_{pred}(t)=\log Z_{base}(t)+\alpha\tanh\left(S(d(t))\right),
$$

其中 $d(t)$ 为网络输出，$S(\cdot)$ 为平滑算子，$\alpha$ 为扰动上界。该设计既保留了模型对局部结构的调整能力，又避免其在无监督条件下产生过大幅度的非物理偏移。

### 3.4 可微正演闭环
阻抗到合成地震的物理链条由两步构成。首先，由阻抗计算反射系数：

$$
r(t)=\frac{Z(t+\Delta t)-Z(t)}{Z(t+\Delta t)+Z(t)+\epsilon}.
$$

随后，将反射系数与子波卷积生成合成地震：

$$
\hat{s}(t)=w(t)\ast r(t),
$$

其中 $w(t)$ 为 Ricker 型子波。为了减弱固定子波假设带来的误差，本文引入低自由度的相位校正参数，但在稳定性优先的训练设置下保持其小幅可控。

### 3.5 损失函数
总体损失可写为：

$$
\mathcal{L}=\lambda_1\mathcal{L}_{LNCC}
+\lambda_2\mathcal{L}_{amp}
+\lambda_3\mathcal{L}_{STFT}
+\lambda_4\mathcal{L}_{prior}
+\lambda_5\mathcal{L}_{refl}
+\lambda_6\mathcal{L}_{struct}
+\lambda_7\mathcal{L}_{var}
+\lambda_8\mathcal{L}_{wavelet}.
$$

其中：

- $\mathcal{L}_{LNCC}$ 约束局部波形相关性，强调事件对齐；
- $\mathcal{L}_{amp}$ 使用尺度不变的幅值误差，避免全局增益差异导致的错误惩罚；
- $\mathcal{L}_{STFT}$ 保证时频一致性；
- $\mathcal{L}_{prior}$ 约束低频先验一致性；
- $\mathcal{L}_{refl}$ 抑制非事件区伪反射；
- $\mathcal{L}_{struct}$ 提升边界与结构可信度；
- $\mathcal{L}_{var}$ 避免方差塌缩或发散；
- $\mathcal{L}_{wavelet}$ 约束子波参数保持稳定。

### 3.6 可解释一致性展示
本文特别区分“主反演结果”和“展示层结果”。对最终阻抗 $Z_{pred}$，通过同一正演链条生成 seismic-like rendered view，再在同一显示域下与观测地震比较，记为 rendered consistency。该展示层仅用于说明“反演结果可产生与观测一致的地震响应”，而不用于证明阻抗本体图像应直接与原始地震图像相似。

## 4 数据与实验设置
### 4.1 真实数据
本文使用真实叠后地震 SGY 文件 `0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy`。该数据包含 517,655 道，每道 1,751 个采样点，采样间隔为 0.002 s，对应时间窗约 2500–6000 ms。

本文所称“浅部目标层段”是指这一分析时间窗中的上部解释层段，而不是近地表意义上的浅层。按采样点估算，前 256 个采样点对应约 0.512 s，前 400 个采样点对应约 0.800 s，可视为后续频谱分析和浅部薄层表征讨论的重点窗口。

### 4.2 频谱特性分析与浅部薄层敏感频带
为了说明薄层表征与频带选择的关系，本文对真实 SGY 数据进行了频谱统计。结果表明：

| 窗口 | 主导峰值频率 (Hz) | $f_{50}$ (Hz) | $f_{95}$ (Hz) |
|---|---:|---:|---:|
| 全时窗（0–1751 点） | 12.28 | 12.85 | 23.70 |
| 上部窗口（前 256 点） | 11.72 | 15.63 | 27.34 |
| 上部窗口（前 400 点） | 12.50 | 16.25 | 27.50 |

由此可见，该真实资料全时窗的主导能量峰值约为 12 Hz，但上部目标层段保留了更宽的有效带宽，其有效能量主要分布在 10–30 Hz，高频端延伸至约 27–28 Hz。基于薄层调谐与频率高端敏感性的经典认识，本文将 20–30 Hz 视为浅部薄层敏感表征的重要参考频带，而不是该数据的主频峰值。

### 4.3 先验模型与训练设置
本文使用三组监督模型构建先验集成，对应不同的频带假设。这样做的目的不在于宣称真实数据主频恰为某一固定频率，而在于利用 20/30/40 Hz 等多频模型共同提供不同带宽假设下的结构先验，并由先验不确定性自动调节其作用强度。

主模型采用单道输入的一维反演网络，并在真实 SGY 上进行自监督适配训练。训练过程中从全场数据中均匀抽样用于训练与评估，最终对 5000 道代表性剖面做可视化分析。

### 4.4 对比实验与评估指标
本文对比谱系包括三类代表性复现实验：

1. `comparison01`：CNN-BiLSTM 半监督方法；
2. `comparison02`：FCRSN-CW 方法；
3. `comparison03`：半监督 GAN / WGAN-GP 方法。

这些方法构成方法谱系背景，用于说明本文工作并非孤立调参，而是在已有代表性路线基础上的系统推进。

本文在真实 SGY 主实验中采用以下核心指标：

- `mean_trace_pcc`：合成地震与观测地震逐道相关的平均值；
- `resid_l1_si`：尺度不变残差；
- `edge_alignment`：结构边界与事件对应程度；
- `lat_tv_ratio` / `lat_tv_ratio_weighted`：横向连续性相关指标；
- `final_impedance_mean` 与 `final_impedance_std`：阻抗尺度与方差稳定性；
- `render_display_agreement / pearson / cosine`：渲染一致性展示指标。

## 5 结果与分析
### 5.1 主反演结果：v11_edgeplus
本文将 `v11_edgeplus` 作为主反演结果。该结果在无强后处理条件下（`raw_identity`）取得了较强的物理闭环一致性与较为可信的结构表现：

| 指标 | 数值 |
|---|---:|
| mean_trace_pcc | 0.998464 |
| resid_l1_si | 0.036174 |
| edge_alignment | 0.497706 |
| lat_tv_ratio | 1.942256 |
| final_impedance_mean | $7.14\times10^6$ |
| final_impedance_std | $4.06\times10^5$ |

这一结果说明，在无真实阻抗标签条件下，借助监督先验集成与物理闭环自监督适配，可以在真实 SGY 上获得高观测一致性、合理阻抗量级与较稳定的结构特征。对于浅部薄层目标层段而言，该结果更重要的价值不在于“恢复真值”，而在于提供一条在物理、结构与尺度三方面都相对可信的阻抗表征路径。

### 5.2 可解释一致性展示：v12_rendered_99eval
作为辅助展示结果，`v12_rendered_99eval` 将主反演结果衍生为 seismic-like rendered view，并与观测地震在同显示域下比较。其关键指标为：

| 指标 | 数值 |
|---|---:|
| render_display_agreement | 0.999627 |
| render_display_pearson | 0.999981 |
| render_display_cosine | 0.999981 |
| mean_trace_pcc | 0.998038 |
| resid_l1_si | 0.041454 |
| edge_alignment | 0.494625 |
| lat_tv_ratio_weighted | 1.835066 |

该结果的意义在于：解释人员通常需要“看起来更一致”的对比图，以帮助判断反演结果是否与原始地震的事件走势和空间结构相匹配。rendered consistency 通过比较物理可比对象（观测地震 vs 由阻抗正演得到的 rendered seismic-like view），在不牺牲主反演物理性的前提下实现了超过 99% 的一致性展示。因此，它适合在论文中作为“辅助展示层”和“解释友好评估层”，而非作为阻抗真值准确性的直接证据。

### 5.3 反例：v11_inversion99_restart
`v11_inversion99_restart` 使用 `display_direct_match` 路线，直接追求“原始 seismic 图像 vs 阻抗本体显示图像”的极高相似度。其结果表现为：

| 指标 | 数值 |
|---|---:|
| display_agreement | 0.996073 |
| display_pearson | 0.998692 |
| display_cosine | 0.998691 |
| mean_trace_pcc | 0.111967 |
| resid_l1_si | 0.639954 |
| final_impedance_std | $4.49\times10^6$ |

该结果虽然在视觉上极度接近原始地震图像，但物理闭环严重失效，阻抗方差明显发散，说明 naive display matching 会诱导模型进入显示域作弊。这个反例的学术意义不在于“性能更好”，而在于明确了高视觉一致性与高物理可信度之间的边界，并证明本文提出 rendered consistency 的必要性。

### 5.4 与复现基线的关系
在方法谱系上，本文并非从零开始，而是在复现多类代表性方法基础上的进一步推进。`comparison01`、`comparison02` 和 `comparison03` 分别代表半监督时序建模、全卷积残差收缩网络和对抗式半监督反演路线。与这些方法相比，本文的差异化价值主要在于：

1. 将监督先验集成与真实 SGY 无标签自监督适配统一到同一闭环框架；
2. 建立了面向真实资料的多目标评估与候选选择逻辑，而非仅依赖单一相关系数；
3. 给出了 rendered consistency 作为辅助展示层，并用反例证明直接 display matching 的风险。

## 6 讨论
### 6.1 为什么需要“浅部薄层目标层段”这一定位
从频谱统计看，真实 SGY 的全时窗主导峰值约为 12 Hz，但上部目标层段的有效带宽明显更宽，95% 累积能量上限已接近 27–28 Hz。这意味着在上部解释层段，高频端仍保留了用于薄层敏感表征的重要信息。因此，将 20–30 Hz 作为浅部薄层敏感参考频带是有物理与统计依据的，但这一表述必须克制：它不是说数据主频就是 20–30 Hz，而是说这一频带对于上部目标层段的薄层表征更具解释意义。

### 6.2 rendered consistency 的方法学意义
本文认为，“像”的定义必须建立在物理可比对象之上。观测地震与由反演阻抗正演得到的 rendered seismic-like view 属于同一类可比量，因此它们之间的高一致性可以被用于辅助解释；相反，观测地震与阻抗本体显示图属于不同物理量，直接要求二者像素级一致只会驱动显示域作弊。换言之，rendered consistency 不是在回避“像”的需求，而是在给“像”一个可被物理接受的定义。

### 6.3 无真值标签场景下的证据边界
由于真实资料缺少阻抗真值，本文不能宣称已经准确恢复浅部薄层的真实阻抗。本文能够提供的证据链包括：上部目标层段的频带适配性、物理闭环一致性、结构边界与事件对应关系、横向连续性诊断以及渲染一致性展示。因此，本文更稳妥的结论应是“为浅部薄层目标层段提供了更可信的阻抗表征与更直观的一致性展示”，而非“恢复了真实薄层真值阻抗”。

### 6.4 局限性与未来工作
本文仍存在若干局限。首先，真实资料仅有单条 SGY 案例，尚需更多工区数据验证泛化性。其次，当前浅部薄层解释的结论主要依赖结构一致性与频带适配性证据，而非井约束验证；后续若能引入井资料，可进一步开展井震标定与薄层统计验证。再次，尽管 rendered consistency 解决了展示层的解释问题，但它不能替代对阻抗本体真实性的独立验证。未来工作可围绕多工区验证、有限井约束联合验证以及面向薄层目标层段的更强空间结构建模展开。

## 7 结论
本文面向真实叠后 SGY 的无标签波阻抗反演问题，提出了一种服务于浅部薄层目标层段表征的深度学习工作流。该方法通过监督先验集成提供低频锚定，在可微正演闭环下进行自监督适配训练，并联合多类损失约束物理一致性、结构可信度、横向连续性与尺度稳定性。频谱分析表明，真实资料全时窗主导峰值约为 12 Hz，而上部目标层段的有效能量主要分布在 10–30 Hz，95% 累积能量上限接近 27–28 Hz；据此，本文将 20–30 Hz 视为浅部薄层敏感表征的重要参考频带，而非主频峰值。

在真实资料案例中，主结果 `v11_edgeplus` 在无强后处理条件下取得了高物理闭环一致性与合理阻抗尺度，并表现出较可信的结构与横向特征。与此同时，本文提出的 rendered consistency 展示层通过将反演阻抗正演为 seismic-like view，与观测地震在同域比较，在保持主物理指标基本稳定的前提下实现了大于 99% 的渲染显示一致性，从而增强了浅部薄层目标层段解释的直观一致性与沟通可用性。反例 `v11_inversion99_restart` 则进一步证明：直接追求“原始 seismic 图像 vs 阻抗本体图像”的极高相似度会导致物理闭环崩坏与方差发散。综上，本文强调，在真实 SGY 无标签场景下，高可信反演不应建立在显示域作弊基础上，而应建立在物理闭环、结构一致性和可解释展示层协同支撑的证据链之上。

## 参考文献（待按目标期刊格式统一）
1. 周萍，等. 基于 CNN-BiLSTM 的半监督学习地震波阻抗反演.
2. 王康，等. 基于全卷积残差收缩网络的地震波阻抗反演.
3. 王永昌，等. 基于生成对抗网络的半监督地震波阻抗反演.
4. 陈彦虎，等. 地震波形指示反演方法及其应用.
5. [待补全] SG-CUnet 相关论文.
6. [待补全] 深度学习三种地震波阻抗反演方法比较.

## 附：建议在投稿前补充或替换的材料
1. 将 `comparison01/02/03` 的关键数值指标统一整理成一张基线对比表。
2. 在主文中加入 `v11_edgeplus` 的上部 256–400 点窗口局部放大图。
3. 在补充材料中加入 `v11_inversion99_restart` 的反例图，以支撑方法学讨论。
4. 将参考文献补全为目标期刊格式。
