# Marmousi2 代表性方法对比图说明

本目录包含论文主文使用的 Marmousi2 代表性方法对比图，以及对应的重绘脚本。

## 输出文件

- `fig_marmousi2_baseline_comparison.png`
- `fig_marmousi2_baseline_comparison.pdf`
- `make_fig_marmousi2_baseline_comparison.py`
- `fig_marmousi2_baseline_comparison_sources.json`

## 统一重绘策略

本图不是截图拼接，而是基于仓库中已有的结果数组与最终 checkpoint 重新生成。统一策略如下：

1. 统一比较域为 `2721 traces x 470 time samples`
2. 参考阻抗来自 `new/data.npy`
3. 对原始时间长度不是 470 的结果，沿时间轴做线性重采样对齐
4. 五个子图使用统一色标范围与统一字体、线宽、布局
5. 所有子图共享一个 colorbar，便于公平视觉比较

## 最终使用的源文件

### (a) Reference

- `d:/SEISMIC_CODING/new/data.npy`

说明：
- 参考阻抗取自 `new/data.npy` 中的 `acoustic_impedance`
- 原始长度为 1880，沿时间轴线性重采样到 470，以便与其余方法统一

### (b) comparison01: CNN-BiLSTM 半监督方法

- `d:/SEISMIC_CODING/comparison01/seismic.npy`
- `d:/SEISMIC_CODING/comparison01/norm_params.json`
- `d:/SEISMIC_CODING/comparison01/marmousi_cnn_bilstm_semi.pth`
- `d:/SEISMIC_CODING/comparison01/marmousi_cnn_bilstm.py`

说明：
- 使用最终半监督 checkpoint 重新推理整幅剖面
- 不直接使用旧截图，避免清晰度与裁剪不一致问题

### (c) comparison02: FCRSN-CW

- `d:/SEISMIC_CODING/comparison02/runs/marmousi2/results/pred_impedance_all.npy`
- `d:/SEISMIC_CODING/comparison02/runs/marmousi2/results/true_impedance_all.npy`

说明：
- 原始结果为 1880 采样点的全分辨率剖面
- 为公平对比，沿时间轴线性重采样到 470

### (d) comparison03: 半监督 GAN / WGAN-GP

- `d:/SEISMIC_CODING/comparison03/runs/optimized_v3_advanced/pred_test.npz`
- `d:/SEISMIC_CODING/comparison03/runs/optimized_v3_advanced/stats.json`
- `d:/SEISMIC_CODING/comparison03/data/marmousi2_2721_like_l101.npz`

说明：
- 使用 `optimized_v3_advanced` 结果
- 先依据 `stats.json` 反归一化，再统一显示
- 该方法使用其自带的 Marmousi2 转换包，但与仓库统一真值高度一致

### (e) new: 本文方法

- `d:/SEISMIC_CODING/new/data.npy`
- `d:/SEISMIC_CODING/new/configs/exp_real_data.yaml`
- `d:/SEISMIC_CODING/new/results/real_unet1d_optimized/norm_stats.json`
- `d:/SEISMIC_CODING/new/results/real_unet1d_optimized/checkpoints/best.pt`
- `d:/SEISMIC_CODING/new/seisinv/models/baselines.py`

说明：
- 选用 `real_unet1d_optimized` 这条 Marmousi2 派生主线
- 通过最终最佳 checkpoint 重新推理整幅剖面
- 该结果最适合与 comparison01/02/03 在 Marmousi2 上同台对比

## 统一显示策略

- colormap: `viridis`
- shared range: 以参考阻抗真值的最小值与最大值为全图统一显示范围

采用这一策略的原因是：
- 真值决定统一视觉基准
- 能保证方法间的亮度、色调变化可直接比较
- 避免每个子图单独拉伸造成的“看起来都很好”假象

## 需要说明的限制

1. `comparison02` 原始输出是 1880 采样点，需要重采样到 470 才能与其余方法并排
2. `comparison03` 使用了自己构建的 Marmousi2 数据包，不是直接读取 `new/data.npy`，但与统一真值高度一致
3. `new` 目录下后期 `v11/v12` 主要是针对真实 SGY 的实验，不适合直接作为 Marmousi2 主文对比图；因此这里选用 `real_unet1d_optimized`

## 建议图题

图 X Marmousi2 模型上不同代表性方法的波阻抗反演结果对比

## 建议图注

`comparison01`、`comparison02` 和 `comparison03` 分别对应 CNN-BiLSTM 半监督方法、FCRSN-CW 方法以及半监督 WGAN-GP 方法；“Proposed”对应 `new` 主线中基于 `real_unet1d_optimized` checkpoint 重生成的结果。所有子图均采用统一尺寸、统一裁剪范围和统一色标范围，用于比较不同方法对 Marmousi2 模型整体阻抗结构的表征能力。
