"""用相同配色重新生成对比图"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

OUT = Path('sgy_inversion_v2')

# 加载数据
seis = np.load(OUT / 'seismic_raw.npy')   # [traces, samples]
imp  = np.load(OUT / 'impedance_pred.npy') # [traces, samples]

n_total = seis.shape[0]
T_START, T_END = 2500, 6000

# 下采样
step = max(1, n_total // 3000)
sd = seis[::step, :].T   # [samples, traces]
id_ = imp[::step, :].T

# 统一配色: 彩虹色图
colors = [
    (0.0,  '#000080'),
    (0.12, '#0000FF'),
    (0.24, '#00BFFF'),
    (0.36, '#00FF7F'),
    (0.48, '#ADFF2F'),
    (0.60, '#FFFF00'),
    (0.72, '#FFA500'),
    (0.84, '#FF4500'),
    (1.0,  '#8B0000'),
]
cmap = LinearSegmentedColormap.from_list('rainbow', [(c[0],c[1]) for c in colors])

ext = [0, n_total, T_END, T_START]

# ---- 图1: 原始地震 (彩虹色) ----
fig, ax = plt.subplots(figsize=(18, 10))
vm = np.percentile(np.abs(sd), 99)
im = ax.imshow(sd, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
               extent=ext, interpolation='bilinear')
ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
ax.set_title('地震剖面 - 原始数据\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
cbar.set_label('振幅', fontsize=12)
plt.tight_layout()
plt.savefig(OUT/'seismic_original_rainbow.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  seismic_original_rainbow.png")

# ---- 图2: 反演阻抗 (彩虹色) ----
fig, ax = plt.subplots(figsize=(18, 10))
v1, v2 = np.percentile(id_, [2, 98])
im = ax.imshow(id_, aspect='auto', cmap=cmap, vmin=v1, vmax=v2,
               extent=ext, interpolation='bilinear')
ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
ax.set_title('波阻抗反演剖面\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
cbar.set_label('波阻抗 (kg/m²·s)', fontsize=12)
plt.tight_layout()
plt.savefig(OUT/'impedance_inversion_rainbow.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  impedance_inversion_rainbow.png")

# ---- 图3: 对比图 (统一配色) ----
fig, axes = plt.subplots(2, 1, figsize=(18, 14))

im1 = axes[0].imshow(sd, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
                     extent=ext, interpolation='bilinear')
axes[0].set_ylabel('时间 (ms)', fontsize=13, fontweight='bold')
axes[0].set_title('原始地震剖面', fontsize=15, fontweight='bold')
plt.colorbar(im1, ax=axes[0], pad=0.02, shrink=0.85, label='振幅')

im2 = axes[1].imshow(id_, aspect='auto', cmap=cmap, vmin=v1, vmax=v2,
                     extent=ext, interpolation='bilinear')
axes[1].set_xlabel('道号 (Trace)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('时间 (ms)', fontsize=13, fontweight='bold')
axes[1].set_title('反演波阻抗剖面', fontsize=15, fontweight='bold')
plt.colorbar(im2, ax=axes[1], pad=0.02, shrink=0.85, label='波阻抗 (kg/m²·s)')

plt.suptitle('地震反演对比 (统一配色)\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(OUT/'comparison_same_cmap.png', dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("  comparison_same_cmap.png")

print("完成!")
