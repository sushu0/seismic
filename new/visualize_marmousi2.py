"""
绘制真实的Marmousi2模型图像
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载data.npy
print("加载data.npy...")
data = np.load('data.npy', allow_pickle=True).item()

seismic = data['seismic']  # (2721, 1, 470)
impedance = data['acoustic_impedance']  # (2721, 1, 1880)

print(f"地震数据形状: {seismic.shape}")
print(f"阻抗数据形状: {impedance.shape}")
print(f"地震数据范围: [{seismic.min():.3f}, {seismic.max():.3f}]")
print(f"阻抗数据范围: [{impedance.min():.0f}, {impedance.max():.0f}]")

# 移除channel维度
seismic_2d = seismic[:, 0, :]  # (2721, 470)
impedance_2d = impedance[:, 0, :]  # (2721, 1880)

# 创建输出目录
out_dir = Path('results/marmousi2_visualization')
out_dir.mkdir(parents=True, exist_ok=True)

print("\n生成Marmousi2模型图像...")

# 1. 地震剖面图（全分辨率）
print("1. 绘制地震剖面图...")
fig, ax = plt.subplots(figsize=(16, 8))

im_T = seismic_2d.T  # (470, 2721)
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

levels = 30
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='seismic')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Amplitude', rotation=90, labelpad=15)

ax.set_xlabel('Trace number', fontsize=12)
ax.set_ylabel('Time sample', fontsize=12)
ax.set_title('Marmousi2 Seismic Section', fontsize=14, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'marmousi2_seismic_section.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ 保存: {out_dir / 'marmousi2_seismic_section.png'}")

# 2. 阻抗剖面图（全分辨率）
print("2. 绘制阻抗剖面图...")
fig, ax = plt.subplots(figsize=(16, 8))

im_T = impedance_2d.T  # (1880, 2721)
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

levels = 30
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='jet')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Impedance (m/s·g/cm³)', rotation=90, labelpad=15)

ax.set_xlabel('Trace number', fontsize=12)
ax.set_ylabel('Depth sample', fontsize=12)
ax.set_title('Marmousi2 Acoustic Impedance Section', fontsize=14, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'marmousi2_impedance_section.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ 保存: {out_dir / 'marmousi2_impedance_section.png'}")

# 3. 阻抗剖面图（下采样到与地震数据相同分辨率）
print("3. 绘制阻抗剖面图（下采样版本）...")
impedance_downsampled = impedance_2d[:, ::4][:, :470]  # (2721, 470)

fig, ax = plt.subplots(figsize=(16, 8))

im_T = impedance_downsampled.T  # (470, 2721)
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

levels = 30
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='jet')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Impedance (m/s·g/cm³)', rotation=90, labelpad=15)

ax.set_xlabel('Trace number', fontsize=12)
ax.set_ylabel('Time sample', fontsize=12)
ax.set_title('Marmousi2 Acoustic Impedance Section (Downsampled)', fontsize=14, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'marmousi2_impedance_section_downsampled.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"   ✓ 保存: {out_dir / 'marmousi2_impedance_section_downsampled.png'}")

# 4. 选择特定道进行可视化
trace_ids = [299, 599, 1699, 2299]
valid_traces = [t for t in trace_ids if t < len(seismic_2d)]

if valid_traces:
    print(f"4. 绘制特定道对比图 (道号: {valid_traces})...")
    
    # 4a. 地震道对比
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, trace_id in enumerate(valid_traces[:4]):
        if i < len(axes):
            ax = axes[i]
            t = np.arange(len(seismic_2d[trace_id]))
            
            ax.plot(t, seismic_2d[trace_id], 'b-', linewidth=1)
            ax.set_xlabel('Time sample', fontsize=10)
            ax.set_ylabel('Amplitude', fontsize=10)
            ax.set_title(f'Seismic Trace No. {trace_id}', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'marmousi2_selected_seismic_traces.png', dpi=200)
    plt.close()
    print(f"   ✓ 保存: {out_dir / 'marmousi2_selected_seismic_traces.png'}")
    
    # 4b. 阻抗道对比（原始分辨率）
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, trace_id in enumerate(valid_traces[:4]):
        if i < len(axes):
            ax = axes[i]
            t = np.arange(len(impedance_2d[trace_id]))
            
            ax.plot(t, impedance_2d[trace_id], 'r-', linewidth=1)
            ax.set_xlabel('Depth sample', fontsize=10)
            ax.set_ylabel('Impedance (m/s·g/cm³)', fontsize=10)
            ax.set_title(f'Impedance Trace No. {trace_id}', fontsize=12)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'marmousi2_selected_impedance_traces.png', dpi=200)
    plt.close()
    print(f"   ✓ 保存: {out_dir / 'marmousi2_selected_impedance_traces.png'}")

# 5. 统计信息图
print("5. 生成统计信息图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 地震数据直方图
axes[0, 0].hist(seismic_2d.flatten(), bins=100, color='blue', alpha=0.7)
axes[0, 0].set_xlabel('Amplitude')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Seismic Data Histogram')
axes[0, 0].grid(True, alpha=0.3)

# 阻抗数据直方图
axes[0, 1].hist(impedance_2d.flatten(), bins=100, color='red', alpha=0.7)
axes[0, 1].set_xlabel('Impedance (m/s·g/cm³)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Impedance Data Histogram')
axes[0, 1].grid(True, alpha=0.3)

# 随机选择几道进行wiggle显示
axes[1, 0].set_title('Seismic Wiggle Display (Random Traces)')
sample_traces = np.random.choice(len(seismic_2d), 20, replace=False)
for i, trace_id in enumerate(sorted(sample_traces)):
    offset = i * 0.5
    axes[1, 0].plot(seismic_2d[trace_id] + offset, np.arange(len(seismic_2d[trace_id])), 
                    'k-', linewidth=0.5, alpha=0.6)
axes[1, 0].invert_yaxis()
axes[1, 0].set_xlabel('Amplitude (with offset)')
axes[1, 0].set_ylabel('Time sample')
axes[1, 0].grid(True, alpha=0.3)

# 数据统计文本
stats_text = f"""
Marmousi2 Model Statistics

Seismic Data:
  Shape: {seismic.shape}
  Range: [{seismic.min():.3f}, {seismic.max():.3f}]
  Mean: {seismic.mean():.3f}
  Std: {seismic.std():.3f}

Impedance Data:
  Shape: {impedance.shape}
  Range: [{impedance.min():.0f}, {impedance.max():.0f}]
  Mean: {impedance.mean():.0f}
  Std: {impedance.std():.0f}

Total Traces: {seismic.shape[0]}
Seismic Samples: {seismic.shape[2]}
Impedance Samples: {impedance.shape[2]}
"""

axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig(out_dir / 'marmousi2_statistics.png', dpi=200)
plt.close()
print(f"   ✓ 保存: {out_dir / 'marmousi2_statistics.png'}")

print("\n" + "="*70)
print("Marmousi2模型可视化完成!")
print("="*70)
print(f"输出目录: {out_dir}")
print("\n生成的图像:")
print("  1. marmousi2_seismic_section.png - 地震剖面图（全分辨率）")
print("  2. marmousi2_impedance_section.png - 阻抗剖面图（原始分辨率1880点）")
print("  3. marmousi2_impedance_section_downsampled.png - 阻抗剖面图（下采样470点）")
print("  4. marmousi2_selected_seismic_traces.png - 选定地震道")
print("  5. marmousi2_selected_impedance_traces.png - 选定阻抗道")
print("  6. marmousi2_statistics.png - 数据统计信息")
print("="*70)
