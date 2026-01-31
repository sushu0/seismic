# -*- coding: utf-8 -*-
"""
40Hz模型结果可视化脚本
"""
import numpy as np
import segyio
import torch
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

# 配置
SEISMIC_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_40Hz_re.sgy'
IMPEDANCE_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_40Hz_04.txt'
MODEL_PATH = Path('D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer/checkpoints/best.pt')
NORM_PATH = Path('D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer/norm_stats.json')
OUTPUT_DIR = Path('D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer')

print("="*60)
print("40Hz 模型结果可视化")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 加载数据
print("\n加载数据...")
with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
    seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)

raw_imp = np.loadtxt(IMPEDANCE_PATH, usecols=4, skiprows=1).astype(np.float32)
n_traces = seismic.shape[0]
n_samples = len(raw_imp) // n_traces
impedance = raw_imp.reshape(n_traces, n_samples)

print(f"地震数据: {seismic.shape}")
print(f"阻抗数据: {impedance.shape}")

# 加载归一化参数
with open(NORM_PATH, 'r') as f:
    norm_stats = json.load(f)

seis_norm = (seismic - norm_stats['seis_mean']) / norm_stats['seis_std']

# 从训练脚本导入模型定义
print("\n加载模型...")
import sys
sys.path.insert(0, 'D:/SEISMIC_CODING/new')

# 读取训练脚本获取模型类
script_path = 'D:/SEISMIC_CODING/new/train_40Hz_thinlayer.py'
with open(script_path, 'r', encoding='utf-8') as f:
    script = f.read()

# 找到模型定义部分并执行
model_end = script.find('def main():')
if model_end > 0:
    exec(script[:model_end], globals())

# 加载模型
model = ThinLayerNetV2(1, 1).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

best_epoch = ckpt.get('epoch', 'N/A')
val_metrics = ckpt.get('val_metrics', {})
print(f"模型加载成功 (Epoch {best_epoch})")
print(f"验证集 PCC: {val_metrics.get('pcc', 'N/A'):.4f}")

# 推理所有道
print("\n推理全部数据...")
all_pred = []
with torch.no_grad():
    for i in range(n_traces):
        x = torch.from_numpy(seis_norm[i:i+1]).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x)
        all_pred.append(pred.cpu().numpy().squeeze())

pred_full = np.array(all_pred)
pred_full_denorm = pred_full * norm_stats['imp_std'] + norm_stats['imp_mean']

# 计算全局指标
pcc_full, _ = pearsonr(pred_full_denorm.flatten(), impedance.flatten())
ss_res = np.sum((impedance - pred_full_denorm) ** 2)
ss_tot = np.sum((impedance - np.mean(impedance)) ** 2)
r2_full = 1 - ss_res / ss_tot

print(f"\n全数据集指标:")
print(f"  PCC: {pcc_full:.4f}")
print(f"  R²:  {r2_full:.4f}")

# 坐标参数
dt_ms = 0.01
total_time = n_samples * dt_ms
shot_per_trace = 20

# ===== 剖面对比图 =====
print("\n生成可视化...")
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
extent = [0, n_traces * shot_per_trace, total_time, 0]

vmin, vmax = impedance.min(), impedance.max()
im0 = axes[0].imshow(impedance.T, aspect='auto', cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
axes[0].set_title('True Impedance', fontsize=14)
axes[0].set_xlabel('Shot Number', fontsize=12)
axes[0].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im0, ax=axes[0], label='Impedance')

im1 = axes[1].imshow(pred_full_denorm.T, aspect='auto', cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
axes[1].set_title('Predicted Impedance', fontsize=14)
axes[1].set_xlabel('Shot Number', fontsize=12)
axes[1].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im1, ax=axes[1], label='Impedance')

diff = pred_full_denorm - impedance
diff_max = np.percentile(np.abs(diff), 99)
im2 = axes[2].imshow(diff.T, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-diff_max, vmax=diff_max)
axes[2].set_title('Difference (Pred - True)', fontsize=14)
axes[2].set_xlabel('Shot Number', fontsize=12)
axes[2].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im2, ax=axes[2], label='Difference')

plt.suptitle(f'40Hz Model: PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'section_comparison_40Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: section_comparison_40Hz.png")

# ===== 薄层区域放大图 =====
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
t_start, t_end = 30, 60
sample_start = int(t_start / dt_ms)
sample_end = int(t_end / dt_ms)
extent_zoom = [0, n_traces * shot_per_trace, t_end, t_start]

imp_zoom = impedance[:, sample_start:sample_end]
pred_zoom = pred_full_denorm[:, sample_start:sample_end]
vmin_z, vmax_z = imp_zoom.min(), imp_zoom.max()

im0 = axes[0].imshow(imp_zoom.T, aspect='auto', cmap='seismic', extent=extent_zoom, vmin=vmin_z, vmax=vmax_z)
axes[0].set_title('True Impedance (30-60 ms)', fontsize=14)
axes[0].set_xlabel('Shot Number', fontsize=12)
axes[0].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pred_zoom.T, aspect='auto', cmap='seismic', extent=extent_zoom, vmin=vmin_z, vmax=vmax_z)
axes[1].set_title('Predicted Impedance (30-60 ms)', fontsize=14)
axes[1].set_xlabel('Shot Number', fontsize=12)
axes[1].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im1, ax=axes[1])

diff_zoom = pred_zoom - imp_zoom
diff_max_z = np.percentile(np.abs(diff_zoom), 99)
im2 = axes[2].imshow(diff_zoom.T, aspect='auto', cmap='RdBu_r', extent=extent_zoom, vmin=-diff_max_z, vmax=diff_max_z)
axes[2].set_title('Difference (30-60 ms)', fontsize=14)
axes[2].set_xlabel('Shot Number', fontsize=12)
axes[2].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im2, ax=axes[2])

plt.suptitle('40Hz Model - Thin Layer Zone', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'thin_layer_zone_40Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: thin_layer_zone_40Hz.png")

# ===== 道对比图 =====
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
test_traces = [10, 50, 90]
time_axis = np.arange(n_samples) * dt_ms

for idx, trace_idx in enumerate(test_traces):
    ax = axes[0, idx]
    ax.plot(impedance[trace_idx], time_axis, 'b-', linewidth=1.5, label='True')
    ax.plot(pred_full_denorm[trace_idx], time_axis, 'r--', linewidth=1.5, label='Predicted')
    ax.set_xlabel('Impedance', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title(f'Trace {trace_idx} (Shot {trace_idx*shot_per_trace})', fontsize=12)
    ax.invert_yaxis()
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    trace_pcc, _ = pearsonr(impedance[trace_idx], pred_full_denorm[trace_idx])
    ax.text(0.05, 0.95, f'PCC={trace_pcc:.4f}', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax2 = axes[1, idx]
    ax2.plot(imp_zoom[trace_idx], time_axis[sample_start:sample_end], 'b-', linewidth=1.5, label='True')
    ax2.plot(pred_zoom[trace_idx], time_axis[sample_start:sample_end], 'r--', linewidth=1.5, label='Predicted')
    ax2.set_xlabel('Impedance', fontsize=11)
    ax2.set_ylabel('Time (ms)', fontsize=11)
    ax2.set_title(f'Trace {trace_idx} (30-60 ms)', fontsize=12)
    ax2.invert_yaxis()
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

plt.suptitle(f'40Hz Model - Trace Comparison\nOverall PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trace_comparison_40Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: trace_comparison_40Hz.png")

# 保存完整指标
full_metrics = {
    'full_pcc': float(pcc_full),
    'full_r2': float(r2_full),
    'best_epoch': int(best_epoch) if isinstance(best_epoch, (int, np.integer)) else best_epoch,
    'val_pcc': float(val_metrics.get('pcc', 0))
}
with open(OUTPUT_DIR / 'full_metrics.json', 'w') as f:
    json.dump(full_metrics, f, indent=2)

print("\n" + "="*60)
print("40Hz 可视化完成!")
print("="*60)
print(f"\n输出目录: {OUTPUT_DIR}")
print(f"- section_comparison_40Hz.png")
print(f"- thin_layer_zone_40Hz.png")  
print(f"- trace_comparison_40Hz.png")
print(f"- full_metrics.json")
