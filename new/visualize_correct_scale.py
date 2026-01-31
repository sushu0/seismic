"""
生成与真实地质模型坐标系完全一致的可视化
30Hz数据训练结果
"""
import numpy as np
import matplotlib.pyplot as plt
import segyio
import json
import torch
from pathlib import Path
from scipy.stats import pearsonr

plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14

# ===== 配置 =====
result_dir = Path('results/01_30Hz_verified')
seismic_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'
impedance_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt'

# 关键参数 - 与真实模型图一致
DT = 0.01  # 采样间隔 ms (10001个采样点 → 100ms)
SHOT_SCALE = 20  # 每道对应20个shot (100道 → 2000 shot)

print("=" * 60)
print("30Hz 训练结果可视化 (与真实模型同坐标系)")
print("=" * 60)

# 1. 加载数据
print("\n[1/5] 加载数据...")
with segyio.open(seismic_path, 'r', ignore_geometry=True, strict=False) as f:
    seismic = np.stack([f.trace[i] for i in range(f.tracecount)]).astype(np.float32)

imp_raw = np.loadtxt(impedance_path, usecols=4, skiprows=1)
impedance = imp_raw.reshape(100, 10001).astype(np.float32)

print(f"  地震数据: {seismic.shape}")
print(f"  阻抗数据: {impedance.shape}")
print(f"  时间范围: 0-{10001 * DT:.1f} ms")
print(f"  Shot范围: 0-{100 * SHOT_SCALE}")

# 2. 加载归一化参数
print("\n[2/5] 加载归一化参数...")
with open(result_dir / 'norm_stats.json') as f:
    norm = json.load(f)
print(f"  imp_mean: {norm['imp_mean']:.2e}")
print(f"  imp_std: {norm['imp_std']:.2e}")

# 3. 加载模型
print("\n[3/5] 加载模型...")
from train_30Hz_from_20Hz_script import ThinLayerNetV2, highpass_filter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
ckpt = torch.load(result_dir / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"  设备: {device}")

# 4. 全剖面预测
print("\n[4/5] 生成全剖面预测...")
seismic_hf = highpass_filter(seismic, cutoff=12, fs=1000)

predictions = []
for idx in range(100):
    seis = seismic[idx]
    seis_hf = seismic_hf[idx]
    
    seis_norm = (seis - norm['seis_mean']) / (norm['seis_std'] + 1e-6)
    seis_hf_norm = seis_hf / (np.std(seis_hf) + 1e-6)
    
    x = np.stack([seis_norm, seis_hf_norm], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(x)
    
    pred_denorm = pred.cpu().numpy().squeeze() * norm['imp_std'] + norm['imp_mean']
    predictions.append(pred_denorm)

pred_all = np.array(predictions)
print(f"  预测完成: {pred_all.shape}")
print(f"  预测范围: [{pred_all.min():.2e}, {pred_all.max():.2e}]")
print(f"  真实范围: [{impedance.min():.2e}, {impedance.max():.2e}]")

# 5. 计算指标
print("\n[5/5] 计算指标...")
pred_flat = pred_all.flatten()
true_flat = impedance.flatten()
pcc, _ = pearsonr(pred_flat, true_flat)
ss_res = np.sum((pred_flat - true_flat)**2)
ss_tot = np.sum((true_flat - true_flat.mean())**2)
r2 = 1 - ss_res / ss_tot
mae = np.mean(np.abs(pred_flat - true_flat))
rmse = np.sqrt(np.mean((pred_flat - true_flat)**2))

print(f"  PCC:  {pcc:.4f}")
print(f"  R²:   {r2:.4f}")
print(f"  MAE:  {mae:.2e}")
print(f"  RMSE: {rmse:.2e}")

# ===== 可视化 =====
figures_dir = result_dir / 'figures'
figures_dir.mkdir(exist_ok=True)

# 颜色范围 - 与真实模型一致
vmin, vmax = 6.5e6, 1.0e7

# 图1: 全剖面对比 (与真实模型相同格式)
print("\n生成全剖面对比图...")
fig, axes = plt.subplots(1, 3, figsize=(20, 8))

# 真实阻抗
im0 = axes[0].imshow(impedance.T, aspect='auto', cmap='jet',
                     vmin=vmin, vmax=vmax,
                     extent=[0, 100*SHOT_SCALE, 100, 0])
axes[0].set_xlabel('Shot Number')
axes[0].set_ylabel('Time (ms)')
axes[0].set_title('True Impedance Model')
cbar0 = plt.colorbar(im0, ax=axes[0], format='%.1e')
cbar0.set_label('Impedance (m/s·g/cm³)')

# 预测阻抗
im1 = axes[1].imshow(pred_all.T, aspect='auto', cmap='jet',
                     vmin=vmin, vmax=vmax,
                     extent=[0, 100*SHOT_SCALE, 100, 0])
axes[1].set_xlabel('Shot Number')
axes[1].set_ylabel('Time (ms)')
axes[1].set_title(f'Predicted Impedance (PCC={pcc:.3f}, R²={r2:.3f})')
cbar1 = plt.colorbar(im1, ax=axes[1], format='%.1e')
cbar1.set_label('Impedance (m/s·g/cm³)')

# 误差
error = pred_all - impedance
err_max = np.percentile(np.abs(error), 99)
im2 = axes[2].imshow(error.T, aspect='auto', cmap='RdBu_r',
                     vmin=-err_max, vmax=err_max,
                     extent=[0, 100*SHOT_SCALE, 100, 0])
axes[2].set_xlabel('Shot Number')
axes[2].set_ylabel('Time (ms)')
axes[2].set_title('Error (Pred - True)')
cbar2 = plt.colorbar(im2, ax=axes[2], format='%.1e')
cbar2.set_label('Error')

plt.tight_layout()
plt.savefig(figures_dir / 'section_comparison_ms.png', dpi=150, bbox_inches='tight')
print(f"  已保存: section_comparison_ms.png")

# 图2: 薄层区域放大 (45-70 ms)
print("生成薄层区域放大图...")
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 10))

t_start, t_end = 4500, 7000  # 对应 45-70 ms

# 真实薄层
im0 = axes2[0].imshow(impedance[:, t_start:t_end].T, aspect='auto', cmap='jet',
                      vmin=vmin, vmax=vmax,
                      extent=[0, 100*SHOT_SCALE, 70, 45])
axes2[0].set_xlabel('Shot Number')
axes2[0].set_ylabel('Time (ms)')
axes2[0].set_title('True Impedance - Thin Layer Zone')
plt.colorbar(im0, ax=axes2[0], format='%.1e')

# 预测薄层
im1 = axes2[1].imshow(pred_all[:, t_start:t_end].T, aspect='auto', cmap='jet',
                      vmin=vmin, vmax=vmax,
                      extent=[0, 100*SHOT_SCALE, 70, 45])
axes2[1].set_xlabel('Shot Number')
axes2[1].set_ylabel('Time (ms)')
axes2[1].set_title('Predicted Impedance - Thin Layer Zone')
plt.colorbar(im1, ax=axes2[1], format='%.1e')

# 误差
im2 = axes2[2].imshow(error[:, t_start:t_end].T, aspect='auto', cmap='RdBu_r',
                      vmin=-err_max, vmax=err_max,
                      extent=[0, 100*SHOT_SCALE, 70, 45])
axes2[2].set_xlabel('Shot Number')
axes2[2].set_ylabel('Time (ms)')
axes2[2].set_title('Error - Thin Layer Zone')
plt.colorbar(im2, ax=axes2[2], format='%.1e')

plt.tight_layout()
plt.savefig(figures_dir / 'thin_layer_zone_ms.png', dpi=150, bbox_inches='tight')
print(f"  已保存: thin_layer_zone_ms.png")

# 图3: 多道对比
print("生成单道对比图...")
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))

# 选择不同位置的道进行对比
trace_indices = [10, 30, 50, 70, 90]
time_axis = np.arange(10001) * DT  # Time in ms

for i, trace_idx in enumerate(trace_indices[:3]):
    ax = axes3[0, i]
    ax.plot(time_axis, impedance[trace_idx], 'b-', label='True', linewidth=1.5)
    ax.plot(time_axis, pred_all[trace_idx], 'r--', label='Predicted', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Impedance')
    ax.set_title(f'Trace {trace_idx} (Shot {trace_idx * SHOT_SCALE})')
    ax.legend()
    ax.set_xlim(40, 80)
    ax.grid(True, alpha=0.3)

for i, trace_idx in enumerate(trace_indices[3:]):
    ax = axes3[1, i]
    ax.plot(time_axis, impedance[trace_idx], 'b-', label='True', linewidth=1.5)
    ax.plot(time_axis, pred_all[trace_idx], 'r--', label='Predicted', linewidth=1.5)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Impedance')
    ax.set_title(f'Trace {trace_idx} (Shot {trace_idx * SHOT_SCALE})')
    ax.legend()
    ax.set_xlim(40, 80)
    ax.grid(True, alpha=0.3)

# 第三个子图显示整体统计
ax_stats = axes3[1, 2]
ax_stats.axis('off')
stats_text = f"""
整体指标:
  PCC:  {pcc:.4f}
  R²:   {r2:.4f}
  MAE:  {mae:.2e}
  RMSE: {rmse:.2e}

数据信息:
  道数: 100 (Shot: 0-2000)
  采样: 10001 (Time: 0-100ms)
  
阻抗范围:
  真实: [{impedance.min():.2e}, {impedance.max():.2e}]
  预测: [{pred_all.min():.2e}, {pred_all.max():.2e}]
"""
ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes, fontsize=12,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(figures_dir / 'trace_comparison_ms.png', dpi=150, bbox_inches='tight')
print(f"  已保存: trace_comparison_ms.png")

# 保存指标
metrics = {
    'pcc': float(pcc),
    'r2': float(r2),
    'mae': float(mae),
    'rmse': float(rmse),
    'pred_min': float(pred_all.min()),
    'pred_max': float(pred_all.max()),
    'true_min': float(impedance.min()),
    'true_max': float(impedance.max())
}
with open(figures_dir / 'metrics_fulldata.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  已保存: metrics_fulldata.json")

print("\n" + "=" * 60)
print("完成!")
print("=" * 60)

plt.show()
