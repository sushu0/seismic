"""
生成与真实地质模型完全相同格式的对比可视化
"""
import numpy as np
import matplotlib.pyplot as plt
import segyio
import json
import torch
from pathlib import Path

plt.rcParams['font.size'] = 12

# 加载数据
print("加载数据...")
with segyio.open(r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy', 'r', ignore_geometry=True, strict=False) as f:
    seismic = np.stack([f.trace[i] for i in range(f.tracecount)]).astype(np.float32)

imp_raw = np.loadtxt(r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt', usecols=4, skiprows=1)
impedance = imp_raw.reshape(100, 10001).astype(np.float32)

# 参数 - 与真实模型图一致
DT = 0.01  # ms
SHOT_SCALE = 20  # 每道对应20个shot

# 加载归一化参数和模型
result_dir = Path('results/01_30Hz_verified')
with open(result_dir / 'norm_stats.json') as f:
    norm = json.load(f)

from train_30Hz_from_20Hz_script import ThinLayerNetV2, highpass_filter

model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1)
ckpt = torch.load(result_dir / 'checkpoints' / 'best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# 对所有100道进行预测
seismic_hf = highpass_filter(seismic, cutoff=12, fs=1000)

predictions = []
print("生成预测...")
for idx in range(100):
    seis = seismic[idx]
    seis_hf = seismic_hf[idx]
    
    seis_norm = (seis - norm['seis_mean']) / (norm['seis_std'] + 1e-6)
    seis_hf_norm = seis_hf / (np.std(seis_hf) + 1e-6)
    
    x = np.stack([seis_norm, seis_hf_norm], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0)
    
    with torch.no_grad():
        pred = model(x)
    
    pred_denorm = pred.numpy().squeeze() * norm['imp_std'] + norm['imp_mean']
    predictions.append(pred_denorm)

pred_arr = np.array(predictions)

print(f"预测完成: {pred_arr.shape}")
print(f"真实阻抗范围: [{impedance.min():.2e}, {impedance.max():.2e}]")
print(f"预测阻抗范围: [{pred_arr.min():.2e}, {pred_arr.max():.2e}]")

# ============ 生成与真实模型相同格式的图 ============
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 颜色范围 - 与真实模型一致
vmin, vmax = 6.5e6, 1.0e7

# 1. 真实阻抗模型
ax1 = axes[0]
im1 = ax1.imshow(impedance.T, aspect='auto', cmap='jet',
                 vmin=vmin, vmax=vmax,
                 extent=[0, 100*SHOT_SCALE, 100, 0])
ax1.set_xlabel('Shot Number')
ax1.set_ylabel('Time (ms)')
ax1.set_title('True Impedance Model')
cbar1 = plt.colorbar(im1, ax=ax1, format='%.2e')
cbar1.set_label('Impedance (m/s·g/cm³)')

# 2. 预测阻抗模型
ax2 = axes[1]
im2 = ax2.imshow(pred_arr.T, aspect='auto', cmap='jet',
                 vmin=vmin, vmax=vmax,
                 extent=[0, 100*SHOT_SCALE, 100, 0])
ax2.set_xlabel('Shot Number')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Predicted Impedance Model')
cbar2 = plt.colorbar(im2, ax=ax2, format='%.2e')
cbar2.set_label('Impedance (m/s·g/cm³)')

# 3. 误差图
ax3 = axes[2]
error = pred_arr - impedance
error_max = np.percentile(np.abs(error), 99)
im3 = ax3.imshow(error.T, aspect='auto', cmap='RdBu_r',
                 vmin=-error_max, vmax=error_max,
                 extent=[0, 100*SHOT_SCALE, 100, 0])
ax3.set_xlabel('Shot Number')
ax3.set_ylabel('Time (ms)')
ax3.set_title('Error (Predicted - True)')
cbar3 = plt.colorbar(im3, ax=ax3, format='%.2e')
cbar3.set_label('Impedance Error')

plt.tight_layout()
plt.savefig(result_dir / 'figures' / 'full_model_comparison.png', dpi=150)
print(f"\n已保存: {result_dir / 'figures' / 'full_model_comparison.png'}")

# ============ 只显示薄层区域（45-70ms） ============
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 8))

# 时间范围对应的采样点
t_start, t_end = 4500, 7000  # 45-70ms

# 1. 真实薄层区域
ax1 = axes2[0]
im1 = ax1.imshow(impedance[:, t_start:t_end].T, aspect='auto', cmap='jet',
                 vmin=vmin, vmax=vmax,
                 extent=[0, 100*SHOT_SCALE, 70, 45])
ax1.set_xlabel('Shot Number')
ax1.set_ylabel('Time (ms)')
ax1.set_title('True Impedance - Thin Layer Zone')
cbar1 = plt.colorbar(im1, ax=ax1, format='%.2e')
cbar1.set_label('Impedance (m/s·g/cm³)')

# 2. 预测薄层区域
ax2 = axes2[1]
im2 = ax2.imshow(pred_arr[:, t_start:t_end].T, aspect='auto', cmap='jet',
                 vmin=vmin, vmax=vmax,
                 extent=[0, 100*SHOT_SCALE, 70, 45])
ax2.set_xlabel('Shot Number')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Predicted Impedance - Thin Layer Zone')
cbar2 = plt.colorbar(im2, ax=ax2, format='%.2e')
cbar2.set_label('Impedance (m/s·g/cm³)')

# 3. 薄层区域误差
ax3 = axes2[2]
error_thin = error[:, t_start:t_end]
im3 = ax3.imshow(error_thin.T, aspect='auto', cmap='RdBu_r',
                 vmin=-error_max, vmax=error_max,
                 extent=[0, 100*SHOT_SCALE, 70, 45])
ax3.set_xlabel('Shot Number')
ax3.set_ylabel('Time (ms)')
ax3.set_title('Error - Thin Layer Zone')
cbar3 = plt.colorbar(im3, ax=ax3, format='%.2e')
cbar3.set_label('Impedance Error')

plt.tight_layout()
plt.savefig(result_dir / 'figures' / 'thin_layer_comparison.png', dpi=150)
print(f"已保存: {result_dir / 'figures' / 'thin_layer_comparison.png'}")

# ============ 计算全数据集指标 ============
from scipy.stats import pearsonr

pred_flat = pred_arr.flatten()
true_flat = impedance.flatten()
pcc, _ = pearsonr(pred_flat, true_flat)
ss_res = np.sum((pred_flat - true_flat)**2)
ss_tot = np.sum((true_flat - true_flat.mean())**2)
r2 = 1 - ss_res/ss_tot
mae = np.mean(np.abs(pred_flat - true_flat))

print("\n" + "="*60)
print("全数据集（100道）指标:")
print("="*60)
print(f"PCC: {pcc:.4f}")
print(f"R²:  {r2:.4f}")
print(f"MAE: {mae:.2e}")

# 薄层区域指标
threshold = 7.5e6
thin_mask = impedance > threshold
thin_pred = pred_arr[thin_mask]
thin_true = impedance[thin_mask]
thin_mae = np.mean(np.abs(thin_pred - thin_true))

print(f"\n薄层区域 MAE: {thin_mae:.2e}")
print(f"薄层区域相对误差: {thin_mae / np.mean(thin_true) * 100:.2f}%")

plt.show()
