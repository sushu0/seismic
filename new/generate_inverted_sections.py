"""
绘制反演后的合成地震剖面图
"""
import numpy as np
import torch
from pathlib import Path
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seisinv.models.baselines import UNet1D
from seisinv.losses.physics import ForwardModel
from seisinv.utils.wavelet import ricker

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("="*70)
print("生成反演后的合成地震剖面图")
print("="*70)

# 加载原始数据
print("\n1. 加载原始数据...")
data = np.load('data.npy', allow_pickle=True).item()
seismic_obs = data['seismic'][:, 0, :]  # (2721, 470) - 观测地震
impedance_true = data['acoustic_impedance'][:, 0, ::4][:, :470]  # (2721, 470) - 真实阻抗（下采样）

print(f"   观测地震: {seismic_obs.shape}")
print(f"   真实阻抗: {impedance_true.shape}")

# 加载模型配置
exp_name = 'real_unet1d_optimized'
config_path = 'configs/exp_real_data.yaml'

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载归一化统计量
import json
norm_path = Path(f'results/{exp_name}/norm_stats.json')
with open(norm_path, 'r') as f:
    norm_stats = json.load(f)

# 加载训练好的模型
print("\n2. 加载训练好的模型...")
model = UNet1D(
    in_ch=1,
    out_ch=1,
    base=cfg['model']['base'],
    depth=cfg['model']['depth']
).to(device)

ckpt_path = Path(f'results/{exp_name}/checkpoints/best.pt')
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# 创建物理正演模型
phys_cfg = cfg['physics']
wavelet = ricker(
    f0=phys_cfg['wavelet_f0_hz'],
    length=phys_cfg['wavelet_length_s'],
    dt=phys_cfg['dt_s']
)
wavelet_tensor = torch.from_numpy(wavelet).float().to(device)
fm = ForwardModel(wavelet=wavelet_tensor, eps=float(phys_cfg['eps']))

# 对全部数据进行阻抗反演
print("\n3. 执行阻抗反演...")
seismic_norm = (seismic_obs - norm_stats['seis_mean']) / norm_stats['seis_std']

batch_size = 128
n_total = len(seismic_norm)
impedance_pred_norm = []

with torch.no_grad():
    for i in range(0, n_total, batch_size):
        batch_seis = seismic_norm[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_seis[:, None, :].astype(np.float32)).to(device)
        
        pred = model(batch_tensor)
        if isinstance(pred, tuple):
            pred, _ = pred
        
        impedance_pred_norm.append(pred.cpu().numpy()[:, 0, :])

impedance_pred_norm = np.concatenate(impedance_pred_norm, axis=0)
impedance_pred = impedance_pred_norm * norm_stats['imp_std'] + norm_stats['imp_mean']

print(f"   反演完成: {impedance_pred.shape}")

# 从预测阻抗生成合成地震
print("\n4. 生成合成地震剖面（物理正演）...")
seismic_syn_norm = []

with torch.no_grad():
    for i in range(0, n_total, batch_size):
        batch_imp = impedance_pred_norm[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_imp[:, None, :].astype(np.float32)).to(device)
        
        seis_syn = fm(batch_tensor)
        seismic_syn_norm.append(seis_syn.cpu().numpy()[:, 0, :])

seismic_syn_norm = np.concatenate(seismic_syn_norm, axis=0)
seismic_syn = seismic_syn_norm * norm_stats['seis_std'] + norm_stats['seis_mean']

print(f"   合成地震: {seismic_syn.shape}")
print(f"   合成地震范围: [{seismic_syn.min():.3f}, {seismic_syn.max():.3f}]")

# 创建输出目录
out_dir = Path(f'results/{exp_name}')

print("\n5. 绘制剖面图...")

# ========== 图1: 观测地震剖面 ==========
print("   - 观测地震剖面...")
fig, ax = plt.subplots(figsize=(16, 8))

im_T = seismic_obs.T  # (470, 2721)
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

levels = 30
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='seismic', extend='both')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Amplitude', rotation=90, labelpad=15, fontsize=12)

ax.set_xlabel('Trace number', fontsize=13)
ax.set_ylabel('Time sample', fontsize=13)
ax.set_title('Observed Seismic Section', fontsize=15, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'observed_seismic_section.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"      ✓ observed_seismic_section.png")

# ========== 图2: 反演后的合成地震剖面 ==========
print("   - 反演后的合成地震剖面...")
fig, ax = plt.subplots(figsize=(16, 8))

im_T = seismic_syn.T  # (470, 2721)
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

levels = 30
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='seismic', extend='both')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Amplitude', rotation=90, labelpad=15, fontsize=12)

ax.set_xlabel('Trace number', fontsize=13)
ax.set_ylabel('Time sample', fontsize=13)
ax.set_title('Synthetic Seismic Section (Inverted)', fontsize=15, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'synthetic_seismic_section_inverted.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"      ✓ synthetic_seismic_section_inverted.png")

# ========== 图3: 预测阻抗剖面 ==========
print("   - 预测阻抗剖面...")
fig, ax = plt.subplots(figsize=(16, 8))

im_T = impedance_pred.T  # (470, 2721)
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

levels = 30
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='jet', extend='both')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Impedance (m/s·g/cm³)', rotation=90, labelpad=15, fontsize=12)

ax.set_xlabel('Trace number', fontsize=13)
ax.set_ylabel('Time sample', fontsize=13)
ax.set_title('Predicted Impedance Section (Inverted)', fontsize=15, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'predicted_impedance_section_inverted.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"      ✓ predicted_impedance_section_inverted.png")

# ========== 图4: 真实阻抗剖面（对比用） ==========
print("   - 真实阻抗剖面...")
fig, ax = plt.subplots(figsize=(16, 8))

im_T = impedance_true.T  # (470, 2721)
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

levels = 30
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='jet', extend='both')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Impedance (m/s·g/cm³)', rotation=90, labelpad=15, fontsize=12)

ax.set_xlabel('Trace number', fontsize=13)
ax.set_ylabel('Time sample', fontsize=13)
ax.set_title('True Impedance Section', fontsize=15, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'true_impedance_section.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"      ✓ true_impedance_section.png")

# ========== 图5: 差异图（观测 - 合成） ==========
print("   - 地震差异图...")
diff_seismic = seismic_obs - seismic_syn

fig, ax = plt.subplots(figsize=(16, 8))

im_T = diff_seismic.T
n_samples, n_traces = im_T.shape
X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))

# 使用对称的colormap
vmax = max(abs(diff_seismic.min()), abs(diff_seismic.max()))
levels = np.linspace(-vmax, vmax, 30)
cf = ax.contourf(X, Y, im_T, levels=levels, cmap='seismic', extend='both')

cbar = plt.colorbar(cf, ax=ax, pad=0.02)
cbar.set_label('Amplitude Difference', rotation=90, labelpad=15, fontsize=12)

ax.set_xlabel('Trace number', fontsize=13)
ax.set_ylabel('Time sample', fontsize=13)
ax.set_title('Residual (Observed - Synthetic)', fontsize=15, fontweight='bold')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(out_dir / 'residual_seismic_section.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"      ✓ residual_seismic_section.png")

# 计算整体拟合度
from seisinv.utils.metrics import summarize_metrics
metrics_imp = summarize_metrics(impedance_true, impedance_pred)
metrics_seis = summarize_metrics(seismic_obs, seismic_syn)

print("\n" + "="*70)
print("完成！")
print("="*70)
print(f"\n整体反演精度:")
print(f"  阻抗: PCC={metrics_imp['PCC']:.4f}, R²={metrics_imp['R2']:.4f}")
print(f"  地震: PCC={metrics_seis['PCC']:.4f}, R²={metrics_seis['R2']:.4f}")
print(f"\n输出目录: {out_dir}")
print(f"\n生成的剖面图:")
print(f"  1. observed_seismic_section.png - 观测地震剖面")
print(f"  2. synthetic_seismic_section_inverted.png - 反演后的合成地震剖面")
print(f"  3. predicted_impedance_section_inverted.png - 预测阻抗剖面")
print(f"  4. true_impedance_section.png - 真实阻抗剖面")
print(f"  5. residual_seismic_section.png - 残差剖面")
print("="*70)
