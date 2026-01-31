"""
修正版：使用完整数据集和正确道号绘制真实阻抗和预测对比
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

# 加载原始完整数据
print("加载完整data.npy...")
data = np.load('data.npy', allow_pickle=True).item()
seismic_full = data['seismic'][:, 0, :]  # (2721, 470)
impedance_full = data['acoustic_impedance'][:, 0, :]  # (2721, 1880)

print(f"完整数据: seismic={seismic_full.shape}, impedance={impedance_full.shape}")

# 加载训练好的模型
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

print(f"归一化统计: seis_mean={norm_stats['seis_mean']:.4f}, imp_mean={norm_stats['imp_mean']:.0f}")

# 构建并加载模型
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

# 创建前向模型
phys_cfg = cfg['physics']
wavelet = ricker(
    f0=phys_cfg['wavelet_f0_hz'],
    length=phys_cfg['wavelet_length_s'],
    dt=phys_cfg['dt_s']
)
wavelet_tensor = torch.from_numpy(wavelet).float().to(device)
fm = ForwardModel(wavelet=wavelet_tensor, eps=float(phys_cfg['eps']))

# 对完整数据集进行预测
print("对完整数据集进行预测...")

# 归一化地震数据
seismic_norm = (seismic_full - norm_stats['seis_mean']) / norm_stats['seis_std']

# 批量预测（避免内存溢出）
batch_size = 64
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

impedance_pred_norm = np.concatenate(impedance_pred_norm, axis=0)  # (2721, 470)

# 反归一化预测结果
impedance_pred = impedance_pred_norm * norm_stats['imp_std'] + norm_stats['imp_mean']

print(f"预测完成: impedance_pred={impedance_pred.shape}")
print(f"预测范围: [{impedance_pred.min():.0f}, {impedance_pred.max():.0f}]")

# 对阻抗下采样以匹配预测（用于对比）
impedance_full_downsampled = impedance_full[:, ::4][:, :470]

out_dir = Path(f'results/{exp_name}')

# 指定的四个道号
trace_ids = [299, 599, 1699, 2299]

print(f"\n生成四道对比图 (道号: {trace_ids})...")

# ============ 图1: 真实阻抗 vs 预测阻抗（下采样版本，470点）============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, trace_id in enumerate(trace_ids):
    ax = axes[i]
    t = np.arange(len(impedance_pred[trace_id]))
    
    # 真实阻抗（下采样到470点）
    ax.plot(t, impedance_full_downsampled[trace_id], 'r-', linewidth=1.5, label='真实')
    # 预测阻抗（470点）
    ax.plot(t, impedance_pred[trace_id], 'b-', linewidth=1.5, label='预测')
    
    ax.set_xlabel('t', fontsize=11)
    ax.set_ylabel('Impedance(m/s*g/cm^3)', fontsize=11)
    ax.set_title(f'No. {trace_id}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 470])
    
    if i == 0:
        ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig(out_dir / 'four_trace_impedance_comparison_corrected.png', dpi=200)
plt.close()
print(f"✓ 保存: four_trace_impedance_comparison_corrected.png")

# ============ 图2: 真实原始分辨率阻抗曲线（1880点）============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, trace_id in enumerate(trace_ids):
    ax = axes[i]
    t = np.arange(len(impedance_full[trace_id]))
    
    # 原始高分辨率阻抗（1880点）
    ax.plot(t, impedance_full[trace_id], 'r-', linewidth=1.2)
    
    ax.set_xlabel('Depth sample', fontsize=11)
    ax.set_ylabel('Impedance (m/s * g cm^-3)', fontsize=11)
    ax.set_title(f'Impedance Trace No. {trace_id}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1880])

plt.tight_layout()
plt.savefig(out_dir / 'four_trace_true_impedance_highres.png', dpi=200)
plt.close()
print(f"✓ 保存: four_trace_true_impedance_highres.png")

# ============ 图3: 地震记录对比（观测 vs 合成）============
print("生成地震对比图...")

# 从预测阻抗生成合成地震
seismic_pred_norm = []
with torch.no_grad():
    for i in range(0, n_total, batch_size):
        batch_imp = impedance_pred_norm[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_imp[:, None, :].astype(np.float32)).to(device)
        
        seis_syn = fm(batch_tensor)
        seismic_pred_norm.append(seis_syn.cpu().numpy()[:, 0, :])

seismic_pred_norm = np.concatenate(seismic_pred_norm, axis=0)
seismic_pred = seismic_pred_norm * norm_stats['seis_std'] + norm_stats['seis_mean']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, trace_id in enumerate(trace_ids):
    ax = axes[i]
    t = np.arange(len(seismic_full[trace_id]))
    
    # 观测地震
    ax.plot(t, seismic_full[trace_id], 'r-', linewidth=1.5, label='观测')
    # 合成地震
    ax.plot(t, seismic_pred[trace_id], 'b-', linewidth=1.5, label='合成')
    
    ax.set_xlabel('t', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title(f'No. {trace_id}', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    if i == 0:
        ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig(out_dir / 'four_trace_seismic_comparison_corrected.png', dpi=200)
plt.close()
print(f"✓ 保存: four_trace_seismic_comparison_corrected.png")

# ============ 计算这四道的预测精度 ============
print(f"\n计算指定四道的预测精度...")
from seisinv.utils.metrics import summarize_metrics

for trace_id in trace_ids:
    y_true = impedance_full_downsampled[trace_id:trace_id+1]
    y_pred = impedance_pred[trace_id:trace_id+1]
    
    metrics = summarize_metrics(y_true, y_pred)
    print(f"  道号 {trace_id}: PCC={metrics['PCC']:.4f}, R²={metrics['R2']:.4f}, MSE={metrics['MSE']:.4f}")

print("\n" + "="*70)
print("图像生成完成!")
print("="*70)
print(f"1. four_trace_impedance_comparison_corrected.png - 真实vs预测阻抗(470点)")
print(f"2. four_trace_true_impedance_highres.png - 真实阻抗原始分辨率(1880点)")
print(f"3. four_trace_seismic_comparison_corrected.png - 观测vs合成地震")
print("="*70)
