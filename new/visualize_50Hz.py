# -*- coding: utf-8 -*-
"""
50Hz模型结果可视化脚本 - 双通道输入版本
"""
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import butter, filtfilt
from pathlib import Path

# 配置
SEISMIC_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_50Hz_re.sgy'
IMPEDANCE_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_50Hz_04.txt'
MODEL_PATH = Path('D:/SEISMIC_CODING/new/results/01_50Hz_thinlayer/checkpoints/best.pt')
NORM_PATH = Path('D:/SEISMIC_CODING/new/results/01_50Hz_thinlayer/norm_stats.json')
OUTPUT_DIR = Path('D:/SEISMIC_CODING/new/results/01_50Hz_thinlayer')

print("="*60)
print("50Hz 模型结果可视化")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# ==================== 模型定义 (从训练脚本复制) ====================
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.branches = nn.ModuleList()
        branch_ch = out_ch // len(dilations)
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(branch_ch),
                nn.GELU()
            ))
        self.fusion = nn.Sequential(
            nn.Conv1d(branch_ch * len(dilations), out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        branches = [branch(x) for branch in self.branches]
        out = torch.cat(branches, dim=1)
        out = self.fusion(out)
        return out + self.skip(x)


class BoundaryEnhanceModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.edge_conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        with torch.no_grad():
            edge_kernel = torch.tensor([-1.0, 2.0, -1.0]).view(1, 1, 3)
            self.edge_conv.weight.data = edge_kernel.repeat(ch, 1, 1)
        self.refine = nn.Sequential(
            nn.Conv1d(ch * 2, ch, kernel_size=1),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        edge = torch.abs(self.edge_conv(x))
        combined = torch.cat([x, edge], dim=1)
        attention = self.refine(combined)
        return x * attention + x


class ThinLayerBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_boundary=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.boundary = BoundaryEnhanceModule(out_ch) if use_boundary else nn.Identity()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.boundary(out)
        return out + self.skip(x)


class ThinLayerNetV2(nn.Module):
    """改进版ThinLayerNet - 双通道输入"""
    def __init__(self, in_ch=2, base_ch=64, out_ch=1):
        super().__init__()
        # 输入卷积 - 处理双通道(原始+高频)
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.GELU()
        )
        
        self.multi_scale = DilatedConvBlock(base_ch, base_ch, dilations=[1, 2, 4, 8])
        
        # Encoder - 减少下采样次数
        self.enc1 = ThinLayerBlock(base_ch, base_ch * 2)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = ThinLayerBlock(base_ch * 2, base_ch * 4)
        self.pool2 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DilatedConvBlock(base_ch * 4, base_ch * 8, dilations=[1, 2, 4, 8, 16]),
            ThinLayerBlock(base_ch * 8, base_ch * 8)
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec2 = ThinLayerBlock(base_ch * 8, base_ch * 4)
        
        self.up1 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1 = ThinLayerBlock(base_ch * 4, base_ch * 2)
        
        # 细化模块
        self.refine = nn.Sequential(
            ThinLayerBlock(base_ch * 2 + base_ch, base_ch * 2),
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.GELU(),
            BoundaryEnhanceModule(base_ch),
        )
        
        # 输出
        self.output = nn.Sequential(
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_ch // 2, out_ch, kernel_size=1)
        )
    
    def forward(self, x):
        x0 = self.input_conv(x)
        x0 = self.multi_scale(x0)
        
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = self.up2(b)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        if d1.shape[-1] != x0.shape[-1]:
            d1 = F.interpolate(d1, size=x0.shape[-1], mode='linear', align_corners=False)
        
        out = torch.cat([d1, x0], dim=1)
        out = self.refine(out)
        out = self.output(out)
        
        return out


def highpass_filter(data, cutoff=20, fs=1000, order=4):
    """高通滤波提取高频成分 - 50Hz数据用20Hz截止"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


# ==================== 加载数据 ====================
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

# 预处理 - 归一化和高频提取
seis_norm = (seismic - norm_stats['seis_mean']) / norm_stats['seis_std']
seismic_hf = highpass_filter(seismic, cutoff=15, fs=1000)
seis_hf_norm = seismic_hf / (np.std(seismic_hf, axis=1, keepdims=True) + 1e-6)

# ==================== 加载模型 ====================
print("\n加载模型...")
model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

best_epoch = ckpt.get('epoch', 'N/A')
val_metrics = ckpt.get('val_metrics', {})
print(f"模型加载成功 (Epoch {best_epoch})")
if val_metrics:
    print(f"验证集 PCC: {val_metrics.get('pcc', 'N/A'):.4f}")

# ==================== 推理所有道 ====================
print("\n推理全部数据...")
all_pred = []
with torch.no_grad():
    for i in range(n_traces):
        # 双通道输入: [原始地震, 高频成分]
        x = np.stack([seis_norm[i], seis_hf_norm[i]], axis=0)  # (2, n_samples)
        x_t = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, 2, n_samples)
        pred = model(x_t)
        all_pred.append(pred.cpu().numpy().squeeze())
        if (i + 1) % 20 == 0:
            print(f"  推理进度: {i+1}/{n_traces}")

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

plt.suptitle(f'50Hz Model: PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'section_comparison_50Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: section_comparison_50Hz.png")

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

plt.suptitle('50Hz Model - Thin Layer Zone', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'thin_layer_zone_50Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: thin_layer_zone_50Hz.png")

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

plt.suptitle(f'50Hz Model - Trace Comparison\nOverall PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'trace_comparison_50Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print("保存: trace_comparison_50Hz.png")

# 保存完整指标
full_metrics = {
    'full_pcc': float(pcc_full),
    'full_r2': float(r2_full),
    'best_epoch': int(best_epoch) if isinstance(best_epoch, (int, np.integer)) else best_epoch,
    'val_pcc': float(val_metrics.get('pcc', 0)) if val_metrics else 0
}
with open(OUTPUT_DIR / 'full_metrics.json', 'w') as f:
    json.dump(full_metrics, f, indent=2)

print("\n" + "="*60)
print("50Hz 可视化完成!")
print("="*60)
print(f"\n输出目录: {OUTPUT_DIR}")
