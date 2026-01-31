# -*- coding: utf-8 -*-
"""
V6模型可视化脚本 - 生成高质量对比图
"""
import os
import sys
import json
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
import pandas as pd

matplotlib.use('Agg')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2

# ==================== 模型定义 ====================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(ch, ch // r),
            nn.ReLU(),
            nn.Linear(ch // r, ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        w = self.fc(x).unsqueeze(-1)
        return x * w


class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 2, 4, 8]):
        super().__init__()
        b_ch = out_ch // len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, b_ch, 3, padding=d, dilation=d),
                nn.BatchNorm1d(b_ch),
                nn.GELU()
            ) for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(b_ch * len(dilations), out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        out = self.fuse(out)
        out = self.se(out)
        return out + self.skip(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.se(out)
        return out + self.skip(x)


class InversionNet(nn.Module):
    def __init__(self, in_ch=2, base=48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3),
            nn.BatchNorm1d(base),
            nn.GELU()
        )
        self.ms = DilatedBlock(base, base)
        
        self.e1 = ResBlock(base, base * 2)
        self.p1 = nn.MaxPool1d(2)
        self.e2 = ResBlock(base * 2, base * 4)
        self.p2 = nn.MaxPool1d(2)
        
        self.neck = nn.Sequential(
            DilatedBlock(base * 4, base * 8),
            ResBlock(base * 8, base * 8)
        )
        
        self.u2 = nn.ConvTranspose1d(base * 8, base * 4, 2, 2)
        self.d2 = ResBlock(base * 8, base * 4)
        self.u1 = nn.ConvTranspose1d(base * 4, base * 2, 2, 2)
        self.d1 = ResBlock(base * 4, base * 2)
        
        self.refine = nn.Sequential(
            ResBlock(base * 2 + base, base * 2),
            nn.Conv1d(base * 2, base, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv1d(base, 1, 1)
    
    def forward(self, x):
        x0 = self.stem(x)
        x0 = self.ms(x0)
        
        e1 = self.e1(x0)
        e2 = self.e2(self.p1(e1))
        
        b = self.neck(self.p2(e2))
        
        d2 = self.u2(b)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], 1))
        
        d1 = self.u1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], 1))
        
        if d1.shape[-1] != x0.shape[-1]:
            d1 = F.interpolate(d1, x0.shape[-1], mode='linear', align_corners=False)
        
        out = self.refine(torch.cat([d1, x0], 1))
        return self.out(out)


def highpass_filter(data, cutoff, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, data, axis=-1)


def load_data_and_model(freq):
    """加载数据和模型"""
    FREQ_CONFIGS = {
        '20Hz': {'highpass_cutoff': 8},
        '30Hz': {'highpass_cutoff': 12},
        '40Hz': {'highpass_cutoff': 15},
        '50Hz': {'highpass_cutoff': 20},
    }
    
    config = FREQ_CONFIGS[freq]
    HIGHPASS_CUTOFF = config['highpass_cutoff']
    
    SEISMIC_PATH = rf'D:\SEISMIC_CODING\zmy_data\01\data\01_{freq}_re.sgy'
    IMPEDANCE_PATH = rf'D:\SEISMIC_CODING\zmy_data\01\data\01_{freq}_04.txt'
    OUTPUT_DIR = Path(rf'D:\SEISMIC_CODING\new\results\01_{freq}_v6')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载地震数据
    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
        seismic = np.array([f.trace[i] for i in range(f.tracecount)], dtype=np.float32)
    
    # 加载波阻抗
    df = pd.read_csv(IMPEDANCE_PATH, sep=r'\s+', skiprows=1, header=None)
    impedance_raw = df.iloc[:, -1].values.astype(np.float32)
    n_traces = seismic.shape[0]
    n_samples = seismic.shape[1]
    impedance = impedance_raw.reshape(n_traces, n_samples)
    
    # 加载归一化参数
    with open(OUTPUT_DIR / 'norm_stats.json') as f:
        stats = json.load(f)
    
    sm, ss = stats['seis_mean'], stats['seis_std']
    im, ist = stats['imp_mean'], stats['imp_std']
    
    # 归一化
    seis_n = (seismic - sm) / ss
    seis_hf = highpass_filter(seismic, HIGHPASS_CUTOFF)
    seis_hf_n = (seis_hf - seis_hf.mean()) / seis_hf.std()
    
    # 加载模型
    model = InversionNet(in_ch=2, base=48).to(device)
    ckpt = torch.load(OUTPUT_DIR / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # 预测所有道
    predictions = []
    with torch.no_grad():
        for i in range(n_traces):
            x = np.stack([seis_n[i], seis_hf_n[i]], axis=0).astype(np.float32)
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            pred = model(x).cpu().numpy().flatten()
            # 反归一化
            pred_denorm = pred * ist + im
            predictions.append(pred_denorm)
    
    predictions = np.array(predictions)
    
    return seismic, impedance, predictions, stats


def plot_trace_comparison(freq, trace_idx, output_dir):
    """绘制单道对比图"""
    seismic, true_imp, pred_imp, stats = load_data_and_model(freq)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    
    time = np.arange(seismic.shape[1]) * 0.001  # 假设1ms采样
    
    # 地震道
    ax1 = axes[0]
    ax1.plot(seismic[trace_idx], time, 'b-', linewidth=0.8)
    ax1.set_ylabel('Time (s)', fontsize=14)
    ax1.set_xlabel('Amplitude', fontsize=14)
    ax1.set_title(f'Seismic Trace #{trace_idx+1}', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3)
    
    # 波阻抗对比
    ax2 = axes[1]
    ax2.plot(true_imp[trace_idx], time, 'b-', linewidth=1.2, label='True', alpha=0.8)
    ax2.plot(pred_imp[trace_idx], time, 'r--', linewidth=1.2, label='Predicted', alpha=0.8)
    ax2.set_xlabel('Impedance', fontsize=14)
    ax2.set_title(f'Impedance Comparison ({freq})', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(loc='upper right', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 误差
    ax3 = axes[2]
    error = pred_imp[trace_idx] - true_imp[trace_idx]
    ax3.fill_betweenx(time, 0, error, where=error>=0, color='red', alpha=0.5, label='Over')
    ax3.fill_betweenx(time, 0, error, where=error<0, color='blue', alpha=0.5, label='Under')
    ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
    ax3.set_xlabel('Error', fontsize=14)
    ax3.set_title('Prediction Error', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.legend(loc='upper right', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # 计算指标
    pcc, _ = pearsonr(pred_imp[trace_idx], true_imp[trace_idx])
    ss_res = np.sum((true_imp[trace_idx] - pred_imp[trace_idx]) ** 2)
    ss_tot = np.sum((true_imp[trace_idx] - true_imp[trace_idx].mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    fig.suptitle(f'{freq} Model - Trace #{trace_idx+1} | PCC={pcc:.4f} | R²={r2:.4f}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{freq}_trace_{trace_idx+1}_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {freq}_trace_{trace_idx+1}_comparison.png")


def plot_section_comparison(freq, output_dir):
    """绘制剖面对比图"""
    seismic, true_imp, pred_imp, stats = load_data_and_model(freq)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    n_traces, n_samples = true_imp.shape
    extent = [0, n_traces, n_samples*0.001, 0]  # 假设1ms采样
    
    # 地震剖面
    ax1 = axes[0, 0]
    vmax = np.percentile(np.abs(seismic), 98)
    im1 = ax1.imshow(seismic.T, aspect='auto', cmap='seismic', extent=extent, vmin=-vmax, vmax=vmax)
    ax1.set_xlabel('Trace Number', fontsize=12)
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('Seismic Section', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Amplitude')
    
    # 真实波阻抗
    ax2 = axes[0, 1]
    vmin_imp = np.percentile(true_imp, 2)
    vmax_imp = np.percentile(true_imp, 98)
    im2 = ax2.imshow(true_imp.T, aspect='auto', cmap='jet', extent=extent, vmin=vmin_imp, vmax=vmax_imp)
    ax2.set_xlabel('Trace Number', fontsize=12)
    ax2.set_ylabel('Time (s)', fontsize=12)
    ax2.set_title('True Impedance', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Impedance')
    
    # 预测波阻抗
    ax3 = axes[1, 0]
    im3 = ax3.imshow(pred_imp.T, aspect='auto', cmap='jet', extent=extent, vmin=vmin_imp, vmax=vmax_imp)
    ax3.set_xlabel('Trace Number', fontsize=12)
    ax3.set_ylabel('Time (s)', fontsize=12)
    ax3.set_title('Predicted Impedance', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Impedance')
    
    # 误差剖面
    ax4 = axes[1, 1]
    error = pred_imp - true_imp
    err_max = np.percentile(np.abs(error), 98)
    im4 = ax4.imshow(error.T, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-err_max, vmax=err_max)
    ax4.set_xlabel('Trace Number', fontsize=12)
    ax4.set_ylabel('Time (s)', fontsize=12)
    ax4.set_title('Prediction Error', fontsize=14, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Error')
    
    # 计算整体指标
    pcc, _ = pearsonr(pred_imp.flatten(), true_imp.flatten())
    ss_res = np.sum((true_imp - pred_imp) ** 2)
    ss_tot = np.sum((true_imp - true_imp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    fig.suptitle(f'{freq} V6 Model - Section Comparison | PCC={pcc:.4f} | R²={r2:.4f}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{freq}_section_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {freq}_section_comparison.png")


def plot_scatter_and_histogram(freq, output_dir):
    """绘制散点图和直方图"""
    seismic, true_imp, pred_imp, stats = load_data_and_model(freq)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    true_flat = true_imp.flatten()
    pred_flat = pred_imp.flatten()
    
    # 散点图
    ax1 = axes[0]
    # 随机采样以避免过多点
    n_points = min(50000, len(true_flat))
    idx = np.random.choice(len(true_flat), n_points, replace=False)
    ax1.scatter(true_flat[idx], pred_flat[idx], alpha=0.1, s=1, c='blue')
    
    # 1:1线
    lims = [min(true_flat.min(), pred_flat.min()), max(true_flat.max(), pred_flat.max())]
    ax1.plot(lims, lims, 'r-', linewidth=2, label='1:1 Line')
    
    # 线性回归
    z = np.polyfit(true_flat, pred_flat, 1)
    p = np.poly1d(z)
    ax1.plot(lims, p(lims), 'g--', linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.0f}')
    
    ax1.set_xlabel('True Impedance', fontsize=12)
    ax1.set_ylabel('Predicted Impedance', fontsize=12)
    ax1.set_title('Scatter Plot', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 误差直方图
    ax2 = axes[1]
    error = pred_flat - true_flat
    ax2.hist(error, bins=100, density=True, alpha=0.7, color='steelblue', edgecolor='white')
    ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax2.axvline(x=error.mean(), color='g', linestyle='-', linewidth=2, label=f'Mean={error.mean():.2f}')
    ax2.set_xlabel('Error', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 相对误差直方图
    ax3 = axes[2]
    rel_error = (pred_flat - true_flat) / (np.abs(true_flat) + 1e-8) * 100
    rel_error_clipped = np.clip(rel_error, -20, 20)
    ax3.hist(rel_error_clipped, bins=100, density=True, alpha=0.7, color='coral', edgecolor='white')
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Relative Error (%)', fontsize=12)
    ax3.set_ylabel('Density', fontsize=12)
    ax3.set_title('Relative Error Distribution', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 计算指标
    pcc, _ = pearsonr(pred_flat, true_flat)
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
    
    fig.suptitle(f'{freq} V6 Model | PCC={pcc:.4f} | R²={r2:.4f} | RMSE={rmse:.2f}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{freq}_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {freq}_statistics.png")


def plot_multi_freq_comparison(output_dir):
    """绘制多频率对比图"""
    freqs = ['20Hz', '30Hz', '40Hz']
    colors = {'20Hz': '#1f77b4', '30Hz': '#2ca02c', '40Hz': '#d62728'}
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    metrics_list = []
    
    for i, freq in enumerate(freqs):
        try:
            seismic, true_imp, pred_imp, stats = load_data_and_model(freq)
            
            # 选择一个代表性道
            trace_idx = 50
            
            # 上行：单道对比
            ax_top = axes[0, i]
            time = np.arange(true_imp.shape[1]) * 0.001
            ax_top.plot(true_imp[trace_idx], time, 'b-', linewidth=1.2, label='True', alpha=0.8)
            ax_top.plot(pred_imp[trace_idx], time, 'r--', linewidth=1.2, label='Predicted', alpha=0.8)
            ax_top.set_xlabel('Impedance', fontsize=12)
            if i == 0:
                ax_top.set_ylabel('Time (s)', fontsize=12)
            ax_top.invert_yaxis()
            ax_top.legend(loc='upper right', fontsize=10)
            ax_top.grid(True, alpha=0.3)
            
            # 计算该道的指标
            pcc_trace, _ = pearsonr(pred_imp[trace_idx], true_imp[trace_idx])
            ax_top.set_title(f'{freq} - Trace #{trace_idx+1}\nPCC={pcc_trace:.4f}', fontsize=13, fontweight='bold')
            
            # 下行：剖面
            ax_bot = axes[1, i]
            n_traces, n_samples = pred_imp.shape
            extent = [0, n_traces, n_samples*0.001, 0]
            vmin_imp = np.percentile(true_imp, 2)
            vmax_imp = np.percentile(true_imp, 98)
            im = ax_bot.imshow(pred_imp.T, aspect='auto', cmap='jet', extent=extent, vmin=vmin_imp, vmax=vmax_imp)
            ax_bot.set_xlabel('Trace Number', fontsize=12)
            if i == 0:
                ax_bot.set_ylabel('Time (s)', fontsize=12)
            
            # 整体指标
            pcc_all, _ = pearsonr(pred_imp.flatten(), true_imp.flatten())
            ss_res = np.sum((true_imp - pred_imp) ** 2)
            ss_tot = np.sum((true_imp - true_imp.mean()) ** 2)
            r2_all = 1 - ss_res / ss_tot
            
            ax_bot.set_title(f'PCC={pcc_all:.4f} | R²={r2_all:.4f}', fontsize=13, fontweight='bold')
            
            plt.colorbar(im, ax=ax_bot, label='Impedance')
            
            metrics_list.append({
                'freq': freq,
                'pcc': pcc_all,
                'r2': r2_all
            })
            
        except Exception as e:
            print(f"  Warning: {freq} skipped - {e}")
            axes[0, i].text(0.5, 0.5, f'{freq}\nNot Available', ha='center', va='center', fontsize=14)
            axes[1, i].text(0.5, 0.5, f'{freq}\nNot Available', ha='center', va='center', fontsize=14)
    
    fig.suptitle('V6 Model - Multi-Frequency Comparison', fontsize=18, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_freq_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: multi_freq_comparison.png")
    
    return metrics_list


def plot_summary_metrics(metrics_list, output_dir):
    """绘制汇总指标图"""
    if not metrics_list:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    freqs = [m['freq'] for m in metrics_list]
    pccs = [m['pcc'] for m in metrics_list]
    r2s = [m['r2'] for m in metrics_list]
    
    colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']
    
    # PCC条形图
    ax1 = axes[0]
    bars1 = ax1.bar(freqs, pccs, color=colors[:len(freqs)], edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('PCC', fontsize=14)
    ax1.set_title('Pearson Correlation Coefficient', fontsize=14, fontweight='bold')
    ax1.set_ylim([0.85, 1.0])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, pcc in zip(bars1, pccs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{pcc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # R²条形图
    ax2 = axes[1]
    bars2 = ax2.bar(freqs, r2s, color=colors[:len(freqs)], edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('R²', fontsize=14)
    ax2.set_title('Coefficient of Determination', fontsize=14, fontweight='bold')
    ax2.set_ylim([0.7, 1.0])
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar, r2 in zip(bars2, r2s):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{r2:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    fig.suptitle('V6 Model Performance Summary', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: performance_summary.png")


def main():
    output_dir = Path(r'D:\SEISMIC_CODING\new\results\visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Generating V6 Model Visualizations")
    print("="*60)
    
    freqs = ['20Hz', '30Hz', '40Hz']
    
    for freq in freqs:
        ckpt_path = Path(rf'D:\SEISMIC_CODING\new\results\01_{freq}_v6\checkpoints\best.pt')
        if not ckpt_path.exists():
            print(f"\n{freq}: Model not found, skipping...")
            continue
        
        print(f"\n{freq}:")
        try:
            # 单道对比图
            for trace_idx in [25, 50, 75]:
                plot_trace_comparison(freq, trace_idx, output_dir)
            
            # 剖面对比图
            plot_section_comparison(freq, output_dir)
            
            # 统计图
            plot_scatter_and_histogram(freq, output_dir)
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nGenerating summary plots...")
    
    # 多频率对比图
    metrics_list = plot_multi_freq_comparison(output_dir)
    
    # 汇总指标图
    plot_summary_metrics(metrics_list, output_dir)
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
