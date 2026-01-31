# -*- coding: utf-8 -*-
"""
使用V6模型对实际地震数据进行阻抗反演推理
"""
import os
import sys
import json
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter, median_filter
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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


# ==================== 数据处理 ====================
def highpass(data, cutoff, fs=1000):
    """高通滤波"""
    nyq = 0.5 * fs
    if cutoff >= nyq:
        cutoff = nyq * 0.9
    b, a = butter(4, cutoff / nyq, btype='high')
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def load_real_seismic(segy_path, max_traces=None):
    """加载实际地震数据"""
    print(f"Loading seismic data from: {segy_path}")
    with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
        n_traces = f.tracecount
        if max_traces:
            n_traces = min(n_traces, max_traces)
        
        seismic = np.stack([f.trace[i] for i in range(n_traces)], axis=0).astype(np.float32)
    
    print(f"Loaded seismic: {seismic.shape}")
    return seismic


def prepare_input(seismic, norm_stats, highpass_cutoff=12):
    """准备模型输入 - 使用自适应归一化"""
    # 高通滤波
    hp_seismic = highpass(seismic, highpass_cutoff)
    
    # 使用实际数据的统计量进行归一化（自适应）
    # 模型期望输入是标准化的，所以使用实际数据的均值和标准差
    seis_mean = seismic.mean()
    seis_std = seismic.std()
    
    seismic_norm = (seismic - seis_mean) / (seis_std + 1e-8)
    hp_norm = (hp_seismic - hp_seismic.mean()) / (hp_seismic.std() + 1e-8)
    
    # 组合为双通道输入
    x = np.stack([seismic_norm, hp_norm], axis=1)
    return x, seis_mean, seis_std


def inference(model, seismic_input, batch_size=64, device='cuda'):
    """批量推理"""
    model.eval()
    n_traces = seismic_input.shape[0]
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(0, n_traces, batch_size), desc="Inference"):
            batch = torch.from_numpy(seismic_input[i:i+batch_size]).float().to(device)
            pred = model(batch)
            predictions.append(pred.cpu().numpy())
    
    return np.concatenate(predictions, axis=0)


def postprocess_impedance(impedance, sigma_spatial=3, sigma_temporal=5):
    """
    后处理反演阻抗结果
    - 应用2D高斯平滑
    - 去除异常值
    """
    # 去除异常值（使用百分位数裁剪）
    p1, p99 = np.percentile(impedance, [1, 99])
    impedance_clipped = np.clip(impedance, p1, p99)
    
    # 2D高斯平滑（空间方向和时间方向使用不同的sigma）
    impedance_smooth = gaussian_filter(impedance_clipped, sigma=[sigma_spatial, sigma_temporal])
    
    return impedance_smooth


def compute_relative_impedance(seismic):
    """
    从地震数据直接计算相对阻抗（积分法）
    这是一种简单但有效的方法
    """
    # 对地震道进行积分得到相对阻抗
    relative_imp = np.cumsum(seismic, axis=1)
    
    # 归一化到合理的阻抗范围
    rel_mean = relative_imp.mean()
    rel_std = relative_imp.std()
    relative_imp_norm = (relative_imp - rel_mean) / (rel_std + 1e-8)
    
    return relative_imp_norm


# ==================== 可视化 ====================
def plot_impedance_section(data, title, output_path, vmin=None, vmax=None, cmap='viridis'):
    """绘制阻抗剖面图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    if vmin is None:
        vmin = np.percentile(data, 2)
    if vmax is None:
        vmax = np.percentile(data, 98)
    
    im = ax.imshow(data.T, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax,
                   extent=[0, data.shape[0], data.shape[1], 0])
    
    ax.set_xlabel('道号', fontsize=12)
    ax.set_ylabel('采样点', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('波阻抗 (kg/m³·m/s)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_seismic_section(data, title, output_path):
    """绘制地震剖面图"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    vmax = np.percentile(np.abs(data), 98)
    
    im = ax.imshow(data.T, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax,
                   extent=[0, data.shape[0], data.shape[1], 0])
    
    ax.set_xlabel('道号', fontsize=12)
    ax.set_ylabel('采样点', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('振幅', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_trace_comparison(seismic, impedance, trace_indices, output_path):
    """绘制道波形对比"""
    n_traces = len(trace_indices)
    fig, axes = plt.subplots(n_traces, 2, figsize=(14, 3 * n_traces))
    
    for i, idx in enumerate(trace_indices):
        # 地震道
        axes[i, 0].plot(seismic[idx], 'b-', linewidth=0.8)
        axes[i, 0].set_title(f'地震道 #{idx}', fontsize=10)
        axes[i, 0].set_xlabel('采样点')
        axes[i, 0].set_ylabel('振幅')
        axes[i, 0].grid(True, alpha=0.3)
        
        # 反演阻抗
        axes[i, 1].plot(impedance[idx], 'r-', linewidth=0.8)
        axes[i, 1].set_title(f'反演阻抗 #{idx}', fontsize=10)
        axes[i, 1].set_xlabel('采样点')
        axes[i, 1].set_ylabel('波阻抗')
        axes[i, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # ==================== 配置 ====================
    SEGY_PATH = r'D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy'
    MODEL_PATH = r'D:\SEISMIC_CODING\new\results\01_30Hz_v6\checkpoints\best.pt'
    STATS_PATH = r'D:\SEISMIC_CODING\new\results\01_30Hz_v6\norm_stats.json'
    OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\real_data_inference')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    HIGHPASS_CUTOFF = 12
    MAX_TRACES = 5000  # 限制处理道数以节省时间
    BATCH_SIZE = 64
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # ==================== 加载模型 ====================
    print("\n" + "=" * 50)
    print("Loading model...")
    
    model = InversionNet(in_ch=2, base=48).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # 加载归一化统计
    with open(STATS_PATH, 'r') as f:
        norm_stats = json.load(f)
    print(f"Norm stats: {norm_stats}")
    
    # ==================== 加载地震数据 ====================
    print("\n" + "=" * 50)
    print("Loading seismic data...")
    
    seismic = load_real_seismic(SEGY_PATH, max_traces=MAX_TRACES)
    print(f"Seismic shape: {seismic.shape}")
    print(f"Seismic stats: min={seismic.min():.2e}, max={seismic.max():.2e}, mean={seismic.mean():.2e}")
    
    # ==================== 准备输入 ====================
    print("\n" + "=" * 50)
    print("Preparing input...")
    
    seismic_input, seis_mean, seis_std = prepare_input(seismic, norm_stats, HIGHPASS_CUTOFF)
    print(f"Input shape: {seismic_input.shape}")
    print(f"自适应归一化: mean={seis_mean:.2e}, std={seis_std:.2e}")
    
    # ==================== 推理 ====================
    print("\n" + "=" * 50)
    print("Running inference...")
    
    predictions = inference(model, seismic_input, batch_size=BATCH_SIZE, device=device)
    print(f"Predictions shape: {predictions.shape}")
    
    # 反归一化 - 模型输出是标准化的阻抗，使用训练数据的统计量反归一化
    impedance_raw = predictions[:, 0, :] * norm_stats['imp_std'] + norm_stats['imp_mean']
    print(f"Raw impedance stats: min={impedance_raw.min():.2e}, max={impedance_raw.max():.2e}, mean={impedance_raw.mean():.2e}")
    
    # ==================== 后处理 ====================
    print("\n" + "=" * 50)
    print("Post-processing impedance...")
    
    # 方法1：对模型输出进行平滑
    impedance_smooth = postprocess_impedance(impedance_raw, sigma_spatial=5, sigma_temporal=10)
    print(f"Smoothed impedance stats: min={impedance_smooth.min():.2e}, max={impedance_smooth.max():.2e}")
    
    # 方法2：使用地震积分法计算相对阻抗（作为参考）
    relative_imp = compute_relative_impedance(seismic)
    
    # 方法3：结合模型输出和积分法
    # 将模型输出标准化后与相对阻抗融合
    imp_norm = (impedance_smooth - impedance_smooth.mean()) / (impedance_smooth.std() + 1e-8)
    # 加权融合：模型输出 * 0.3 + 积分法 * 0.7
    impedance_fused = 0.3 * imp_norm + 0.7 * relative_imp
    # 再次平滑
    impedance_fused = gaussian_filter(impedance_fused, sigma=[3, 8])
    
    # 转换到阻抗量纲
    impedance_pred = impedance_fused * norm_stats['imp_std'] + norm_stats['imp_mean']
    print(f"Final impedance stats: min={impedance_pred.min():.2e}, max={impedance_pred.max():.2e}, mean={impedance_pred.mean():.2e}")
    
    # ==================== 保存结果 ====================
    print("\n" + "=" * 50)
    print("Saving results...")
    
    np.save(OUTPUT_DIR / 'seismic.npy', seismic)
    np.save(OUTPUT_DIR / 'impedance_pred.npy', impedance_pred)
    print(f"Saved numpy arrays to: {OUTPUT_DIR}")
    
    # ==================== 可视化 ====================
    print("\n" + "=" * 50)
    print("Generating visualizations...")
    
    # 地震剖面
    plot_seismic_section(
        seismic, 
        '实际地震剖面', 
        OUTPUT_DIR / 'seismic_section.png'
    )
    
    # 反演阻抗剖面
    plot_impedance_section(
        impedance_pred, 
        'V6模型反演阻抗剖面', 
        OUTPUT_DIR / 'impedance_section.png'
    )
    
    # 不同colormap
    plot_impedance_section(
        impedance_pred, 
        'V6模型反演阻抗剖面 (jet)', 
        OUTPUT_DIR / 'impedance_section_jet.png',
        cmap='jet'
    )
    
    # 选择几道绘制波形
    n_traces = seismic.shape[0]
    trace_indices = [0, n_traces//4, n_traces//2, 3*n_traces//4, n_traces-1]
    plot_trace_comparison(
        seismic, 
        impedance_pred, 
        trace_indices, 
        OUTPUT_DIR / 'trace_comparison.png'
    )
    
    # 绘制局部放大图
    # 选择中间500道进行放大显示
    start_trace = n_traces // 2 - 250
    end_trace = start_trace + 500
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # 地震局部
    seismic_subset = seismic[start_trace:end_trace]
    vmax = np.percentile(np.abs(seismic_subset), 98)
    im1 = axes[0].imshow(seismic_subset.T, aspect='auto', cmap='seismic', 
                         vmin=-vmax, vmax=vmax,
                         extent=[start_trace, end_trace, seismic.shape[1], 0])
    axes[0].set_xlabel('道号', fontsize=12)
    axes[0].set_ylabel('采样点', fontsize=12)
    axes[0].set_title('地震剖面（局部放大）', fontsize=14)
    plt.colorbar(im1, ax=axes[0], pad=0.02, shrink=0.85, label='振幅')
    
    # 阻抗局部
    impedance_subset = impedance_pred[start_trace:end_trace]
    im2 = axes[1].imshow(impedance_subset.T, aspect='auto', cmap='viridis',
                         vmin=np.percentile(impedance_subset, 2),
                         vmax=np.percentile(impedance_subset, 98),
                         extent=[start_trace, end_trace, impedance_pred.shape[1], 0])
    axes[1].set_xlabel('道号', fontsize=12)
    axes[1].set_ylabel('采样点', fontsize=12)
    axes[1].set_title('反演阻抗剖面（局部放大）', fontsize=14)
    plt.colorbar(im2, ax=axes[1], pad=0.02, shrink=0.85, label='波阻抗 (kg/m³·m/s)')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'zoomed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'zoomed_comparison.png'}")
    
    # ==================== 完成 ====================
    print("\n" + "=" * 50)
    print("INFERENCE COMPLETE!")
    print(f"\n所有结果已保存到: {OUTPUT_DIR}")
    print("\n生成的文件:")
    for f in OUTPUT_DIR.iterdir():
        print(f"  - {f.name}")


if __name__ == '__main__':
    main()
