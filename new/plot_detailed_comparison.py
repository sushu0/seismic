# -*- coding: utf-8 -*-
"""
多频率波阻抗反演详细可视化
包含：剖面对比、道对比、误差分布、频谱分析、薄层区域放大等
"""
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.gridspec import GridSpec
from scipy.signal import butter, filtfilt
from scipy import stats
from pathlib import Path

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ==================== 模型定义 ====================
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
    def __init__(self, in_ch=2, base_ch=64, out_ch=1):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.GELU()
        )
        self.multi_scale = DilatedConvBlock(base_ch, base_ch, dilations=[1, 2, 4, 8])
        self.enc1 = ThinLayerBlock(base_ch, base_ch * 2)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = ThinLayerBlock(base_ch * 2, base_ch * 4)
        self.pool2 = nn.MaxPool1d(2)
        self.bottleneck = nn.Sequential(
            DilatedConvBlock(base_ch * 4, base_ch * 8, dilations=[1, 2, 4, 8, 16]),
            ThinLayerBlock(base_ch * 8, base_ch * 8)
        )
        self.up2 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec2 = ThinLayerBlock(base_ch * 8, base_ch * 4)
        self.up1 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1 = ThinLayerBlock(base_ch * 4, base_ch * 2)
        self.refine = nn.Sequential(
            ThinLayerBlock(base_ch * 2 + base_ch, base_ch * 2),
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.GELU(),
            BoundaryEnhanceModule(base_ch),
        )
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


def highpass_filter(data, cutoff, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


# ==================== 配置 ====================
FREQ_CONFIGS = {
    '20Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\norm_stats.json',
        'highpass_cutoff': 8,
        'color': '#1f77b4',
    },
    '30Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v3\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v3\norm_stats.json',
        'highpass_cutoff': 12,
        'color': '#ff7f0e',
    },
    '40Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_40Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_40Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\norm_stats.json',
        'highpass_cutoff': 15,
        'color': '#2ca02c',
    },
    '50Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_50Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_50Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_50Hz_thinlayer_v2\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_50Hz_thinlayer_v2\norm_stats.json',
        'highpass_cutoff': 20,
        'color': '#d62728',
    },
}

OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\results\detailed_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")


def load_data_and_predict(config):
    """加载数据并进行预测"""
    with segyio.open(config['seismic'], 'r', ignore_geometry=True) as f:
        seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)
    
    raw_imp = np.loadtxt(config['impedance'], usecols=4, skiprows=1).astype(np.float32)
    n_traces = seismic.shape[0]
    n_samples = len(raw_imp) // n_traces
    impedance = raw_imp.reshape(n_traces, n_samples)
    
    with open(config['norm'], 'r') as f:
        norm_stats = json.load(f)
    
    seis_norm = (seismic - norm_stats['seis_mean']) / norm_stats['seis_std']
    seismic_hf = highpass_filter(seismic, cutoff=config['highpass_cutoff'], fs=1000)
    seis_hf_norm = seismic_hf / (np.std(seismic_hf, axis=1, keepdims=True) + 1e-6)
    
    ckpt = torch.load(config['model'], map_location=device, weights_only=False)
    input_shape = ckpt['model']['input_conv.0.weight'].shape
    in_ch = input_shape[1]
    
    model = ThinLayerNetV2(in_ch=in_ch, base_ch=64, out_ch=1).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    all_pred = []
    with torch.no_grad():
        for i in range(n_traces):
            if in_ch == 1:
                x = seis_norm[i:i+1]
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            else:
                x = np.stack([seis_norm[i], seis_hf_norm[i]], axis=0)
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            pred = model(x_t)
            all_pred.append(pred.cpu().numpy().squeeze())
    
    pred_full = np.array(all_pred)
    pred_full_denorm = pred_full * norm_stats['imp_std'] + norm_stats['imp_mean']
    
    return seismic, impedance, pred_full_denorm, norm_stats


def compute_metrics(true, pred):
    """计算评估指标"""
    true_flat = true.flatten()
    pred_flat = pred.flatten()
    
    mse = np.mean((true_flat - pred_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(true_flat - pred_flat))
    
    # PCC
    pcc = np.corrcoef(true_flat, pred_flat)[0, 1]
    
    # R²
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - np.mean(true_flat)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    # 相对误差
    rel_error = np.mean(np.abs(true_flat - pred_flat) / (np.abs(true_flat) + 1e-8)) * 100
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'PCC': pcc,
        'R2': r2,
        'RelError': rel_error
    }


def plot_single_freq_detailed(freq, seismic, true_imp, pred_imp, config, output_dir):
    """为单个频率绘制详细可视化"""
    n_traces, n_samples = true_imp.shape
    dt_ms = 0.01
    total_time = n_samples * dt_ms
    
    metrics = compute_metrics(true_imp, pred_imp)
    error = pred_imp - true_imp
    
    # ==================== 图1: 综合剖面对比 ====================
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.8], hspace=0.25, wspace=0.25)
    
    extent = [0, n_traces, total_time, 0]
    vmin, vmax = 6e6, 1e7
    
    # 真实剖面
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(true_imp.T, aspect='auto', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    ax1.set_title(f'{freq} 真实波阻抗', fontsize=14, fontweight='bold')
    ax1.set_xlabel('道号')
    ax1.set_ylabel('时间 (ms)')
    plt.colorbar(im1, ax=ax1, label='波阻抗')
    
    # 预测剖面
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pred_imp.T, aspect='auto', cmap='viridis', extent=extent, vmin=vmin, vmax=vmax)
    ax2.set_title(f'{freq} 预测波阻抗', fontsize=14, fontweight='bold')
    ax2.set_xlabel('道号')
    ax2.set_ylabel('时间 (ms)')
    plt.colorbar(im2, ax=ax2, label='波阻抗')
    
    # 误差剖面
    ax3 = fig.add_subplot(gs[0, 2])
    err_max = np.percentile(np.abs(error), 95)
    im3 = ax3.imshow(error.T, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-err_max, vmax=err_max)
    ax3.set_title(f'{freq} 预测误差', fontsize=14, fontweight='bold')
    ax3.set_xlabel('道号')
    ax3.set_ylabel('时间 (ms)')
    plt.colorbar(im3, ax=ax3, label='误差')
    
    # 地震数据
    ax4 = fig.add_subplot(gs[1, 0])
    seis_extent = [0, n_traces, seismic.shape[1] * dt_ms, 0]
    seis_max = np.percentile(np.abs(seismic), 98)
    im4 = ax4.imshow(seismic.T, aspect='auto', cmap='seismic', extent=seis_extent, vmin=-seis_max, vmax=seis_max)
    ax4.set_title(f'{freq} 输入地震数据', fontsize=14, fontweight='bold')
    ax4.set_xlabel('道号')
    ax4.set_ylabel('时间 (ms)')
    plt.colorbar(im4, ax=ax4, label='振幅')
    
    # 道对比 (选择3道)
    ax5 = fig.add_subplot(gs[1, 1])
    trace_indices = [20, 50, 80]
    time_axis = np.arange(n_samples) * dt_ms
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, (trace_idx, color) in enumerate(zip(trace_indices, colors)):
        ax5.plot(true_imp[trace_idx], time_axis, '-', color=color, linewidth=1.5, 
                label=f'真实 (道{trace_idx})', alpha=0.8)
        ax5.plot(pred_imp[trace_idx], time_axis, '--', color=color, linewidth=1.5, 
                label=f'预测 (道{trace_idx})', alpha=0.8)
    ax5.set_xlabel('波阻抗')
    ax5.set_ylabel('时间 (ms)')
    ax5.set_title('道对比', fontsize=14, fontweight='bold')
    ax5.invert_yaxis()
    ax5.legend(loc='lower right', fontsize=8, ncol=2)
    ax5.grid(True, alpha=0.3)
    
    # 散点图 + 回归线
    ax6 = fig.add_subplot(gs[1, 2])
    sample_idx = np.random.choice(true_imp.size, min(5000, true_imp.size), replace=False)
    true_sample = true_imp.flatten()[sample_idx]
    pred_sample = pred_imp.flatten()[sample_idx]
    ax6.scatter(true_sample, pred_sample, alpha=0.3, s=5, c=config['color'])
    
    # 回归线
    slope, intercept, r_value, p_value, std_err = stats.linregress(true_sample, pred_sample)
    x_line = np.array([true_sample.min(), true_sample.max()])
    y_line = slope * x_line + intercept
    ax6.plot(x_line, y_line, 'r-', linewidth=2, label=f'拟合线 (R²={r_value**2:.4f})')
    ax6.plot(x_line, x_line, 'k--', linewidth=1, label='理想线 y=x')
    
    ax6.set_xlabel('真实波阻抗')
    ax6.set_ylabel('预测波阻抗')
    ax6.set_title('真实 vs 预测 散点图', fontsize=14, fontweight='bold')
    ax6.legend(loc='upper left')
    ax6.grid(True, alpha=0.3)
    
    # 误差直方图
    ax7 = fig.add_subplot(gs[2, 0])
    error_flat = error.flatten()
    ax7.hist(error_flat, bins=100, density=True, alpha=0.7, color=config['color'], edgecolor='black', linewidth=0.5)
    ax7.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax7.axvline(x=np.mean(error_flat), color='g', linestyle='-', linewidth=2, label=f'均值={np.mean(error_flat):.2e}')
    ax7.set_xlabel('预测误差')
    ax7.set_ylabel('概率密度')
    ax7.set_title('误差分布', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 道平均误差
    ax8 = fig.add_subplot(gs[2, 1])
    trace_mae = np.mean(np.abs(error), axis=1)
    trace_rmse = np.sqrt(np.mean(error**2, axis=1))
    ax8.plot(range(n_traces), trace_mae, '-', color='blue', linewidth=1.5, label='MAE')
    ax8.plot(range(n_traces), trace_rmse, '-', color='red', linewidth=1.5, label='RMSE')
    ax8.fill_between(range(n_traces), 0, trace_mae, alpha=0.3, color='blue')
    ax8.set_xlabel('道号')
    ax8.set_ylabel('误差')
    ax8.set_title('各道误差统计', fontsize=14, fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 指标文本框
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    metrics_text = f"""
    {freq} 模型评估指标
    ══════════════════════════
    
    PCC (皮尔逊相关系数):  {metrics['PCC']:.4f}
    R² (决定系数):         {metrics['R2']:.4f}
    
    MSE (均方误差):        {metrics['MSE']:.4e}
    RMSE (均方根误差):     {metrics['RMSE']:.4e}
    MAE (平均绝对误差):    {metrics['MAE']:.4e}
    
    相对误差:              {metrics['RelError']:.2f}%
    """
    ax9.text(0.1, 0.5, metrics_text, fontsize=14, family='monospace',
             verticalalignment='center', transform=ax9.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{freq} 波阻抗反演详细分析', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(output_dir / f'{freq}_detailed_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {freq}_detailed_analysis.png")
    
    return metrics


def plot_thin_layer_zoom(freq, true_imp, pred_imp, config, output_dir):
    """薄层区域放大对比"""
    n_traces, n_samples = true_imp.shape
    dt_ms = 0.01
    
    # 寻找变化最剧烈的区域（可能是薄层）
    gradient = np.abs(np.gradient(true_imp, axis=1))
    avg_gradient = np.mean(gradient, axis=0)
    
    # 找到梯度最大的区域
    window_size = 500
    max_grad_sum = 0
    best_start = 0
    for i in range(len(avg_gradient) - window_size):
        grad_sum = np.sum(avg_gradient[i:i+window_size])
        if grad_sum > max_grad_sum:
            max_grad_sum = grad_sum
            best_start = i
    
    # 放大区域
    zoom_start = best_start
    zoom_end = min(best_start + window_size, n_samples)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    extent_zoom = [0, n_traces, zoom_end * dt_ms, zoom_start * dt_ms]
    vmin, vmax = 6e6, 1e7
    
    # 真实（放大）
    im1 = axes[0, 0].imshow(true_imp[:, zoom_start:zoom_end].T, aspect='auto', 
                             cmap='viridis', extent=extent_zoom, vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'{freq} 真实波阻抗 (薄层区域)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('道号')
    axes[0, 0].set_ylabel('时间 (ms)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # 预测（放大）
    im2 = axes[0, 1].imshow(pred_imp[:, zoom_start:zoom_end].T, aspect='auto', 
                             cmap='viridis', extent=extent_zoom, vmin=vmin, vmax=vmax)
    axes[0, 1].set_title(f'{freq} 预测波阻抗 (薄层区域)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('道号')
    axes[0, 1].set_ylabel('时间 (ms)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # 误差（放大）
    error_zoom = pred_imp[:, zoom_start:zoom_end] - true_imp[:, zoom_start:zoom_end]
    err_max = np.percentile(np.abs(error_zoom), 95)
    im3 = axes[0, 2].imshow(error_zoom.T, aspect='auto', cmap='RdBu_r', 
                             extent=extent_zoom, vmin=-err_max, vmax=err_max)
    axes[0, 2].set_title(f'{freq} 预测误差 (薄层区域)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('道号')
    axes[0, 2].set_ylabel('时间 (ms)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # 单道放大对比
    trace_idx = 50
    time_axis = np.arange(zoom_start, zoom_end) * dt_ms
    
    axes[1, 0].plot(true_imp[trace_idx, zoom_start:zoom_end], time_axis, 'b-', 
                    linewidth=2, label='真实')
    axes[1, 0].plot(pred_imp[trace_idx, zoom_start:zoom_end], time_axis, 'r--', 
                    linewidth=2, label='预测')
    axes[1, 0].fill_betweenx(time_axis, true_imp[trace_idx, zoom_start:zoom_end], 
                              pred_imp[trace_idx, zoom_start:zoom_end], alpha=0.3, color='gray')
    axes[1, 0].set_xlabel('波阻抗')
    axes[1, 0].set_ylabel('时间 (ms)')
    axes[1, 0].set_title(f'道 {trace_idx} 放大对比', fontsize=12, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 梯度对比
    true_grad = np.abs(np.gradient(true_imp[trace_idx, zoom_start:zoom_end]))
    pred_grad = np.abs(np.gradient(pred_imp[trace_idx, zoom_start:zoom_end]))
    
    axes[1, 1].plot(true_grad, time_axis, 'b-', linewidth=1.5, label='真实梯度')
    axes[1, 1].plot(pred_grad, time_axis, 'r--', linewidth=1.5, label='预测梯度')
    axes[1, 1].set_xlabel('梯度幅值')
    axes[1, 1].set_ylabel('时间 (ms)')
    axes[1, 1].set_title('边界梯度对比', fontsize=12, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 局部相关性
    local_pcc = []
    window = 50
    for i in range(zoom_start, zoom_end - window):
        true_win = true_imp[:, i:i+window].flatten()
        pred_win = pred_imp[:, i:i+window].flatten()
        pcc = np.corrcoef(true_win, pred_win)[0, 1]
        local_pcc.append(pcc)
    
    local_time = np.arange(zoom_start, zoom_end - window) * dt_ms
    axes[1, 2].plot(local_pcc, local_time, color=config['color'], linewidth=1.5)
    axes[1, 2].axvline(x=0.9, color='g', linestyle='--', label='PCC=0.9')
    axes[1, 2].axvline(x=np.mean(local_pcc), color='r', linestyle='-', 
                       label=f'均值={np.mean(local_pcc):.3f}')
    axes[1, 2].set_xlabel('局部PCC')
    axes[1, 2].set_ylabel('时间 (ms)')
    axes[1, 2].set_title('局部相关性', fontsize=12, fontweight='bold')
    axes[1, 2].invert_yaxis()
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlim(0.5, 1.0)
    
    plt.suptitle(f'{freq} 薄层区域详细分析 (时间 {zoom_start*dt_ms:.1f}-{zoom_end*dt_ms:.1f} ms)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'{freq}_thin_layer_zoom.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {freq}_thin_layer_zoom.png")


def plot_multi_freq_metrics_comparison(all_metrics, output_dir):
    """绘制多频率指标对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    freqs = list(all_metrics.keys())
    colors = [FREQ_CONFIGS[f]['color'] for f in freqs]
    
    # PCC对比
    pcc_values = [all_metrics[f]['PCC'] for f in freqs]
    bars1 = axes[0, 0].bar(freqs, pcc_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 0].set_ylabel('PCC')
    axes[0, 0].set_title('皮尔逊相关系数 (PCC)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim(0.9, 1.0)
    for bar, val in zip(bars1, pcc_values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                        f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0, 0].axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # R²对比
    r2_values = [all_metrics[f]['R2'] for f in freqs]
    bars2 = axes[0, 1].bar(freqs, r2_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].set_title('决定系数 (R²)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim(0.8, 1.0)
    for bar, val in zip(bars2, r2_values):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                        f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[0, 1].axhline(y=0.9, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # RMSE对比
    rmse_values = [all_metrics[f]['RMSE'] for f in freqs]
    bars3 = axes[0, 2].bar(freqs, rmse_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[0, 2].set_ylabel('RMSE')
    axes[0, 2].set_title('均方根误差 (RMSE)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars3, rmse_values):
        axes[0, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.02, 
                        f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    
    # MAE对比
    mae_values = [all_metrics[f]['MAE'] for f in freqs]
    bars4 = axes[1, 0].bar(freqs, mae_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('平均绝对误差 (MAE)', fontsize=14, fontweight='bold')
    for bar, val in zip(bars4, mae_values):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.02, 
                        f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 相对误差对比
    rel_values = [all_metrics[f]['RelError'] for f in freqs]
    bars5 = axes[1, 1].bar(freqs, rel_values, color=colors, edgecolor='black', linewidth=1.5)
    axes[1, 1].set_ylabel('相对误差 (%)')
    axes[1, 1].set_title('相对误差', fontsize=14, fontweight='bold')
    for bar, val in zip(bars5, rel_values):
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                        f'{val:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 雷达图
    ax_radar = axes[1, 2]
    ax_radar.axis('off')
    
    # 创建汇总表格
    summary_text = "模型性能汇总表\n" + "="*50 + "\n\n"
    summary_text += f"{'频率':<8} {'PCC':<10} {'R²':<10} {'RMSE':<12}\n"
    summary_text += "-"*50 + "\n"
    for f in freqs:
        summary_text += f"{f:<8} {all_metrics[f]['PCC']:.4f}     {all_metrics[f]['R2']:.4f}     {all_metrics[f]['RMSE']:.2e}\n"
    summary_text += "-"*50 + "\n"
    
    best_pcc = max(freqs, key=lambda x: all_metrics[x]['PCC'])
    best_r2 = max(freqs, key=lambda x: all_metrics[x]['R2'])
    summary_text += f"\n最佳 PCC: {best_pcc} ({all_metrics[best_pcc]['PCC']:.4f})\n"
    summary_text += f"最佳 R²: {best_r2} ({all_metrics[best_r2]['R2']:.4f})"
    
    ax_radar.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
                  verticalalignment='center', transform=ax_radar.transAxes,
                  bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('多频率模型性能对比', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_freq_metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: multi_freq_metrics_comparison.png")


def plot_frequency_resolution_analysis(all_data, output_dir):
    """频率分辨率分析对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    freqs = list(all_data.keys())
    
    # 选择同一道进行对比
    trace_idx = 50
    
    # 所有频率真实值叠加对比
    ax1 = axes[0, 0]
    for freq in freqs:
        true_imp = all_data[freq]['true']
        n_samples = true_imp.shape[1]
        time_axis = np.arange(n_samples) * 0.01
        ax1.plot(true_imp[trace_idx], time_axis, linewidth=1.5, 
                label=freq, color=FREQ_CONFIGS[freq]['color'], alpha=0.8)
    ax1.set_xlabel('波阻抗')
    ax1.set_ylabel('时间 (ms)')
    ax1.set_title(f'道 {trace_idx} - 不同频率真实阻抗对比', fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 所有频率预测值叠加对比
    ax2 = axes[0, 1]
    for freq in freqs:
        pred_imp = all_data[freq]['pred']
        n_samples = pred_imp.shape[1]
        time_axis = np.arange(n_samples) * 0.01
        ax2.plot(pred_imp[trace_idx], time_axis, linewidth=1.5, 
                label=freq, color=FREQ_CONFIGS[freq]['color'], alpha=0.8)
    ax2.set_xlabel('波阻抗')
    ax2.set_ylabel('时间 (ms)')
    ax2.set_title(f'道 {trace_idx} - 不同频率预测阻抗对比', fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 误差随深度变化
    ax3 = axes[1, 0]
    for freq in freqs:
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        error = np.mean(np.abs(pred_imp - true_imp), axis=0)
        n_samples = len(error)
        time_axis = np.arange(n_samples) * 0.01
        ax3.plot(error, time_axis, linewidth=1.5, 
                label=freq, color=FREQ_CONFIGS[freq]['color'], alpha=0.8)
    ax3.set_xlabel('平均绝对误差')
    ax3.set_ylabel('时间 (ms)')
    ax3.set_title('误差随深度变化', fontsize=14, fontweight='bold')
    ax3.invert_yaxis()
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 梯度分辨率对比（边界清晰度）
    ax4 = axes[1, 1]
    for freq in freqs:
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        true_grad = np.mean(np.abs(np.gradient(true_imp, axis=1)), axis=0)
        pred_grad = np.mean(np.abs(np.gradient(pred_imp, axis=1)), axis=0)
        
        # 梯度相关性
        grad_corr = []
        window = 100
        for i in range(0, len(true_grad) - window, window//2):
            corr = np.corrcoef(true_grad[i:i+window], pred_grad[i:i+window])[0, 1]
            grad_corr.append(corr if not np.isnan(corr) else 0)
        
        time_axis = np.arange(len(grad_corr)) * (window//2) * 0.01
        ax4.plot(grad_corr, time_axis, linewidth=1.5, 
                label=freq, color=FREQ_CONFIGS[freq]['color'], alpha=0.8)
    
    ax4.set_xlabel('梯度相关性')
    ax4.set_ylabel('时间 (ms)')
    ax4.set_title('边界梯度相关性（分辨率指标）', fontsize=14, fontweight='bold')
    ax4.invert_yaxis()
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.axvline(x=0.8, color='r', linestyle='--', alpha=0.5)
    
    plt.suptitle('多频率分辨率分析', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'frequency_resolution_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: frequency_resolution_analysis.png")


# ==================== 主程序 ====================
print("="*60)
print("多频率波阻抗反演详细可视化")
print("="*60)

all_metrics = {}
all_data = {}

for freq, config in FREQ_CONFIGS.items():
    print(f"\n{'='*40}")
    print(f"处理 {freq}...")
    print(f"{'='*40}")
    try:
        seismic, true_imp, pred_imp, norm_stats = load_data_and_predict(config)
        
        # 存储数据
        all_data[freq] = {'true': true_imp, 'pred': pred_imp, 'seismic': seismic}
        
        # 绘制详细分析图
        metrics = plot_single_freq_detailed(freq, seismic, true_imp, pred_imp, config, OUTPUT_DIR)
        all_metrics[freq] = metrics
        
        # 绘制薄层放大图
        plot_thin_layer_zoom(freq, true_imp, pred_imp, config, OUTPUT_DIR)
        
        print(f"  {freq} 完成: PCC={metrics['PCC']:.4f}, R²={metrics['R2']:.4f}")
    except Exception as e:
        print(f"  {freq} 错误: {e}")
        import traceback
        traceback.print_exc()

# 绘制多频率对比图
if len(all_metrics) >= 2:
    print(f"\n{'='*40}")
    print("生成多频率对比图...")
    print(f"{'='*40}")
    plot_multi_freq_metrics_comparison(all_metrics, OUTPUT_DIR)
    plot_frequency_resolution_analysis(all_data, OUTPUT_DIR)

print("\n" + "="*60)
print("详细可视化完成!")
print("="*60)
print(f"\n输出目录: {OUTPUT_DIR}")
print(f"\n生成的文件:")
for f in OUTPUT_DIR.glob("*.png"):
    print(f"  - {f.name}")
