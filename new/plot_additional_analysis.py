# -*- coding: utf-8 -*-
"""
附加详细可视化：单道对比、层位追踪、频率响应分析
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
from scipy.signal import butter, filtfilt, spectrogram, welch
from scipy import stats
from pathlib import Path

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ==================== 模型定义（同上）====================
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


def plot_multi_trace_comparison(all_data, output_dir):
    """多道详细对比图"""
    trace_indices = [10, 30, 50, 70, 90]
    freqs = list(all_data.keys())
    
    fig, axes = plt.subplots(len(trace_indices), len(freqs) + 1, figsize=(24, 18))
    
    for i, trace_idx in enumerate(trace_indices):
        # 各频率对比
        for j, freq in enumerate(freqs):
            true_imp = all_data[freq]['true']
            pred_imp = all_data[freq]['pred']
            n_samples = true_imp.shape[1]
            time_axis = np.arange(n_samples) * 0.01
            
            ax = axes[i, j]
            ax.plot(true_imp[trace_idx], time_axis, 'b-', linewidth=1.5, label='True')
            ax.plot(pred_imp[trace_idx], time_axis, 'r--', linewidth=1.5, label='Pred')
            ax.fill_betweenx(time_axis, true_imp[trace_idx], pred_imp[trace_idx], 
                             alpha=0.2, color='gray')
            
            # 计算该道的PCC
            pcc = np.corrcoef(true_imp[trace_idx], pred_imp[trace_idx])[0, 1]
            
            if i == 0:
                ax.set_title(f'{freq}\nPCC={pcc:.3f}', fontsize=12, fontweight='bold')
            else:
                ax.set_title(f'PCC={pcc:.3f}', fontsize=10)
            
            ax.invert_yaxis()
            ax.set_ylabel(f'Trace {trace_idx}\nTime (ms)', fontsize=9)
            ax.grid(True, alpha=0.3)
            
            if i == 0 and j == 0:
                ax.legend(loc='lower right', fontsize=8)
        
        # 最后一列：所有频率叠加
        ax_all = axes[i, -1]
        for freq in freqs:
            pred_imp = all_data[freq]['pred']
            n_samples = pred_imp.shape[1]
            time_axis = np.arange(n_samples) * 0.01
            ax_all.plot(pred_imp[trace_idx], time_axis, linewidth=1, 
                       label=freq, color=FREQ_CONFIGS[freq]['color'], alpha=0.8)
        
        # 用第一个频率的真实值作为参考
        first_freq = freqs[0]
        true_imp = all_data[first_freq]['true']
        ax_all.plot(true_imp[trace_idx], time_axis, 'k-', linewidth=2, label='True', alpha=0.6)
        
        if i == 0:
            ax_all.set_title('All Freq\nComparison', fontsize=12, fontweight='bold')
            ax_all.legend(loc='lower right', fontsize=7)
        ax_all.invert_yaxis()
        ax_all.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Trace Impedance Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'multi_trace_detailed_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: multi_trace_detailed_comparison.png")


def plot_layer_boundary_analysis(all_data, output_dir):
    """层位边界分析"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    freqs = list(all_data.keys())
    
    # 第一行：各频率的边界检测结果
    for i, freq in enumerate(freqs):
        ax = fig.add_subplot(gs[0, i])
        
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        # 计算真实和预测的边界（梯度）
        true_grad = np.abs(np.gradient(true_imp, axis=1))
        pred_grad = np.abs(np.gradient(pred_imp, axis=1))
        
        n_traces, n_samples = true_imp.shape
        extent = [0, n_traces, n_samples * 0.01, 0]
        
        # 显示预测的边界
        grad_max = np.percentile(pred_grad, 98)
        im = ax.imshow(pred_grad.T, aspect='auto', cmap='hot', extent=extent, vmin=0, vmax=grad_max)
        ax.set_title(f'{freq} Boundary Detection', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trace')
        ax.set_ylabel('Time (ms)')
        plt.colorbar(im, ax=ax, label='Gradient')
    
    # 第二行：边界追踪对比
    trace_idx = 50
    ax_boundary = fig.add_subplot(gs[1, :2])
    
    for freq in freqs:
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        true_grad = np.abs(np.gradient(true_imp[trace_idx]))
        pred_grad = np.abs(np.gradient(pred_imp[trace_idx]))
        
        n_samples = len(true_grad)
        time_axis = np.arange(n_samples) * 0.01
        
        ax_boundary.plot(pred_grad, time_axis, linewidth=1.5, 
                        label=f'{freq}', color=FREQ_CONFIGS[freq]['color'])
    
    # 真实边界
    true_grad = np.abs(np.gradient(all_data[freqs[0]]['true'][trace_idx]))
    ax_boundary.plot(true_grad, time_axis, 'k--', linewidth=2, label='True', alpha=0.7)
    
    ax_boundary.set_xlabel('Gradient Magnitude')
    ax_boundary.set_ylabel('Time (ms)')
    ax_boundary.set_title(f'Trace {trace_idx} - Boundary Gradient Comparison', fontsize=14, fontweight='bold')
    ax_boundary.invert_yaxis()
    ax_boundary.legend(loc='lower right')
    ax_boundary.grid(True, alpha=0.3)
    
    # 边界检测准确性分析
    ax_accuracy = fig.add_subplot(gs[1, 2:])
    
    boundary_accuracy = {}
    for freq in freqs:
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        # 寻找主要边界（梯度超过阈值的位置）
        true_grad = np.mean(np.abs(np.gradient(true_imp, axis=1)), axis=0)
        pred_grad = np.mean(np.abs(np.gradient(pred_imp, axis=1)), axis=0)
        
        threshold = np.percentile(true_grad, 90)
        true_boundaries = true_grad > threshold
        pred_boundaries = pred_grad > threshold
        
        # 计算准确率
        tp = np.sum(true_boundaries & pred_boundaries)
        fp = np.sum(~true_boundaries & pred_boundaries)
        fn = np.sum(true_boundaries & ~pred_boundaries)
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        boundary_accuracy[freq] = {'precision': precision, 'recall': recall, 'f1': f1}
    
    metrics = ['precision', 'recall', 'f1']
    x = np.arange(len(freqs))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [boundary_accuracy[f][metric] for f in freqs]
        bars = ax_accuracy.bar(x + i * width, values, width, 
                               label=metric.capitalize(), alpha=0.8)
        for bar, val in zip(bars, values):
            ax_accuracy.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    
    ax_accuracy.set_xlabel('Frequency')
    ax_accuracy.set_ylabel('Score')
    ax_accuracy.set_title('Boundary Detection Metrics', fontsize=14, fontweight='bold')
    ax_accuracy.set_xticks(x + width)
    ax_accuracy.set_xticklabels(freqs)
    ax_accuracy.legend()
    ax_accuracy.grid(True, alpha=0.3, axis='y')
    ax_accuracy.set_ylim(0, 1.1)
    
    # 第三行：层位厚度分析
    ax_thickness = fig.add_subplot(gs[2, :2])
    
    for freq in freqs:
        true_imp = all_data[freq]['true']
        
        # 计算层位厚度分布（相邻样本变化小于阈值认为是同一层）
        layer_thicknesses = []
        for t in range(true_imp.shape[0]):
            grad = np.abs(np.gradient(true_imp[t]))
            threshold = np.percentile(grad, 75)
            
            # 找边界
            boundaries = np.where(grad > threshold)[0]
            if len(boundaries) > 1:
                thicknesses = np.diff(boundaries)
                layer_thicknesses.extend(thicknesses)
        
        if layer_thicknesses:
            ax_thickness.hist(layer_thicknesses, bins=50, alpha=0.5, 
                            label=freq, color=FREQ_CONFIGS[freq]['color'])
    
    ax_thickness.set_xlabel('Layer Thickness (samples)')
    ax_thickness.set_ylabel('Count')
    ax_thickness.set_title('Layer Thickness Distribution', fontsize=14, fontweight='bold')
    ax_thickness.legend()
    ax_thickness.grid(True, alpha=0.3)
    
    # 误差随层位厚度的关系
    ax_err_thickness = fig.add_subplot(gs[2, 2:])
    
    for freq in freqs:
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        # 按区域计算误差
        error = np.abs(pred_imp - true_imp)
        grad = np.abs(np.gradient(true_imp, axis=1))
        
        # 分类：高梯度区域（边界）vs 低梯度区域（层内）
        high_grad_mask = grad > np.percentile(grad, 80)
        low_grad_mask = grad <= np.percentile(grad, 80)
        
        boundary_error = np.mean(error[high_grad_mask])
        interior_error = np.mean(error[low_grad_mask])
        
        ax_err_thickness.bar([freq + '\nBoundary', freq + '\nInterior'], 
                            [boundary_error, interior_error],
                            color=[FREQ_CONFIGS[freq]['color'], FREQ_CONFIGS[freq]['color']],
                            alpha=0.8)
    
    ax_err_thickness.set_ylabel('Mean Absolute Error')
    ax_err_thickness.set_title('Error: Boundary vs Interior', fontsize=14, fontweight='bold')
    ax_err_thickness.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Layer Boundary Analysis', fontsize=18, fontweight='bold')
    plt.savefig(output_dir / 'layer_boundary_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: layer_boundary_analysis.png")


def plot_seismic_impedance_correlation(all_data, output_dir):
    """地震数据与波阻抗相关性分析"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    freqs = list(all_data.keys())
    
    for i, freq in enumerate(freqs):
        seismic = all_data[freq]['seismic']
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        # 地震振幅与阻抗的关系
        ax1 = axes[0, i]
        
        # 下采样以便绘图
        n_samples = min(5000, seismic.size)
        idx = np.random.choice(seismic.size, n_samples, replace=False)
        
        seis_flat = seismic.flatten()[idx]
        imp_flat = true_imp.flatten()[idx]
        
        ax1.scatter(seis_flat, imp_flat, alpha=0.3, s=3, c=FREQ_CONFIGS[freq]['color'])
        ax1.set_xlabel('Seismic Amplitude')
        ax1.set_ylabel('True Impedance')
        ax1.set_title(f'{freq} - Seis vs True Imp', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 计算相关性
        corr = np.corrcoef(seis_flat, imp_flat)[0, 1]
        ax1.text(0.05, 0.95, f'Corr={corr:.3f}', transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 地震振幅与预测误差的关系
        ax2 = axes[1, i]
        
        error_flat = np.abs(pred_imp.flatten() - true_imp.flatten())[idx]
        
        ax2.scatter(np.abs(seis_flat), error_flat, alpha=0.3, s=3, c=FREQ_CONFIGS[freq]['color'])
        ax2.set_xlabel('|Seismic Amplitude|')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title(f'{freq} - |Seis| vs Error', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 相关性
        corr_err = np.corrcoef(np.abs(seis_flat), error_flat)[0, 1]
        ax2.text(0.05, 0.95, f'Corr={corr_err:.3f}', transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Seismic-Impedance Correlation Analysis', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'seismic_impedance_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: seismic_impedance_correlation.png")


def plot_error_spatial_distribution(all_data, output_dir):
    """误差空间分布图"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    freqs = list(all_data.keys())
    
    for i, freq in enumerate(freqs):
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        error = np.abs(pred_imp - true_imp)
        
        n_traces, n_samples = true_imp.shape
        extent = [0, n_traces, n_samples * 0.01, 0]
        
        # 绝对误差分布
        ax1 = axes[0, i]
        err_max = np.percentile(error, 95)
        im1 = ax1.imshow(error.T, aspect='auto', cmap='Reds', extent=extent, vmin=0, vmax=err_max)
        ax1.set_title(f'{freq} Absolute Error', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Trace')
        ax1.set_ylabel('Time (ms)')
        plt.colorbar(im1, ax=ax1)
        
        # 相对误差分布
        ax2 = axes[1, i]
        rel_error = error / (np.abs(true_imp) + 1e-8) * 100
        rel_max = np.percentile(rel_error, 95)
        im2 = ax2.imshow(rel_error.T, aspect='auto', cmap='YlOrRd', extent=extent, vmin=0, vmax=rel_max)
        ax2.set_title(f'{freq} Relative Error (%)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Trace')
        ax2.set_ylabel('Time (ms)')
        plt.colorbar(im2, ax=ax2, label='%')
    
    plt.suptitle('Error Spatial Distribution', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'error_spatial_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: error_spatial_distribution.png")


def plot_pcc_r2_by_trace(all_data, output_dir):
    """逐道PCC和R²分布"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    freqs = list(all_data.keys())
    
    # PCC by trace
    ax1 = axes[0]
    for freq in freqs:
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        n_traces = true_imp.shape[0]
        pcc_by_trace = []
        for t in range(n_traces):
            pcc = np.corrcoef(true_imp[t], pred_imp[t])[0, 1]
            pcc_by_trace.append(pcc)
        
        ax1.plot(range(n_traces), pcc_by_trace, linewidth=1.5, 
                label=f'{freq} (mean={np.mean(pcc_by_trace):.3f})',
                color=FREQ_CONFIGS[freq]['color'])
    
    ax1.set_xlabel('Trace Number')
    ax1.set_ylabel('PCC')
    ax1.set_title('Pearson Correlation Coefficient by Trace', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.9, color='r', linestyle='--', alpha=0.5, label='PCC=0.9')
    ax1.set_ylim(0.8, 1.0)
    
    # R² by trace
    ax2 = axes[1]
    for freq in freqs:
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        n_traces = true_imp.shape[0]
        r2_by_trace = []
        for t in range(n_traces):
            ss_res = np.sum((true_imp[t] - pred_imp[t]) ** 2)
            ss_tot = np.sum((true_imp[t] - np.mean(true_imp[t])) ** 2)
            r2 = 1 - ss_res / ss_tot
            r2_by_trace.append(r2)
        
        ax2.plot(range(n_traces), r2_by_trace, linewidth=1.5, 
                label=f'{freq} (mean={np.mean(r2_by_trace):.3f})',
                color=FREQ_CONFIGS[freq]['color'])
    
    ax2.set_xlabel('Trace Number')
    ax2.set_ylabel('R²')
    ax2.set_title('R² by Trace', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.85, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylim(0.6, 1.0)
    
    plt.suptitle('Per-Trace Performance Metrics', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'pcc_r2_by_trace.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: pcc_r2_by_trace.png")


def create_summary_report(all_data, output_dir):
    """创建汇总报告"""
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("多频率波阻抗反演详细分析报告")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    for freq in all_data.keys():
        true_imp = all_data[freq]['true']
        pred_imp = all_data[freq]['pred']
        
        # 全局指标
        pcc = np.corrcoef(true_imp.flatten(), pred_imp.flatten())[0, 1]
        ss_res = np.sum((true_imp - pred_imp) ** 2)
        ss_tot = np.sum((true_imp - np.mean(true_imp)) ** 2)
        r2 = 1 - ss_res / ss_tot
        mse = np.mean((true_imp - pred_imp) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(true_imp - pred_imp))
        
        # 逐道统计
        n_traces = true_imp.shape[0]
        pcc_list = [np.corrcoef(true_imp[t], pred_imp[t])[0, 1] for t in range(n_traces)]
        
        report_lines.append(f"\n{freq} 模型性能报告")
        report_lines.append("-" * 50)
        report_lines.append(f"  全局指标:")
        report_lines.append(f"    PCC:  {pcc:.4f}")
        report_lines.append(f"    R²:   {r2:.4f}")
        report_lines.append(f"    MSE:  {mse:.4e}")
        report_lines.append(f"    RMSE: {rmse:.4e}")
        report_lines.append(f"    MAE:  {mae:.4e}")
        report_lines.append(f"  逐道PCC统计:")
        report_lines.append(f"    最小: {np.min(pcc_list):.4f}")
        report_lines.append(f"    最大: {np.max(pcc_list):.4f}")
        report_lines.append(f"    均值: {np.mean(pcc_list):.4f}")
        report_lines.append(f"    标准差: {np.std(pcc_list):.4f}")
        report_lines.append(f"  数据形状: {true_imp.shape}")
    
    report_lines.append("\n" + "=" * 70)
    report_lines.append("分析完成")
    report_lines.append("=" * 70)
    
    report_text = "\n".join(report_lines)
    
    with open(output_dir / 'analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(f"保存: analysis_report.txt")
    print(report_text)


# ==================== 主程序 ====================
print("=" * 60)
print("附加详细可视化")
print("=" * 60)

all_data = {}

for freq, config in FREQ_CONFIGS.items():
    print(f"\n加载 {freq}...")
    try:
        seismic, true_imp, pred_imp, norm_stats = load_data_and_predict(config)
        all_data[freq] = {'true': true_imp, 'pred': pred_imp, 'seismic': seismic}
        print(f"  {freq} 加载完成")
    except Exception as e:
        print(f"  {freq} 错误: {e}")

if len(all_data) >= 2:
    print("\n生成附加可视化图表...")
    plot_multi_trace_comparison(all_data, OUTPUT_DIR)
    plot_layer_boundary_analysis(all_data, OUTPUT_DIR)
    plot_seismic_impedance_correlation(all_data, OUTPUT_DIR)
    plot_error_spatial_distribution(all_data, OUTPUT_DIR)
    plot_pcc_r2_by_trace(all_data, OUTPUT_DIR)
    create_summary_report(all_data, OUTPUT_DIR)

print("\n" + "=" * 60)
print("附加可视化完成!")
print("=" * 60)
print(f"\n输出目录: {OUTPUT_DIR}")
