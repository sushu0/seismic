# -*- coding: utf-8 -*-
"""
多频率波阻抗剖面对比图
仿照用户提供的图像样式绘制 20Hz, 30Hz, 40Hz, 50Hz 的真实图和预测图
"""
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.signal import butter, filtfilt
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
# 使用优化后的模型版本：20Hz v2, 30Hz v3, 40Hz v2, 50Hz v2
FREQ_CONFIGS = {
    '20Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2\norm_stats.json',
        'highpass_cutoff': 8,
    },
    '30Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v3\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v3\norm_stats.json',
        'highpass_cutoff': 12,
    },
    '40Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_40Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_40Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_40Hz_thinlayer_v2\norm_stats.json',
        'highpass_cutoff': 15,
    },
    '50Hz': {
        'seismic': r'D:\SEISMIC_CODING\zmy_data\01\data\01_50Hz_re.sgy',
        'impedance': r'D:\SEISMIC_CODING\zmy_data\01\data\01_50Hz_04.txt',
        'model': r'D:\SEISMIC_CODING\new\results\01_50Hz_thinlayer_v2\checkpoints\best.pt',
        'norm': r'D:\SEISMIC_CODING\new\results\01_50Hz_thinlayer_v2\norm_stats.json',
        'highpass_cutoff': 20,
    },
}

OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\results\multi_freq_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")


def load_data_and_predict(config):
    """加载数据并进行预测"""
    # 加载地震数据
    with segyio.open(config['seismic'], 'r', ignore_geometry=True) as f:
        seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)
    
    # 加载阻抗数据
    raw_imp = np.loadtxt(config['impedance'], usecols=4, skiprows=1).astype(np.float32)
    n_traces = seismic.shape[0]
    n_samples = len(raw_imp) // n_traces
    impedance = raw_imp.reshape(n_traces, n_samples)
    
    # 加载归一化参数
    with open(config['norm'], 'r') as f:
        norm_stats = json.load(f)
    
    # 预处理
    seis_norm = (seismic - norm_stats['seis_mean']) / norm_stats['seis_std']
    seismic_hf = highpass_filter(seismic, cutoff=config['highpass_cutoff'], fs=1000)
    seis_hf_norm = seismic_hf / (np.std(seismic_hf, axis=1, keepdims=True) + 1e-6)
    
    # 检查模型输入通道数
    ckpt = torch.load(config['model'], map_location=device, weights_only=False)
    input_shape = ckpt['model']['input_conv.0.weight'].shape
    in_ch = input_shape[1]  # 获取输入通道数
    
    # 加载模型
    model = ThinLayerNetV2(in_ch=in_ch, base_ch=64, out_ch=1).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # 推理
    all_pred = []
    with torch.no_grad():
        for i in range(n_traces):
            if in_ch == 1:
                # 单通道输入
                x = seis_norm[i:i+1]
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            else:
                # 双通道输入
                x = np.stack([seis_norm[i], seis_hf_norm[i]], axis=0)
                x_t = torch.from_numpy(x).unsqueeze(0).to(device)
            pred = model(x_t)
            all_pred.append(pred.cpu().numpy().squeeze())
    
    pred_full = np.array(all_pred)
    pred_full_denorm = pred_full * norm_stats['imp_std'] + norm_stats['imp_mean']
    
    return impedance, pred_full_denorm


def plot_impedance_section(data, title, output_path, use_chinese=True):
    """绘制单个波阻抗剖面图"""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    n_traces, n_samples = data.shape
    dt_ms = 0.01  # 采样间隔 ms
    total_time = n_samples * dt_ms
    
    # 使用道号作为X轴
    extent = [0, n_traces, total_time, 0]
    
    # 使用viridis色图，与用户提供的图一致
    im = ax.imshow(data.T, aspect='auto', cmap='viridis', extent=extent,
                   vmin=6e6, vmax=1e7)
    
    if use_chinese:
        ax.set_xlabel('道号', fontsize=14)
        ax.set_ylabel('时间 (ms)', fontsize=14)
        cbar = plt.colorbar(im, ax=ax, label='波阻抗 (m/s*g/cm3)')
    else:
        ax.set_xlabel('Shot Number', fontsize=14)
        ax.set_ylabel('Time (ms)', fontsize=14)
        cbar = plt.colorbar(im, ax=ax, label='impedance (m/s*g/cm3)')
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.tick_params(labelsize=12)
    
    # 调整colorbar的科学计数法显示
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path.name}")


def plot_comparison_grid(true_data_dict, pred_data_dict, output_path):
    """绘制4x2网格对比图（真实 vs 预测）"""
    fig, axes = plt.subplots(4, 2, figsize=(20, 24))
    
    freqs = ['20Hz', '30Hz', '40Hz', '50Hz']
    
    for i, freq in enumerate(freqs):
        true_data = true_data_dict[freq]
        pred_data = pred_data_dict[freq]
        
        n_traces, n_samples = true_data.shape
        dt_ms = 0.01
        total_time = n_samples * dt_ms
        extent = [0, n_traces, total_time, 0]
        
        # 真实图
        im0 = axes[i, 0].imshow(true_data.T, aspect='auto', cmap='viridis', 
                                 extent=extent, vmin=6e6, vmax=1e7)
        axes[i, 0].set_title(f'{freq} 真实波阻抗剖面', fontsize=14, fontweight='bold')
        axes[i, 0].set_xlabel('道号', fontsize=12)
        axes[i, 0].set_ylabel('时间 (ms)', fontsize=12)
        cbar0 = plt.colorbar(im0, ax=axes[i, 0], label='波阻抗 (m/s*g/cm3)')
        cbar0.formatter.set_powerlimits((0, 0))
        cbar0.update_ticks()
        
        # 预测图
        im1 = axes[i, 1].imshow(pred_data.T, aspect='auto', cmap='viridis', 
                                 extent=extent, vmin=6e6, vmax=1e7)
        axes[i, 1].set_title(f'{freq} 预测波阻抗剖面', fontsize=14, fontweight='bold')
        axes[i, 1].set_xlabel('道号', fontsize=12)
        axes[i, 1].set_ylabel('时间 (ms)', fontsize=12)
        cbar1 = plt.colorbar(im1, ax=axes[i, 1], label='波阻抗 (m/s*g/cm3)')
        cbar1.formatter.set_powerlimits((0, 0))
        cbar1.update_ticks()
    
    plt.suptitle('多频率波阻抗反演对比 (真实 vs 预测)', fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: {output_path.name}")


# ==================== 主程序 ====================
print("="*60)
print("多频率波阻抗剖面绘制")
print("="*60)

true_data_dict = {}
pred_data_dict = {}

for freq, config in FREQ_CONFIGS.items():
    print(f"\n处理 {freq}...")
    try:
        true_imp, pred_imp = load_data_and_predict(config)
        true_data_dict[freq] = true_imp
        pred_data_dict[freq] = pred_imp
        
        # 绘制单独的真实图和预测图
        plot_impedance_section(true_imp, f'{freq} 真实波阻抗剖面', 
                              OUTPUT_DIR / f'{freq}_true.png')
        plot_impedance_section(pred_imp, f'{freq} 预测波阻抗剖面', 
                              OUTPUT_DIR / f'{freq}_pred.png')
        
        print(f"  {freq} 完成")
    except Exception as e:
        print(f"  {freq} 错误: {e}")

# 绘制4x2网格对比图
if len(true_data_dict) == 4:
    print("\n绘制对比网格图...")
    plot_comparison_grid(true_data_dict, pred_data_dict, 
                        OUTPUT_DIR / 'multi_freq_comparison_grid.png')

print("\n" + "="*60)
print("绘图完成!")
print("="*60)
print(f"\n输出目录: {OUTPUT_DIR}")
