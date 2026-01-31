# -*- coding: utf-8 -*-
"""
ThinLayerNet V2 可视化脚本
生成训练完成后的预测结果可视化图
"""
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配置 ====================
class Config:
    SEISMIC_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy'
    IMPEDANCE_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt'
    OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2')
    CHECKPOINT_PATH = OUTPUT_DIR / 'checkpoints' / 'best.pt'
    
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    SEED = 42

CFG = Config()

# ==================== 模型定义 (与训练脚本完全一致) ====================
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


# ==================== 数据加载 ====================
def load_seismic_data(path):
    with segyio.open(path, "r", ignore_geometry=True, strict=False) as f:
        n_traces = f.tracecount
        data = np.stack([np.copy(f.trace[i]) for i in range(n_traces)])
    return data.astype(np.float32)


def load_impedance_data(path, n_traces):
    raw = np.loadtxt(path, usecols=4, skiprows=1)
    n_samples = len(raw) // n_traces
    return raw.reshape(n_traces, n_samples).astype(np.float32)


def highpass_filter(data, cutoff=15, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


# ==================== 可视化函数 ====================
def plot_section_comparison(true_imp, pred_imp, title_suffix='', save_path=None):
    """绘制剖面对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # 使用更严格的百分位数裁剪，突出细节
    vmin, vmax = np.percentile(true_imp, [5, 95])
    
    # 真实阻抗 - 使用 jet colormap，色彩更丰富
    im0 = axes[0].imshow(true_imp.T, aspect='auto', cmap='jet', 
                         vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].set_title('真实阻抗', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('道号', fontsize=13)
    axes[0].set_ylabel('采样点', fontsize=13)
    cbar0 = plt.colorbar(im0, ax=axes[0], shrink=0.85)
    cbar0.ax.tick_params(labelsize=10)
    
    # 预测阻抗 - 使用相同的 colormap 和范围
    im1 = axes[1].imshow(pred_imp.T, aspect='auto', cmap='jet',
                         vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].set_title('预测阻抗 (ThinLayerNet V2)', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('道号', fontsize=13)
    axes[1].set_ylabel('采样点', fontsize=13)
    cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.85)
    cbar1.ax.tick_params(labelsize=10)
    
    # 误差 - 使用更严格的裁剪，突出误差模式
    error = pred_imp - true_imp
    err_limit = np.percentile(np.abs(error), 95)
    im2 = axes[2].imshow(error.T, aspect='auto', cmap='RdBu_r',
                         vmin=-err_limit, vmax=err_limit, interpolation='bilinear')
    axes[2].set_title('预测误差', fontsize=16, fontweight='bold')
    axes[2].set_xlabel('道号', fontsize=13)
    axes[2].set_ylabel('采样点', fontsize=13)
    cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.85)
    cbar2.ax.tick_params(labelsize=10)
    
    plt.suptitle(f'ThinLayerNet V2 波阻抗反演结果 {title_suffix}', fontsize=18, y=1.01, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'  保存: {save_path}')
    plt.close()


def plot_trace_comparison(true_imp, pred_imp, trace_indices, save_path=None):
    """绘制单道对比图"""
    n_traces = len(trace_indices)
    fig, axes = plt.subplots(n_traces, 1, figsize=(15, 4*n_traces))
    if n_traces == 1:
        axes = [axes]
    
    # 使用更鲜艳的颜色
    colors_true = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    colors_pred = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#8e44ad']
    
    for i, idx in enumerate(trace_indices):
        ax = axes[i]
        samples = np.arange(true_imp.shape[1])
        color_idx = i % len(colors_true)
        
        ax.plot(samples, true_imp[idx], color=colors_true[color_idx], 
        mae = np.mean(np.abs(true_imp[idx] - pred_imp[idx]))
        
        ax.set_title(f'道 #{idx}  |  PCC={pcc:.4f}  |  MSE={mse:.2e}  |  MAE={mae:.1f}', 
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('采样点', fontsize=12)
        ax.set_ylabel('阻抗值', fontsize=12)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
        ax.grid(True, alpha=0.25, linestyle='--')6, y=1.005, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white{mse:.6f}', fontsize=12)
        ax.set_xlabel('采样点')
        ax.set_ylabel('阻抗值')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('ThinLayerNet V2 单道对比', fontsize=14, y=1.01)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  保存: {save_path}')
    plt.close()

6, 9))
    
    true_grad = np.diff(true_imp[trace_idx])
    pred_grad = np.diff(pred_imp[trace_idx])
    samples = np.arange(len(true_grad))
    
    # 反射系数对比 - 使用渐变填充突出差异
    axes[0].plot(samples, true_grad, color='#2ecc71', label='真实梯度', 
                 linewidth=2.5, alpha=0.9)
    axes[0].plot(samples, pred_grad, color='#e74c3c', linestyle='--', 
                 label='预测梯度', linewidth=2.5, alpha=0.85)
    axes[0].fill_between(samples, true_grad, alpha=0.15, color='#2ecc71')
    axes[0].fill_between(samples, pred_grad, alpha=0.15, color='#e74c3c')
    axes[0].axhline(y=0, color='k', linestyle='-', linewidth=0.8, alpha=0.3)
    axes[0].set_title(f'道 #{trace_idx} 阻抗梯度(反射系数)对比', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('采样点', fontsize=12)
    axes[0].set_ylabel('梯度值', fontsize=12)
    axes[0].legend(fontsize=11, framealpha=0.95, loc='best')
    axes[0].grid(True, alpha=0.25, linestyle='--')
    axes[0].set_facecolor('#f8f9fa')
    
    # 阻抗对比 - 增强视觉效果
    time = np.arange(len(true_imp[trace_idx]))
    axes[1].plot(time, true_imp[trace_idx], color='#3498db', 
                 label='真实阻抗', linewidth=2.5, alpha=0.9)
    axes[1].plot(time, pred_imp[trace_idx], color='#f39c12', 
                 linestyle='--', label='预测阻抗', linewidth=2.5, alpha=0.85)
    axes[1].fill_between(time, true200, bbox_inches='tight', facecolor='white0.12, color='#3498db')
    axes[1].set_title(f'道 #{trace_idx} 阻抗对比', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('采样点', fontsize=12)
    axes[1].set_ylabel('阻抗值', fontsize=12)
    axes[1].legend(fontsize=11, framealpha=0.95, loc='best')
    axes[1].grid(True, alpha=0.25, linestyle='--')
    axes[1].set_facecolor('#f8f9fa'
    axes[1].set_ylabel('阻抗值')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    6, 6))
    
    # 误差直方图 - 使用更漂亮的渐变色
    n, bins, patches = axes[0].hist(error, bins=80, color='steelblue', 
                                      alpha=0.75, edgecolor='#34495e', linewidth=0.5)
    
    # 为直方图添加渐变色
    cm = plt.cm.get_cmap('coolwarm')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    
    axes[0].axvline(x=0, color='#e74c3c', linestyle='--', linewidth=2.5, label='零误差线')
    axes[0].set_title('预测误差分布', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('误差值', fontsize=12)
    axes[0].set_ylabel('频次', fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.25, linestyle='--')
    axes[0].set_facecolor('#f8f9fa')
    
    mean_err = np.mean(error)
    std_err = np.std(error)
    median_err = np.median(error)
    axes[0].text(0.97, 0.97, f'均值: {mean_err:.2f}\n中位数: {median_err:.2f}\n标准差: {std_err:.2f}', 
                 transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
                 horizontalalignment='right', 
                 bbox=dict(boxstyle='round,pad=0.8', facecolor='#ecf0f1', 
                          edgecolor='#34495e', linewidth=1.5, alpha=0.9))
    
    # 散点图 - 使用密度色彩
    sample_idx = np.random.choice(len(error), min(15000, len(error)), replace=False)
    true_sample = true_imp.flatten()[sample_idx]
    pred_sample = pred_imp.flatten()[sample_idx]
    
    # 使用 hexbin 创建密度图
    hb = axes[1].hexbin(true_sample, pred_sample, gridsize=50, cmap='YlOrRd', 
                        mincnt=1, alpha=0.8, edgecolors='none')
    cb = plt.colorbar(hb, ax=axes[1], label='样本密度')
    cb.ax.tick_params(labelsize=10)
    
    # 理想线
    vmin = np.percentile(true_imp, 5)
    vmax = np.percentile(true_imp, 95)
    axes[1].plot([vmin, vmax], [vmin, vmax], color='#2ecc71', 
                 linestyle='--', linewidth=3, label='理想线 y=x', alpha=0.9)
    axes[1].set_title('真实 vs 预测 密度图', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('真实阻抗', fontsize=12)
    axes[1].set_ylabel('预测阻抗', fontsize=12)
    axes[1].legend(fontsize=11, loc='upper left')
    axes[1].grid(True, alpha=0.25, linestyle='--')
    axes[1].set_facecolor('#f8f9fa'
    sample_idx = np.random.choice(len(error), min(10000, len(error)), replace=False)
    axes[1].scatter(true_imp.flatten()[sample_idx], pred_imp.flatten()[sample_idx], 
                    alpha=0.3, s=1, c='steelblue')
    
    # 理想线
    min_val, max_val = true_imp.min(), true_imp.max()
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线 y=x')
    axes[1].set_title('真实 vs 预测 散点图', fontsize=12)
    axes[1].set_xlabel('真实阻抗')
    axes[1].set_ylabel('预测阻抗')
    axes[1].legend()
    axes[1].set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f'  保存: {save_path}')
    plt.close()


def compute_metrics(true_imp, pred_imp):
    """计算评估指标"""
    metrics = {}
    
    # 整体指标
    metrics['mse'] = np.mean((pred_imp - true_imp)**2)
    metrics['mae'] = np.mean(np.abs(pred_imp - true_imp))
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # PCC
    flat_true = true_imp.flatten()
    flat_pred = pred_imp.flatten()
    metrics['pcc'] = np.corrcoef(flat_true, flat_pred)[0, 1]
    
    # R²
    ss_res = np.sum((flat_true - flat_pred)**2)
    ss_tot = np.sum((flat_true - np.mean(flat_true))**2)
    metrics['r2'] = 1 - ss_res / (ss_tot + 1e-10)
    
    # 逐道 PCC
    n_traces = true_imp.shape[0]
    trace_pccs = []
    for i in range(n_traces):
        pcc = np.corrcoef(true_imp[i], pred_imp[i])[0, 1]
        if not np.isnan(pcc):
            trace_pccs.append(pcc)
    metrics['mean_trace_pcc'] = np.mean(trace_pccs)
    metrics['min_trace_pcc'] = np.min(trace_pccs)
    metrics['max_trace_pcc'] = np.max(trace_pccs)
    
    return metrics


def main():
    print("=" * 60)
    print("ThinLayerNet V2 可视化脚本")
    print("=" * 60)
    
    # 创建图像输出目录
    fig_dir = CFG.OUTPUT_DIR / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    seismic = load_seismic_data(CFG.SEISMIC_PATH)
    n_traces = seismic.shape[0]
    impedance = load_impedance_data(CFG.IMPEDANCE_PATH, n_traces)
    print(f"  地震数据: {seismic.shape}")
    print(f"  阻抗数据: {impedance.shape}")
    
    # 加载归一化参数
    norm_path = CFG.OUTPUT_DIR / 'norm_stats.json'
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    print(f"  归一化参数已加载: {norm_path}")
    
    # 划分数据集
    np.random.seed(CFG.SEED)
    indices = np.random.permutation(n_traces)
    n_train = int(n_traces * CFG.TRAIN_RATIO)
    n_val = int(n_traces * CFG.VAL_RATIO)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"  训练集: {len(train_idx)} 道, 验证集: {len(val_idx)} 道, 测试集: {len(test_idx)} 道")
    
    # 加载模型
    print("\n[2/5] 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
    
    # 加载 checkpoint - 使用 weights_only=False 兼容 PyTorch 2.6
    ckpt = torch.load(CFG.CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  模型已加载: {CFG.CHECKPOINT_PATH}")
    print(f"  最佳 epoch: {ckpt.get('epoch', 'N/A')}")
    
    # 准备输入数据
    print("\n[3/5] 准备双通道输入...")
    # 高频通道 - 先对原始数据做滤波（与训练一致）
    seismic_hf = highpass_filter(seismic, cutoff=12, fs=1000)
    
    # 归一化地震数据
    seis_mean = norm_params['seis_mean']
    seis_std = norm_params['seis_std']
    seismic_norm = (seismic - seis_mean) / (seis_std + 1e-6)
    
    # 进行预测
    print("\n[4/5] 进行预测...")
    all_preds = []
    
    with torch.no_grad():
        for i in range(n_traces):
            # 归一化高频通道（与训练一致）
            seis_hf_i = seismic_hf[i]
            seis_hf_norm = seis_hf_i / (np.std(seis_hf_i) + 1e-6)
            
            # 双通道输入
            x_orig = torch.tensor(seismic_norm[i:i+1, :], dtype=torch.float32).unsqueeze(0).to(device)  # (1, 1, L)
            x_hf = torch.tensor(seis_hf_norm[np.newaxis, :], dtype=torch.float32).unsqueeze(0).to(device)
            x = torch.cat([x_orig, x_hf], dim=1)  # (1, 2, L)
            
            pred = model(x)
            all_preds.append(pred.cpu().numpy().squeeze())
    
    pred_norm = np.array(all_preds)
    
    # 反归一化
    imp_mean = norm_params['imp_mean']
    imp_std = norm_params['imp_std']
    pred_imp = pred_norm * imp_std + imp_mean
    
    print(f"  预测完成: {pred_imp.shape}")
    
    # 生成可视化图
    print("\n[5/5] 生成可视化图...")
    
    # 测试集
    test_true = impedance[test_idx]
    test_pred = pred_imp[test_idx]
    
    # 1. 剖面对比图
    print("  绘制剖面对比图...")
    plot_section_comparison(test_true, test_pred, '(测试集)', 
                           fig_dir / 'section_comparison_test.png')
    
    # 全部数据剖面
    plot_section_comparison(impedance, pred_imp, '(全部数据)', 
                           fig_dir / 'section_comparison_all.png')
    
    # 2. 单道对比图
    print("  绘制单道对比图...")
    # 从测试集选几道
    if len(test_idx) >= 5:
        sample_traces = [test_idx[0], test_idx[len(test_idx)//4], 
                        test_idx[len(test_idx)//2], test_idx[3*len(test_idx)//4],
                        test_idx[-1]]
    else:
        sample_traces = list(test_idx)
    
    plot_trace_comparison(impedance, pred_imp, sample_traces,
                         fig_dir / 'trace_comparison.png')
    
    # 3. 梯度对比图
    print("  绘制梯度对比图...")
    plot_gradient_comparison(impedance, pred_imp, test_idx[len(test_idx)//2],
                            fig_dir / 'gradient_comparison.png')
    
    # 4. 误差分布图
    print("  绘制误差分布图...")
    plot_error_histogram(test_true, test_pred,
                        fig_dir / 'error_histogram.png')
    
    # 计算并打印指标
    print("\n" + "=" * 60)
    print("测试集评估指标")
    print("=" * 60)
    metrics = compute_metrics(test_true, test_pred)
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")
    
    # 保存指标
    metrics_path = fig_dir / 'test_metrics.json'
    # 转换为 Python float 以便 JSON 序列化
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"\n指标已保存: {metrics_path}")
    
    print("\n" + "=" * 60)
    print(f"所有图像已保存到: {fig_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
