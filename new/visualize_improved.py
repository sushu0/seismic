# -*- coding: utf-8 -*-
"""
ThinLayerNet V2 改进版可视化 - 更好看的配色和细节
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

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# 配置
class Config:
    SEISMIC_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy'
    IMPEDANCE_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt'
    OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2')
    CHECKPOINT_PATH = OUTPUT_DIR / 'checkpoints' / 'best.pt'
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    SEED = 42

CFG = Config()

# 模型定义
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

# 数据加载函数
def load_seismic_data(path):
    with segyio.open(path, "r", ignore_geometry=True, strict=False) as f:
        n_traces = f.tracecount
        data = np.stack([np.copy(f.trace[i]) for i in range(n_traces)])
    return data.astype(np.float32)

def load_impedance_data(path, n_traces):
    raw = np.loadtxt(path, usecols=4, skiprows=1)
    n_samples = len(raw) // n_traces
    return raw.reshape(n_traces, n_samples).astype(np.float32)

def highpass_filter(data, cutoff=12, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)

# 改进的可视化函数
def plot_beautiful_section(true_imp, pred_imp, save_path):
    """绘制更漂亮的剖面对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(22, 8), facecolor='white')
    
    # 使用百分位数裁剪突出细节
    vmin, vmax = np.percentile(true_imp, [3, 97])
    
    # 真实阻抗 - 使用jet colormap
    im0 = axes[0].imshow(true_imp.T, aspect='auto', cmap='jet', 
                         vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[0].set_title('真实阻抗', fontsize=18, fontweight='bold', pad=15)
    axes[0].set_xlabel('道号', fontsize=14)
    axes[0].set_ylabel('采样点', fontsize=14)
    cbar0 = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    cbar0.ax.tick_params(labelsize=11)
    
    # 预测阻抗
    im1 = axes[1].imshow(pred_imp.T, aspect='auto', cmap='jet',
                         vmin=vmin, vmax=vmax, interpolation='bilinear')
    axes[1].set_title('预测阻抗 (ThinLayerNet V2)', fontsize=18, fontweight='bold', pad=15)
    axes[1].set_xlabel('道号', fontsize=14)
    axes[1].set_ylabel('采样点', fontsize=14)
    cbar1 = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=11)
    
    # 误差 - 使用RdBu_r colormap
    error = pred_imp - true_imp
    err_limit = np.percentile(np.abs(error), 97)
    im2 = axes[2].imshow(error.T, aspect='auto', cmap='RdBu_r',
                         vmin=-err_limit, vmax=err_limit, interpolation='bilinear')
    axes[2].set_title('预测误差', fontsize=18, fontweight='bold', pad=15)
    axes[2].set_xlabel('道号', fontsize=14)
    axes[2].set_ylabel('采样点', fontsize=14)
    cbar2 = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(labelsize=11)
    
    plt.suptitle('ThinLayerNet V2 波阻抗反演结果', fontsize=20, y=0.98, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=250, bbox_inches='tight', facecolor='white')
    print(f'✓ 保存: {save_path}')
    plt.close()

def main():
    print("=" * 70)
    print("ThinLayerNet V2 改进版可视化")
    print("=" * 70)
    
    # 创建输出目录
    fig_dir = CFG.OUTPUT_DIR / 'figures_improved'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n[1/4] 加载数据...")
    seismic = load_seismic_data(CFG.SEISMIC_PATH)
    n_traces = seismic.shape[0]
    impedance = load_impedance_data(CFG.IMPEDANCE_PATH, n_traces)
    print(f"  地震数据: {seismic.shape}")
    print(f"  阻抗数据: {impedance.shape}")
    
    # 加载归一化参数
    norm_path = CFG.OUTPUT_DIR / 'norm_stats.json'
    with open(norm_path, 'r') as f:
        norm_params = json.load(f)
    
    # 划分数据集
    np.random.seed(CFG.SEED)
    indices = np.random.permutation(n_traces)
    n_train = int(n_traces * CFG.TRAIN_RATIO)
    n_val = int(n_traces * CFG.VAL_RATIO)
    test_idx = indices[n_train + n_val:]
    print(f"  测试集: {len(test_idx)} 道")
    
    # 加载模型
    print("\n[2/4] 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
    ckpt = torch.load(CFG.CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  模型已加载 (epoch {ckpt['epoch']})")
    
    # 准备输入
    print("\n[3/4] 准备输入...")
    seismic_hf = highpass_filter(seismic, cutoff=12, fs=1000)
    seis_mean = norm_params['seis_mean']
    seis_std = norm_params['seis_std']
    seismic_norm = (seismic - seis_mean) / (seis_std + 1e-6)
    
    # 预测
    print("\n[4/4] 进行预测...")
    all_preds = []
    with torch.no_grad():
        for i in range(n_traces):
            seis_hf_i = seismic_hf[i]
            seis_hf_norm = seis_hf_i / (np.std(seis_hf_i) + 1e-6)
            
            x_orig = torch.tensor(seismic_norm[i:i+1, :], dtype=torch.float32).unsqueeze(0).to(device)
            x_hf = torch.tensor(seis_hf_norm[np.newaxis, :], dtype=torch.float32).unsqueeze(0).to(device)
            x = torch.cat([x_orig, x_hf], dim=1)
            
            pred = model(x)
            all_preds.append(pred.cpu().numpy().squeeze())
    
    pred_norm = np.array(all_preds)
    pred_imp = pred_norm * norm_params['imp_std'] + norm_params['imp_mean']
    print(f"  预测完成: {pred_imp.shape}")
    
    # 生成图像
    print("\n[5/5] 生成可视化...")
    test_true = impedance[test_idx]
    test_pred = pred_imp[test_idx]
    
    plot_beautiful_section(test_true, test_pred, 
                          fig_dir / 'beautiful_comparison_test.png')
    plot_beautiful_section(impedance, pred_imp,
                          fig_dir / 'beautiful_comparison_all.png')
    
    # 计算指标
    pcc = np.corrcoef(test_true.flatten(), test_pred.flatten())[0, 1]
    mse = np.mean((test_true - test_pred)**2)
    print(f"\n测试集 PCC: {pcc:.4f}")
    print(f"测试集 MSE: {mse:.2e}")
    
    print("\n" + "=" * 70)
    print(f"完成！图像已保存到: {fig_dir}")
    print("=" * 70)

if __name__ == '__main__':
    main()
