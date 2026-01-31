# -*- coding: utf-8 -*-
"""
使用优化后的 ImprovedUNet1D 模型进行预测
"""
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 路径配置 ====================
SEISMIC_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy'
IMPEDANCE_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt'
MODEL_PATH = r'D:\SEISMIC_CODING\new\results\01_20Hz_unet1d_optimized\checkpoints\best.pt'
NORM_PATH = r'D:\SEISMIC_CODING\new\results\01_20Hz_unet1d_optimized\norm_stats.json'
OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\output_images')

# ==================== 模型定义 ====================
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, p: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class AttentionGate(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.W_g = nn.Conv1d(ch, ch, 1)
        self.W_x = nn.Conv1d(ch, ch, 1)
        self.psi = nn.Conv1d(ch, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[-1] != x1.shape[-1]:
            diff = x1.shape[-1] - g1.shape[-1]
            if diff > 0:
                g1 = F.pad(g1, (0, diff))
            else:
                g1 = g1[..., :x1.shape[-1]]
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class ImprovedUNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 48, depth: int = 5, out_ch: int = 1, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth)]
        
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(ConvBlock(prev, c, dropout=dropout))
            self.pool.append(nn.MaxPool1d(2))
            prev = c
        
        self.bottleneck = ConvBlock(prev, prev * 2, dropout=dropout)
        prev = prev * 2
        
        self.up = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose1d(prev, c, kernel_size=2, stride=2))
            self.attn.append(AttentionGate(c))
            self.dec.append(ConvBlock(prev, c, dropout=dropout))
            prev = c
        
        self.out = nn.Sequential(
            nn.Conv1d(prev, prev // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(prev // 2, out_ch, kernel_size=1)
        )

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = self.pool[i](x)
        
        x = self.bottleneck(x)
        
        for i in range(self.depth):
            x = self.up[i](x)
            skip = skips[-(i + 1)]
            
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    x = x[..., :skip.shape[-1]]
            
            skip = self.attn[i](x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.dec[i](x)
        
        return self.out(x)


# ==================== 数据加载函数 ====================
def load_seismic_data(path):
    with segyio.open(path, "r", ignore_geometry=True, strict=False) as f:
        n_traces = f.tracecount
        data = np.stack([np.copy(f.trace[i]) for i in range(n_traces)])
        dt = f.bin[segyio.BinField.Interval] * 1e-6
    print(f"加载地震数据: {n_traces} 道, 每道 {data.shape[1]} 采样点")
    print(f"采样间隔 dt = {dt*1000:.2f} ms")
    return data.astype(np.float32), dt


def load_impedance_data(path, n_traces):
    raw_data = np.loadtxt(path, usecols=4, skiprows=1)
    n_samples = len(raw_data) // n_traces
    data = raw_data.reshape(n_traces, n_samples)
    print(f"加载真实波阻抗: {data.shape}")
    return data.astype(np.float32)


# ==================== 预测函数 ====================
def predict():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    seismic, dt = load_seismic_data(SEISMIC_PATH)
    n_traces = seismic.shape[0]
    true_impedance = load_impedance_data(IMPEDANCE_PATH, n_traces)
    
    # 对齐长度
    min_len = min(seismic.shape[1], true_impedance.shape[1])
    seismic = seismic[:, :min_len]
    true_impedance = true_impedance[:, :min_len]
    
    # 加载归一化参数
    with open(NORM_PATH, 'r') as f:
        norm_stats = json.load(f)
    
    # 归一化
    seis_norm = (seismic - norm_stats['seis_mean']) / (norm_stats['seis_std'] + 1e-6)
    
    # 加载模型
    model = ImprovedUNet1D(in_ch=1, base=48, depth=5, out_ch=1).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    print(f"加载 Epoch {checkpoint['epoch']} 的模型权重")
    
    # 预测
    predictions = []
    with torch.no_grad():
        for i in range(n_traces):
            x = torch.from_numpy(seis_norm[i:i+1]).unsqueeze(0).to(device)  # [1, 1, T]
            pred = model(x)
            predictions.append(pred.squeeze().cpu().numpy())
    
    pred_norm = np.stack(predictions)
    
    # 反归一化
    prediction = pred_norm * norm_stats['imp_std'] + norm_stats['imp_mean']
    
    print(f"\n预测波阻抗统计:")
    print(f"  形状: {prediction.shape}")
    print(f"  最小值: {prediction.min():.2f}")
    print(f"  最大值: {prediction.max():.2f}")
    print(f"  平均值: {prediction.mean():.2f}")
    
    return prediction, dt, seismic, true_impedance


# ==================== 可视化函数 ====================
def visualize_results(prediction, dt, seismic, true_impedance):
    n_traces, n_samples = seismic.shape
    time_axis = np.arange(n_samples) * dt * 1000  # ms
    trace_axis = np.arange(n_traces)
    
    # 计算显示范围
    vmin_seis, vmax_seis = np.percentile(seismic, [2, 98])
    vmin_imp = min(np.percentile(true_impedance, 2), np.percentile(prediction, 2))
    vmax_imp = max(np.percentile(true_impedance, 98), np.percentile(prediction, 98))
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    # 地震数据
    im0 = axes[0].imshow(seismic.T, aspect='auto', cmap='seismic',
                          extent=[0, n_traces, time_axis[-1], time_axis[0]],
                          vmin=vmin_seis, vmax=vmax_seis)
    axes[0].set_xlabel('道号')
    axes[0].set_ylabel('时间 (ms)')
    axes[0].set_title('输入地震剖面 (20Hz)')
    plt.colorbar(im0, ax=axes[0], label='振幅')
    
    # 真实波阻抗
    im1 = axes[1].imshow(true_impedance.T, aspect='auto', cmap='jet',
                          extent=[0, n_traces, time_axis[-1], time_axis[0]],
                          vmin=vmin_imp, vmax=vmax_imp)
    axes[1].set_xlabel('道号')
    axes[1].set_ylabel('时间 (ms)')
    axes[1].set_title('真实波阻抗剖面')
    plt.colorbar(im1, ax=axes[1], label='波阻抗 (m/s·g/cm³)')
    
    # 预测波阻抗
    im2 = axes[2].imshow(prediction.T, aspect='auto', cmap='jet',
                          extent=[0, n_traces, time_axis[-1], time_axis[0]],
                          vmin=vmin_imp, vmax=vmax_imp)
    axes[2].set_xlabel('道号')
    axes[2].set_ylabel('时间 (ms)')
    axes[2].set_title('ImprovedUNet1D 预测波阻抗剖面 (优化版)')
    plt.colorbar(im2, ax=axes[2], label='波阻抗 (m/s·g/cm³)')
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / '01_20Hz_prediction_optimized.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n剖面图保存到: {save_path}")
    plt.close()
    
    # 单道对比
    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10))
    trace_indices = [25, 50, 75]
    
    for ax, idx in zip(axes2, trace_indices):
        ax.plot(time_axis, true_impedance[idx], 'b-', label='真实波阻抗', linewidth=1)
        ax.plot(time_axis, prediction[idx], 'r--', label='预测波阻抗', linewidth=1, alpha=0.8)
        ax.set_xlabel('时间 (ms)')
        ax.set_ylabel('波阻抗')
        ax.set_title(f'道号 {idx} 波阻抗对比')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path2 = OUTPUT_DIR / '01_20Hz_traces_optimized.png'
    plt.savefig(save_path2, dpi=150, bbox_inches='tight')
    print(f"单道对比图保存到: {save_path2}")
    plt.close()


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("ImprovedUNet1D (优化版) 地震波阻抗预测")
    print("=" * 60)
    
    prediction, dt, seismic, true_impedance = predict()
    visualize_results(prediction, dt, seismic, true_impedance)
    
    # 计算指标
    pred_flat = prediction.flatten()
    true_flat = true_impedance.flatten()
    
    mse = np.mean((pred_flat - true_flat) ** 2)
    pcc = np.corrcoef(pred_flat, true_flat)[0, 1]
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\n全剖面评估指标:")
    print(f"  MSE: {mse:.2f}")
    print(f"  PCC: {pcc:.4f}")
    print(f"  R²:  {r2:.4f}")
    
    print("\n预测完成!")


if __name__ == '__main__':
    main()
