# -*- coding: utf-8 -*-
"""
针对 0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy 文件的地震反演训练与可视化
- 使用物理约束的自监督学习方法
- 从地震数据积分生成伪阻抗标签
- 使用 UNet1D 模型 + 物理正演约束
- CUDA 加速训练
- 反演结果可视化
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.signal import hilbert
from scipy.ndimage import gaussian_filter, uniform_filter1d
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from tqdm import tqdm
import segyio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================== 工具函数 ==============================

def ricker_wavelet(f0, dt, length):
    """生成Ricker子波"""
    t = np.arange(-length/2, length/2 + dt, dt, dtype=np.float64)
    pi2 = np.pi ** 2
    w = (1.0 - 2.0*pi2*(f0**2)*(t**2)) * np.exp(-pi2*(f0**2)*(t**2))
    return w.astype(np.float32)

# ============================== 模型定义 ==============================

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )
    def forward(self, x):
        return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_ch=1, base=48, depth=5, out_ch=1):
        super().__init__()
        self.depth = depth
        chs = [base * (2**i) for i in range(depth)]
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(ConvBlock(prev, c))
            self.pool.append(nn.MaxPool1d(2))
            prev = c
        self.bottleneck = ConvBlock(prev, prev * 2)
        prev = prev * 2
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose1d(prev, c, kernel_size=2, stride=2))
            self.dec.append(ConvBlock(prev, c))
            prev = c
        self.out = nn.Conv1d(prev, out_ch, kernel_size=1)

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
            x = torch.cat([x, skip], dim=1)
            x = self.dec[i](x)
        return self.out(x)

class ForwardModel(nn.Module):
    """物理正演模型：阻抗 -> 反射系数 -> 地震道"""
    def __init__(self, wavelet, eps=1e-6):
        super().__init__()
        assert wavelet.ndim == 1
        self.register_buffer("wavelet", wavelet.clone().detach().float())
        self.eps = eps

    def reflectivity(self, imp):
        imp_prev = torch.roll(imp, shifts=1, dims=-1)
        num = imp - imp_prev
        den = imp + imp_prev + self.eps
        r = num / den
        r[..., 0] = 0.0
        return r

    def forward(self, imp):
        r = self.reflectivity(imp)
        w = self.wavelet.view(1, 1, -1)
        pad = (w.shape[-1] - 1) // 2
        s = F.conv1d(r, w, padding=pad)
        if s.shape[-1] != r.shape[-1]:
            s = s[..., :r.shape[-1]]
        return s

# ============================== 数据集 ==============================

class SeismicTraceDataset(Dataset):
    """地震道数据集"""
    def __init__(self, seis, imp=None):
        self.seis = seis.astype(np.float32)
        self.imp = imp.astype(np.float32) if imp is not None else None

    def __len__(self):
        return self.seis.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.seis[idx]).unsqueeze(0)  # [1, T]
        if self.imp is not None:
            y = torch.from_numpy(self.imp[idx]).unsqueeze(0)  # [1, T]
            return {"seis": x, "imp": y}
        return {"seis": x}

# ============================== 数据准备 ==============================

def load_sgy_data(sgy_path, max_traces=None):
    """加载SGY文件"""
    print(f"正在读取: {sgy_path}")
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as f:
        n_traces = f.tracecount
        n_samples = len(f.samples)
        dt = f.bin[segyio.BinField.Interval]
        print(f"  总道数: {n_traces}, 采样点: {n_samples}, dt: {dt}μs")
        
        if max_traces and n_traces > max_traces:
            # 均匀抽样
            indices = np.linspace(0, n_traces - 1, max_traces, dtype=int)
            seismic = np.stack([f.trace[int(i)] for i in indices], axis=0).astype(np.float32)
            print(f"  抽样 {max_traces} 道用于训练")
        else:
            seismic = segyio.tools.collect(f.trace[:]).astype(np.float32)
        
    return seismic, n_traces, n_samples, dt


def prepare_training_data(seismic, trace_len=1024, stride=256, train_ratio=0.8):
    """
    准备训练数据：
    1. 将长地震道切割成固定长度的片段
    2. 利用地震积分法生成伪阻抗标签
    3. 分割训练/验证集
    """
    n_traces, n_samples = seismic.shape
    print(f"原始数据: {n_traces} 道 × {n_samples} 采样点")
    
    # 如果采样点>trace_len,切片; 否则直接使用
    if n_samples > trace_len:
        segments_seis = []
        for start in range(0, n_samples - trace_len + 1, stride):
            seg = seismic[:, start:start + trace_len]
            segments_seis.append(seg)
        seismic_all = np.concatenate(segments_seis, axis=0)
    else:
        trace_len = n_samples
        seismic_all = seismic
    
    print(f"切片后: {seismic_all.shape[0]} 道 × {trace_len} 采样点")
    
    # Z-score 归一化地震数据
    seis_mean = seismic_all.mean()
    seis_std = seismic_all.std() + 1e-8
    seismic_norm = (seismic_all - seis_mean) / seis_std
    
    # 生成伪阻抗标签(积分法)
    # 用地震道的积分近似相对阻抗变化
    imp_relative = np.cumsum(seismic_all, axis=1)
    
    # 使用指数积分：阻抗 ∝ exp(2 * ∫反射系数)
    # r(t) ≈ s(t) 的简化关系下
    imp_exp = np.exp(2.0 * np.cumsum(seismic_norm * 0.01, axis=1))
    
    # 缩放到合理的阻抗范围 (类似于Marmousi2的范围)
    base_impedance = 5000.0  # 基准阻抗 kg/m²·s
    pseudo_imp = base_impedance * imp_exp
    
    # 对伪阻抗进行平滑以减少噪声
    for i in range(pseudo_imp.shape[0]):
        pseudo_imp[i] = gaussian_filter(pseudo_imp[i], sigma=3)
    
    # 归一化阻抗
    imp_mean = pseudo_imp.mean()
    imp_std = pseudo_imp.std() + 1e-8
    imp_norm = (pseudo_imp - imp_mean) / imp_std
    
    # 分割训练/验证集
    n_total = seismic_norm.shape[0]
    indices = np.random.permutation(n_total)
    n_train = int(n_total * train_ratio)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    train_seis = seismic_norm[train_idx]
    train_imp = imp_norm[train_idx]
    val_seis = seismic_norm[val_idx]
    val_imp = imp_norm[val_idx]
    
    stats = {
        'seis_mean': float(seis_mean),
        'seis_std': float(seis_std),
        'imp_mean': float(imp_mean),
        'imp_std': float(imp_std),
        'trace_len': trace_len,
        'base_impedance': base_impedance,
    }
    
    print(f"训练集: {train_seis.shape[0]} 道")
    print(f"验证集: {val_seis.shape[0]} 道")
    print(f"地震归一化: mean={seis_mean:.4e}, std={seis_std:.4e}")
    print(f"阻抗归一化: mean={imp_mean:.4e}, std={imp_std:.4e}")
    
    return train_seis, train_imp, val_seis, val_imp, stats


# ============================== 训练 ==============================

def train_model(model, fm, train_loader, val_loader, device, stats, output_dir,
                epochs=150, lr=5e-4, weight_decay=1e-5, lambda_phys=0.1,
                lambda_freq=0.05, grad_clip=1.0):
    """训练模型"""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    mse_loss = nn.MSELoss()
    l1_loss = nn.SmoothL1Loss()
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    ckpt_dir = output_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n开始训练 (epochs={epochs}, lr={lr})...")
    print(f"物理约束权重: {lambda_phys}, 频域约束权重: {lambda_freq}")
    print(f"设备: {device}")
    
    for epoch in range(1, epochs + 1):
        # ---- 训练 ----
        model.train()
        epoch_loss = 0.0
        epoch_sup = 0.0
        epoch_phys = 0.0
        n_batches = 0
        
        for batch in train_loader:
            seis = batch['seis'].to(device)   # [B, 1, T]
            imp_gt = batch['imp'].to(device)  # [B, 1, T]
            
            # 前向传播
            imp_pred = model(seis)
            
            # 监督损失
            loss_sup = l1_loss(imp_pred, imp_gt) + 0.5 * mse_loss(imp_pred, imp_gt)
            
            # 物理约束损失：预测阻抗 -> 正演地震道 -> 与观测对比
            loss_phys = torch.tensor(0.0, device=device)
            if lambda_phys > 0:
                # 将预测阻抗反归一化到物理域
                imp_phys = imp_pred * stats['imp_std'] + stats['imp_mean']
                seis_synth = fm(imp_phys)
                # 归一化合成地震道
                seis_synth_norm = (seis_synth - seis_synth.mean()) / (seis_synth.std() + 1e-8)
                seis_norm = (seis - seis.mean()) / (seis.std() + 1e-8)
                loss_phys = mse_loss(seis_synth_norm, seis_norm)
            
            # 频域损失
            loss_freq = torch.tensor(0.0, device=device)
            if lambda_freq > 0:
                # 简化的频域损失：比较FFT幅度谱
                pred_fft = torch.fft.rfft(imp_pred, dim=-1).abs()
                gt_fft = torch.fft.rfft(imp_gt, dim=-1).abs()
                loss_freq = mse_loss(pred_fft, gt_fft)
            
            total_loss = loss_sup + lambda_phys * loss_phys + lambda_freq * loss_freq
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            epoch_loss += total_loss.item()
            epoch_sup += loss_sup.item()
            epoch_phys += loss_phys.item()
            n_batches += 1
        
        scheduler.step()
        avg_train_loss = epoch_loss / n_batches
        train_losses.append(avg_train_loss)
        
        # ---- 验证 ----
        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for batch in val_loader:
                seis = batch['seis'].to(device)
                imp_gt = batch['imp'].to(device)
                imp_pred = model(seis)
                loss = mse_loss(imp_pred, imp_gt)
                val_loss += loss.item()
                n_val += 1
        
        avg_val_loss = val_loss / max(n_val, 1)
        val_losses.append(avg_val_loss)
        
        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'stats': stats,
            }, ckpt_dir / 'best.pt')
        
        # 日志
        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"Train: {avg_train_loss:.6f} (sup={epoch_sup/n_batches:.6f}, phys={epoch_phys/n_batches:.6f}) | "
                  f"Val: {avg_val_loss:.6f} | LR: {lr_now:.2e}")
    
    # 保存最终模型
    torch.save({
        'epoch': epochs,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'val_loss': avg_val_loss,
        'stats': stats,
    }, ckpt_dir / 'last.pt')
    
    print(f"\n训练完成! 最优验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses


# ============================== 推理 ==============================

def infer_all_traces(model, sgy_path, stats, device, batch_size=128, max_traces=None):
    """对所有道进行推理"""
    
    print("\n加载地震数据进行推理...")
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as f:
        n_traces_total = f.tracecount
        n_samples = len(f.samples)
        
        if max_traces and n_traces_total > max_traces:
            # 均匀抽样推理
            indices = np.linspace(0, n_traces_total - 1, max_traces, dtype=int)
            seismic = np.stack([f.trace[int(i)] for i in indices], axis=0).astype(np.float32)
            print(f"  抽样推理: {max_traces} 道")
        else:
            seismic = segyio.tools.collect(f.trace[:]).astype(np.float32)
            print(f"  全部推理: {n_traces_total} 道")
    
    # 归一化
    seismic_norm = (seismic - stats['seis_mean']) / stats['seis_std']
    
    # 如果训练时trace_len < n_samples，需要滑窗推理
    trace_len = stats['trace_len']
    
    model.eval()
    all_preds = np.zeros_like(seismic)  # [N, T]
    count = np.zeros_like(seismic)      # 用于重叠平均
    
    print(f"  推理中... (trace_len={trace_len}, total_samples={n_samples})")
    
    with torch.no_grad():
        if n_samples <= trace_len:
            # 直接推理
            for i in tqdm(range(0, seismic_norm.shape[0], batch_size), desc="推理"):
                batch = seismic_norm[i:i+batch_size]
                x = torch.from_numpy(batch).unsqueeze(1).float().to(device)  # [B, 1, T]
                pred = model(x)
                all_preds[i:i+batch_size] = pred[:, 0, :].cpu().numpy()
                count[i:i+batch_size] = 1.0
        else:
            # 滑窗推理
            stride = trace_len // 2
            for start in range(0, n_samples - trace_len + 1, stride):
                end = start + trace_len
                seg = seismic_norm[:, start:end]
                
                for i in range(0, seg.shape[0], batch_size):
                    batch = seg[i:i+batch_size]
                    x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
                    pred = model(x)
                    all_preds[i:i+batch_size, start:end] += pred[:, 0, :].cpu().numpy()
                    count[i:i+batch_size, start:end] += 1.0
            
            # 处理最后一个窗口
            if (n_samples - trace_len) % stride != 0:
                start = n_samples - trace_len
                seg = seismic_norm[:, start:]
                for i in range(0, seg.shape[0], batch_size):
                    batch = seg[i:i+batch_size]
                    x = torch.from_numpy(batch).unsqueeze(1).float().to(device)
                    pred = model(x)
                    all_preds[i:i+batch_size, start:] += pred[:, 0, :].cpu().numpy()
                    count[i:i+batch_size, start:] += 1.0
    
    # 重叠区域取平均
    count = np.maximum(count, 1.0)
    all_preds /= count
    
    # 反归一化
    impedance = all_preds * stats['imp_std'] + stats['imp_mean']
    
    # 后处理平滑
    impedance = gaussian_filter(impedance, sigma=[3, 5])
    
    print(f"  反演结果: shape={impedance.shape}, range=[{impedance.min():.2f}, {impedance.max():.2f}]")
    
    return seismic, impedance, n_traces_total


# ============================== 可视化 ==============================

def create_seismic_colormap():
    """鲜艳红蓝色图"""
    colors = [
        (0.0, '#00008B'),
        (0.2, '#0066FF'),
        (0.4, '#99CCFF'),
        (0.5, '#FFFFFF'),
        (0.6, '#FFAAAA'),
        (0.8, '#FF3300'),
        (1.0, '#8B0000'),
    ]
    return LinearSegmentedColormap.from_list('seismic_vivid', [(c[0], c[1]) for c in colors])


def plot_results(seismic, impedance, n_traces_total, time_start, time_end, output_dir):
    """生成反演结果可视化"""
    n_traces_vis, n_samples = seismic.shape
    
    # 下采样用于可视化
    target_traces = min(3000, n_traces_vis)
    step = max(1, n_traces_vis // target_traces)
    seis_ds = seismic[::step, :].T    # [samples, traces]
    imp_ds = impedance[::step, :].T   # [samples, traces]
    
    seismic_cmap = create_seismic_colormap()
    
    # ---- 图1: 原始地震剖面 ----
    fig, ax = plt.subplots(figsize=(18, 10))
    vmax = np.percentile(np.abs(seis_ds), 99)
    im = ax.imshow(seis_ds, aspect='auto', cmap=seismic_cmap,
                   vmin=-vmax, vmax=vmax,
                   extent=[0, n_traces_total, time_end, time_start],
                   interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('地震剖面 - 原始数据\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', 
                 fontsize=16, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('振幅', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'sgy_seismic_original.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: sgy_seismic_original.png")
    
    # ---- 图2: 反演阻抗剖面 (同样式) ----
    fig, ax = plt.subplots(figsize=(18, 10))
    vmin = np.percentile(imp_ds, 1)
    vmax = np.percentile(imp_ds, 99)
    im = ax.imshow(imp_ds, aspect='auto', cmap=seismic_cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[0, n_traces_total, time_end, time_start],
                   interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('波阻抗反演剖面\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', 
                 fontsize=16, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('波阻抗 (kg/m²·s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'sgy_impedance_inversion.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: sgy_impedance_inversion.png")
    
    # ---- 图3: 训练损失曲线 ----
    # (在main中单独绘制)
    
    return seis_ds, imp_ds


def plot_training_curves(train_losses, val_losses, output_dir):
    """绘制训练损失曲线"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, 'b-', label='训练损失', linewidth=1.5)
    ax.plot(val_losses, 'r-', label='验证损失', linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    ax.set_title('训练过程损失变化', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  已保存: training_loss.png")


# ============================== 主函数 ==============================

def main():
    # ==================== 配置 ====================
    SGY_FILE = Path(r'D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy')
    OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\sgy_inversion_output')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 时间参数
    TIME_START = 2500  # ms
    TIME_END = 6000    # ms
    DT_S = 0.002       # 采样间隔 2ms = 0.002s
    
    # 模型超参数
    MODEL_BASE = 48     # UNet1D基础通道数
    MODEL_DEPTH = 5     # 网络深度
    
    # 训练超参数
    TRAIN_TRACES = 8000     # 用于训练的道数
    TRACE_LEN = 1024        # 每道切片长度
    STRIDE = 512            # 切片步长
    BATCH_SIZE = 32
    EPOCHS = 150
    LR = 5e-4
    LAMBDA_PHYS = 0.1       # 物理约束权重
    LAMBDA_FREQ = 0.05      # 频域约束权重
    
    # 推理参数
    INFER_TRACES = 5000     # 推理道数（均匀采样）
    
    # 物理参数
    WAVELET_F0 = 25.0       # Ricker子波主频 (Hz)
    WAVELET_LENGTH = 0.128  # 子波持续时间 (s)
    
    # 随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=" * 60)
    print(f"地震反演训练系统")
    print(f"=" * 60)
    print(f"设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SGY文件: {SGY_FILE}")
    print(f"输出目录: {OUTPUT_DIR}")
    
    # ==================== 步骤1: 加载SGY数据 ====================
    print(f"\n{'='*60}")
    print("步骤1: 加载SGY数据")
    print(f"{'='*60}")
    
    seismic_raw, n_traces_total, n_samples, dt = load_sgy_data(SGY_FILE, max_traces=TRAIN_TRACES)
    print(f"加载数据: {seismic_raw.shape}")
    
    # ==================== 步骤2: 准备训练数据 ====================
    print(f"\n{'='*60}")
    print("步骤2: 准备训练数据（自监督伪标签）")
    print(f"{'='*60}")
    
    train_seis, train_imp, val_seis, val_imp, stats = prepare_training_data(
        seismic_raw, trace_len=TRACE_LEN, stride=STRIDE, train_ratio=0.85
    )
    
    # 保存归一化参数
    with open(OUTPUT_DIR / 'norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 创建数据加载器
    train_ds = SeismicTraceDataset(train_seis, train_imp)
    val_ds = SeismicTraceDataset(val_seis, val_imp)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=0, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE*2, shuffle=False, 
                            num_workers=0, pin_memory=True)
    
    # ==================== 步骤3: 构建模型 ====================
    print(f"\n{'='*60}")
    print("步骤3: 构建模型")
    print(f"{'='*60}")
    
    model = UNet1D(in_ch=1, base=MODEL_BASE, depth=MODEL_DEPTH).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型: UNet1D (base={MODEL_BASE}, depth={MODEL_DEPTH})")
    print(f"参数量: {n_params:,}")
    
    # 构建物理正演模型
    wavelet = ricker_wavelet(f0=WAVELET_F0, dt=DT_S, length=WAVELET_LENGTH)
    wavelet_t = torch.from_numpy(wavelet).to(device)
    fm = ForwardModel(wavelet_t).to(device)
    print(f"物理正演模型: Ricker子波 f0={WAVELET_F0}Hz, dt={DT_S}s")
    
    # ==================== 步骤4: 训练 ====================
    print(f"\n{'='*60}")
    print("步骤4: CUDA训练")
    print(f"{'='*60}")
    
    train_losses, val_losses = train_model(
        model=model,
        fm=fm,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        stats=stats,
        output_dir=OUTPUT_DIR,
        epochs=EPOCHS,
        lr=LR,
        lambda_phys=LAMBDA_PHYS,
        lambda_freq=LAMBDA_FREQ,
    )
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, OUTPUT_DIR)
    
    # ==================== 步骤5: 加载最优模型并推理 ====================
    print(f"\n{'='*60}")
    print("步骤5: 加载最优模型, 反演推理")
    print(f"{'='*60}")
    
    # 加载最优模型
    best_ckpt = torch.load(OUTPUT_DIR / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model'])
    print(f"已加载最优模型 (epoch={best_ckpt['epoch']}, val_loss={best_ckpt['val_loss']:.6f})")
    
    # 对全部数据进行推理
    seismic_all, impedance_all, n_total = infer_all_traces(
        model, SGY_FILE, stats, device, batch_size=128, max_traces=INFER_TRACES
    )
    
    # 保存推理结果
    np.save(OUTPUT_DIR / 'seismic.npy', seismic_all)
    np.save(OUTPUT_DIR / 'impedance_pred.npy', impedance_all)
    print(f"  已保存推理结果 numpy 文件")
    
    # ==================== 步骤6: 可视化 ====================
    print(f"\n{'='*60}")
    print("步骤6: 可视化反演结果")
    print(f"{'='*60}")
    
    plot_results(seismic_all, impedance_all, n_total, TIME_START, TIME_END, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print(f"全部完成! 结果保存在: {OUTPUT_DIR}")
    print(f"{'='*60}")
    print(f"\n生成的文件:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}")


if __name__ == '__main__':
    main()
