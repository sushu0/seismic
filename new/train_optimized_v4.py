# -*- coding: utf-8 -*-
"""
优化版波阻抗反演模型 V4
改进：
1. CBAM注意力机制
2. 多尺度特征融合
3. 组合损失函数（MSE + 梯度损失 + 频域损失）
4. 余弦退火学习率调度
5. 更强的数据增强
6. 更长的训练周期
"""
import os
import json
import random
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from scipy.signal import butter, filtfilt
from pathlib import Path
from datetime import datetime
import math

# ==================== 配置 ====================
FREQ = '30Hz'  # 可修改为 20Hz, 40Hz, 50Hz
FREQ_CONFIGS = {
    '20Hz': {'highpass_cutoff': 8, 'thin_min': 8, 'thin_max': 50},
    '30Hz': {'highpass_cutoff': 12, 'thin_min': 6, 'thin_max': 35},
    '40Hz': {'highpass_cutoff': 15, 'thin_min': 5, 'thin_max': 25},
    '50Hz': {'highpass_cutoff': 20, 'thin_min': 4, 'thin_max': 20},
}

config = FREQ_CONFIGS[FREQ]
HIGHPASS_CUTOFF = config['highpass_cutoff']
THIN_MIN = config['thin_min']
THIN_MAX = config['thin_max']

SEISMIC_PATH = rf'D:\SEISMIC_CODING\zmy_data\01\data\01_{FREQ}_re.sgy'
IMPEDANCE_PATH = rf'D:\SEISMIC_CODING\zmy_data\01\data\01_{FREQ}_04.txt'
OUTPUT_DIR = Path(rf'D:\SEISMIC_CODING\new\results\01_{FREQ}_optimized_v4')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'checkpoints').mkdir(exist_ok=True)

# 训练参数
EPOCHS = 800
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")
print(f"频率: {FREQ}, 高通截止: {HIGHPASS_CUTOFF}Hz")


# ==================== 注意力模块 ====================
class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        out = self.sigmoid(avg_out + max_out).view(b, c, 1)
        return x * out


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv(out))
        return x * out


class CBAM(nn.Module):
    """CBAM注意力模块"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# ==================== 多尺度特征提取 ====================
class MultiScaleDilatedConv(nn.Module):
    """多尺度空洞卷积"""
    def __init__(self, in_ch, out_ch, dilations=[1, 2, 4, 8, 16]):
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
        self.attention = CBAM(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        branches = [branch(x) for branch in self.branches]
        out = torch.cat(branches, dim=1)
        out = self.fusion(out)
        out = self.attention(out)
        return out + self.skip(x)


class BoundaryEnhanceModule(nn.Module):
    """边界增强模块"""
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


class EnhancedResBlock(nn.Module):
    """增强残差块：带CBAM注意力"""
    def __init__(self, in_ch, out_ch, use_attention=True):
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
        self.attention = CBAM(out_ch) if use_attention else nn.Identity()
        self.boundary = BoundaryEnhanceModule(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.attention(out)
        out = self.boundary(out)
        out = self.dropout(out)
        return out + self.skip(x)


# ==================== 优化模型 V4 ====================
class OptimizedNetV4(nn.Module):
    """优化波阻抗反演网络 V4"""
    def __init__(self, in_ch=2, base_ch=64, out_ch=1):
        super().__init__()
        
        # 输入处理
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.GELU()
        )
        
        # 多尺度特征提取
        self.multi_scale = MultiScaleDilatedConv(base_ch, base_ch, dilations=[1, 2, 4, 8, 16])
        
        # 编码器
        self.enc1 = EnhancedResBlock(base_ch, base_ch * 2)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = EnhancedResBlock(base_ch * 2, base_ch * 4)
        self.pool2 = nn.MaxPool1d(2)
        
        self.enc3 = EnhancedResBlock(base_ch * 4, base_ch * 8)
        self.pool3 = nn.MaxPool1d(2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            MultiScaleDilatedConv(base_ch * 8, base_ch * 16, dilations=[1, 2, 4, 8, 16, 32]),
            EnhancedResBlock(base_ch * 16, base_ch * 16),
            EnhancedResBlock(base_ch * 16, base_ch * 16)
        )
        
        # 解码器
        self.up3 = nn.ConvTranspose1d(base_ch * 16, base_ch * 8, kernel_size=2, stride=2)
        self.dec3 = EnhancedResBlock(base_ch * 16, base_ch * 8)
        
        self.up2 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec2 = EnhancedResBlock(base_ch * 8, base_ch * 4)
        
        self.up1 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1 = EnhancedResBlock(base_ch * 4, base_ch * 2)
        
        # 精细化
        self.refine = nn.Sequential(
            EnhancedResBlock(base_ch * 2 + base_ch, base_ch * 2),
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.GELU(),
            CBAM(base_ch),
            BoundaryEnhanceModule(base_ch),
        )
        
        # 输出
        self.output = nn.Sequential(
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_ch // 2, out_ch, kernel_size=1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # 输入
        x0 = self.input_conv(x)
        x0 = self.multi_scale(x0)
        
        # 编码
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # 瓶颈
        b = self.bottleneck(p3)
        
        # 解码
        d3 = self.up3(b)
        if d3.shape[-1] != e3.shape[-1]:
            d3 = F.interpolate(d3, size=e3.shape[-1], mode='linear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # 精细化
        if d1.shape[-1] != x0.shape[-1]:
            d1 = F.interpolate(d1, size=x0.shape[-1], mode='linear', align_corners=False)
        out = torch.cat([d1, x0], dim=1)
        out = self.refine(out)
        out = self.output(out)
        
        return out


# ==================== 组合损失函数 ====================
class CombinedLoss(nn.Module):
    """组合损失函数：MSE + 梯度损失 + 频域损失 + 边界损失"""
    def __init__(self, mse_weight=1.0, grad_weight=0.5, fft_weight=0.2, boundary_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.fft_weight = fft_weight
        self.boundary_weight = boundary_weight
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def gradient_loss(self, pred, target):
        """梯度损失：保持边界清晰"""
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        return self.l1(pred_grad, target_grad)
    
    def fft_loss(self, pred, target):
        """频域损失：保持频率特性"""
        # 转换为float32以避免cuFFT半精度限制
        pred_f = pred.float()
        target_f = target.float()
        
        pred_fft = torch.fft.rfft(pred_f, dim=-1)
        target_fft = torch.fft.rfft(target_f, dim=-1)
        
        # 幅度谱损失
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)
        mag_loss = self.l1(pred_mag, target_mag)
        
        # 相位谱损失
        pred_phase = torch.angle(pred_fft)
        target_phase = torch.angle(target_fft)
        phase_loss = self.l1(pred_phase, target_phase)
        
        return mag_loss + 0.1 * phase_loss
    
    def boundary_loss(self, pred, target):
        """边界损失：聚焦于高梯度区域"""
        target_grad = torch.abs(target[:, :, 1:] - target[:, :, :-1])
        # 找到高梯度区域
        grad_threshold = target_grad.mean() + target_grad.std()
        boundary_mask = (target_grad > grad_threshold).float()
        
        # 扩展mask以匹配原始尺寸
        boundary_mask = F.pad(boundary_mask, (0, 1), mode='replicate')
        
        # 边界区域的MSE
        pred_boundary = pred * boundary_mask
        target_boundary = target * boundary_mask
        
        return self.mse(pred_boundary, target_boundary)
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        fft_loss = self.fft_loss(pred, target)
        boundary_loss = self.boundary_loss(pred, target)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.grad_weight * grad_loss + 
                     self.fft_weight * fft_loss +
                     self.boundary_weight * boundary_loss)
        
        return total_loss, {
            'mse': mse_loss.item(),
            'grad': grad_loss.item(),
            'fft': fft_loss.item(),
            'boundary': boundary_loss.item()
        }


# ==================== 数据处理 ====================
def highpass_filter(data, cutoff, fs=1000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


class SeismicDataset(Dataset):
    def __init__(self, seismic, seismic_hf, impedance, indices, augment=False):
        self.seismic = seismic
        self.seismic_hf = seismic_hf
        self.impedance = impedance
        self.indices = indices
        self.augment = augment
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        seis = self.seismic[i].copy()
        seis_hf = self.seismic_hf[i].copy()
        imp = self.impedance[i].copy()
        
        if self.augment:
            # 数据增强
            # 1. 随机幅度缩放
            if random.random() < 0.5:
                scale = random.uniform(0.9, 1.1)
                seis = seis * scale
                seis_hf = seis_hf * scale
            
            # 2. 添加高斯噪声
            if random.random() < 0.3:
                noise_level = random.uniform(0.01, 0.05)
                seis = seis + np.random.randn(*seis.shape).astype(np.float32) * noise_level * np.std(seis)
                seis_hf = seis_hf + np.random.randn(*seis_hf.shape).astype(np.float32) * noise_level * np.std(seis_hf)
            
            # 3. 随机时移
            if random.random() < 0.3:
                shift = random.randint(-20, 20)
                if shift > 0:
                    seis = np.concatenate([np.zeros(shift, dtype=np.float32), seis[:-shift]])
                    seis_hf = np.concatenate([np.zeros(shift, dtype=np.float32), seis_hf[:-shift]])
                    imp = np.concatenate([np.zeros(shift, dtype=np.float32) + imp[0], imp[:-shift]])
                elif shift < 0:
                    seis = np.concatenate([seis[-shift:], np.zeros(-shift, dtype=np.float32)])
                    seis_hf = np.concatenate([seis_hf[-shift:], np.zeros(-shift, dtype=np.float32)])
                    imp = np.concatenate([imp[-shift:], np.zeros(-shift, dtype=np.float32) + imp[-1]])
            
            # 4. 随机翻转
            if random.random() < 0.3:
                seis = seis[::-1].copy()
                seis_hf = seis_hf[::-1].copy()
                imp = imp[::-1].copy()
        
        x = np.stack([seis, seis_hf], axis=0)
        y = imp[np.newaxis, :]
        
        return torch.from_numpy(x), torch.from_numpy(y)


# ==================== 训练函数 ====================
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    loss_components = {'mse': 0, 'grad': 0, 'fft': 0, 'boundary': 0}
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                pred = model(x)
                loss, components = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(x)
            loss, components = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
    
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_pred, all_true = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss, _ = criterion(pred, y)
            total_loss += loss.item()
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
    
    all_pred = np.concatenate(all_pred, axis=0).flatten()
    all_true = np.concatenate(all_true, axis=0).flatten()
    
    pcc = np.corrcoef(all_pred, all_true)[0, 1]
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    return total_loss / len(loader), pcc, r2


# ==================== 主程序 ====================
def main():
    print("=" * 60)
    print(f"优化训练 V4 - {FREQ}")
    print("=" * 60)
    
    # 加载数据
    print("\n加载数据...")
    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
        seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)
    
    raw_imp = np.loadtxt(IMPEDANCE_PATH, usecols=4, skiprows=1).astype(np.float32)
    n_traces = seismic.shape[0]
    n_samples = len(raw_imp) // n_traces
    impedance = raw_imp.reshape(n_traces, n_samples)
    
    print(f"地震数据形状: {seismic.shape}")
    print(f"阻抗数据形状: {impedance.shape}")
    
    # 归一化
    seis_mean, seis_std = seismic.mean(), seismic.std()
    imp_mean, imp_std = impedance.mean(), impedance.std()
    
    seis_norm = (seismic - seis_mean) / seis_std
    imp_norm = (impedance - imp_mean) / imp_std
    
    # 高通滤波
    seismic_hf = highpass_filter(seismic, cutoff=HIGHPASS_CUTOFF, fs=1000)
    seis_hf_norm = seismic_hf / (np.std(seismic_hf, axis=1, keepdims=True) + 1e-6)
    
    # 保存归一化参数
    norm_stats = {
        'seis_mean': float(seis_mean),
        'seis_std': float(seis_std),
        'imp_mean': float(imp_mean),
        'imp_std': float(imp_std),
        'highpass_cutoff': HIGHPASS_CUTOFF
    }
    with open(OUTPUT_DIR / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # 划分数据集
    indices = np.arange(n_traces)
    np.random.shuffle(indices)
    
    n_train = int(n_traces * TRAIN_RATIO)
    n_val = int(n_traces * VAL_RATIO)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    
    print(f"训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")
    
    # 创建数据集
    train_ds = SeismicDataset(seis_norm, seis_hf_norm, imp_norm, train_idx, augment=True)
    val_ds = SeismicDataset(seis_norm, seis_hf_norm, imp_norm, val_idx, augment=False)
    test_ds = SeismicDataset(seis_norm, seis_hf_norm, imp_norm, test_idx, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # 创建模型
    model = OptimizedNetV4(in_ch=2, base_ch=64, out_ch=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 损失函数、优化器
    criterion = CombinedLoss(mse_weight=1.0, grad_weight=0.5, fft_weight=0.2, boundary_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # 余弦退火学习率
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=100, T_mult=2, eta_min=1e-6)
    
    # 混合精度
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    # 训练
    best_val_pcc = 0
    best_epoch = 0
    log_file = open(OUTPUT_DIR / 'train_log.txt', 'w', encoding='utf-8')
    log_file.write(f"训练开始: {datetime.now()}\n")
    log_file.write(f"频率: {FREQ}, 高通截止: {HIGHPASS_CUTOFF}Hz\n")
    log_file.write(f"模型参数量: {total_params:,}\n\n")
    
    print("\n开始训练...")
    for epoch in range(1, EPOCHS + 1):
        train_loss, loss_comp = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_pcc, val_r2 = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        
        if val_pcc > best_val_pcc:
            best_val_pcc = val_pcc
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_metrics': {'pcc': val_pcc, 'r2': val_r2, 'loss': val_loss}
            }, OUTPUT_DIR / 'checkpoints' / 'best.pt')
        
        if epoch % 10 == 0 or epoch <= 5:
            log_msg = (f"Epoch {epoch:4d}/{EPOCHS} | LR: {current_lr:.2e} | "
                      f"Train Loss: {train_loss:.4f} (MSE:{loss_comp['mse']:.4f}, Grad:{loss_comp['grad']:.4f}) | "
                      f"Val Loss: {val_loss:.4f} | Val PCC: {val_pcc:.4f} | Val R²: {val_r2:.4f}")
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()
        
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, OUTPUT_DIR / 'checkpoints' / f'epoch_{epoch}.pt')
    
    # 保存最终模型
    torch.save({
        'epoch': EPOCHS,
        'model': model.state_dict(),
    }, OUTPUT_DIR / 'checkpoints' / 'last.pt')
    
    # 测试
    print("\n" + "=" * 40)
    print("测试最佳模型...")
    ckpt = torch.load(OUTPUT_DIR / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    test_loss, test_pcc, test_r2 = validate(model, test_loader, criterion, device)
    
    print(f"测试结果:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  PCC:  {test_pcc:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    print(f"  最佳Epoch: {best_epoch}")
    
    log_file.write(f"\n测试结果:\n")
    log_file.write(f"  Loss: {test_loss:.4f}\n")
    log_file.write(f"  PCC:  {test_pcc:.4f}\n")
    log_file.write(f"  R²:   {test_r2:.4f}\n")
    log_file.write(f"  最佳Epoch: {best_epoch}\n")
    log_file.write(f"\n训练结束: {datetime.now()}\n")
    log_file.close()
    
    # 保存测试指标
    test_metrics = {
        'test_loss': float(test_loss),
        'test_pcc': float(test_pcc),
        'test_r2': float(test_r2),
        'best_epoch': best_epoch,
        'total_params': total_params
    }
    with open(OUTPUT_DIR / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n结果已保存到: {OUTPUT_DIR}")
    return test_pcc, test_r2


if __name__ == '__main__':
    main()
