# -*- coding: utf-8 -*-
"""
优化版波阻抗反演模型 V6 - 稳定版
改进：
1. 更稳健的训练流程
2. 定期保存检查点
3. 支持断点续训
"""
import os
import sys
import json
import random
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from pathlib import Path
from datetime import datetime
import math

# 禁用不必要的警告
import warnings
warnings.filterwarnings('ignore')

# ==================== 命令行参数 ====================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--freq', type=str, default='30Hz', choices=['20Hz', '30Hz', '40Hz', '50Hz'])
parser.add_argument('--epochs', type=int, default=800)
parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
args = parser.parse_args()

FREQ = args.freq

# ==================== 配置 ====================
FREQ_CONFIGS = {
    '20Hz': {'highpass_cutoff': 8},
    '30Hz': {'highpass_cutoff': 12},
    '40Hz': {'highpass_cutoff': 15},
    '50Hz': {'highpass_cutoff': 20},
}

config = FREQ_CONFIGS[FREQ]
HIGHPASS_CUTOFF = config['highpass_cutoff']

SEISMIC_PATH = rf'D:\SEISMIC_CODING\zmy_data\01\data\01_{FREQ}_re.sgy'
IMPEDANCE_PATH = rf'D:\SEISMIC_CODING\zmy_data\01\data\01_{FREQ}_04.txt'
OUTPUT_DIR = Path(rf'D:\SEISMIC_CODING\new\results\01_{FREQ}_v6')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'checkpoints').mkdir(exist_ok=True)

EPOCHS = args.epochs
BATCH_SIZE = 4
LR = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== 模型 ====================
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


# ==================== 损失函数 ====================
class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.HuberLoss(delta=1.0)
        self.l1 = nn.L1Loss()
    
    def forward(self, pred, target):
        huber = self.huber(pred, target)
        
        # 梯度损失
        p_grad = pred[:, :, 1:] - pred[:, :, :-1]
        t_grad = target[:, :, 1:] - target[:, :, :-1]
        grad = self.l1(p_grad, t_grad)
        
        return huber + 0.3 * grad


# ==================== 数据 ====================
def highpass(data, cutoff, fs=1000):
    nyq = 0.5 * fs
    b, a = butter(4, cutoff / nyq, btype='high')
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


class Dataset1D(Dataset):
    def __init__(self, seis, seis_hf, imp, idx, aug=False):
        self.seis, self.seis_hf, self.imp = seis, seis_hf, imp
        self.idx, self.aug = idx, aug
    
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, i):
        j = self.idx[i]
        s, sh, im = self.seis[j].copy(), self.seis_hf[j].copy(), self.imp[j].copy()
        
        if self.aug and random.random() < 0.5:
            scale = random.uniform(0.95, 1.05)
            s, sh = s * scale, sh * scale
        
        x = np.stack([s, sh], axis=0)
        y = im[np.newaxis, :]
        return torch.from_numpy(x), torch.from_numpy(y)


# ==================== 训练 ====================
def train_one(model, loader, criterion, optim, dev):
    model.train()
    total = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        optim.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item()
    return total / len(loader)


def validate(model, loader, criterion, dev):
    model.eval()
    total, preds, trues = 0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            p = model(x)
            total += criterion(p, y).item()
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())
    
    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()
    pcc = np.corrcoef(preds, trues)[0, 1]
    r2 = 1 - np.sum((trues - preds)**2) / np.sum((trues - trues.mean())**2)
    return total / len(loader), pcc, r2


def main():
    print(f"Device: {device}")
    print(f"Frequency: {FREQ}, Highpass: {HIGHPASS_CUTOFF}Hz")
    print(f"Epochs: {EPOCHS}")
    print("=" * 50)
    
    # 加载数据
    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
        seis = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)
    
    raw = np.loadtxt(IMPEDANCE_PATH, usecols=4, skiprows=1).astype(np.float32)
    n_tr, n_samp = seis.shape[0], len(raw) // seis.shape[0]
    imp = raw.reshape(n_tr, n_samp)
    
    print(f"Data: {seis.shape}")
    
    # 归一化
    sm, ss = seis.mean(), seis.std()
    im, ist = imp.mean(), imp.std()
    seis_n = (seis - sm) / ss
    imp_n = (imp - im) / ist
    
    seis_hf = highpass(seis, HIGHPASS_CUTOFF)
    seis_hf_n = seis_hf / (np.std(seis_hf, axis=1, keepdims=True) + 1e-6)
    
    # 保存归一化参数
    with open(OUTPUT_DIR / 'norm_stats.json', 'w') as f:
        json.dump({
            'seis_mean': float(sm), 'seis_std': float(ss),
            'imp_mean': float(im), 'imp_std': float(ist),
            'highpass_cutoff': HIGHPASS_CUTOFF
        }, f, indent=2)
    
    # 划分
    idx = np.arange(n_tr)
    np.random.shuffle(idx)
    n_train = int(n_tr * TRAIN_RATIO)
    n_val = int(n_tr * VAL_RATIO)
    tr_idx = idx[:n_train]
    va_idx = idx[n_train:n_train + n_val]
    te_idx = idx[n_train + n_val:]
    
    print(f"Train: {len(tr_idx)}, Val: {len(va_idx)}, Test: {len(te_idx)}")
    
    tr_ds = Dataset1D(seis_n, seis_hf_n, imp_n, tr_idx, aug=True)
    va_ds = Dataset1D(seis_n, seis_hf_n, imp_n, va_idx)
    te_ds = Dataset1D(seis_n, seis_hf_n, imp_n, te_idx)
    
    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 模型
    model = InversionNet(in_ch=2, base=48).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = CombinedLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, EPOCHS, eta_min=1e-6)
    
    # 恢复训练
    start_ep = 1
    best_pcc = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optim'])
        start_ep = ckpt['epoch'] + 1
        best_pcc = ckpt.get('best_pcc', 0)
        print(f"Resumed from epoch {start_ep - 1}")
    
    # 训练
    no_improve = 0
    log = open(OUTPUT_DIR / 'train_log.txt', 'a', encoding='utf-8')
    log.write(f"\n--- Training started at {datetime.now()} ---\n")
    
    for ep in range(start_ep, EPOCHS + 1):
        tr_loss = train_one(model, tr_ld, criterion, optim, device)
        va_loss, va_pcc, va_r2 = validate(model, va_ld, criterion, device)
        sched.step()
        
        if va_pcc > best_pcc:
            best_pcc = va_pcc
            no_improve = 0
            torch.save({
                'epoch': ep, 'model': model.state_dict(), 'optim': optim.state_dict(),
                'best_pcc': best_pcc, 'metrics': {'pcc': va_pcc, 'r2': va_r2}
            }, OUTPUT_DIR / 'checkpoints' / 'best.pt')
        else:
            no_improve += 1
        
        if ep % 10 == 0 or ep <= 5:
            msg = f"Ep {ep:4d}/{EPOCHS} | Tr: {tr_loss:.4f} | Va: {va_loss:.4f} | PCC: {va_pcc:.4f} | R2: {va_r2:.4f} | Best: {best_pcc:.4f}"
            print(msg)
            log.write(msg + '\n')
            log.flush()
        
        if ep % 50 == 0:
            torch.save({
                'epoch': ep, 'model': model.state_dict(), 'optim': optim.state_dict(),
                'best_pcc': best_pcc
            }, OUTPUT_DIR / 'checkpoints' / f'epoch_{ep}.pt')
        
        if no_improve >= 100:
            print(f"Early stop at epoch {ep}")
            break
    
    # 保存最终
    torch.save({'epoch': ep, 'model': model.state_dict()}, OUTPUT_DIR / 'checkpoints' / 'last.pt')
    
    # 测试
    ckpt = torch.load(OUTPUT_DIR / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    te_loss, te_pcc, te_r2 = validate(model, te_ld, criterion, device)
    
    print("\n" + "=" * 50)
    print(f"Test Results:")
    print(f"  PCC: {te_pcc:.4f}")
    print(f"  R2:  {te_r2:.4f}")
    print(f"  Best epoch: {ckpt['epoch']}")
    
    log.write(f"\nTest: PCC={te_pcc:.4f}, R2={te_r2:.4f}, BestEp={ckpt['epoch']}\n")
    log.close()
    
    with open(OUTPUT_DIR / 'test_metrics.json', 'w') as f:
        json.dump({'test_pcc': float(te_pcc), 'test_r2': float(te_r2), 
                   'best_epoch': ckpt['epoch']}, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
