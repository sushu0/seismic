# -*- coding: utf-8 -*-
"""
评估V6模型并保存测试指标
"""
import os
import sys
import json
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from pathlib import Path
from scipy.stats import pearsonr

# ==================== 命令行参数 ====================
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--freq', type=str, default='40Hz', choices=['20Hz', '30Hz', '40Hz', '50Hz'])
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

SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

np.random.seed(SEED)
torch.manual_seed(SEED)
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


# ==================== 数据处理 ====================
def highpass_filter(data, cutoff, fs=1000, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high')
    return filtfilt(b, a, data, axis=-1)


class Dataset1D(Dataset):
    def __init__(self, seis, seis_hf, imp, indices):
        self.seis = seis[indices]
        self.seis_hf = seis_hf[indices]
        self.imp = imp[indices]
    
    def __len__(self):
        return len(self.seis)
    
    def __getitem__(self, idx):
        x = np.stack([self.seis[idx], self.seis_hf[idx]], axis=0).astype(np.float32)
        y = self.imp[idx:idx+1].astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


# ==================== 主函数 ====================
def main():
    print(f"Evaluating {FREQ} V6 model...")
    print(f"Device: {device}")
    
    # 加载数据
    with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
        seismic = np.array([f.trace[i] for i in range(f.tracecount)], dtype=np.float32)
    
    # 加载波阻抗 - 跳过标题行，取最后一列
    import pandas as pd
    df = pd.read_csv(IMPEDANCE_PATH, sep=r'\s+', skiprows=1, header=None)
    impedance_raw = df.iloc[:, -1].values.astype(np.float32)
    # 重塑为 (n_traces, n_samples)
    n_traces = seismic.shape[0]
    n_samples = seismic.shape[1]
    impedance = impedance_raw.reshape(n_traces, n_samples)
    
    print(f"Data: {seismic.shape}")
    
    # 加载归一化参数
    with open(OUTPUT_DIR / 'norm_stats.json') as f:
        stats = json.load(f)
    
    sm, ss = stats['seis_mean'], stats['seis_std']
    im, ist = stats['imp_mean'], stats['imp_std']
    
    # 归一化
    seis_n = (seismic - sm) / ss
    imp_n = (impedance - im) / ist
    seis_hf = highpass_filter(seismic, HIGHPASS_CUTOFF)
    seis_hf_n = (seis_hf - seis_hf.mean()) / seis_hf.std()
    
    # 划分数据
    n_tr = len(seismic)
    idx = np.arange(n_tr)
    np.random.shuffle(idx)
    n_train = int(n_tr * TRAIN_RATIO)
    n_val = int(n_tr * VAL_RATIO)
    te_idx = idx[n_train + n_val:]
    
    print(f"Test samples: {len(te_idx)}")
    
    te_ds = Dataset1D(seis_n, seis_hf_n, imp_n, te_idx)
    te_ld = DataLoader(te_ds, batch_size=8, shuffle=False)
    
    # 加载模型
    model = InversionNet(in_ch=2, base=48).to(device)
    ckpt = torch.load(OUTPUT_DIR / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    best_epoch = ckpt['epoch']
    best_pcc = ckpt['best_pcc']
    
    print(f"Loaded model from epoch {best_epoch}, val_pcc={best_pcc:.4f}")
    
    # 评估
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in te_ld:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    
    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()
    
    pcc, _ = pearsonr(preds, trues)
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\nTest Results:")
    print(f"  PCC: {pcc:.4f}")
    print(f"  R2:  {r2:.4f}")
    
    # 保存结果
    metrics = {
        'test_pcc': float(pcc),
        'test_r2': float(r2),
        'best_epoch': int(best_epoch)
    }
    
    with open(OUTPUT_DIR / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nResults saved to {OUTPUT_DIR / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
