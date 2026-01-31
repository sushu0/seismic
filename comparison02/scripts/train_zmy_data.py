#!/usr/bin/env python
"""Train FCRSN-CW on 20Hz/30Hz/40Hz seismic data (zmy_data)."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segyio
from scipy.signal import resample

from fcrsn_cw.utils.seed import set_global_seed
from fcrsn_cw.models.fcrsn_cw import FCRSN_CW


class CombinedLoss(nn.Module):
    """Huber loss + Gradient loss for better edge preservation."""
    def __init__(self, delta=1.0, grad_weight=0.3):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta)
        self.l1 = nn.L1Loss()
        self.grad_weight = grad_weight
    
    def forward(self, pred, target):
        huber_loss = self.huber(pred, target)
        
        # Gradient loss for preserving edges
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        grad_loss = self.l1(pred_grad, target_grad)
        
        return huber_loss + self.grad_weight * grad_loss


class ZMYDataset(Dataset):
    """Dataset for zmy seismic/impedance data."""
    
    def __init__(self, seismic, impedance, indices, augment=False):
        self.seismic = seismic
        self.impedance = impedance
        self.indices = indices
        self.augment = augment
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.seismic[i:i+1, :].astype(np.float32)  # (1, T)
        y = self.impedance[i:i+1, :].astype(np.float32)  # (1, T)
        
        if self.augment:
            # 幅度缩放
            if np.random.rand() < 0.5:
                scale = np.random.uniform(0.9, 1.1)
                x = x * scale
            # 添加噪声
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 0.02, x.shape).astype(np.float32)
                x = x + noise
        
        return torch.from_numpy(x), torch.from_numpy(y)


def load_data(freq: int):
    """Load seismic and impedance data for given frequency."""
    base_path = Path(r'D:\SEISMIC_CODING\zmy_data\01\data')
    
    # Load seismic using segyio
    segy_path = base_path / f'01_{freq}Hz_re.sgy'
    with segyio.open(str(segy_path), 'r', ignore_geometry=True) as f:
        seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)
    
    # Load impedance
    imp_path = base_path / f'01_{freq}Hz_04.txt'
    imp_raw = np.loadtxt(str(imp_path), usecols=4, skiprows=1).astype(np.float32)
    n_traces = seismic.shape[0]
    n_samples = len(imp_raw) // n_traces
    impedance = imp_raw.reshape(n_traces, n_samples)
    
    # Match lengths
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]
    
    print(f'Loaded {freq}Hz: seismic {seismic.shape}, impedance {impedance.shape}')
    print(f'  Seismic range: {seismic.min():.4f} - {seismic.max():.4f}')
    print(f'  Impedance range: {impedance.min():.2e} - {impedance.max():.2e}')
    
    return seismic, impedance


def normalize_data(seismic, impedance, train_idx):
    """Normalize data using training set statistics."""
    # Z-score for seismic
    seis_mean = seismic[train_idx].mean()
    seis_std = seismic[train_idx].std()
    seismic_norm = (seismic - seis_mean) / (seis_std + 1e-8)
    
    # MinMax for impedance (to [0, 1])
    imp_min = impedance[train_idx].min()
    imp_max = impedance[train_idx].max()
    impedance_norm = (impedance - imp_min) / (imp_max - imp_min + 1e-8)
    
    stats = {
        'seis_mean': float(seis_mean),
        'seis_std': float(seis_std),
        'imp_min': float(imp_min),
        'imp_max': float(imp_max),
    }
    
    return seismic_norm, impedance_norm, stats


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            total_loss += criterion(pred, y).item()
            preds.append(pred.cpu().numpy())
            trues.append(y.cpu().numpy())
    
    preds = np.concatenate(preds).flatten()
    trues = np.concatenate(trues).flatten()
    pcc = np.corrcoef(preds, trues)[0, 1]
    r2 = 1 - np.sum((trues - preds)**2) / (np.sum((trues - trues.mean())**2) + 1e-8)
    
    return total_loss / len(loader), pcc, r2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=int, default=30, choices=[20, 30, 40])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    # Model kernel sizes - optimized for our data
    parser.add_argument('--k_first', type=int, default=51, help='First conv kernel size')
    parser.add_argument('--k_res1', type=int, default=51, help='RSBU first conv kernel size')
    args = parser.parse_args()
    
    set_global_seed(args.seed)
    device = torch.device(args.device)
    
    # Output directory
    run_dir = Path(f'D:/SEISMIC_CODING/comparison02/runs/zmy_{args.freq}Hz')
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'results').mkdir(exist_ok=True)
    
    print(f'Training FCRSN-CW on {args.freq}Hz data')
    print(f'Device: {device}')
    print('=' * 50)
    
    # Load data
    seismic, impedance = load_data(args.freq)
    n_traces = seismic.shape[0]
    
    # Split: 70% train, 15% val, 15% test
    idx = np.arange(n_traces)
    np.random.shuffle(idx)
    n_train = int(n_traces * 0.7)
    n_val = int(n_traces * 0.15)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    print(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')
    
    # Normalize
    seismic_norm, impedance_norm, stats = normalize_data(seismic, impedance, train_idx)
    
    # Save normalization stats
    with open(run_dir / 'scalers.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Save split
    with open(run_dir / 'split.json', 'w') as f:
        json.dump({
            'train': train_idx.tolist(),
            'val': val_idx.tolist(),
            'test': test_idx.tolist(),
        }, f)
    
    # Datasets
    train_ds = ZMYDataset(seismic_norm, impedance_norm, train_idx, augment=True)
    val_ds = ZMYDataset(seismic_norm, impedance_norm, val_idx)
    test_ds = ZMYDataset(seismic_norm, impedance_norm, test_idx)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Model - optimized architecture
    model = FCRSN_CW(
        k_first=args.k_first,
        k_last=3,
        k_res1=args.k_res1,
        k_res2=3,
        last_relu=False,
        output_activation='sigmoid',
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model params: {n_params:,}')
    
    # Loss and optimizer - use combined loss
    criterion = CombinedLoss(delta=0.5, grad_weight=0.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Training
    best_pcc = -1
    history = {'train_loss': [], 'val_loss': [], 'val_pcc': [], 'val_r2': []}
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_pcc, val_r2 = eval_epoch(model, val_loader, criterion, device)
        
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['val_pcc'].append(float(val_pcc))
        history['val_r2'].append(float(val_r2))
        
        scheduler.step()
        
        if val_pcc > best_pcc:
            best_pcc = val_pcc
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_pcc': val_pcc,
                'val_r2': val_r2,
            }, run_dir / 'checkpoints' / 'best.pt')
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch+1:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | PCC: {val_pcc:.4f} | R²: {val_r2:.4f} | LR: {lr:.2e}')
    
    # Save last checkpoint
    torch.save({
        'epoch': args.epochs - 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, run_dir / 'checkpoints' / 'last.pt')
    
    # Save history
    with open(run_dir / 'results' / 'loss_curve.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Evaluate on test set
    print('\n' + '=' * 50)
    print('Evaluating on test set...')
    
    # Load best model
    ckpt = torch.load(run_dir / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    test_loss, test_pcc, test_r2 = eval_epoch(model, test_loader, criterion, device)
    print(f'Test Results: Loss={test_loss:.6f}, PCC={test_pcc:.4f}, R²={test_r2:.4f}')
    
    # Save test metrics
    metrics = {
        'test_loss': float(test_loss),
        'test_pcc': float(test_pcc),
        'test_r2': float(test_r2),
        'best_val_pcc': float(best_pcc),
    }
    with open(run_dir / 'results' / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate predictions for all data
    print('Generating predictions for all traces...')
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(n_traces):
            x = torch.from_numpy(seismic_norm[i:i+1, :][np.newaxis, :]).to(device)
            pred = model(x).cpu().numpy().squeeze()
            all_preds.append(pred)
    
    all_preds = np.array(all_preds)
    
    # Denormalize predictions
    pred_denorm = all_preds * (stats['imp_max'] - stats['imp_min']) + stats['imp_min']
    
    # Save predictions
    np.save(run_dir / 'results' / 'pred_impedance_all.npy', pred_denorm)
    np.save(run_dir / 'results' / 'true_impedance_all.npy', impedance)
    
    print(f'\nResults saved to: {run_dir}')
    print(f'Best validation PCC: {best_pcc:.4f}')
    print(f'Test PCC: {test_pcc:.4f}')


if __name__ == '__main__':
    main()
