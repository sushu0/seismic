#!/usr/bin/env python
"""Train UNet1D (from SS-GAN) in supervised mode on 20Hz/30Hz zmy seismic data."""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import segyio
from tqdm import tqdm

sys.path.insert(0, str(os.path.dirname(__file__)))
from src.ss_gan.models import UNet1D


def seed_everything(seed: int):
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ZMYDataset(Dataset):
    """Dataset for zmy seismic data."""
    def __init__(self, seismic, impedance, augment=False):
        self.seismic = torch.from_numpy(seismic).float().unsqueeze(1)
        self.impedance = torch.from_numpy(impedance).float().unsqueeze(1)
        self.augment = augment
        
    def __len__(self):
        return len(self.seismic)
    
    def __getitem__(self, idx):
        x = self.seismic[idx]
        y = self.impedance[idx]
        if self.augment and np.random.rand() > 0.5:
            # Flip along time axis
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])
        return x, y


class CombinedLoss(nn.Module):
    """Combined Huber + Gradient + Correlation Loss."""
    def __init__(self, huber_delta=0.5, grad_weight=0.3, corr_weight=0.2):
        super().__init__()
        self.huber = nn.SmoothL1Loss(beta=huber_delta)
        self.grad_weight = grad_weight
        self.corr_weight = corr_weight
        
    def forward(self, pred, target):
        # Huber loss
        huber_loss = self.huber(pred, target)
        
        # Gradient loss (preserve edges)
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        grad_loss = F.smooth_l1_loss(pred_grad, target_grad)
        
        # Negative correlation loss (maximize correlation)
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)
        pred_mean = pred_flat.mean(dim=1, keepdim=True)
        target_mean = target_flat.mean(dim=1, keepdim=True)
        pred_centered = pred_flat - pred_mean
        target_centered = target_flat - target_mean
        cov = (pred_centered * target_centered).sum(dim=1)
        pred_std = pred_centered.norm(dim=1) + 1e-8
        target_std = target_centered.norm(dim=1) + 1e-8
        corr = cov / (pred_std * target_std)
        corr_loss = 1 - corr.mean()
        
        return huber_loss + self.grad_weight * grad_loss + self.corr_weight * corr_loss


def load_zmy_data(freq: int):
    """Load zmy seismic and impedance data."""
    base_path = r'D:\SEISMIC_CODING\zmy_data\01\data'
    
    # Load seismic using segyio
    segy_path = os.path.join(base_path, f'01_{freq}Hz_re.sgy')
    with segyio.open(segy_path, 'r', ignore_geometry=True) as f:
        seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)
    
    # Load impedance
    imp_path = os.path.join(base_path, f'01_{freq}Hz_04.txt')
    imp_raw = np.loadtxt(imp_path, usecols=4, skiprows=1).astype(np.float32)
    n_traces = seismic.shape[0]
    n_samples = len(imp_raw) // n_traces
    impedance = imp_raw.reshape(n_traces, n_samples)
    
    # Match lengths
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]
    
    print(f'Loaded {freq}Hz: seismic {seismic.shape}, impedance {impedance.shape}')
    return seismic, impedance


def normalize_data(seismic, impedance):
    """Normalize data and return statistics."""
    x_mean, x_std = seismic.mean(), seismic.std()
    y_mean, y_std = impedance.mean(), impedance.std()
    
    seismic_norm = (seismic - x_mean) / (x_std + 1e-8)
    impedance_norm = (impedance - y_mean) / (y_std + 1e-8)
    
    stats = {
        'x_mean': float(x_mean),
        'x_std': float(x_std),
        'y_mean': float(y_mean),
        'y_std': float(y_std)
    }
    return seismic_norm, impedance_norm, stats


def compute_metrics(pred, target):
    """Compute PCC, R2, and MSE."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # PCC
    pcc = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    # R2
    ss_res = np.sum((target_flat - pred_flat) ** 2)
    ss_tot = np.sum((target_flat - target_flat.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    # MSE
    mse = np.mean((pred_flat - target_flat) ** 2)
    
    return {'pcc': float(pcc), 'r2': float(r2), 'mse': float(mse)}


def train_model(args):
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Training UNet1D (supervised) on {args.freq}Hz data')
    print('=' * 50)
    
    # Setup directories
    run_dir = f'D:/SEISMIC_CODING/comparison03/runs/zmy_{args.freq}Hz_supervised'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f'{run_dir}/checkpoints', exist_ok=True)
    
    # Load and prepare data
    seismic, impedance = load_zmy_data(args.freq)
    seismic_norm, impedance_norm, stats = normalize_data(seismic, impedance)
    
    # Save stats
    with open(f'{run_dir}/stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Split data
    n_traces = seismic.shape[0]
    idx = np.arange(n_traces)
    np.random.shuffle(idx)
    
    n_train = int(n_traces * 0.7)
    n_val = int(n_traces * 0.15)
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    print(f'Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')
    
    # Create datasets
    train_dataset = ZMYDataset(seismic_norm[train_idx], impedance_norm[train_idx], augment=True)
    val_dataset = ZMYDataset(seismic_norm[val_idx], impedance_norm[val_idx], augment=False)
    test_dataset = ZMYDataset(seismic_norm[test_idx], impedance_norm[test_idx], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # Create model
    model = UNet1D(
        in_ch=1, 
        out_ch=1, 
        base_ch=args.base_ch,
        k_large=args.k_large,
        k_small=3
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model parameters: {n_params:,}')
    
    # Loss and optimizer
    criterion = CombinedLoss(huber_delta=0.5, grad_weight=0.3, corr_weight=0.3)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_pcc': [], 'val_r2': []}
    best_val_pcc = -float('inf')
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_metrics = compute_metrics(val_preds, val_targets)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pcc'].append(val_metrics['pcc'])
        history['val_r2'].append(val_metrics['r2'])
        
        # Save best model
        if val_metrics['pcc'] > best_val_pcc:
            best_val_pcc = val_metrics['pcc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_pcc': val_metrics['pcc'],
                'stats': stats
            }, f'{run_dir}/checkpoints/best.pt')
        
        if epoch % 10 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch:3d}/{args.epochs}: train_loss={train_loss:.6f}, '
                  f'val_loss={val_loss:.6f}, val_pcc={val_metrics["pcc"]:.4f}, '
                  f'val_r2={val_metrics["r2"]:.4f}, lr={lr:.2e}')
    
    # Save last model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'stats': stats
    }, f'{run_dir}/checkpoints/last.pt')
    
    # Test evaluation
    print('\n' + '=' * 50)
    print('Test Evaluation (using best model)')
    checkpoint = torch.load(f'{run_dir}/checkpoints/best.pt', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_preds.append(pred.cpu().numpy())
            test_targets.append(y.cpu().numpy())
    
    test_preds = np.concatenate(test_preds, axis=0)
    test_targets = np.concatenate(test_targets, axis=0)
    test_metrics = compute_metrics(test_preds, test_targets)
    
    print(f'Test PCC: {test_metrics["pcc"]:.4f}')
    print(f'Test R²:  {test_metrics["r2"]:.4f}')
    print(f'Test MSE: {test_metrics["mse"]:.6f}')
    
    # Save history with test metrics
    history['test_pcc'] = test_metrics['pcc']
    history['test_r2'] = test_metrics['r2']
    history['test_mse'] = test_metrics['mse']
    with open(f'{run_dir}/history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save test predictions (denormalized)
    test_preds_denorm = test_preds * stats['y_std'] + stats['y_mean']
    test_targets_denorm = test_targets * stats['y_std'] + stats['y_mean']
    np.save(f'{run_dir}/test_predictions.npy', test_preds_denorm)
    np.save(f'{run_dir}/test_targets.npy', test_targets_denorm)
    
    print(f'\nResults saved to: {run_dir}')
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=int, default=30, choices=[20, 30, 40])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base_ch', type=int, default=16)
    parser.add_argument('--k_large', type=int, default=31)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train_model(args)


if __name__ == '__main__':
    main()
