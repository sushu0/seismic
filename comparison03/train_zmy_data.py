#!/usr/bin/env python
"""Train SS-GAN on 20Hz/30Hz zmy seismic data."""

from __future__ import annotations
import os
import sys
import json
import argparse
import numpy as np
import torch
import segyio

sys.path.insert(0, str(os.path.dirname(__file__)))
from src.ss_gan.utils import seed_everything
from src.ss_gan.data import NPZDatasetConfig, make_loader
from src.ss_gan.trainer import TrainConfig, train


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


def create_npz_dataset(seismic, impedance, output_path, train_ratio=0.7, val_ratio=0.15):
    """Create NPZ dataset in SS-GAN format."""
    n_traces = seismic.shape[0]
    idx = np.arange(n_traces)
    np.random.shuffle(idx)
    
    n_train = int(n_traces * train_ratio)
    n_val = int(n_traces * val_ratio)
    
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    # Use most of training as labeled for better supervised learning
    n_labeled = int(n_train * 0.8)  # 80% labeled
    labeled_idx = train_idx[:n_labeled]
    unlabeled_idx = train_idx[n_labeled:]
    
    np.savez(
        output_path,
        x_labeled=seismic[labeled_idx],
        y_labeled=impedance[labeled_idx],
        x_unlabeled=seismic[unlabeled_idx],
        x_val=seismic[val_idx],
        y_val=impedance[val_idx],
        x_test=seismic[test_idx],
        y_test=impedance[test_idx],
    )
    
    print(f'Created dataset: labeled={len(labeled_idx)}, unlabeled={len(unlabeled_idx)}, val={len(val_idx)}, test={len(test_idx)}')
    return output_path


def compute_stats(npz_path: str) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    x = z["x_labeled"].astype("float32")
    y = z["y_labeled"].astype("float32")
    return {
        "x_mean": float(x.mean()),
        "x_std": float(x.std() + 1e-12),
        "y_mean": float(y.mean()),
        "y_std": float(y.std() + 1e-12),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=int, default=30, choices=[20, 30, 40])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    seed_everything(args.seed, deterministic=True, benchmark=False, tf32=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Device: {device}')
    print(f'Training SS-GAN on {args.freq}Hz data')
    print('=' * 50)
    
    # Setup directories
    run_dir = f'D:/SEISMIC_CODING/comparison03/runs/zmy_{args.freq}Hz'
    data_dir = 'D:/SEISMIC_CODING/comparison03/data'
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Load and prepare data
    seismic, impedance = load_zmy_data(args.freq)
    
    # Create NPZ dataset
    npz_path = os.path.join(data_dir, f'zmy_{args.freq}Hz.npz')
    create_npz_dataset(seismic, impedance, npz_path)
    
    # Compute stats
    stats = compute_stats(npz_path)
    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f'Stats: {stats}')
    
    # Create data loaders
    labeled = make_loader(NPZDatasetConfig(npz_path, "labeled", True), args.batch_size, True, 0, stats)
    unlabeled = make_loader(NPZDatasetConfig(npz_path, "unlabeled", True), args.batch_size, True, 0, stats)
    val = make_loader(NPZDatasetConfig(npz_path, "val", True), args.batch_size, False, 0, stats)
    
    # Training config - optimized for our data
    cfg = TrainConfig(
        run_dir=run_dir,
        seed=args.seed,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_critic=1,   # Reduced for smaller dataset - focus on generator
        alpha=1000.0,  # Higher supervised loss weight
        beta=100.0,    # Lower physics loss weight
        lambda_gp=10.0,
        k_large=21,    # Smaller kernels for our data
        k_small=3,
        base_ch_g=16,
        base_ch_d=8,
        wavelet_freq=float(args.freq),
        wavelet_dt=0.001,
        wavelet_dur=0.1,
        save_every=20,
        amp=False,
        normalize=True,
        imp_loss="huber",
        huber_delta=1.0,
        grad_loss_weight=0.5,
        loss_in_physical=False,
        warmup_epochs=10,
        use_ema=True,
        ema_decay=0.999,
    )
    
    # Train
    train(cfg, labeled, unlabeled, val, stats=stats)
    
    print(f'\nResults saved to: {run_dir}')


if __name__ == '__main__':
    main()
