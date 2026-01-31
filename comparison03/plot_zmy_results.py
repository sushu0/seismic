#!/usr/bin/env python
"""Plot UNet1D (SS-GAN based) predictions for zmy data."""

from __future__ import annotations
import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import segyio

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(os.path.dirname(__file__)))
from src.ss_gan.models import UNet1D


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
    
    return seismic, impedance


def load_model(freq: int, device):
    """Load trained model."""
    run_dir = f'D:/SEISMIC_CODING/comparison03/runs/zmy_{freq}Hz_supervised'
    ckpt_path = f'{run_dir}/checkpoints/best.pt'
    
    checkpoint = torch.load(ckpt_path, weights_only=True)
    stats = checkpoint['stats']
    
    model = UNet1D(in_ch=1, out_ch=1, base_ch=16, k_large=31, k_small=3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, stats


def predict(model, seismic, stats, device):
    """Make predictions."""
    x = (seismic - stats['x_mean']) / (stats['x_std'] + 1e-8)
    x = torch.from_numpy(x).float().unsqueeze(1).to(device)
    
    with torch.no_grad():
        pred = model(x)
    
    pred = pred.squeeze(1).cpu().numpy()
    pred = pred * stats['y_std'] + stats['y_mean']
    return pred


def plot_impedance_section(data, title, output_path, vmin=None, vmax=None):
    """Plot impedance section."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    n_traces, n_samples = data.shape
    time_axis = np.arange(n_samples) * 0.001 * 1000  # ms
    
    if vmin is None:
        vmin = np.percentile(data, 2)
    if vmax is None:
        vmax = np.percentile(data, 98)
    
    im = ax.imshow(data.T, aspect='auto', cmap='viridis',
                   extent=[0, n_traces, time_axis[-1], time_axis[0]],
                   vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('道号', fontsize=12)
    ax.set_ylabel('时间 (ms)', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('阻抗 (kg/m³·m/s)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_comparison(pred, true, freq, output_path):
    """Plot side-by-side comparison."""
    vmin = min(np.percentile(pred, 2), np.percentile(true, 2))
    vmax = max(np.percentile(pred, 98), np.percentile(true, 98))
    
    n_traces, n_samples = pred.shape
    time_axis = np.arange(n_samples) * 0.001 * 1000  # ms
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # True
    im0 = axes[0].imshow(true.T, aspect='auto', cmap='viridis',
                         extent=[0, n_traces, time_axis[-1], time_axis[0]],
                         vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('道号', fontsize=12)
    axes[0].set_ylabel('时间 (ms)', fontsize=12)
    axes[0].set_title(f'真实阻抗 ({freq}Hz)', fontsize=14)
    fig.colorbar(im0, ax=axes[0], shrink=0.85, pad=0.02)
    
    # Predicted
    im1 = axes[1].imshow(pred.T, aspect='auto', cmap='viridis',
                         extent=[0, n_traces, time_axis[-1], time_axis[0]],
                         vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('道号', fontsize=12)
    axes[1].set_ylabel('时间 (ms)', fontsize=12)
    axes[1].set_title(f'UNet1D预测阻抗 ({freq}Hz)', fontsize=14)
    fig.colorbar(im1, ax=axes[1], shrink=0.85, pad=0.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def plot_trace_comparison(pred, true, trace_idx, freq, output_path):
    """Plot trace comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    n_samples = pred.shape[1]
    time_axis = np.arange(n_samples) * 0.001 * 1000  # ms
    
    for i, idx in enumerate(trace_idx[:2]):
        axes[i].plot(true[idx], time_axis, 'b-', linewidth=1.5, label='真实')
        axes[i].plot(pred[idx], time_axis, 'r--', linewidth=1.5, label='预测')
        axes[i].invert_yaxis()
        axes[i].set_xlabel('阻抗 (kg/m³·m/s)', fontsize=12)
        axes[i].set_ylabel('时间 (ms)', fontsize=12)
        axes[i].set_title(f'第{idx}道 ({freq}Hz)', fontsize=14)
        axes[i].legend(fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def compute_metrics(pred, true):
    """Compute metrics."""
    pred_flat = pred.flatten()
    true_flat = true.flatten()
    pcc = np.corrcoef(pred_flat, true_flat)[0, 1]
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return pcc, r2


def plot_error_section(pred, true, freq, output_path):
    """Plot error distribution section."""
    error = pred - true
    
    n_traces, n_samples = error.shape
    time_axis = np.arange(n_samples) * 0.001 * 1000
    
    vabs = np.percentile(np.abs(error), 98)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(error.T, aspect='auto', cmap='RdBu_r',
                   extent=[0, n_traces, time_axis[-1], time_axis[0]],
                   vmin=-vabs, vmax=vabs)
    
    ax.set_xlabel('道号', fontsize=12)
    ax.set_ylabel('时间 (ms)', fontsize=12)
    ax.set_title(f'预测误差分布 ({freq}Hz)', fontsize=14)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('误差 (kg/m³·m/s)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Output directory
    output_dir = 'D:/SEISMIC_CODING/comparison03/figures'
    os.makedirs(output_dir, exist_ok=True)
    
    for freq in [20, 30]:
        print(f'\n{"="*50}')
        print(f'Processing {freq}Hz data')
        print('='*50)
        
        # Load data and model
        seismic, true_impedance = load_zmy_data(freq)
        model, stats = load_model(freq, device)
        
        # Predict
        pred_impedance = predict(model, seismic, stats, device)
        
        # Compute metrics
        pcc, r2 = compute_metrics(pred_impedance, true_impedance)
        print(f'{freq}Hz - PCC: {pcc:.4f}, R²: {r2:.4f}')
        
        # Plot impedance sections
        plot_impedance_section(
            true_impedance, 
            f'真实阻抗 ({freq}Hz)', 
            f'{output_dir}/unet1d_{freq}Hz_true.png'
        )
        
        plot_impedance_section(
            pred_impedance,
            f'UNet1D预测阻抗 ({freq}Hz) - PCC={pcc:.4f}',
            f'{output_dir}/unet1d_{freq}Hz_pred.png'
        )
        
        # Comparison plot
        plot_comparison(
            pred_impedance, true_impedance, freq,
            f'{output_dir}/unet1d_{freq}Hz_comparison.png'
        )
        
        # Trace comparison
        trace_indices = [20, 50, 80]
        plot_trace_comparison(
            pred_impedance, true_impedance, trace_indices, freq,
            f'{output_dir}/unet1d_{freq}Hz_traces.png'
        )
        
        # Error plot
        plot_error_section(
            pred_impedance, true_impedance, freq,
            f'{output_dir}/unet1d_{freq}Hz_error.png'
        )
    
    print(f'\nAll figures saved to: {output_dir}')


if __name__ == '__main__':
    main()
