#!/usr/bin/env python
"""Plot FCRSN-CW prediction results for zmy data."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12


def plot_impedance_section(data, title, save_path, vmin=None, vmax=None, time_max=100):
    """绘制波阻抗剖面图"""
    n_traces, n_samples = data.shape
    
    # 下采样显示
    target_samples = 100
    if n_samples > target_samples:
        step = n_samples // target_samples
        data_display = data[:, ::step][:, :target_samples]
    else:
        data_display = data
    
    n_traces_disp, n_samples_disp = data_display.shape
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    
    if vmin is None:
        vmin = np.percentile(data_display, 1)
    if vmax is None:
        vmax = np.percentile(data_display, 99)
    
    extent = [0, n_traces_disp, time_max, 0]
    im = ax.imshow(data_display.T, aspect='auto', cmap='viridis', 
                   extent=extent, vmin=vmin, vmax=vmax)
    
    ax.set_xlabel('道号', fontsize=14)
    ax.set_ylabel('时间（ms）', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlim(0, n_traces_disp)
    ax.set_ylim(time_max, 0)
    
    cbar = plt.colorbar(im, ax=ax, pad=0.08, shrink=0.9)
    cbar.set_label('波阻抗（m/s*g/cm3）', fontsize=12)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'已保存: {save_path}')


def plot_comparison(true_data, pred_data, freq, save_path):
    """绘制真实与预测对比图"""
    n_traces, n_samples = true_data.shape
    
    # 下采样
    target_samples = 100
    if n_samples > target_samples:
        step = n_samples // target_samples
        true_disp = true_data[:, ::step][:, :target_samples]
        pred_disp = pred_data[:, ::step][:, :target_samples]
    else:
        true_disp = true_data
        pred_disp = pred_data
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    n_traces_disp, n_samples_disp = true_disp.shape
    time_max = 100
    
    vmin = min(np.percentile(true_disp, 1), np.percentile(pred_disp, 1))
    vmax = max(np.percentile(true_disp, 99), np.percentile(pred_disp, 99))
    
    extent = [0, n_traces_disp, time_max, 0]
    
    im1 = axes[0].imshow(true_disp.T, aspect='auto', cmap='viridis',
                          extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('道号', fontsize=12)
    axes[0].set_ylabel('时间（ms）', fontsize=12)
    axes[0].set_title(f'{freq}Hz 真实波阻抗剖面', fontsize=14, fontweight='bold')
    
    im2 = axes[1].imshow(pred_disp.T, aspect='auto', cmap='viridis',
                          extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('道号', fontsize=12)
    axes[1].set_ylabel('时间（ms）', fontsize=12)
    axes[1].set_title(f'{freq}Hz FCRSN-CW预测波阻抗剖面', fontsize=14, fontweight='bold')
    
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', 
                        fraction=0.03, pad=0.08, shrink=0.85)
    cbar.set_label('波阻抗（m/s*g/cm3）', fontsize=11)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'已保存: {save_path}')


def plot_trace_comparison(true_data, pred_data, freq, trace_idx, save_path):
    """绘制单道对比图"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    
    true_trace = true_data[trace_idx]
    pred_trace = pred_data[trace_idx]
    error = pred_trace - true_trace
    
    # 时间轴 (假设100ms总长)
    t = np.linspace(0, 100, len(true_trace))
    
    # 真实vs预测
    axes[0].plot(t, true_trace, 'b-', label='真实', linewidth=1)
    axes[0].plot(t, pred_trace, 'r--', label='预测', linewidth=1)
    axes[0].set_xlabel('时间（ms）', fontsize=11)
    axes[0].set_ylabel('波阻抗', fontsize=11)
    axes[0].set_title(f'{freq}Hz 道{trace_idx} 波阻抗对比', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # 散点图
    axes[1].scatter(true_trace[::100], pred_trace[::100], alpha=0.5, s=5)
    min_val = min(true_trace.min(), pred_trace.min())
    max_val = max(true_trace.max(), pred_trace.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1')
    axes[1].set_xlabel('真实波阻抗', fontsize=11)
    axes[1].set_ylabel('预测波阻抗', fontsize=11)
    axes[1].set_title('预测vs真实', fontsize=12, fontweight='bold')
    axes[1].ticklabel_format(style='scientific', scilimits=(0,0))
    
    # 误差
    axes[2].fill_between(t, 0, error, where=error>0, color='red', alpha=0.5, label='高估')
    axes[2].fill_between(t, 0, error, where=error<0, color='blue', alpha=0.5, label='低估')
    axes[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    axes[2].set_xlabel('时间（ms）', fontsize=11)
    axes[2].set_ylabel('误差', fontsize=11)
    axes[2].set_title('预测误差', fontsize=12, fontweight='bold')
    axes[2].legend()
    axes[2].ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f'已保存: {save_path}')


def main():
    base_dir = Path('D:/SEISMIC_CODING/comparison02/runs')
    output_dir = Path('D:/SEISMIC_CODING/comparison02/figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    frequencies = [20, 30]
    
    for freq in frequencies:
        run_dir = base_dir / f'zmy_{freq}Hz'
        
        if not run_dir.exists():
            print(f'跳过 {freq}Hz: 目录不存在')
            continue
        
        print(f'\n处理 {freq}Hz...')
        
        # 加载数据
        pred = np.load(run_dir / 'results' / 'pred_impedance_all.npy')
        true = np.load(run_dir / 'results' / 'true_impedance_all.npy')
        
        # 加载指标
        with open(run_dir / 'results' / 'metrics.json') as f:
            metrics = json.load(f)
        
        print(f'  Test PCC: {metrics["test_pcc"]:.4f}')
        print(f'  Test R²: {metrics["test_r2"]:.4f}')
        print(f'  Pred shape: {pred.shape}')
        print(f'  Pred range: {pred.min():.2e} - {pred.max():.2e}')
        
        # 绘制单独的预测剖面
        plot_impedance_section(
            pred,
            f'{freq}Hz FCRSN-CW预测波阻抗剖面',
            output_dir / f'{freq}Hz_fcrsn_predicted_impedance.png'
        )
        
        # 绘制对比图
        plot_comparison(
            true, pred, freq,
            output_dir / f'{freq}Hz_fcrsn_true_vs_pred.png'
        )
        
        # 绘制单道对比
        for trace_idx in [25, 50, 75]:
            plot_trace_comparison(
                true, pred, freq, trace_idx,
                output_dir / f'{freq}Hz_fcrsn_trace_{trace_idx}.png'
            )
    
    print(f'\n✓ 所有图片已保存到: {output_dir}')


if __name__ == '__main__':
    main()
