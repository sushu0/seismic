# -*- coding: utf-8 -*-
"""
简化版可视化脚本 - 直接对比预测与真实阻抗
"""
import numpy as np
import torch
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt

from train_30Hz_thinlayer_v2 import (
    ThinLayerNetV2, load_seismic_data, load_impedance_data, 
    ThinLayerDatasetV2, ThinLayerLabeler
)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = Path(r'D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v2')
    figures_dir = result_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("加载数据...")
    seismic_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'
    impedance_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt'
    
    seismic, dt = load_seismic_data(seismic_path)
    n_traces = seismic.shape[0]
    impedance = load_impedance_data(impedance_path, n_traces)
    
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]
    
    print(f"数据形状: seismic={seismic.shape}, impedance={impedance.shape}")
    
    # 加载归一化参数
    with open(result_dir / 'norm_stats.json', 'r') as f:
        norm_stats = json.load(f)
    
    # 加载模型
    print("加载模型...")
    model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
    checkpoint = torch.load(
        result_dir / 'checkpoints' / 'best.pt',
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 生成预测
    print("生成预测...")
    np.random.seed(42)
    n_traces_data = seismic.shape[0]
    indices = np.random.permutation(n_traces_data)
    n_train = int(n_traces_data * 0.6)
    n_val = int(n_traces_data * 0.2)
    test_idx = indices[n_train + n_val:]
    
    print(f"测试集索引: {test_idx}, 共 {len(test_idx)} 道")
    
    labeler = ThinLayerLabeler(dt=0.001, dominant_freq=30.0)
    test_ds = ThinLayerDatasetV2(seismic, impedance, test_idx, norm_stats,
                                  augmentor=None, labeler=labeler)
    
    all_pred = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y, _, _ = test_ds[i]
            x = x.unsqueeze(0).to(device)
            pred_out = model(x).cpu().numpy()
            all_pred.append(pred_out[0, 0])
    
    pred_norm = np.array(all_pred)
    pred = pred_norm * norm_stats['imp_std'] + norm_stats['imp_mean']
    
    # 真实阻抗 - 直接使用原始值
    true = impedance[test_idx]
    
    print(f"pred 形状: {pred.shape}, true 形状: {true.shape}")
    print(f"pred 范围: [{pred.min():.0f}, {pred.max():.0f}]")
    print(f"true 范围: [{true.min():.0f}, {true.max():.0f}]")
    
    # 对齐形状
    min_traces = min(true.shape[0], pred.shape[0])
    min_samples = min(true.shape[1], pred.shape[1])
    pred = pred[:min_traces, :min_samples]
    true = true[:min_traces, :min_samples]
    
    print(f"对齐后 pred 形状: {pred.shape}, true 形状: {true.shape}")
    
    # ========== 绘制对比图 ==========
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)
    
    vmin = min(pred.min(), true.min())
    vmax = max(pred.max(), true.max())
    
    # 预测阻抗
    im1 = axes[0].imshow(pred, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    axes[0].set_title('Predicted Impedance', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sample', fontsize=12)
    axes[0].set_ylabel('Trace', fontsize=12)
    plt.colorbar(im1, ax=axes[0], shrink=0.8)
    
    # 真实阻抗
    im2 = axes[1].imshow(true, cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    axes[1].set_title('True Impedance', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Sample', fontsize=12)
    axes[1].set_ylabel('Trace', fontsize=12)
    plt.colorbar(im2, ax=axes[1], shrink=0.8)
    
    # 误差
    error = np.abs(pred - true)
    im3 = axes[2].imshow(error, cmap='hot', aspect='auto')
    axes[2].set_title('Absolute Error', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Sample', fontsize=12)
    axes[2].set_ylabel('Trace', fontsize=12)
    plt.colorbar(im3, ax=axes[2], shrink=0.8)
    
    # 计算指标
    pcc = np.corrcoef(pred.flatten(), true.flatten())[0, 1]
    mae = np.mean(np.abs(pred - true))
    
    plt.suptitle(f'30Hz Test Set Comparison (PCC={pcc:.4f}, MAE={mae:.0f})', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = figures_dir / 'comparison_fixed.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ 保存: {save_path}")
    plt.close()
    
    # ========== 单道对比 ==========
    fig2, axes2 = plt.subplots(2, 1, figsize=(14, 8), dpi=200)
    
    mid_trace = len(pred) // 2
    samples = np.arange(pred.shape[1])
    
    axes2[0].plot(samples, pred[mid_trace], 'b-', linewidth=1.5, label='Predicted', alpha=0.8)
    axes2[0].plot(samples, true[mid_trace], 'r-', linewidth=1.5, label='True', alpha=0.8)
    axes2[0].set_xlabel('Sample', fontsize=12)
    axes2[0].set_ylabel('Impedance', fontsize=12)
    axes2[0].set_title(f'Trace #{mid_trace} Comparison', fontsize=14, fontweight='bold')
    axes2[0].legend(fontsize=10)
    axes2[0].grid(True, alpha=0.3)
    
    error_trace = pred[mid_trace] - true[mid_trace]
    axes2[1].fill_between(samples, error_trace, alpha=0.6, color='steelblue')
    axes2[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes2[1].set_xlabel('Sample', fontsize=12)
    axes2[1].set_ylabel('Error', fontsize=12)
    axes2[1].set_title('Error', fontsize=14, fontweight='bold')
    axes2[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path2 = figures_dir / 'trace_comparison_fixed.png'
    plt.savefig(save_path2, dpi=200, bbox_inches='tight')
    print(f"✓ 保存: {save_path2}")
    plt.close()
    
    print("\n完成!")

if __name__ == '__main__':
    main()
