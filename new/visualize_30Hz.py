# -*- coding: utf-8 -*-
"""
30Hz 数据可视化脚本
生成高质量的训练结果对比图像
"""
import numpy as np
import segyio
import torch
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from train_30Hz_thinlayer_v2 import (
    ThinLayerNetV2, load_seismic_data, load_impedance_data, 
    ThinLayerDatasetV2, ThinLayerLabeler, highpass_filter
)

# 设置中文字体和样式
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

def visualize_30Hz():
    """生成30Hz数据的可视化图像"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = Path(r'D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v2')
    figures_dir = result_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("30Hz 数据 - 可视化")
    print("=" * 70)
    
    # 加载数据和模型
    print("\n加载数据...")
    seismic_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'
    impedance_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt'
    
    seismic, dt = load_seismic_data(seismic_path)
    n_traces = seismic.shape[0]
    impedance = load_impedance_data(impedance_path, n_traces)
    
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]
    
    print(f"数据形状: {seismic.shape}")
    
    # 加载归一化参数
    print("加载模型...")
    with open(result_dir / 'norm_stats.json', 'r') as f:
        norm_stats = json.load(f)
    
    # 加载模型
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
    
    labeler = ThinLayerLabeler(dt=0.001, dominant_freq=30.0)
    test_ds = ThinLayerDatasetV2(seismic, impedance, test_idx, norm_stats,
                                  augmentor=None, labeler=labeler)
    
    all_pred = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, y, _, _ = test_ds[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).cpu().numpy()
            all_pred.append(pred[0, 0])
    
    pred_norm = np.array(all_pred)
    print(f"pred_norm 形状: {pred_norm.shape}")
    pred = pred_norm * norm_stats['imp_std'] + norm_stats['imp_mean']
    print(f"pred 形状: {pred.shape}")

    # impedance 本身就是原始值，不需要反归一化
    true = impedance[test_idx]
    print(f"true 形状 (索引后): {true.shape}")
    
    # 对齐形状
    min_traces = min(true.shape[0], pred.shape[0])
    min_samples = min(true.shape[1], pred.shape[1])
    pred = pred[:min_traces, :min_samples]
    true = true[:min_traces, :min_samples]
    
    print(f"对齐后 - pred: {pred.shape}, true: {true.shape}")
    print(f"pred 范围: [{pred.min():.2e}, {pred.max():.2e}]")
    print(f"true 范围: [{true.min():.2e}, {true.max():.2e}]")
    
    # ========== 图1: 测试集截面对比（单道） ==========
    def plot_beautiful_section(title, test_pred, test_true, save_path):
        """绘制高质量的地震截面对比"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=250)
        
        # 预测阻抗
        im1 = axes[0].imshow(test_pred, cmap='jet', aspect='auto', 
                             vmin=np.percentile(test_pred, 3), 
                             vmax=np.percentile(test_pred, 97))
        axes[0].set_title('预测阻抗', fontsize=18, fontweight='bold')
        axes[0].set_xlabel('采样点', fontsize=14)
        axes[0].set_ylabel('地震道', fontsize=14)
        plt.colorbar(im1, ax=axes[0], label='阻抗', shrink=0.8)
        
        # 真实阻抗
        im2 = axes[1].imshow(test_true, cmap='jet', aspect='auto',
                             vmin=np.percentile(test_true, 3),
                             vmax=np.percentile(test_true, 97))
        axes[1].set_title('真实阻抗', fontsize=18, fontweight='bold')
        axes[1].set_xlabel('采样点', fontsize=14)
        axes[1].set_ylabel('地震道', fontsize=14)
        plt.colorbar(im2, ax=axes[1], label='阻抗', shrink=0.8)
        
        # 误差
        error = np.abs(test_pred - test_true)
        im3 = axes[2].imshow(error, cmap='hot', aspect='auto')
        axes[2].set_title('绝对误差', fontsize=18, fontweight='bold')
        axes[2].set_xlabel('采样点', fontsize=14)
        axes[2].set_ylabel('地震道', fontsize=14)
        plt.colorbar(im3, ax=axes[2], label='误差', shrink=0.8)
        
        plt.suptitle(title, fontsize=20, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
        plt.close()
    
    if not (figures_dir / 'beautiful_comparison_test.png').exists():
        plot_beautiful_section(
            '30Hz 数据 - 测试集阻抗预测对比',
            pred, true,
            figures_dir / 'beautiful_comparison_test.png'
        )
    
    # ========== 图2: 全数据集对比 ==========
    all_pred_full = []
    with torch.no_grad():
        for i in range(min(len(test_ds), 50)):  # 抽取前50个道进行可视化
            x, y, _, _ = test_ds[i]
            x = x.unsqueeze(0).to(device)
            pred = model(x).cpu().numpy()
            all_pred_full.append(pred[0, 0])
    
    if all_pred_full:
        pred_all = np.array(all_pred_full) * norm_stats['imp_std'] + norm_stats['imp_mean']
        true_all = impedance[test_idx[:len(all_pred_full)]]  # 原始值，不需反归一化
        
        if not (figures_dir / 'beautiful_comparison_all.png').exists():
            plot_beautiful_section(
                '30Hz 数据 - 截断后数据对比',
                pred_all, true_all,
                figures_dir / 'beautiful_comparison_all.png'
            )
    
    # ========== 图3: 误差分析 ==========
    def plot_error_analysis(pred, true, save_path):
        """绘制误差分析图"""
        print(f"plot_error_analysis - pred: {pred.shape}, true: {true.shape}")
        
        # 确保形状完全一致
        if pred.shape != true.shape:
            print(f"警告: 形状不匹配! 尝试对齐...")
            min_traces = min(pred.shape[0], true.shape[0])
            min_samples = min(pred.shape[1], true.shape[1])
            pred = pred[:min_traces, :min_samples]
            true = true[:min_traces, :min_samples]
            print(f"对齐后 - pred: {pred.shape}, true: {true.shape}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=250)
        
        # 全局误差分布
        error_flat = (pred - true).flatten()
        axes[0, 0].hist(error_flat, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('误差值', fontsize=12)
        axes[0, 0].set_ylabel('频次', fontsize=12)
        axes[0, 0].set_title('误差分布直方图', fontsize=14, fontweight='bold')
        axes[0, 0].axvline(np.mean(error_flat), color='red', linestyle='--', linewidth=2, label=f'均值={np.mean(error_flat):.2f}')
        axes[0, 0].legend()
        
        # 预测 vs 真实散点图（十六进制颜色密度）
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        axes[0, 1].hexbin(true_flat, pred_flat, gridsize=30, cmap='YlOrRd', mincnt=1)
        axes[0, 1].set_xlabel('真实阻抗', fontsize=12)
        axes[0, 1].set_ylabel('预测阻抗', fontsize=12)
        axes[0, 1].set_title('预测 vs 真实（密度图）', fontsize=14, fontweight='bold')
        # 添加对角线
        min_val = min(pred_flat.min(), true_flat.min())
        max_val = max(pred_flat.max(), true_flat.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测')
        axes[0, 1].legend()
        
        # 道均误差
        trace_mse = np.mean((pred - true) ** 2, axis=1)
        axes[1, 0].plot(trace_mse, 'o-', color='darkgreen', markersize=4, linewidth=1.5)
        axes[1, 0].set_xlabel('地震道号', fontsize=12)
        axes[1, 0].set_ylabel('MSE', fontsize=12)
        axes[1, 0].set_title('各道均方误差', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 全局统计
        mae = np.mean(np.abs(pred - true))
        rmse = np.sqrt(np.mean((pred - true) ** 2))
        pcc = np.corrcoef(pred_flat, true_flat)[0, 1]
        
        stats_text = f"""
全局指标统计：

MAE:  {mae:.4f}
RMSE: {rmse:.4f}
PCC:  {pcc:.4f}

数据范围：
预测: [{pred.min():.0f}, {pred.max():.0f}]
真实: [{true.min():.0f}, {true.max():.0f}]
"""
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=13, family='monospace',
                       verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.suptitle('30Hz 数据 - 误差分析', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
        plt.close()
    
    if not (figures_dir / 'error_analysis.png').exists():
        plot_error_analysis(pred, true, figures_dir / 'error_analysis.png')

    # ========== 图4: 单道对比 ==========
    def plot_trace_comparison(pred, true, trace_idx, save_path):
        """绘制单道对比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=250)
        
        # 第一个图：单道波形对比
        samples = np.arange(pred.shape[1])
        axes[0].plot(samples, pred[trace_idx], 'b-', linewidth=2.5, label='预测', alpha=0.8)
        axes[0].plot(samples, true[trace_idx], 'r--', linewidth=2.5, label='真实', alpha=0.8)
        axes[0].fill_between(samples, pred[trace_idx], true[trace_idx], 
                            alpha=0.2, color='gray', label='误差区间')
        axes[0].set_xlabel('采样点', fontsize=14)
        axes[0].set_ylabel('阻抗', fontsize=14)
        axes[0].set_title(f'单道对比 (地震道#{trace_idx})', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=12, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # 第二个图：误差分布
        error_trace = pred[trace_idx] - true[trace_idx]
        axes[1].fill_between(samples, error_trace, alpha=0.6, color='steelblue')
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
        axes[1].set_xlabel('采样点', fontsize=14)
        axes[1].set_ylabel('误差', fontsize=14)
        axes[1].set_title(f'误差时间序列', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        mae_trace = np.mean(np.abs(error_trace))
        rmse_trace = np.sqrt(np.mean(error_trace ** 2))
        axes[1].text(0.02, 0.95, f'MAE={mae_trace:.2f}\nRMSE={rmse_trace:.2f}',
                    transform=axes[1].transAxes, fontsize=12,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.suptitle('30Hz 数据 - 单道详细对比', fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
        print(f"✓ 保存: {save_path}")
        plt.close()
    
    # 选择中间的道进行绘制
    mid_trace = len(pred) // 2
    trace_path = figures_dir / 'trace_comparison.png'
    if not trace_path.exists():
        try:
            plot_trace_comparison(pred, true, mid_trace, trace_path)
        except Exception as exc:
            print(f"生成单道对比图失败: {exc}")
    
    # 保存测试指标
    with open(result_dir / 'test_metrics.json', 'r') as f:
        test_metrics = json.load(f)
    
    metrics_text = f"""
30Hz 数据 - 测试集结果汇总
{'=' * 50}

全局指标：
  PCC (皮尔逊相关系数):  {test_metrics.get('pcc', 'N/A'):.4f}
  R²  (决定系数):        {test_metrics.get('r2', 'N/A'):.4f}
  MSE (均方误差):        {test_metrics.get('mse', 'N/A'):.4f}

薄层指标：
  薄层 PCC:              {test_metrics.get('thin_pcc', 'N/A'):.4f}
  薄层 MSE:              {test_metrics.get('thin_mse', 'N/A'):.4f}
  薄层 F1:               {test_metrics.get('thin_f1', 'N/A'):.4f}
  薄层 Precision:        {test_metrics.get('thin_precision', 'N/A'):.4f}
  薄层 Recall:           {test_metrics.get('thin_recall', 'N/A'):.4f}
  
  双峰距误差 (DPDE):     {test_metrics.get('dpde_mean', 'N/A'):.2f} 采样点
  分离度:                {test_metrics.get('separability_mean', 'N/A'):.4f}

生成日期: 2026-01-03
数据类型: 30Hz 地震数据
模型: ThinLayerNet V2
训练轮数: 500 epochs
"""
    
    with open(figures_dir / 'metrics_summary.txt', 'w', encoding='utf-8') as f:
        f.write(metrics_text)
    
    print("\n" + "=" * 70)
    print("可视化完成！")
    print("=" * 70)
    print(f"\n生成的图像保存在:")
    print(f"  {figures_dir}")
    print(f"  ├── beautiful_comparison_test.png  (测试集对比)")
    print(f"  ├── beautiful_comparison_all.png   (截断数据对比)")
    print(f"  ├── error_analysis.png             (误差分析)")
    print(f"  ├── trace_comparison.png           (单道对比)")
    print(f"  └── metrics_summary.txt            (指标汇总)")

if __name__ == '__main__':
    visualize_30Hz()
