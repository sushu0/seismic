"""
完整可视化脚本 - 30Hz训练结果
独立运行，不依赖缓存
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import sys
import os
from pathlib import Path

# 确保可以导入模块
os.chdir(r'D:\SEISMIC_CODING\new')
sys.path.insert(0, r'D:\SEISMIC_CODING\new')

from train_30Hz_thinlayer_v2 import (
    ThinLayerNetV2, load_seismic_data, load_impedance_data, 
    ThinLayerDatasetV2, ThinLayerLabeler
)

# 配置 - 使用交叉验证结果
result_dir = Path('results/01_30Hz_verified')
seismic_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'
impedance_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt'
figures_dir = result_dir / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("30Hz 交叉验证结果可视化 (使用20Hz脚本训练)")
print("=" * 60)

# 1. 加载数据
print("\n[1/6] 加载数据...")
seismic, dt = load_seismic_data(seismic_path)
n_traces = seismic.shape[0]
impedance = load_impedance_data(impedance_path, n_traces)

min_len = min(seismic.shape[1], impedance.shape[1])
seismic = seismic[:, :min_len]
impedance = impedance[:, :min_len]

print(f"  seismic: {seismic.shape}")
print(f"  impedance: {impedance.shape}")
print(f"  impedance范围: [{impedance.min():.2e}, {impedance.max():.2e}]")

# 2. 加载归一化参数
print("\n[2/6] 加载归一化参数...")
with open(result_dir / 'norm_stats.json', 'r') as f:
    norm_stats = json.load(f)
print(f"  imp_mean: {norm_stats['imp_mean']:.2e}")
print(f"  imp_std: {norm_stats['imp_std']:.2e}")

# 3. 计算测试集索引
print("\n[3/6] 计算测试集索引...")
np.random.seed(42)
indices = np.random.permutation(n_traces)
n_train = int(n_traces * 0.6)
n_val = int(n_traces * 0.2)
test_idx = indices[n_train + n_val:]
print(f"  测试道数: {len(test_idx)}")

# 4. 加载模型
print("\n[4/6] 加载模型...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  设备: {device}")

checkpoint = torch.load(result_dir / 'checkpoints' / 'best.pt', 
                       map_location=device, weights_only=False)
model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# 5. 生成预测
print("\n[5/6] 生成预测...")
labeler = ThinLayerLabeler(dt=0.001, dominant_freq=30.0)
test_ds = ThinLayerDatasetV2(seismic, impedance, test_idx, norm_stats,
                             augmentor=None, labeler=labeler)

all_pred = []
with torch.no_grad():
    for i in range(len(test_ds)):
        x, y, _, _ = test_ds[i]
        x = x.unsqueeze(0).to(device)
        p = model(x).cpu().numpy()
        all_pred.append(p[0, 0])

# 预测值反归一化
pred = np.array(all_pred) * norm_stats['imp_std'] + norm_stats['imp_mean']
# 真实值直接使用原始数据
true = impedance[test_idx]

print(f"  pred形状: {pred.shape}")
print(f"  true形状: {true.shape}")
print(f"  pred范围: [{pred.min():.2e}, {pred.max():.2e}]")
print(f"  true范围: [{true.min():.2e}, {true.max():.2e}]")

# 确保形状一致
if pred.shape != true.shape:
    print("  警告: 形状不一致，对齐中...")
    min_t = min(pred.shape[0], true.shape[0])
    min_s = min(pred.shape[1], true.shape[1])
    pred = pred[:min_t, :min_s]
    true = true[:min_t, :min_s]
    print(f"  对齐后 pred: {pred.shape}, true: {true.shape}")

# 6. 生成可视化
print("\n[6/6] 生成可视化...")

# 图1: 剖面对比
print("  生成剖面对比图...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=200)

im0 = axes[0].imshow(pred.T, aspect='auto', cmap='jet')
axes[0].set_title('Predicted Impedance', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Trace')
axes[0].set_ylabel('Sample')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

im1 = axes[1].imshow(true.T, aspect='auto', cmap='jet')
axes[1].set_title('True Impedance', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Trace')
axes[1].set_ylabel('Sample')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

error = pred - true
im2 = axes[2].imshow(error.T, aspect='auto', cmap='bwr')
axes[2].set_title('Error (Pred - True)', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Trace')
axes[2].set_ylabel('Sample')
plt.colorbar(im2, ax=axes[2], shrink=0.8)

plt.suptitle('30Hz Data - Test Set Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'section_comparison_fixed.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {figures_dir / 'section_comparison_fixed.png'}")

# 图2: 误差分析
print("  生成误差分析图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

# 误差直方图
error_flat = error.flatten()
axes[0, 0].hist(error_flat, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].axvline(np.mean(error_flat), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean={np.mean(error_flat):.2e}')
axes[0, 0].set_xlabel('Error Value')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Error Distribution', fontweight='bold')
axes[0, 0].legend()

# 散点图
pred_flat = pred.flatten()
true_flat = true.flatten()
print(f"  pred_flat: {pred_flat.shape}, true_flat: {true_flat.shape}")

# 采样避免太多点
n_pts = min(50000, len(pred_flat))
idx = np.random.choice(len(pred_flat), n_pts, replace=False)
axes[0, 1].scatter(true_flat[idx], pred_flat[idx], alpha=0.1, s=1)
vmin, vmax = true_flat.min(), true_flat.max()
axes[0, 1].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=2, label='Perfect Prediction')
axes[0, 1].set_xlabel('True Impedance')
axes[0, 1].set_ylabel('Predicted Impedance')
axes[0, 1].set_title('Prediction vs True', fontweight='bold')
axes[0, 1].legend()

# 道均MSE
trace_mse = np.mean(error ** 2, axis=1)
axes[1, 0].plot(trace_mse, 'o-', color='darkgreen', markersize=4, linewidth=1.5)
axes[1, 0].set_xlabel('Trace Number')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_title('MSE per Trace', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 统计信息
mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error ** 2))
pcc = np.corrcoef(pred_flat, true_flat)[0, 1]
r2 = 1 - np.sum(error ** 2) / np.sum((true - np.mean(true)) ** 2)

stats_text = f"""
Metrics Summary:

MAE:  {mae:.2e}
RMSE: {rmse:.2e}
PCC:  {pcc:.4f}
R²:   {r2:.4f}

Data Range:
Pred: [{pred.min():.2e}, {pred.max():.2e}]
True: [{true.min():.2e}, {true.max():.2e}]
"""
axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
               verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.suptitle('30Hz Data - Error Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'error_analysis_fixed.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {figures_dir / 'error_analysis_fixed.png'}")

# 图3: 单道对比
print("  生成单道对比图...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)

for i, trace_i in enumerate([0, 5, 10, 15]):
    ax = axes[i // 2, i % 2]
    if trace_i < pred.shape[0]:
        ax.plot(true[trace_i], 'b-', label='True', alpha=0.8, linewidth=1)
        ax.plot(pred[trace_i], 'r-', label='Pred', alpha=0.8, linewidth=1)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Impedance')
        ax.set_title(f'Trace {trace_i}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.suptitle('30Hz Data - Trace Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(figures_dir / 'trace_comparison_fixed.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"  ✓ {figures_dir / 'trace_comparison_fixed.png'}")

# 保存指标
metrics = {
    'mae': float(mae),
    'rmse': float(rmse),
    'pcc': float(pcc),
    'r2': float(r2),
    'pred_min': float(pred.min()),
    'pred_max': float(pred.max()),
    'true_min': float(true.min()),
    'true_max': float(true.max()),
}
with open(figures_dir / 'metrics_fixed.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"  ✓ {figures_dir / 'metrics_fixed.json'}")

print("\n" + "=" * 60)
print("完成! 所有可视化已保存到:")
print(f"  {figures_dir}")
print("=" * 60)
