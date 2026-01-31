"""
诊断可视化脚本 - 检查数据形状
"""
import numpy as np
import json
import torch
from pathlib import Path
import sys
sys.path.insert(0, str(Path('.').resolve()))

from src.data import load_seismic_data, load_impedance_data

# 配置
result_dir = Path('results/01_30Hz_thinlayer_v2')
seismic_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'
impedance_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt'
norm_file = result_dir / 'norm_stats.json'
ckpt_file = result_dir / 'checkpoints' / 'best.pt'

print("=" * 60)
print("诊断可视化脚本")
print("=" * 60)

# 加载数据
print("\n1. 加载数据...")
seismic, dt = load_seismic_data(seismic_path)
n_traces = seismic.shape[0]
impedance = load_impedance_data(impedance_path, n_traces)

min_len = min(seismic.shape[1], impedance.shape[1])
seismic = seismic[:, :min_len]
impedance = impedance[:, :min_len]

print(f"   seismic 形状: {seismic.shape}")
print(f"   impedance 形状: {impedance.shape}")
print(f"   impedance 范围: [{impedance.min():.2e}, {impedance.max():.2e}]")

# 加载归一化参数
print("\n2. 加载归一化参数...")
with open(norm_file) as f:
    norm_stats = json.load(f)
print(f"   imp_mean: {norm_stats['imp_mean']:.2e}")
print(f"   imp_std: {norm_stats['imp_std']:.2e}")

# 计算测试集索引
print("\n3. 计算测试集索引...")
np.random.seed(42)
n_traces = seismic.shape[0]
indices = np.random.permutation(n_traces)
n_train = int(n_traces * 0.6)
n_val = int(n_traces * 0.2)
test_idx = indices[n_train + n_val:]
print(f"   测试集道数: {len(test_idx)}")
print(f"   测试集索引: {test_idx}")

# 提取测试集真实值
print("\n4. 提取测试集真实阻抗...")
true_impedance = impedance[test_idx]
print(f"   形状: {true_impedance.shape}")
print(f"   范围: [{true_impedance.min():.2e}, {true_impedance.max():.2e}]")

# 加载模型并预测
print("\n5. 加载模型...")
import sys
sys.path.insert(0, str(Path('.').resolve()))
from src.model import ThinLayerNetV2
from src.data import ThinLayerDatasetV2, ThinLayerLabeler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   设备: {device}")

checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
model.load_state_dict(checkpoint['model'])
model.eval()

# 生成预测
print("\n6. 生成预测...")
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
print(f"   pred_norm 形状: {pred_norm.shape}")
print(f"   pred_norm 范围: [{pred_norm.min():.4f}, {pred_norm.max():.4f}]")

# 反归一化预测值
pred = pred_norm * norm_stats['imp_std'] + norm_stats['imp_mean']
print(f"   pred (反归一化后) 形状: {pred.shape}")
print(f"   pred (反归一化后) 范围: [{pred.min():.2e}, {pred.max():.2e}]")

# 对比形状
print("\n7. 形状对比...")
print(f"   pred 形状: {pred.shape}")
print(f"   true 形状: {true_impedance.shape}")

if pred.shape == true_impedance.shape:
    print("   ✓ 形状匹配!")
else:
    print("   ✗ 形状不匹配!")
    print("   尝试对齐...")
    min_traces = min(pred.shape[0], true_impedance.shape[0])
    min_samples = min(pred.shape[1], true_impedance.shape[1])
    pred_aligned = pred[:min_traces, :min_samples]
    true_aligned = true_impedance[:min_traces, :min_samples]
    print(f"   对齐后 pred: {pred_aligned.shape}")
    print(f"   对齐后 true: {true_aligned.shape}")

# 简单可视化测试
print("\n8. 生成简单可视化...")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 使用对齐后的数据
if pred.shape != true_impedance.shape:
    pred_plot = pred_aligned
    true_plot = true_aligned
else:
    pred_plot = pred
    true_plot = true_impedance

# 图1: 单道对比
axes[0].plot(true_plot[0], 'b-', label='True', alpha=0.7)
axes[0].plot(pred_plot[0], 'r-', label='Pred', alpha=0.7)
axes[0].set_xlabel('Sample')
axes[0].set_ylabel('Impedance')
axes[0].set_title('Trace 0 Comparison')
axes[0].legend()

# 图2: 散点图
pred_flat = pred_plot.flatten()
true_flat = true_plot.flatten()
print(f"   pred_flat 大小: {pred_flat.shape}")
print(f"   true_flat 大小: {true_flat.shape}")

# 采样以避免过多点
idx = np.random.choice(len(pred_flat), min(10000, len(pred_flat)), replace=False)
axes[1].scatter(true_flat[idx], pred_flat[idx], alpha=0.1, s=1)
axes[1].plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()], 'r--')
axes[1].set_xlabel('True Impedance')
axes[1].set_ylabel('Predicted Impedance')
axes[1].set_title('Scatter Plot')

# 图3: 误差直方图
error = pred_plot - true_plot
axes[2].hist(error.flatten(), bins=100)
axes[2].set_xlabel('Error')
axes[2].set_ylabel('Count')
axes[2].set_title('Error Distribution')

plt.tight_layout()
save_path = result_dir / 'figures' / 'debug_plot.png'
plt.savefig(save_path, dpi=150)
print(f"   ✓ 保存: {save_path}")

# 计算指标
print("\n9. 计算指标...")
mse = np.mean((pred_plot - true_plot) ** 2)
pcc = np.corrcoef(pred_flat, true_flat)[0, 1]
print(f"   MSE: {mse:.4e}")
print(f"   PCC: {pcc:.4f}")

print("\n完成!")
