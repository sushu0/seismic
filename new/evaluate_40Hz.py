"""
40Hz模型评估和可视化脚本
"""
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

# 添加路径
sys.path.insert(0, 'D:/SEISMIC_CODING/new')

# 从训练脚本导入需要的类
exec(open('D:/SEISMIC_CODING/new/train_40Hz_thinlayer.py', encoding='utf-8').read().split('if __name__')[0])

# 配置
MODEL_PATH = 'D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer/checkpoints/best.pt'
SEISMIC_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_40Hz_re.sgy'
IMPEDANCE_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_40Hz_04.txt'
NORM_STATS_PATH = 'D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer/norm_stats.json'
OUTPUT_DIR = 'D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer'

print("="*50)
print("40Hz 模型评估")
print("="*50)

# 加载设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# 加载数据
import segyio
with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
    seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)

# 40Hz阻抗数据: 5列格式, 第5列(索引4)是阻抗值
raw_imp = np.loadtxt(IMPEDANCE_PATH, usecols=4, skiprows=1).astype(np.float32)
n_traces = seismic.shape[0]
n_samples = len(raw_imp) // n_traces
impedance = raw_imp.reshape(n_traces, n_samples)

print(f"数据形状: seismic={seismic.shape}, impedance={impedance.shape}")

# 加载归一化参数 (使用 mean/std)
with open(NORM_STATS_PATH, 'r') as f:
    norm_stats = json.load(f)

# 归一化
seis_norm = (seismic - norm_stats['seis_mean']) / norm_stats['seis_std']
imp_norm = (impedance - norm_stats['imp_mean']) / norm_stats['imp_std']

# 划分数据集 (与训练相同: 60/20/20)
n_traces = seismic.shape[0]
n_train = int(n_traces * 0.6)
n_val = int(n_traces * 0.2)

test_seis = seis_norm[n_train + n_val:]
test_imp = imp_norm[n_train + n_val:]
test_imp_raw = impedance[n_train + n_val:]

print(f"测试集大小: {len(test_seis)}")

# 加载模型
model = ThinLayerNetV2(1, 1).to(device)
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
best_epoch = ckpt.get('epoch', 'N/A')
best_val_pcc = ckpt.get('val_metrics', {}).get('pcc', 'N/A')
print(f"加载模型 (Epoch {best_epoch})")

# 推理
all_pred = []
all_true = []
with torch.no_grad():
    for i in range(len(test_seis)):
        x = torch.from_numpy(test_seis[i:i+1]).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x)
        pred_np = pred.cpu().numpy().squeeze()
        all_pred.append(pred_np)
        all_true.append(test_imp[i])

pred_arr = np.array(all_pred)
true_arr = np.array(all_true)

# 反归一化
pred_denorm = pred_arr * norm_stats['imp_std'] + norm_stats['imp_mean']
true_denorm = true_arr * norm_stats['imp_std'] + norm_stats['imp_mean']

# 计算指标
from scipy.stats import pearsonr
mse = np.mean((pred_denorm - true_denorm) ** 2)
pcc, _ = pearsonr(pred_denorm.flatten(), true_denorm.flatten())
ss_res = np.sum((true_denorm - pred_denorm) ** 2)
ss_tot = np.sum((true_denorm - np.mean(true_denorm)) ** 2)
r2 = 1 - ss_res / ss_tot

print("\n" + "="*50)
print("测试集评估结果:")
print("="*50)
print(f"MSE:  {mse:.6f}")
print(f"PCC:  {pcc:.4f}")
print(f"R²:   {r2:.4f}")

# 保存完整的指标
test_metrics = {
    'mse': float(mse),
    'pcc': float(pcc),
    'r2': float(r2),
    'best_epoch': best_epoch,
    'best_val_pcc': best_val_pcc
}
with open(f'{OUTPUT_DIR}/test_metrics_full.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

# 全数据集评估
print("\n评估全数据集...")
all_pred_full = []
with torch.no_grad():
    for i in range(n_traces):
        x = torch.from_numpy(seis_norm[i:i+1]).unsqueeze(0).unsqueeze(0).to(device)
        pred = model(x)
        pred_np = pred.cpu().numpy().squeeze()
        all_pred_full.append(pred_np)

pred_full = np.array(all_pred_full)
pred_full_denorm = pred_full * norm_stats['imp_std'] + norm_stats['imp_mean']

pcc_full, _ = pearsonr(pred_full_denorm.flatten(), impedance.flatten())
ss_res_full = np.sum((impedance - pred_full_denorm) ** 2)
ss_tot_full = np.sum((impedance - np.mean(impedance)) ** 2)
r2_full = 1 - ss_res_full / ss_tot_full

print(f"\n全数据集 ({n_traces} traces):")
print(f"PCC:  {pcc_full:.4f}")
print(f"R²:   {r2_full:.4f}")

# ========== 可视化（使用正确的坐标系统）==========
print("\n生成可视化...")

# 坐标转换参数
dt_ms = 0.01  # 采样间隔 ms
n_samples = impedance.shape[1]  # 10001
total_time = n_samples * dt_ms  # 约100.01 ms
shot_per_trace = 20  # 每个trace代表20个shot

# 剖面对比图
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# 创建extent (Shot Number x Time)
extent = [0, n_traces * shot_per_trace, total_time, 0]

# 真实阻抗
vmin, vmax = impedance.min(), impedance.max()
im0 = axes[0].imshow(impedance.T, aspect='auto', cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
axes[0].set_title('True Impedance', fontsize=14)
axes[0].set_xlabel('Shot Number', fontsize=12)
axes[0].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im0, ax=axes[0], label='Impedance')

# 预测阻抗
im1 = axes[1].imshow(pred_full_denorm.T, aspect='auto', cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
axes[1].set_title('Predicted Impedance', fontsize=14)
axes[1].set_xlabel('Shot Number', fontsize=12)
axes[1].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im1, ax=axes[1], label='Impedance')

# 差异
diff = pred_full_denorm - impedance
diff_max = np.percentile(np.abs(diff), 99)
im2 = axes[2].imshow(diff.T, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-diff_max, vmax=diff_max)
axes[2].set_title('Difference (Pred - True)', fontsize=14)
axes[2].set_xlabel('Shot Number', fontsize=12)
axes[2].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im2, ax=axes[2], label='Difference')

plt.suptitle(f'40Hz Model: PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/section_comparison_40Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: section_comparison_40Hz.png")

# 薄层区域放大图 (40Hz分辨率更高，选择30-60ms区域)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
t_start, t_end = 30, 60  # ms
sample_start = int(t_start / dt_ms)
sample_end = int(t_end / dt_ms)

extent_zoom = [0, n_traces * shot_per_trace, t_end, t_start]

imp_zoom = impedance[:, sample_start:sample_end]
pred_zoom = pred_full_denorm[:, sample_start:sample_end]

vmin_z, vmax_z = imp_zoom.min(), imp_zoom.max()

im0 = axes[0].imshow(imp_zoom.T, aspect='auto', cmap='seismic', extent=extent_zoom, vmin=vmin_z, vmax=vmax_z)
axes[0].set_title('True Impedance (30-60 ms)', fontsize=14)
axes[0].set_xlabel('Shot Number', fontsize=12)
axes[0].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(pred_zoom.T, aspect='auto', cmap='seismic', extent=extent_zoom, vmin=vmin_z, vmax=vmax_z)
axes[1].set_title('Predicted Impedance (30-60 ms)', fontsize=14)
axes[1].set_xlabel('Shot Number', fontsize=12)
axes[1].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im1, ax=axes[1])

diff_zoom = pred_zoom - imp_zoom
diff_max_z = np.percentile(np.abs(diff_zoom), 99)
im2 = axes[2].imshow(diff_zoom.T, aspect='auto', cmap='RdBu_r', extent=extent_zoom, vmin=-diff_max_z, vmax=diff_max_z)
axes[2].set_title('Difference (30-60 ms)', fontsize=14)
axes[2].set_xlabel('Shot Number', fontsize=12)
axes[2].set_ylabel('Time (ms)', fontsize=12)
plt.colorbar(im2, ax=axes[2])

plt.suptitle('40Hz Model - Thin Layer Zone', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/thin_layer_zone_40Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: thin_layer_zone_40Hz.png")

# 道对比图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
test_traces = [10, 50, 90]  # 选择不同位置的道
time_axis = np.arange(n_samples) * dt_ms

for idx, trace_idx in enumerate(test_traces):
    # 完整道
    ax = axes[0, idx]
    ax.plot(impedance[trace_idx], time_axis, 'b-', linewidth=1.5, label='True')
    ax.plot(pred_full_denorm[trace_idx], time_axis, 'r--', linewidth=1.5, label='Predicted')
    ax.set_xlabel('Impedance', fontsize=11)
    ax.set_ylabel('Time (ms)', fontsize=11)
    ax.set_title(f'Trace {trace_idx} (Shot {trace_idx*shot_per_trace})', fontsize=12)
    ax.invert_yaxis()
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 局部相关系数
    trace_pcc, _ = pearsonr(impedance[trace_idx], pred_full_denorm[trace_idx])
    ax.text(0.05, 0.95, f'PCC={trace_pcc:.4f}', transform=ax.transAxes, fontsize=10, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 薄层区域放大
    ax2 = axes[1, idx]
    ax2.plot(imp_zoom[trace_idx], time_axis[sample_start:sample_end], 'b-', linewidth=1.5, label='True')
    ax2.plot(pred_zoom[trace_idx], time_axis[sample_start:sample_end], 'r--', linewidth=1.5, label='Predicted')
    ax2.set_xlabel('Impedance', fontsize=11)
    ax2.set_ylabel('Time (ms)', fontsize=11)
    ax2.set_title(f'Trace {trace_idx} (30-60 ms)', fontsize=12)
    ax2.invert_yaxis()
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

plt.suptitle(f'40Hz Model - Trace Comparison\nOverall PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/trace_comparison_40Hz.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"保存: trace_comparison_40Hz.png")

print("\n" + "="*50)
print("40Hz 评估完成!")
print("="*50)
print(f"\n输出目录: {OUTPUT_DIR}")
print(f"- section_comparison_40Hz.png")
print(f"- thin_layer_zone_40Hz.png")
print(f"- trace_comparison_40Hz.png")
print(f"- test_metrics_full.json")
