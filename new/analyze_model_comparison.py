"""
分析训练结果与真实地质模型的对比
"""
import numpy as np
import matplotlib.pyplot as plt
import segyio
import json
import torch
from pathlib import Path
from scipy.stats import pearsonr

# 加载数据
print("加载数据...")
with segyio.open(r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy', 'r', ignore_geometry=True, strict=False) as f:
    seismic = np.stack([f.trace[i] for i in range(f.tracecount)]).astype(np.float32)

imp_raw = np.loadtxt(r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt', usecols=4, skiprows=1)
impedance = imp_raw.reshape(100, 10001).astype(np.float32)

print(f"地震数据形状: {seismic.shape}")
print(f"阻抗数据形状: {impedance.shape}")

# 关键参数
# 真实模型图显示: X轴 Shot Number 0-2000, Y轴 Time 0-100ms
# 所以: 100道对应2000 shot (每道=20 shot), 10001采样点对应100ms (dt=0.01ms)
DT = 0.01  # ms
N_TRACES = 100
N_SAMPLES = 10001

print(f"\n采样间隔: {DT} ms")
print(f"总时间: {N_SAMPLES * DT:.1f} ms")

# ============ 创建与真实模型相同坐标系的可视化 ============
fig = plt.figure(figsize=(16, 12))

# 1. 训练数据阻抗（与真实模型相同坐标系）
ax1 = fig.add_subplot(2, 2, 1)
im1 = ax1.imshow(impedance.T, aspect='auto', cmap='jet',
                 vmin=6.5e6, vmax=1e7,
                 extent=[0, 2000, 100, 0])  # Shot 0-2000, Time 0-100ms
ax1.set_xlabel('Shot Number')
ax1.set_ylabel('Time (ms)')
ax1.set_title('Training Data - Impedance Model')
plt.colorbar(im1, ax=ax1, label='Impedance (m/s·g/cm³)')

# 2. 聚焦薄层区域 (45-70 ms)
ax2 = fig.add_subplot(2, 2, 2)
# 45-70ms 对应采样点 4500-7000
start_sample = 4500
end_sample = 7000
im2 = ax2.imshow(impedance[:, start_sample:end_sample].T, aspect='auto', cmap='jet',
                 vmin=6.5e6, vmax=1e7,
                 extent=[0, 2000, end_sample*DT, start_sample*DT])
ax2.set_xlabel('Shot Number')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Training Data - Thin Layer Zone (45-70 ms)')
plt.colorbar(im2, ax=ax2, label='Impedance (m/s·g/cm³)')

# 加载预测结果
result_dir = Path('results/01_30Hz_verified')
with open(result_dir / 'norm_stats.json') as f:
    norm = json.load(f)

# 测试集索引
np.random.seed(42)
indices = np.random.permutation(100)
train_idx = indices[:60]
val_idx = indices[60:80]
test_idx = indices[80:]

print(f"\n测试集索引: {sorted(test_idx)}")

# 加载模型
from train_30Hz_from_20Hz_script import ThinLayerNetV2, highpass_filter

model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1)
ckpt = torch.load(result_dir / 'checkpoints' / 'best.pt', map_location='cpu', weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()

# 预测
seismic_hf = highpass_filter(seismic, cutoff=12, fs=1000)

predictions = []
ground_truth = []
test_indices_sorted = sorted(test_idx)

for idx in test_indices_sorted:
    seis = seismic[idx]
    seis_hf = seismic_hf[idx]
    imp = impedance[idx]
    
    seis_norm = (seis - norm['seis_mean']) / (norm['seis_std'] + 1e-6)
    seis_hf_norm = seis_hf / (np.std(seis_hf) + 1e-6)
    
    x = np.stack([seis_norm, seis_hf_norm], axis=0)
    x = torch.from_numpy(x).float().unsqueeze(0)
    
    with torch.no_grad():
        pred = model(x)
    
    pred_denorm = pred.numpy().squeeze() * norm['imp_std'] + norm['imp_mean']
    predictions.append(pred_denorm)
    ground_truth.append(imp)

pred_arr = np.array(predictions)
true_arr = np.array(ground_truth)

print(f"\n预测形状: {pred_arr.shape}")
print(f"预测范围: [{pred_arr.min():.2e}, {pred_arr.max():.2e}]")
print(f"真实范围: [{true_arr.min():.2e}, {true_arr.max():.2e}]")

# 计算指标
pred_flat = pred_arr.flatten()
true_flat = true_arr.flatten()
pcc, _ = pearsonr(pred_flat, true_flat)
ss_res = np.sum((pred_flat - true_flat)**2)
ss_tot = np.sum((true_flat - true_flat.mean())**2)
r2 = 1 - ss_res/ss_tot

print(f"\nPCC: {pcc:.4f}")
print(f"R²:  {r2:.4f}")

# 3. 预测结果
ax3 = fig.add_subplot(2, 2, 3)
# 将测试集预测按原始索引排列以便可视化
pred_full = np.full((100, 10001), np.nan)
for i, idx in enumerate(test_indices_sorted):
    pred_full[idx] = pred_arr[i]

im3 = ax3.imshow(pred_full.T, aspect='auto', cmap='jet',
                 vmin=6.5e6, vmax=1e7,
                 extent=[0, 2000, 100, 0])
ax3.set_xlabel('Shot Number')
ax3.set_ylabel('Time (ms)')
ax3.set_title(f'Predicted Impedance (Test Set, PCC={pcc:.3f})')
plt.colorbar(im3, ax=ax3, label='Impedance (m/s·g/cm³)')

# 4. 单道对比
ax4 = fig.add_subplot(2, 2, 4)
# 选择一个有薄层的测试道
trace_with_thin = None
for i, idx in enumerate(test_indices_sorted):
    if np.any(true_arr[i] > 7.5e6):
        trace_with_thin = i
        trace_idx = idx
        break

if trace_with_thin is not None:
    time_axis = np.arange(10001) * DT
    ax4.plot(time_axis, true_arr[trace_with_thin], 'b-', label='True', linewidth=1)
    ax4.plot(time_axis, pred_arr[trace_with_thin], 'r--', label='Predicted', linewidth=1)
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Impedance')
    ax4.set_title(f'Trace {trace_idx} Comparison')
    ax4.legend()
    ax4.set_xlim(40, 80)  # 聚焦薄层区域
    ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(result_dir / 'figures' / 'model_comparison_analysis.png', dpi=150)
print(f"\n已保存: {result_dir / 'figures' / 'model_comparison_analysis.png'}")

# ============ 详细误差分析 ============
print("\n" + "="*60)
print("详细误差分析")
print("="*60)

# 薄层区域分析
threshold = 7.5e6
thin_mask = true_arr > threshold
non_thin_mask = ~thin_mask

print(f"\n薄层区域 (阻抗 > {threshold:.1e}):")
print(f"  占比: {thin_mask.sum() / thin_mask.size * 100:.2f}%")

if thin_mask.sum() > 0:
    thin_pred = pred_arr[thin_mask]
    thin_true = true_arr[thin_mask]
    thin_pcc, _ = pearsonr(thin_pred, thin_true)
    thin_mae = np.mean(np.abs(thin_pred - thin_true))
    print(f"  PCC: {thin_pcc:.4f}")
    print(f"  MAE: {thin_mae:.2e}")

print(f"\n非薄层区域:")
print(f"  占比: {non_thin_mask.sum() / non_thin_mask.size * 100:.2f}%")

if non_thin_mask.sum() > 0:
    non_thin_pred = pred_arr[non_thin_mask]
    non_thin_true = true_arr[non_thin_mask]
    non_thin_pcc, _ = pearsonr(non_thin_pred, non_thin_true)
    non_thin_mae = np.mean(np.abs(non_thin_pred - non_thin_true))
    print(f"  PCC: {non_thin_pcc:.4f}")
    print(f"  MAE: {non_thin_mae:.2e}")

# 检查预测是否正确识别了薄层位置
print("\n薄层识别分析:")
for i, idx in enumerate(test_indices_sorted[:5]):
    true_trace = true_arr[i]
    pred_trace = pred_arr[i]
    
    true_thin = np.where(true_trace > threshold)[0]
    pred_thin = np.where(pred_trace > threshold)[0]
    
    if len(true_thin) > 0:
        true_center = (true_thin.min() + true_thin.max()) // 2
        true_thickness = true_thin.max() - true_thin.min()
        
        if len(pred_thin) > 0:
            pred_center = (pred_thin.min() + pred_thin.max()) // 2
            pred_thickness = pred_thin.max() - pred_thin.min()
            center_error = pred_center - true_center
            thickness_error = pred_thickness - true_thickness
            print(f"  道{idx}: 真实中心={true_center*DT:.1f}ms 厚度={true_thickness} | "
                  f"预测中心={pred_center*DT:.1f}ms 厚度={pred_thickness} | "
                  f"中心误差={center_error}点 厚度误差={thickness_error}点")
        else:
            print(f"  道{idx}: 真实有薄层(中心={true_center*DT:.1f}ms)，预测未识别!")
    else:
        if len(pred_thin) > 0:
            pred_center = (pred_thin.min() + pred_thin.max()) // 2
            print(f"  道{idx}: 真实无薄层，预测误报(中心={pred_center*DT:.1f}ms)")
        else:
            print(f"  道{idx}: 真实无薄层，预测正确")

plt.show()
