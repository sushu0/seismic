"""
从data.npy准备真实数据集
"""
import numpy as np
from pathlib import Path

# 加载data.npy
print("加载data.npy...")
data = np.load('data.npy', allow_pickle=True).item()

seismic = data['seismic']  # (2721, 1, 470)
impedance = data['acoustic_impedance']  # (2721, 1, 1880)

print(f"地震数据形状: {seismic.shape}")
print(f"阻抗数据形状: {impedance.shape}")
print(f"地震数据范围: [{seismic.min():.3f}, {seismic.max():.3f}]")
print(f"阻抗数据范围: [{impedance.min():.0f}, {impedance.max():.0f}]")

# 注意：seismic是470采样点，impedance是1880采样点
# 需要对齐或者只使用impedance的部分采样点

# 方案：对impedance进行下采样，使其与seismic长度匹配
# 1880 / 470 = 4，所以每4个点取1个
impedance_downsampled = impedance[:, :, ::4][:, :, :470]
print(f"下采样后阻抗形状: {impedance_downsampled.shape}")

# 移除channel维度 (2721, 1, 470) -> (2721, 470)
seismic = seismic[:, 0, :]
impedance = impedance_downsampled[:, 0, :]

print(f"最终地震形状: {seismic.shape}")
print(f"最终阻抗形状: {impedance.shape}")

# 划分数据集: 80% train, 10% val, 10% test
n_total = len(seismic)
n_train = int(0.8 * n_total)
n_val = int(0.1 * n_total)
n_test = n_total - n_train - n_val

print(f"\n数据集划分: train={n_train}, val={n_val}, test={n_test}")

# 随机打乱
np.random.seed(42)
indices = np.random.permutation(n_total)

train_idx = indices[:n_train]
val_idx = indices[n_train:n_train+n_val]
test_idx = indices[n_train+n_val:]

# 保存数据
out_dir = Path('data/real')
out_dir.mkdir(parents=True, exist_ok=True)

np.save(out_dir / 'train_labeled_seis.npy', seismic[train_idx])
np.save(out_dir / 'train_labeled_imp.npy', impedance[train_idx])
np.save(out_dir / 'val_seis.npy', seismic[val_idx])
np.save(out_dir / 'val_imp.npy', impedance[val_idx])
np.save(out_dir / 'test_seis.npy', seismic[test_idx])
np.save(out_dir / 'test_imp.npy', impedance[test_idx])

print(f"\n数据已保存到: {out_dir}")
print("文件列表:")
for f in sorted(out_dir.glob('*.npy')):
    d = np.load(f)
    print(f"  {f.name}: {d.shape}")
