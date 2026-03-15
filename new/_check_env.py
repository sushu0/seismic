import torch, json, os, numpy as np

# Check real checkpoints
real_ckpt = r'D:\SEISMIC_CODING\new\results\real_unet1d_optimized\checkpoints'
if os.path.exists(real_ckpt):
    print('Real checkpoints:', os.listdir(real_ckpt))

# Check real data shapes
for name in ['train_labeled_seis','train_labeled_imp','val_seis','val_imp','test_seis','test_imp']:
    path = rf'D:\SEISMIC_CODING\new\data\real\{name}.npy'
    arr = np.load(path)
    print(f'{name}: {arr.shape}, dtype={arr.dtype}, range=[{arr.min():.4f}, {arr.max():.4f}]')

# norm stats
with open(r'D:\SEISMIC_CODING\new\results\real_unet1d_optimized\norm_stats.json') as f:
    stats = json.load(f)
    print('Norm stats:', stats)

# 01_20Hz data
for name in ['train_labeled_seis','train_labeled_imp']:
    path = rf'D:\SEISMIC_CODING\new\data\01_20Hz\{name}.npy'
    arr = np.load(path)
    print(f'01_20Hz {name}: {arr.shape}')

print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
