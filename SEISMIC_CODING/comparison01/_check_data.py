import os
import numpy as np

print("="*60)
print("数据文件检查 - Data Files Check")
print("="*60)

root = r"D:\SEISMIC_CODING\comparison01"

# 检查原始数据
data_npy = os.path.join(root, "data.npy")
if os.path.exists(data_npy):
    print(f"✓ data.npy 存在")
    data = np.load(data_npy, allow_pickle=True).item()
    print(f"  键: {list(data.keys())}")
    for k, v in data.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
else:
    print(f"✗ data.npy 不存在")

print()

# 检查训练数据
seis_path = os.path.join(root, "seismic.npy")
imp_path = os.path.join(root, "impedance.npy")

if os.path.exists(seis_path) and os.path.exists(imp_path):
    seismic = np.load(seis_path)
    impedance = np.load(imp_path)
    print(f"✓ seismic.npy: shape={seismic.shape}, dtype={seismic.dtype}")
    print(f"  值范围: [{seismic.min():.6f}, {seismic.max():.6f}]")
    print(f"✓ impedance.npy: shape={impedance.shape}, dtype={impedance.dtype}")
    print(f"  值范围: [{impedance.min():.2f}, {impedance.max():.2f}]")
    
    if seismic.shape == impedance.shape:
        print(f"✓ Shape一致性检查通过")
        T, Nx = seismic.shape
        print(f"  时间采样点 T={T}, 道数 Nx={Nx}")
    else:
        print(f"✗ Shape不一致！seismic={seismic.shape}, impedance={impedance.shape}")
else:
    print(f"✗ 训练数据文件不完整")
    print(f"  seismic.npy: {'存在' if os.path.exists(seis_path) else '不存在'}")
    print(f"  impedance.npy: {'存在' if os.path.exists(imp_path) else '不存在'}")
    print(f"\n需要运行: python split_marmousi2_from_data_npy.py")

print()

# 检查归一化参数
norm_path = os.path.join(root, "norm_params.json")
if os.path.exists(norm_path):
    import json
    with open(norm_path, 'r') as f:
        norm = json.load(f)
    print(f"✓ norm_params.json 存在")
    print(f"  seis范围: [{norm['seis_min']:.6f}, {norm['seis_max']:.6f}]")
    print(f"  imp范围: [{norm['imp_min']:.2f}, {norm['imp_max']:.2f}]")
else:
    print(f"✗ norm_params.json 不存在（训练后会生成）")

print("="*60)
