import segyio, numpy as np, pandas as pd

# 1. 读 SEG-Y
with segyio.open('MODEL_P-WAVE_VELOCITY_1.25m.segy') as f:
    vp = segyio.tools.collect(f.trace[:]).T    # 转置成 (time, trace)

with segyio.open('MODEL_DENSITY_1.25m.segy') as f:
    rho = segyio.tools.collect(f.trace[:]).T   # 同上

# 2. 坐标轴
nx, nz = vp.shape
dx = dz = 1.25                                   # 米
x = np.arange(nx) * dx
t = np.arange(nz) * dz / 1000.                   # 毫秒

# 3. 计算阻抗
imp = vp * rho

# 4. 拉平成 TSV
X, T = np.meshgrid(x, t, indexing='ij')
df = pd.DataFrame({
    'X Distance': X.ravel(),
    'Time': T.ravel(),
    'Impedance': imp.ravel()
})
df.to_csv('marmousi2_impedance.tsv', sep='\t', index=False, float_format='%.2f')
print('✅ 已生成 marmousi2_impedance.tsv')
