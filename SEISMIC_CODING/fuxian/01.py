import numpy as np

# 如果本地没有数据文件，可使用 wget 从 Zenodo 下载
# 这里提供 Zenodo 链接供参考:contentReference[oaicite:6]{index=6}:
# !wget -O marmousi_Ip_model.npy "https://zenodo.org/record/14233581/files/marmousi_Ip_model.npy?download=1"
# !wget -O marmousi_synthetic_seismic.npy "https://zenodo.org/record/14233581/files/marmousi_synthetic_seismic.npy?download=1"

# 加载 Marmousi2 声阻抗模型和对应的合成地震数据
impedance = np.load('marmousi_Ip_model.npy')        # 2D 声阻抗剖面
seismic = np.load('marmousi_synthetic_seismic.npy')  # 2D 合成地震记录

# 检查数据形状
print("Impedance model shape:", impedance.shape)
print("Seismic data shape:", seismic.shape)

# 基本统计信息
print("Impedance values - min:", impedance.min(), "max:", impedance.max(), "mean:", impedance.mean())
print("Seismic values - min:", seismic.min(), "max:", seismic.max(), "mean:", seismic.mean())
