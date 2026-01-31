import os
import numpy as np


def resample_time_axis(arr: np.ndarray, new_T: int) -> np.ndarray:
    """把 2D 数组 [N_traces, T] 沿时间轴重采样到 new_T（线性插值）。"""
    if arr.ndim != 2:
        raise ValueError(f"resample_time_axis expects 2D (N_traces, T), got {arr.shape}")
    n_traces, old_T = arr.shape
    if old_T == new_T:
        return arr
    old_x = np.linspace(0.0, 1.0, old_T, dtype=np.float64)
    new_x = np.linspace(0.0, 1.0, new_T, dtype=np.float64)
    out = np.empty((n_traces, new_T), dtype=np.float64)
    for i in range(n_traces):
        out[i] = np.interp(new_x, old_x, arr[i].astype(np.float64, copy=False))
    return out

# 目标目录
target_dir = r"D:\SEISMIC_CODING\comparison01"
data_npy_path = os.path.join(target_dir, "data.npy")

if not os.path.exists(data_npy_path):
    raise FileNotFoundError(
        f"data.npy 没找到：{data_npy_path}\n请先从 GitHub 下载放到该目录。"
    )

print("正在读取 data.npy ...")

# 读取 data.npy，兼容 dict / array(object)
data_raw = np.load(data_npy_path, allow_pickle=True)
if isinstance(data_raw, dict):
    data_dic = data_raw
else:
    data_dic = data_raw.item()

print("data.npy 中包含的键 (keys)：", list(data_dic.keys()))

# 自动识别 seismic 和 impedance 的 key
keys = list(data_dic.keys())
seis_key_candidates = [k for k in keys if "seis" in k.lower()]
imp_key_candidates  = [k for k in keys
                       if ("imp" in k.lower()) or ("acoustic" in k.lower())]

if not seis_key_candidates:
    raise KeyError("没找到包含 'seis' 的 key，请手动检查 keys。")
if not imp_key_candidates:
    raise KeyError("没找到包含 'imp' 或 'acoustic' 的 key，请手动检查 keys。")

seis_key = seis_key_candidates[0]
imp_key  = imp_key_candidates[0]
print(f"自动识别：seismic key = {seis_key}, impedance key = {imp_key}")

seismic_raw   = data_dic[seis_key]
impedance_raw = data_dic[imp_key]

print("原始 seismic_raw 形状：", seismic_raw.shape)
print("原始 impedance_raw 形状：", impedance_raw.shape)

# ---- 关键：处理 3D [N_traces, 1, T] 的情况 ----
if seismic_raw.ndim == 3:
    seismic_raw = np.squeeze(seismic_raw)       # 去掉维度 1 -> [N_traces, T_seis]
if impedance_raw.ndim == 3:
    impedance_raw = np.squeeze(impedance_raw)   # 去掉维度 1 -> [N_traces, T_imp]

print("去掉通道维度后 seismic_raw 形状：", seismic_raw.shape)
print("去掉通道维度后 impedance_raw 形状：", impedance_raw.shape)

# 此时应为 2D
if seismic_raw.ndim != 2 or impedance_raw.ndim != 2:
    raise ValueError(
        "期望 seismic 和 impedance 都是 2D 数组 (N_traces, T)，"
        f"但是实际形状分别为 {seismic_raw.shape}, {impedance_raw.shape}"
    )

# 对齐时间长度：优先保持地震长度不变，把另一方重采样到相同长度。
T_seis = int(seismic_raw.shape[1])
T_imp = int(impedance_raw.shape[1])

if T_seis != T_imp:
    if T_imp > T_seis:
        print(
            f"注意：地震长度 = {T_seis}, 阻抗长度 = {T_imp}。"
            f"将把阻抗从 T={T_imp} 重采样到 T={T_seis}（覆盖全时间范围，避免只截取浅层）。"
        )
        impedance_raw = resample_time_axis(impedance_raw, new_T=T_seis)
    else:
        print(
            f"注意：地震长度 = {T_seis}, 阻抗长度 = {T_imp}。"
            f"将把地震从 T={T_seis} 重采样到 T={T_imp}。"
        )
        seismic_raw = resample_time_axis(seismic_raw, new_T=T_imp)

print("对齐后 seismic_raw 形状：", seismic_raw.shape)
print("对齐后 impedance_raw 形状：", impedance_raw.shape)

# 现在两者都是 [N_traces, T]，转置成我们代码习惯的 [T, N_traces]
seismic   = seismic_raw.T.astype("float32")     # [T, Nx]
impedance = impedance_raw.T.astype("float32")   # [T, Nx]

print("转置后 seismic 形状（T, Nx）：", seismic.shape)
print("转置后 impedance 形状（T, Nx）：", impedance.shape)

# 保存成我们训练脚本要用的文件名
seis_out = os.path.join(target_dir, "seismic.npy")
imp_out  = os.path.join(target_dir, "impedance.npy")

np.save(seis_out, seismic)
np.save(imp_out, impedance)

print("已保存：", seis_out)
print("已保存：", imp_out)
print("OK！现在可以直接用训练脚本 marmousi_cnn_bilstm.py 了。")
