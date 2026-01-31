import numpy as np
import torch
from torch.utils.data import Dataset

def compute_reflectivity_from_impedance(Z):
    Z_next = np.roll(Z, -1, axis=0)
    Z_next[-1, :] = Z[-1, :]
    R = (Z_next - Z) / (Z_next + Z + 1e-8)
    return R.astype(np.float32)

def normalize(x):
    m, s = x.mean(), x.std() + 1e-8
    return (x - m)/s, float(m), float(s)

def denorm(x, m, s):
    return x * s + m

class PatchDataset(Dataset):
    """
    从完整 (T,X) 剖面切 2D 小块：(1, T_win, X_win)
    返回：地震 S、阻抗 Z、反射 R（由 Z 计算）
    """
    def __init__(self, Z_full, S_full, T_win=128, X_win=16, stride_t=64, stride_x=8,
                 aug_fn=None):
        self.Z = torch.from_numpy(Z_full).float()  # (T,X)
        self.S = torch.from_numpy(S_full).float()
        self.R = torch.from_numpy(compute_reflectivity_from_impedance(Z_full)).float()
        self.T, self.X = self.Z.shape
        self.T_win, self.X_win = T_win, X_win
        self.stride_t, self.stride_x = stride_t, stride_x
        self.aug_fn = aug_fn
        idxs = []
        for t0 in range(0, self.T - T_win + 1, stride_t):
            for x0 in range(0, self.X - X_win + 1, stride_x):
                idxs.append((t0, x0))
        self.idxs = idxs
    def __len__(self): return len(self.idxs)
    def __getitem__(self, i):
        t0, x0 = self.idxs[i]
        t1, x1 = t0+self.T_win, x0+self.X_win
        Zp = self.Z[t0:t1, x0:x1].unsqueeze(0)  # (1,Tw,Xw)
        Rp = self.R[t0:t1, x0:x1].unsqueeze(0)
        Sp = self.S[t0:t1, x0:x1].unsqueeze(0)
        if self.aug_fn is not None:
            Zp, Rp, Sp = self.aug_fn(Zp, Rp, Sp)
        return Sp, Zp, Rp
