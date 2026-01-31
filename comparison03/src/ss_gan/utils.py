from __future__ import annotations
import random
import numpy as np
import torch

def seed_everything(
    seed: int = 1234,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
    tf32: bool = False,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark) if not deterministic else False
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)

def pcc(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    return float((a*b).mean() / (a.std()+eps) / (b.std()+eps))

def r2(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    ss_res = np.sum((a-b)**2)
    ss_tot = np.sum((a-a.mean())**2) + eps
    return float(1.0 - ss_res/ss_tot)

def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
