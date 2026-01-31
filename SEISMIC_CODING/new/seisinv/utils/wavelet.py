from __future__ import annotations
import numpy as np
import torch

def ricker(f0: float, dt: float, length: float) -> np.ndarray:
    """Ricker wavelet.
    Args:
        f0: dominant frequency (Hz)
        dt: sample interval (s)
        length: total length (s), symmetric about 0
    Returns:
        w: shape [K]
    """
    t = np.arange(-length/2, length/2 + dt, dt, dtype=np.float64)
    pi2 = (np.pi ** 2)
    w = (1.0 - 2.0*pi2*(f0**2)*(t**2)) * np.exp(-pi2*(f0**2)*(t**2))
    return w.astype(np.float32)

def wavelet_tensor(f0: float, dt: float, length: float, device=None) -> torch.Tensor:
    w = ricker(f0=f0, dt=dt, length=length)
    wt = torch.from_numpy(w).to(device=device)
    return wt
