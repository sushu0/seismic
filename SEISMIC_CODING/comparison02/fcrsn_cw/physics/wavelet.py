from __future__ import annotations
import numpy as np
from scipy.signal import hilbert

def ricker(f0_hz: float, dt_s: float, nt: int) -> np.ndarray:
    """Ricker wavelet (Mexican hat) of length nt samples."""
    t0 = (nt - 1) / 2.0
    t = (np.arange(nt) - t0) * dt_s
    a = (np.pi * f0_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w.astype(np.float32)

def phase_rotate(w: np.ndarray, phase_deg: float) -> np.ndarray:
    """Constant phase rotation using analytic signal."""
    if abs(phase_deg) < 1e-9:
        return w
    phase = np.deg2rad(phase_deg)
    analytic = hilbert(w.astype(np.float64))
    rotated = np.real(analytic * np.exp(1j * phase))
    return rotated.astype(np.float32)
