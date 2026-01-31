from __future__ import annotations
import numpy as np
from scipy.signal import fftconvolve
from .wavelet import ricker, phase_rotate

def impedance_to_reflectivity(Z: np.ndarray) -> np.ndarray:
    """Compute reflectivity series r_i = (Z_{i+1}-Z_i)/(Z_{i+1}+Z_i).
    Z: (N, T) or (T,) impedance.
    """
    Z = np.asarray(Z).astype(np.float64)
    if Z.ndim == 1:
        num = Z[1:] - Z[:-1]
        den = Z[1:] + Z[:-1] + 1e-12
        r = num / den
        return np.concatenate([r, r[-1:]], axis=0).astype(np.float32)
    elif Z.ndim == 2:
        num = Z[:, 1:] - Z[:, :-1]
        den = Z[:, 1:] + Z[:, :-1] + 1e-12
        r = num / den
        return np.concatenate([r, r[:, -1:]], axis=1).astype(np.float32)
    else:
        raise ValueError("Z must be 1D or 2D.")

def reflectivity_to_seismic(r: np.ndarray, f0_hz: float, dt_s: float, wavelet_nt: int = 299, phase_deg: float = 0.0) -> np.ndarray:
    """Convolve reflectivity with (phase-rotated) Ricker wavelet."""
    w = ricker(f0_hz=f0_hz, dt_s=dt_s, nt=wavelet_nt)
    w = phase_rotate(w, phase_deg=phase_deg)
    r = np.asarray(r).astype(np.float32)
    if r.ndim == 1:
        return fftconvolve(r, w, mode="same").astype(np.float32)
    if r.ndim == 2:
        out = np.empty_like(r, dtype=np.float32)
        for i in range(r.shape[0]):
            out[i] = fftconvolve(r[i], w, mode="same").astype(np.float32)
        return out
    raise ValueError("r must be 1D or 2D.")

def make_synthetic_pair_from_impedance(
    impedance: np.ndarray,
    f0_hz: float = 30.0,
    dt_s: float = 0.001,
    wavelet_nt: int = 299,
    phase_deg: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r = impedance_to_reflectivity(impedance)
    s = reflectivity_to_seismic(r, f0_hz=f0_hz, dt_s=dt_s, wavelet_nt=wavelet_nt, phase_deg=phase_deg)
    return s, impedance.astype(np.float32)
