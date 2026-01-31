from __future__ import annotations
import numpy as np

def mse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    return float(np.mean((y_pred - y_true) ** 2))

def pearsonr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    x = np.asarray(x).astype(np.float64)
    y = np.asarray(y).astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x*x).sum()) * np.sqrt((y*y).sum())) + eps
    return float((x*y).sum() / denom)

def _pearsonr_batch(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row Pearson correlation for 2D arrays.

    x, y: (N, T)
    returns: (N,)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    num = np.sum(x * y, axis=1)
    den = (np.sqrt(np.sum(x * x, axis=1)) * np.sqrt(np.sum(y * y, axis=1))) + eps
    return num / den

def add_gaussian_noise_snr(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise to reach target SNR(dB) based on power ratio.

    SNR(dB) = 10 log10(P_signal / P_noise)
    """
    sig = np.asarray(signal).astype(np.float64)
    p_sig = np.mean(sig**2)
    if p_sig <= 0:
        return sig.copy()
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(p_noise), size=sig.shape)
    return (sig + noise).astype(signal.dtype)

def pcc_shallow_deep(y_pred: np.ndarray, y_true: np.ndarray, shallow_frac: float = 0.2) -> tuple[float, float]:
    """Compute average PCC for shallow and deep segments over a batch of traces.
    y_pred/y_true: (N, T) arrays.
    shallow_frac: fraction of samples considered shallow (paper uses 1/5).
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    assert y_pred.shape == y_true.shape and y_pred.ndim == 2
    n, t = y_pred.shape
    cut = int(round(t * shallow_frac))
    if cut < 1:
        cut = 1
    shallow_corr = _pearsonr_batch(y_pred[:, :cut], y_true[:, :cut])
    deep_corr = _pearsonr_batch(y_pred[:, cut:], y_true[:, cut:])
    return float(np.mean(shallow_corr)), float(np.mean(deep_corr))
