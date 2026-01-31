from __future__ import annotations
import numpy as np

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)
    return float(np.mean((y_true - y_pred) ** 2))

def pcc(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = y_true.astype(np.float64).reshape(-1)
    y_pred = y_pred.astype(np.float64).reshape(-1)
    x = y_true - y_true.mean()
    y = y_pred - y_pred.mean()
    denom = (np.sqrt((x**2).sum()) * np.sqrt((y**2).sum())) + eps
    return float((x*y).sum() / denom)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    y_true = y_true.astype(np.float64).reshape(-1)
    y_pred = y_pred.astype(np.float64).reshape(-1)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum() + eps
    return float(1.0 - ss_res / ss_tot)

def summarize_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MSE": mse(y_true, y_pred),
        "PCC": pcc(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }
