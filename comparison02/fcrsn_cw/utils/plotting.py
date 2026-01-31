from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def plot_trace_compare(
    out_path: str | Path,
    t: np.ndarray,
    true_imp: np.ndarray,
    pred_imp: np.ndarray,
    title: str = "",
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(true_imp, t, label="True")
    plt.plot(pred_imp, t, label="Pred")
    plt.gca().invert_yaxis()
    plt.xlabel("Impedance (scaled)")
    plt.ylabel("Time/depth sample")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_section(out_path: str | Path, section: np.ndarray, title: str = "", xlabel: str = "Trace", ylabel: str = "Sample") -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(section.T, aspect="auto", origin="upper")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
