from __future__ import annotations
from pathlib import Path
import numpy as np

def save_trace_plot(y_true: np.ndarray, y_pred: np.ndarray, out_path: str | Path, max_traces: int = 6):
    """Plot several traces (true vs pred) in one figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(max_traces, y_true.shape[0])
    idx = np.linspace(0, y_true.shape[0]-1, n, dtype=int)

    plt.figure()
    for i, k in enumerate(idx):
        plt.plot(y_true[k], label=f"true[{k}]" if i == 0 else None)
        plt.plot(y_pred[k], linestyle="--", label=f"pred[{k}]" if i == 0 else None)
    plt.xlabel("t index")
    plt.ylabel("impedance (normalized)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_section(im: np.ndarray, out_path: str | Path, title: str):
    """Save a 2D image (trace x time) as a figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(im, aspect="auto")
    plt.title(title)
    plt.xlabel("time sample")
    plt.ylabel("trace")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_section_with_contour(im: np.ndarray, out_path: str | Path, title: str, 
                               xlabel: str = "Trace number", ylabel: str = "Sample",
                               cmap: str = "seismic"):
    """Save a 2D section with contour overlay (similar to the reference image)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Transpose to have traces on x-axis and samples on y-axis
    im_T = im.T  # shape: (n_samples, n_traces)
    
    # Create meshgrid for contour
    n_samples, n_traces = im_T.shape
    X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))
    
    # Plot filled contours
    levels = 20
    cf = ax.contourf(X, Y, im_T, levels=levels, cmap=cmap)
    
    # Add colorbar with label
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label(f'[0,30,1,228]', rotation=0, labelpad=15)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.invert_yaxis()  # Invert y-axis to have 0 at top
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_four_trace_comparison(s_obs: np.ndarray, s_pred: np.ndarray, 
                                trace_ids: list, out_path: str | Path,
                                ylabel: str = "Impedance(m/s*g/cm^3)"):
    """Save 4-panel comparison of observed vs predicted seismic traces."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, trace_id in enumerate(trace_ids):
        ax = axes[i]
        t = np.arange(len(s_obs[trace_id]))
        
        # Plot observed (red) and predicted (blue)
        ax.plot(t, s_obs[trace_id], 'r-', linewidth=1, label='观测')
        ax.plot(t, s_pred[trace_id], 'b-', linewidth=1, label='预测')
        
        ax.set_xlabel('t')
        ax.set_ylabel(ylabel)
        ax.set_title(f'No. {trace_id}')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
