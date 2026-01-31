from __future__ import annotations

import os
import argparse
from typing import Iterable

import numpy as np
import matplotlib.pyplot as plt


def _set_cjk_font_if_available() -> None:
    try:
        from matplotlib import font_manager
    except Exception:
        return

    preferred = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]

    available = set()
    try:
        for fp in font_manager.fontManager.ttflist:
            available.add(fp.name)
    except Exception:
        return

    for name in preferred:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


def _as_1d_trace(arr: np.ndarray) -> np.ndarray:
    """Accept [T] or [1,T] and return [T]."""
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(f"Expected a 1D trace, got shape={arr.shape}")


def _get_pred_true(z: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray]:
    pred = z["pred"]
    true = z["true"]

    # infer.py exports [N,1,T] by default
    if pred.ndim == 3 and pred.shape[1] == 1:
        pred = pred[:, 0, :]
    if true.ndim == 3 and true.shape[1] == 1:
        true = true[:, 0, :]

    if pred.ndim != 2 or true.ndim != 2:
        raise ValueError(f"Expected pred/true to be 2D [N,T]; got pred={pred.shape}, true={true.shape}")
    if pred.shape != true.shape:
        raise ValueError(f"pred/true shape mismatch: pred={pred.shape}, true={true.shape}")
    return pred, true


def _maybe_denorm(y: np.ndarray, mean: float | None, std: float | None) -> np.ndarray:
    if mean is None or std is None:
        return y
    return y * float(std) + float(mean)


def _iter_trace_ids_default() -> list[int]:
    # Layout: (top-left)299 (top-right)599 (bottom-left)1699 (bottom-right)2299
    return [299, 599, 1699, 2299]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help=".npz produced by infer.py (contains pred/true)")
    ap.add_argument("--out", required=True, help="Output .png path")
    ap.add_argument(
        "--trace_ids",
        nargs="*",
        type=int,
        default=None,
        help="Trace numbers to plot (default: 299 2299 599 1699)",
    )
    ap.add_argument("--one_based", action=argparse.BooleanOptionalAction, default=True, help="Interpret trace_ids as 1-based indices")
    ap.add_argument("--time_ms_max", type=float, default=2350.0, help="Time axis max in ms")
    ap.add_argument("--dt_ms", type=float, default=None, help="Optional dt (ms). If set, overrides time_ms_max")
    ap.add_argument("--ckpt", default=None, help="Optional checkpoint to denormalize if pred is normalized")
    ap.add_argument("--phys", action=argparse.BooleanOptionalAction, default=True, help="If ckpt provided, plot in physical units")
    ap.add_argument("--title_prefix", default="No.")
    args = ap.parse_args()

    _set_cjk_font_if_available()

    trace_ids = _iter_trace_ids_default() if args.trace_ids is None or len(args.trace_ids) == 0 else list(args.trace_ids)

    z = np.load(args.pred, allow_pickle=True)
    pred, true = _get_pred_true(z)
    n_traces, t_len = pred.shape

    y_mean = None
    y_std = None
    if args.ckpt is not None and args.phys:
        import torch

        ck = torch.load(args.ckpt, map_location="cpu")
        stats = ck.get("stats", {}) or {}
        y_mean = stats.get("y_mean", None)
        y_std = stats.get("y_std", None)

    if args.dt_ms is not None:
        t_ms = np.arange(t_len, dtype=np.float32) * float(args.dt_ms)
    else:
        t_ms = np.linspace(0.0, float(args.time_ms_max), t_len, dtype=np.float32)

    # Figure layout: 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.0), sharex=False, sharey=False)
    axes = axes.reshape(-1)

    # Map in the same order as the reference figure
    for ax, trace_no in zip(axes, trace_ids):
        idx = int(trace_no) - 1 if args.one_based else int(trace_no)
        if idx < 0 or idx >= n_traces:
            raise IndexError(f"Trace {trace_no} -> index {idx} out of range [0,{n_traces-1}]")

        p = _as_1d_trace(pred[idx])
        y = _as_1d_trace(true[idx])

        p_plot = _maybe_denorm(p, y_mean, y_std)
        y_plot = _maybe_denorm(y, y_mean, y_std)

        ax.plot(t_ms, p_plot, color="red", linewidth=1.5, label="预测")
        ax.plot(t_ms, y_plot, color="blue", linewidth=1.5, label="标签")
        ax.set_title(f"{args.title_prefix} {trace_no}")
        ax.set_xlabel("t/ms")
        ax.set_ylabel("Impedance/(m/s · g/cm³)")
        ax.grid(False)
        ax.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
