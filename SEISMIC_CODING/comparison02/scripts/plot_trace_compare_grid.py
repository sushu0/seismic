#!/usr/bin/env python
"""Plot true vs predicted impedance curves for selected traces in a 2x2 grid.

Matches the user's reference style:
- 2x2 subplots
- titles like "No. 299"
- x-axis: t/ms (if dt_s available)
- y-axis: impedance

Inputs:
- run_dir with scalers.json, split.json, checkpoints/*.pt
- dataset: npz (--data) or Marmousi2-style dict npy (--data_npy)

Output:
- runs/<run_dir>/results/trace_compare_grid_<split>.png
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add project root

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.signal import resample_poly

import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str((_Path(__file__).resolve().parents[1] / ".mplconfig").resolve()))
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

from fcrsn_cw.data.dataset import SeismicImpedanceDataset
from fcrsn_cw.data.scaler import load_scalers
from fcrsn_cw.models.fcrsn_cw import FCRSN_CW


def _load_dataset_from_npy(path: Path, key_seismic: str, key_impedance: str, npy_resample: str) -> tuple[np.ndarray, np.ndarray, dict]:
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError("--data_npy expects a pickled dict-like object.")
    if key_seismic not in obj or key_impedance not in obj:
        raise ValueError(f"Missing required keys in {path}. Available keys: {sorted(list(obj.keys()))}")

    seismic = np.asarray(obj[key_seismic])
    impedance = np.asarray(obj[key_impedance])

    if seismic.ndim == 3 and seismic.shape[1] == 1:
        seismic = seismic[:, 0, :]
    if impedance.ndim == 3 and impedance.shape[1] == 1:
        impedance = impedance[:, 0, :]

    if seismic.ndim != 2 or impedance.ndim != 2:
        raise ValueError(f"Expected 2D arrays after squeeze. Got seismic.ndim={seismic.ndim}, impedance.ndim={impedance.ndim}.")
    if seismic.shape[0] != impedance.shape[0]:
        raise ValueError(f"Trace count mismatch: seismic.shape[0]={seismic.shape[0]} vs impedance.shape[0]={impedance.shape[0]}")

    if seismic.shape[1] != impedance.shape[1]:
        t_s = int(seismic.shape[1])
        t_z = int(impedance.shape[1])

        def is_int(x: float) -> bool:
            return abs(x - round(x)) < 1e-12

        ratio_up = t_z / max(t_s, 1)
        ratio_down = t_s / max(t_z, 1)
        mode = npy_resample
        if mode == "auto":
            if is_int(ratio_up):
                mode = "upsample_seismic"
            elif is_int(ratio_down):
                mode = "downsample_impedance"
            else:
                mode = "none"
        if mode == "none":
            raise ValueError(f"Shape mismatch: seismic.shape={seismic.shape}, impedance.shape={impedance.shape}. Set --npy_resample to align.")
        if mode == "upsample_seismic":
            if not is_int(ratio_up):
                raise ValueError(f"Cannot upsample seismic: {t_z}/{t_s} not integer")
            up = int(round(ratio_up))
            seismic = resample_poly(seismic.astype(np.float32), up=up, down=1, axis=1).astype(np.float32)
            seismic = seismic[:, :t_z]
        elif mode == "downsample_impedance":
            if not is_int(ratio_down):
                raise ValueError(f"Cannot downsample impedance: {t_s}/{t_z} not integer")
            down = int(round(ratio_down))
            impedance = resample_poly(impedance.astype(np.float32), up=1, down=down, axis=1).astype(np.float32)
            impedance = impedance[:, :t_s]
        if seismic.shape != impedance.shape:
            raise ValueError(f"After resampling, shapes still mismatch: seismic.shape={seismic.shape}, impedance.shape={impedance.shape}")

    meta = {"source": "npy", "data_npy": str(path)}
    return seismic.astype(np.float32), impedance.astype(np.float32), meta


def _load_dataset_npz(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path, allow_pickle=True)
    seismic = data["seismic"].astype(np.float32)
    impedance = data["impedance"].astype(np.float32)
    meta: dict = {}
    if "meta" in data:
        try:
            meta = json.loads(data["meta"].item())
        except Exception:
            meta = {"meta_raw": str(data["meta"])}
    meta.setdefault("source", "npz")
    meta.setdefault("data", str(path))
    return seismic, impedance, meta


@torch.no_grad()
def _predict_all(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    yhats: list[np.ndarray] = []
    for x, _y in loader:
        x = x.to(device)
        yhat = model(x).detach().cpu().numpy()
        yhats.append(yhat)
    return np.concatenate(yhats, axis=0)[:, 0, :]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")

    ap.add_argument("--data", type=str, default="")
    ap.add_argument("--data_npy", type=str, default="")
    ap.add_argument("--npy_key_seismic", type=str, default="seismic")
    ap.add_argument("--npy_key_impedance", type=str, default="acoustic_impedance")
    ap.add_argument("--npy_resample", type=str, default="auto", choices=["auto", "none", "upsample_seismic", "downsample_impedance"])

    ap.add_argument("--split", type=str, default="all", choices=["all", "train", "val", "test"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--trace_nos", type=int, nargs="*", default=[299, 599, 1699, 2299], help="Trace numbers to plot.")
    ap.add_argument("--one_based", action="store_true", help="Interpret trace_nos as 1-based (recommended, matches 'No. k').")

    ap.add_argument("--k_first", type=int, default=299)
    ap.add_argument("--k_last", type=int, default=3)
    ap.add_argument("--k_res1", type=int, default=299)
    ap.add_argument("--k_res2", type=int, default=3)
    ap.add_argument(
        "--auto_kernel_from_dt",
        action="store_true",
        help="Optional: scale k_first/k_res1 to preserve ~kernel_ms window using dt_s from dataset meta (npz).",
    )
    ap.add_argument("--kernel_ms", type=float, default=299.0)
    ap.add_argument(
        "--last_relu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ReLU at output (paper uses ReLU).",
    )

    ap.add_argument(
        "--output_activation",
        type=str,
        default="",
        choices=["", "none", "relu", "sigmoid"],
        help="Optional: override model output activation. '' keeps legacy behavior controlled by --last_relu.",
    )

    ap.add_argument("--ylabel", type=str, default="Impedance/(m/s*g/cm^3)")

    ap.add_argument(
        "--postprocess",
        type=str,
        default="none",
        choices=["none", "median"],
        help="Optional inference post-processing applied to predicted impedance (physical scale).",
    )
    ap.add_argument("--pp_kernel", type=int, default=5, help="Median kernel (odd) along time axis when --postprocess median")

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if bool(args.data) == bool(args.data_npy):
        raise ValueError("Provide exactly one of --data (npz) or --data_npy.")

    if args.data:
        seismic, impedance, meta = _load_dataset_npz(Path(args.data))
    else:
        seismic, impedance, meta = _load_dataset_from_npy(Path(args.data_npy), args.npy_key_seismic, args.npy_key_impedance, args.npy_resample)

    dt_s = None
    if isinstance(meta, dict) and "dt_s" in meta:
        try:
            dt_s = float(meta["dt_s"])
        except Exception:
            dt_s = None

    if args.auto_kernel_from_dt:
        if dt_s is None:
            raise ValueError("--auto_kernel_from_dt requires dt_s in dataset meta (npz).")
        dt_ms = float(dt_s) * 1000.0
        if dt_ms <= 0:
            raise ValueError(f"Invalid dt_s: {dt_s}")
        k = int(round(float(args.kernel_ms) / dt_ms))
        k = max(3, k)
        if k % 2 == 0:
            k += 1
        if args.k_first == ap.get_default("k_first"):
            args.k_first = k
        if args.k_res1 == ap.get_default("k_res1"):
            args.k_res1 = k

    if args.split == "all":
        idx = np.arange(seismic.shape[0])
        split_tag = "all"
    else:
        split = json.loads((run_dir / "split.json").read_text(encoding="utf-8"))
        idx = np.array(split[args.split], dtype=int)
        split_tag = args.split

    seismic_scaler, imp_scaler = load_scalers(run_dir / "scalers.json")
    ds = SeismicImpedanceDataset(seismic, impedance, idx, seismic_scaler, imp_scaler)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = FCRSN_CW(
        k_first=args.k_first,
        k_last=args.k_last,
        k_res1=args.k_res1,
        k_res2=args.k_res2,
        last_relu=args.last_relu,
        output_activation=(args.output_activation if args.output_activation != "" else None),
    ).to(device)
    ckpt = torch.load(str(run_dir / args.ckpt), map_location="cpu")
    model.load_state_dict(ckpt["model"])

    yhat_scaled = _predict_all(model, dl, device=device)
    y_scaled = imp_scaler.transform(impedance[idx])

    yhat = imp_scaler.inverse_transform(yhat_scaled)
    ytrue = imp_scaler.inverse_transform(y_scaled)

    if args.postprocess == "median":
        k = int(args.pp_kernel)
        if k % 2 == 0:
            k += 1
        yhat = median_filter(yhat, size=(1, k), mode="nearest")

    # x-axis
    n_samples = ytrue.shape[1]
    if dt_s is not None:
        x_ms = np.arange(n_samples, dtype=np.float64) * (dt_s * 1000.0)
        xlabel = "t/ms"
    else:
        x_ms = np.arange(n_samples, dtype=np.float64)
        xlabel = "t"

    # map requested trace numbers to local indices within split
    requested = [int(n) for n in args.trace_nos]
    local_indices: list[int] = []
    for n in requested:
        global_idx = n - 1 if args.one_based else n
        pos = np.where(idx == global_idx)[0]
        if pos.size == 0:
            raise ValueError(f"Trace No. {n} (global idx {global_idx}) not present in split '{split_tag}'.")
        local_indices.append(int(pos[0]))

    # plot
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, n, li in zip(axes, requested, local_indices):
        ax.plot(x_ms, yhat[li], color="red", linewidth=1.5, label="预测")
        ax.plot(x_ms, ytrue[li], color="blue", linewidth=1.5, label="标签")
        ax.set_title(f"No. {n}")
        ax.grid(True, alpha=0.25)

    # labels
    for ax in axes[2:]:
        ax.set_xlabel(xlabel)
    for ax in axes[::2]:
        ax.set_ylabel(args.ylabel)

    # legend (single)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.06, 0.94), frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = run_dir / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trace_compare_grid_{split_tag}.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print("[OK] wrote", out_path)


if __name__ == "__main__":
    main()
