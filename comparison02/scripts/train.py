#!/usr/bin/env python
"""Train FCRSN-CW on a seismic-impedance dataset.

Paper training setup (synthetic Marmousi2 experiment):
- Total traces: 13601; split 10601 train / 1500 val / 1500 test fileciteturn2file0L249-L251
- Loss: MSE fileciteturn2file0L251-L269
- Optimizer: Adam, lr=0.001, weight_decay=1e-7, epochs=50, batch_size=12 fileciteturn2file0L263-L269
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add project root
import argparse
import json
import platform
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.signal import resample_poly

from fcrsn_cw.utils.seed import set_global_seed
from fcrsn_cw.data.dataset import make_split_indices, SeismicImpedanceDataset, fit_default_scalers
from fcrsn_cw.data.scaler import save_scalers
from fcrsn_cw.models.fcrsn_cw import FCRSN_CW
from fcrsn_cw.train.engine import (
    train_one_epoch,
    eval_one_epoch,
    train_one_epoch_cfg,
    eval_one_epoch_cfg,
    save_checkpoint,
    TrainState,
)
from fcrsn_cw.physics.forward import make_synthetic_pair_from_impedance


def load_traces_from_segy(path: Path | str) -> np.ndarray:
    import segyio  # lazy import

    path = Path(path)
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.asarray(segyio.tools.collect(f.trace[:]), dtype=np.float32)


def load_dt_from_segy(path: Path | str) -> float:
    import segyio  # lazy import

    path = Path(path)
    with segyio.open(path, "r", ignore_geometry=True) as f:
        dt_us = int(f.bin.get(segyio.BinField.Interval, 0) or 0)
        if dt_us <= 0:
            dt_us = int(f.header[0].get(segyio.TraceField.TRACE_SAMPLE_INTERVAL, 0) or 0)
        if dt_us <= 0:
            raise ValueError("Failed to read sample interval (dt) from SEG-Y headers.")
        return float(dt_us) / 1e6

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paper",
        action="store_true",
        help="Apply paper-aligned defaults (Marmousi2 synthetic experiment): kernels 299/3, f0=30Hz, phase=0, wavelet_nt=299, lr=1e-3, wd=1e-7, batch=12, epochs=50, seed=42.",
    )
    ap.add_argument(
        "--paper_strict",
        action="store_true",
        help="With --paper: additionally assert dt=1ms and data shape 13601x2800 (paper Marmousi2 setup).",
    )
    ap.add_argument("--data", type=str, default="data/synth_marmousi_like.npz")
    ap.add_argument(
        "--data_npy",
        type=str,
        default="",
        help="Optional: path to a .npy that contains a pickled dict (e.g., keys 'seismic' and 'acoustic_impedance').",
    )
    ap.add_argument(
        "--npy_key_seismic",
        type=str,
        default="seismic",
        help="When --data_npy is set: dict key name for seismic array.",
    )
    ap.add_argument(
        "--npy_key_impedance",
        type=str,
        default="acoustic_impedance",
        help="When --data_npy is set: dict key name for acoustic impedance array.",
    )
    ap.add_argument(
        "--npy_resample",
        type=str,
        default="auto",
        choices=["auto", "none", "upsample_seismic", "downsample_impedance"],
        help=(
            "When loading from --data_npy and seismic/impedance lengths differ: "
            "'auto' tries integer-factor resampling to match lengths; 'none' errors; "
            "or force 'upsample_seismic'/'downsample_impedance'."
        ),
    )
    ap.add_argument(
        "--dt_s",
        type=float,
        default=None,
        help="Optional: sampling interval to record into meta when loading from --data_npy (needed for --paper_strict if not using SEG-Y).",
    )
    ap.add_argument("--vp_segy", type=str, default="", help="Optional: Marmousi2 Vp SEG-Y path")
    ap.add_argument("--rho_segy", type=str, default="", help="Optional: Marmousi2 Density SEG-Y path")
    ap.add_argument(
        "--rho_const",
        type=float,
        default=None,
        help="Optional: use constant density (paper assumes constant rho). If set, ignores --rho_segy.",
    )
    ap.add_argument(
        "--export_npz",
        type=str,
        default="",
        help="Optional: save the derived (seismic, impedance, meta) dataset to an .npz for reproducible evaluation.",
    )
    ap.add_argument("--f0", type=float, default=30.0, help="Ricker central frequency (Hz) for forward modeling")
    ap.add_argument("--phase", type=float, default=0.0, help="Constant phase rotation (deg)")
    ap.add_argument("--wavelet_nt", type=int, default=299, help="Wavelet length (samples) aligning with kernel size")
    ap.add_argument("--run_dir", type=str, default="runs/exp1")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-7)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument(
        "--impedance_scaler",
        type=str,
        default="minmax",
        choices=["minmax", "standard"],
        help="Impedance target scaling. Default is minmax (current code).",
    )
    ap.add_argument(
        "--output_activation",
        type=str,
        default="",
        choices=["", "none", "relu", "sigmoid"],
        help=(
            "Optional: override model output activation. '' keeps legacy behavior controlled by --last_relu. "
            "For minmax-scaled targets, 'sigmoid' bounds outputs to [0,1] and can reduce overshoot."
        ),
    )

    ap.add_argument(
        "--loss_type",
        type=str,
        default="mse",
        choices=["mse", "huber"],
        help="Training loss on scaled targets. 'mse' matches the paper; 'huber' is more robust to outliers.",
    )
    ap.add_argument("--huber_delta", type=float, default=1.0, help="Huber delta when --loss_type huber")
    ap.add_argument(
        "--grad_loss_weight",
        type=float,
        default=0.0,
        help="Optional: add MSE on time derivative (diff along T): loss = base + w * MSE(dyhat, dy).",
    )

    ap.add_argument(
        "--train_snr_db",
        type=float,
        default=None,
        help="Optional: add Gaussian noise to seismic input during training to match target SNR (dB). Default off.",
    )

    ap.add_argument(
        "--scheduler",
        type=str,
        default="none",
        choices=["none", "plateau"],
        help="Optional learning rate scheduler. 'plateau' uses ReduceLROnPlateau on val loss.",
    )
    ap.add_argument("--plateau_factor", type=float, default=0.5)
    ap.add_argument("--plateau_patience", type=int, default=5)
    ap.add_argument("--min_lr", type=float, default=1e-6)

    ap.add_argument(
        "--auto_kernel_from_dt",
        action="store_true",
        help=(
            "Optional: when dt_s is known, scale the paper's 299-sample kernel to preserve ~299ms physical window. "
            "Sets k_first=k_res1=wavelet_nt=round(kernel_ms/dt_ms) (odd)."
        ),
    )
    ap.add_argument(
        "--kernel_ms",
        type=float,
        default=299.0,
        help="Physical window (ms) used when --auto_kernel_from_dt is enabled. Paper corresponds to 299ms.",
    )
    ap.add_argument("--k_first", type=int, default=299, help="Conv kernel size of first layer (paper uses 299).")
    ap.add_argument("--k_last", type=int, default=3, help="Conv kernel size of last layer (paper uses 3).")
    ap.add_argument("--k_res1", type=int, default=299, help="RSBU first conv kernel (paper uses 299; Volve experiment uses 80).")
    ap.add_argument("--k_res2", type=int, default=3, help="RSBU second conv kernel (paper uses 3).")
    ap.add_argument(
        "--last_relu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ReLU at output (paper uses ReLU). Use --no-last-relu to disable.",
    )
    args = ap.parse_args()

    def write_paper_repro_record(run_dir: Path, args, meta: dict, data_note: dict, deviations: list[str]) -> None:
        """Write a paper-mode reproduction record declaring every default/assumption."""
        try:
            import scipy
        except Exception:
            scipy = None

        record = {
            "date": None,
            "paper_mode": bool(args.paper),
            "paper_strict": bool(args.paper_strict),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": {
                    "version": torch.__version__,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "torch_cuda": torch.version.cuda,
                    "device": str(args.device),
                    "device_name": (torch.cuda.get_device_name(0) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else None),
                },
                "numpy": np.__version__,
                "scipy": (getattr(scipy, "__version__", None) if scipy is not None else None),
            },
            "data": {
                "source": meta.get("source", None),
                "shape": data_note.get("shape"),
                "dt_s": meta.get("dt_s", None),
                "notes": data_note,
            },
            "model": {
                "name": "FCRSN-CW",
                "k_first": int(args.k_first),
                "k_last": int(args.k_last),
                "k_res1": int(args.k_res1),
                "k_res2": int(args.k_res2),
                "last_relu": bool(args.last_relu),
                "output_activation": (None if args.output_activation == "" else str(args.output_activation)),
            },
            "training": {
                "optimizer": "Adam",
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "loss": {
                    "type": str(args.loss_type),
                    "huber_delta": float(args.huber_delta),
                    "grad_loss_weight": float(args.grad_loss_weight),
                },
                "scheduler": {
                    "type": str(args.scheduler),
                    "plateau_factor": float(args.plateau_factor),
                    "plateau_patience": int(args.plateau_patience),
                    "min_lr": float(args.min_lr),
                },
                "seed": int(args.seed),
                "split": "10601/1500/1500 (train/val/test)" if args.paper else "random",
            },
            "assumptions_and_defaults": [
                "Unless explicitly stated in the paper, this run uses the minimal default implementation choices used by this codebase.",
                "Data normalization/scaling: z-score on seismic and a configurable scaler on impedance (fit on train split only). If the paper specifies a different preprocessing, this must be changed and explicitly recorded.",
                "Paper-strict mode forbids any implicit resampling or split resizing.",
            ],
            "deviations_from_paper": deviations,
            "meta": meta,
        }
        out_dir = run_dir / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "repro_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    if args.paper:
        # Apply paper-aligned settings as defaults, but allow explicit CLI overrides.
        def set_if_default(name: str, value):
            if getattr(args, name) == ap.get_default(name):
                setattr(args, name, value)

        set_if_default("seed", 42)
        set_if_default("epochs", 50)
        set_if_default("batch_size", 12)
        set_if_default("lr", 1e-3)
        set_if_default("weight_decay", 1e-7)
        set_if_default("f0", 30.0)
        set_if_default("phase", 0.0)
        set_if_default("k_first", 299)
        set_if_default("k_last", 3)
        set_if_default("k_res1", 299)
        set_if_default("k_res2", 3)
        set_if_default("wavelet_nt", 299)
        set_if_default("last_relu", True)
        if args.vp_segy and args.rho_const is None:
            # Paper synthetic experiment assumes constant density.
            args.rho_const = 1.0
        print("[INFO] --paper enabled: using paper-aligned defaults (CLI overrides respected).")

    if args.paper_strict and not args.paper:
        raise ValueError("--paper_strict requires --paper (paper_strict is meaningful only in paper-locked mode).")

    set_global_seed(args.seed)

    deviations: list[str] = []

    # Paper Marmousi2 synthetic experiment expectations
    paper_shape = (13601, 2800)
    paper_dt_s = 0.001

    if args.vp_segy:
        dt_s = load_dt_from_segy(args.vp_segy)
        vp = load_traces_from_segy(args.vp_segy)

        if args.rho_const is not None:
            rho = float(args.rho_const)
            impedance_raw = vp * rho
            rho_meta = {"rho_const": rho}
        else:
            if not args.rho_segy:
                raise ValueError("When --vp_segy is set, provide --rho_segy or set --rho_const.")
            rho_arr = load_traces_from_segy(args.rho_segy)
            if vp.shape != rho_arr.shape:
                raise ValueError(f"Vp shape {vp.shape} != Density shape {rho_arr.shape}")
            impedance_raw = vp * rho_arr
            rho_meta = {"rho": args.rho_segy}

        seismic, impedance = make_synthetic_pair_from_impedance(
            impedance=impedance_raw,
            f0_hz=args.f0,
            dt_s=dt_s,
            wavelet_nt=args.wavelet_nt,
            phase_deg=args.phase,
        )
        meta = {
            "source": "segy",
            "vp": args.vp_segy,
            "dt_s": dt_s,
            "f0_hz": float(args.f0),
            "phase_deg": float(args.phase),
            "wavelet_nt": int(args.wavelet_nt),
            **rho_meta,
        }

        if args.export_npz:
            out = Path(args.export_npz)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, seismic=seismic, impedance=impedance, meta=json.dumps(meta))
    else:
        if args.data_npy:
            obj = np.load(args.data_npy, allow_pickle=True)
            if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
                obj = obj.item()
            if not isinstance(obj, dict):
                raise ValueError("--data_npy expects a pickled dict-like object (e.g., {'seismic': ..., 'acoustic_impedance': ...}).")

            if args.npy_key_seismic not in obj:
                raise ValueError(f"Key '{args.npy_key_seismic}' not found in {args.data_npy}. Available keys: {sorted(list(obj.keys()))}")
            if args.npy_key_impedance not in obj:
                raise ValueError(f"Key '{args.npy_key_impedance}' not found in {args.data_npy}. Available keys: {sorted(list(obj.keys()))}")

            seismic = np.asarray(obj[args.npy_key_seismic])
            impedance = np.asarray(obj[args.npy_key_impedance])

            # Common Marmousi2 exports include singleton channel dimension: (N, 1, T)
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
                ratio_up = t_z / max(t_s, 1)
                ratio_down = t_s / max(t_z, 1)

                def is_int(x: float) -> bool:
                    return abs(x - round(x)) < 1e-12

                mode = args.npy_resample
                if args.paper and args.paper_strict and mode != "none":
                    raise ValueError(
                        "Paper strict mode forbids resampling. Provide a dataset that already matches the paper's sample count and dt. "
                        "Set --npy_resample none to ensure no implicit resampling occurs."
                    )
                if mode == "auto":
                    if is_int(ratio_up):
                        mode = "upsample_seismic"
                    elif is_int(ratio_down):
                        mode = "downsample_impedance"
                    else:
                        mode = "none"

                if mode == "none":
                    raise ValueError(
                        "Shape mismatch for --data_npy: seismic and impedance must have the same sample length. "
                        f"Got seismic.shape={seismic.shape}, impedance.shape={impedance.shape}. "
                        "If they differ by an integer factor (common when seismic is downsampled), set --npy_resample accordingly."
                    )

                if mode == "upsample_seismic":
                    if not is_int(ratio_up):
                        raise ValueError(f"Cannot upsample seismic to match impedance: {t_z}/{t_s} is not an integer factor")
                    up = int(round(ratio_up))
                    seismic = resample_poly(seismic.astype(np.float32), up=up, down=1, axis=1).astype(np.float32)
                    # ensure exact length
                    seismic = seismic[:, :t_z]
                    deviations.append(
                        "Non-paper preprocessing: resampled seismic (upsample) to match impedance length because input data_npy has mismatched sample counts."
                    )
                elif mode == "downsample_impedance":
                    if not is_int(ratio_down):
                        raise ValueError(f"Cannot downsample impedance to match seismic: {t_s}/{t_z} is not an integer factor")
                    down = int(round(ratio_down))
                    impedance = resample_poly(impedance.astype(np.float32), up=1, down=down, axis=1).astype(np.float32)
                    impedance = impedance[:, :t_s]
                    deviations.append(
                        "Non-paper preprocessing: resampled impedance (downsample) to match seismic length because input data_npy has mismatched sample counts."
                    )
                else:
                    raise ValueError(f"Unknown --npy_resample mode: {args.npy_resample}")

                if seismic.shape != impedance.shape:
                    raise ValueError(f"After resampling, shapes still mismatch: seismic.shape={seismic.shape}, impedance.shape={impedance.shape}")

            seismic = seismic.astype(np.float32)
            impedance = impedance.astype(np.float32)
            meta = {
                "source": "npy",
                "data_npy": str(args.data_npy),
                "npy_key_seismic": str(args.npy_key_seismic),
                "npy_key_impedance": str(args.npy_key_impedance),
            }
            if args.dt_s is not None:
                meta["dt_s"] = float(args.dt_s)
            else:
                deviations.append(
                    "Paper detail not disclosed in data_npy: dt_s is unknown/not provided; results may not be comparable to the paper unless dt_s=0.001s is confirmed."
                )
        else:
            data = np.load(args.data, allow_pickle=True)
            seismic = data["seismic"].astype(np.float32)
            impedance = data["impedance"].astype(np.float32)
            meta = json.loads(data["meta"].item()) if "meta" in data else {}

    # Optional: dt-aware kernel sizing to preserve the paper's physical window (~299ms) when dt differs.
    dt_for_kernel = meta.get("dt_s", None)
    if args.auto_kernel_from_dt:
        if dt_for_kernel is None:
            raise ValueError("--auto_kernel_from_dt requires dt_s to be available (SEG-Y headers or npz meta or --dt_s when using --data_npy).")
        dt_s = float(dt_for_kernel)
        dt_ms = dt_s * 1000.0
        if dt_ms <= 0:
            raise ValueError(f"Invalid dt_s for kernel sizing: {dt_s}")
        k = int(round(float(args.kernel_ms) / dt_ms))
        k = max(3, k)
        if k % 2 == 0:
            k += 1
        # Only override if user didn't explicitly set non-default values.
        if args.k_first == ap.get_default("k_first"):
            args.k_first = k
        if args.k_res1 == ap.get_default("k_res1"):
            args.k_res1 = k
        if args.wavelet_nt == ap.get_default("wavelet_nt"):
            args.wavelet_nt = k
        if args.paper and abs(float(dt_for_kernel) - paper_dt_s) > 1e-6:
            deviations.append(
                f"Non-paper adaptation: enabled --auto_kernel_from_dt; set k_first/k_res1/wavelet_nt={k} to preserve ~{args.kernel_ms}ms window under dt_s={dt_s}."
            )

    # Scaling/activation consistency checks
    if args.impedance_scaler == "standard" and (args.output_activation.lower() in ("relu", "sigmoid") or (args.output_activation == "" and args.last_relu)):
        deviations.append(
            "Potential mismatch: impedance_scaler=standard can produce negative targets in scaled space; consider --output_activation none and/or --no-last-relu."
        )

    # Record dataset deviations in non-strict paper mode
    if args.paper and not args.paper_strict:
        if tuple(seismic.shape) != paper_shape:
            deviations.append(
                f"Non-paper dataset shape: observed {tuple(seismic.shape)}, paper uses {paper_shape}. This run is a sanity/engineering run, not a strict reproduction."
            )
        dt_val = meta.get("dt_s", None)
        if dt_val is not None:
            try:
                dt_val = float(dt_val)
            except Exception:
                dt_val = None
        if dt_val is not None and abs(dt_val - paper_dt_s) > 1e-6:
            deviations.append(f"Non-paper sampling interval: observed dt_s={dt_val}, paper uses dt_s={paper_dt_s}.")

    if args.paper and args.train_snr_db is not None:
        deviations.append(f"Non-paper training augmentation: added Gaussian noise to seismic inputs during training (train_snr_db={float(args.train_snr_db)} dB).")

    if args.paper and args.paper_strict:
        # Paper Marmousi2 synthetic experiment: 13601 traces x 2800 samples, dt=1ms.
        expected_shape = (13601, 2800)
        if seismic.shape != expected_shape:
            raise ValueError(
                "Paper strict check failed: dataset shape mismatch. "
                f"Observed seismic shape={seismic.shape}, expected {expected_shape} per paper."
            )

        # dt check only meaningful when we can read dt (SEG-Y path or meta contains dt_s).
        dt_s = None
        if args.vp_segy:
            dt_s = float(meta.get("dt_s")) if "dt_s" in meta else None
        elif "dt_s" in meta:
            dt_s = float(meta.get("dt_s"))
        if dt_s is None:
            raise ValueError(
                "Paper strict check failed: dt_s is not available to validate. "
                "Paper requires dt=0.001s. Provide SEG-Y with dt in headers, or an npz meta field 'dt_s', "
                "or (when using --data_npy) pass --dt_s 0.001."
            )
        if abs(dt_s - 0.001) > 1e-6:
            raise ValueError(
                "Paper strict check failed: sampling interval mismatch. "
                f"Observed dt_s={dt_s}, expected 0.001 per paper."
            )

    n = seismic.shape[0]
    # Paper split: 10601/1500/1500. In strict mode, enforce exactly.
    n_train, n_val, n_test = 10601, 1500, 1500
    if args.paper and args.paper_strict:
        if n != (n_train + n_val + n_test):
            raise ValueError(
                "Paper strict check failed: expected exactly 13601 traces to match the paper split. "
                f"Observed n={n}."
            )
    else:
        # Non-strict: if dataset smaller, proportionally adjust.
        if n < n_train + n_val + n_test:
            n_train = int(n * 0.78)
            n_val = int(n * 0.11)
            n_test = min(n - n_train - n_val, int(n * 0.11))
            if args.paper:
                deviations.append(
                    "Non-paper split: dataset has fewer than 13601 traces, so split sizes were proportionally resized (not the paper's fixed 10601/1500/1500)."
                )
    splits = make_split_indices(n, n_train, n_val, n_test, seed=args.seed)

    # Fit scalers on train split only
    seismic_scaler, imp_scaler = fit_default_scalers(
        seismic[splits.train],
        impedance[splits.train],
        impedance_scaler=args.impedance_scaler,
    )

    ds_train = SeismicImpedanceDataset(seismic, impedance, splits.train, seismic_scaler, imp_scaler)
    ds_val   = SeismicImpedanceDataset(seismic, impedance, splits.val, seismic_scaler, imp_scaler)

    pin = str(args.device).startswith("cuda")
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    output_activation = (args.output_activation if args.output_activation != "" else None)
    model = FCRSN_CW(
        k_first=args.k_first,
        k_last=args.k_last,
        k_res1=args.k_res1,
        k_res2=args.k_res2,
        last_relu=args.last_relu,
        output_activation=output_activation,
    ).to(args.device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="min",
            factor=float(args.plateau_factor),
            patience=int(args.plateau_patience),
            min_lr=float(args.min_lr),
        )

    run_dir = Path(args.run_dir)
    (run_dir/"checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir/"results").mkdir(parents=True, exist_ok=True)

    save_scalers(run_dir/"scalers.json", seismic_scaler, imp_scaler)
    (run_dir/"split.json").write_text(json.dumps({
        "train": splits.train.tolist(), "val": splits.val.tolist(), "test": splits.test.tolist(),
        "meta": meta,
    }, indent=2), encoding="utf-8")

    if args.paper:
        write_paper_repro_record(
            run_dir,
            args,
            meta,
            data_note={
                "shape": tuple(seismic.shape),
                "n_train": int(n_train),
                "n_val": int(n_val),
                "n_test": int(n_test),
                "scaling": f"seismic z-score, impedance {args.impedance_scaler} (fit on train split)",
            },
            deviations=deviations,
        )

    best = float("inf")
    state = TrainState(epoch=0, best_val_loss=best)

    loss_curve = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch_cfg(
            model,
            dl_train,
            optim,
            device=torch.device(args.device),
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            grad_weight=args.grad_loss_weight,
            train_snr_db=args.train_snr_db,
            noise_seed=args.seed + epoch,
        )
        va = eval_one_epoch_cfg(
            model,
            dl_val,
            device=torch.device(args.device),
            loss_type=args.loss_type,
            huber_delta=args.huber_delta,
            grad_weight=args.grad_loss_weight,
        )
        loss_curve["train"].append(tr)
        loss_curve["val"].append(va)
        if scheduler is not None:
            scheduler.step(va)
        print(f"Epoch {epoch:03d}: train_loss={tr:.6f}  val_loss={va:.6f}")
        state.epoch = epoch
        if va < best:
            best = va
            state.best_val_loss = best
            save_checkpoint(run_dir/"checkpoints/best.pt", model, optim, state)
    save_checkpoint(run_dir/"checkpoints/last.pt", model, optim, state)
    (run_dir/"results"/"loss_curve.json").write_text(json.dumps(loss_curve, indent=2), encoding="utf-8")
    print(f"[OK] Done. Best val MSE = {best:.6f}. Checkpoints in {run_dir/'checkpoints'}")

if __name__ == "__main__":
    main()
