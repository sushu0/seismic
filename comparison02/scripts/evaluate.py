#!/usr/bin/env python
"""Evaluate a trained FCRSN-CW model and reproduce key metrics/figures.

Paper evaluations include:
- MSE on predictions fileciteturn2file0L284-L286
- Noise robustness: add Gaussian noise with SNR 35/25/15/5 dB and evaluate fileciteturn2file0L318-L323
- PCC shallow vs deep (shallow: first 1/5 samples) fileciteturn2file0L367-L383

This script writes:
  runs/<exp>/results/metrics.json
  runs/<exp>/results/noise_table.json
  runs/<exp>/results/trace_650.png, trace_1250.png
  runs/<exp>/results/pred_section.png (first 512 traces)
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
from scipy.ndimage import median_filter

from fcrsn_cw.utils.seed import set_global_seed
from fcrsn_cw.data.dataset import SeismicImpedanceDataset
from fcrsn_cw.data.scaler import load_scalers
from fcrsn_cw.models.fcrsn_cw import FCRSN_CW
from fcrsn_cw.utils.metrics import mse, add_gaussian_noise_snr, pcc_shallow_deep
from fcrsn_cw.utils.plotting import plot_trace_compare, plot_section

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys = []
    yhats = []
    for x, y in loader:
        x = x.to(device)
        yhat = model(x).cpu().numpy()
        ys.append(y.numpy())
        yhats.append(yhat)
    y = np.concatenate(ys, axis=0)[:, 0, :]
    yhat = np.concatenate(yhats, axis=0)[:, 0, :]
    return yhat, y

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--paper",
        action="store_true",
        help="Paper-locked evaluation: only compute/report paper-defined metrics (MSE, PCC shallow/deep, noise robustness table).",
    )
    ap.add_argument("--data", type=str, default="data/synth_marmousi_like.npz")
    ap.add_argument(
        "--data_npy",
        type=str,
        default="",
        help="Optional: path to a .npy that contains a pickled dict (e.g., keys 'seismic' and 'acoustic_impedance').",
    )
    ap.add_argument("--npy_key_seismic", type=str, default="seismic")
    ap.add_argument("--npy_key_impedance", type=str, default="acoustic_impedance")
    ap.add_argument(
        "--npy_resample",
        type=str,
        default="auto",
        choices=["auto", "none", "upsample_seismic", "downsample_impedance"],
        help="When loading from --data_npy and lengths differ: try/fix integer-factor resampling to match shapes.",
    )
    ap.add_argument("--run_dir", type=str, default="runs/exp1")
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--plot_split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which split to use for the example trace figures (paper uses validation set).",
    )
    ap.add_argument(
        "--plot_local_indices",
        type=int,
        nargs="*",
        default=[650, 1250],
        help="1-based indices within the chosen split to plot (paper uses 650 and 1250).",
    )

    ap.add_argument(
        "--postprocess",
        type=str,
        default="none",
        choices=["none", "median"],
        help="Optional inference post-processing applied to predicted impedance (physical scale).",
    )
    ap.add_argument(
        "--pp_kernel",
        type=int,
        default=5,
        help="Post-processing kernel size (odd) for --postprocess median (applied along time axis).",
    )

    ap.add_argument("--k_first", type=int, default=299)
    ap.add_argument("--k_last", type=int, default=3)
    ap.add_argument("--k_res1", type=int, default=299)
    ap.add_argument("--k_res2", type=int, default=3)
    ap.add_argument(
        "--output_activation",
        type=str,
        default="",
        choices=["", "none", "relu", "sigmoid"],
        help="Optional: override model output activation. '' keeps legacy behavior controlled by --last_relu.",
    )
    ap.add_argument(
        "--auto_kernel_from_dt",
        action="store_true",
        help="Optional: scale k_first/k_res1 to preserve ~kernel_ms window using dt_s from dataset meta (npz only).",
    )
    ap.add_argument("--kernel_ms", type=float, default=299.0)
    ap.add_argument(
        "--last_relu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ReLU at output (paper uses ReLU). Use --no-last-relu to disable.",
    )
    args = ap.parse_args()

    set_global_seed(args.seed)

    run_dir = Path(args.run_dir)

    deviations: list[str] = []
    paper_shape = (13601, 2800)

    meta: dict = {}

    if args.data_npy:
        obj = np.load(args.data_npy, allow_pickle=True)
        if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
            obj = obj.item()
        if not isinstance(obj, dict):
            raise ValueError("--data_npy expects a pickled dict-like object.")
        if args.npy_key_seismic not in obj or args.npy_key_impedance not in obj:
            raise ValueError(f"Missing required keys in {args.data_npy}. Available keys: {sorted(list(obj.keys()))}")
        seismic = np.asarray(obj[args.npy_key_seismic])
        impedance = np.asarray(obj[args.npy_key_impedance])
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
            mode = args.npy_resample
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
                deviations.append(
                    "Non-paper preprocessing: resampled seismic (upsample) to match impedance length because input data_npy has mismatched sample counts."
                )
            elif mode == "downsample_impedance":
                if not is_int(ratio_down):
                    raise ValueError(f"Cannot downsample impedance: {t_s}/{t_z} not integer")
                down = int(round(ratio_down))
                impedance = resample_poly(impedance.astype(np.float32), up=1, down=down, axis=1).astype(np.float32)
                impedance = impedance[:, :t_s]
                deviations.append(
                    "Non-paper preprocessing: resampled impedance (downsample) to match seismic length because input data_npy has mismatched sample counts."
                )
            if seismic.shape != impedance.shape:
                raise ValueError(f"After resampling, shapes still mismatch: seismic.shape={seismic.shape}, impedance.shape={impedance.shape}")
        seismic = seismic.astype(np.float32)
        impedance = impedance.astype(np.float32)
        meta = {"source": "npy", "data_npy": str(args.data_npy)}
    else:
        data = np.load(args.data, allow_pickle=True)
        seismic = data["seismic"].astype(np.float32)
        impedance = data["impedance"].astype(np.float32)
        if "meta" in data:
            try:
                meta = json.loads(data["meta"].item())
            except Exception:
                meta = {}

    # Optional: dt-aware kernel sizing (npz meta only)
    if args.auto_kernel_from_dt:
        dt_s = meta.get("dt_s", None)
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
        deviations.append(f"Non-paper adaptation: enabled --auto_kernel_from_dt; set k_first/k_res1={k} using dt_s={float(dt_s)}.")

    split = json.loads((run_dir/"split.json").read_text(encoding="utf-8"))
    test_idx = np.array(split["test"], dtype=int)
    val_idx = np.array(split["val"], dtype=int)

    seismic_scaler, imp_scaler = load_scalers(run_dir/"scalers.json")
    ds_test = SeismicImpedanceDataset(seismic, impedance, test_idx, seismic_scaler, imp_scaler)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    output_activation = (args.output_activation if args.output_activation != "" else None)
    model = FCRSN_CW(
        k_first=args.k_first,
        k_last=args.k_last,
        k_res1=args.k_res1,
        k_res2=args.k_res2,
        last_relu=args.last_relu,
        output_activation=output_activation,
    ).to(args.device)
    ckpt = torch.load(str(run_dir/args.ckpt), map_location="cpu")
    model.load_state_dict(ckpt["model"])

    yhat_scaled, y_scaled = predict(model, dl_test, device=torch.device(args.device))
    # Undo scaling to compute metrics in both spaces
    yhat = imp_scaler.inverse_transform(yhat_scaled)
    y = imp_scaler.inverse_transform(y_scaled)

    if args.postprocess != "none":
        if args.postprocess == "median":
            k = int(args.pp_kernel)
            if k < 1:
                raise ValueError("--pp_kernel must be >= 1")
            if k % 2 == 0:
                k += 1
            yhat = median_filter(yhat, size=(1, k), mode="nearest")
            deviations.append(f"Non-paper inference: applied median postprocess with kernel={k} along time axis.")
        else:
            raise ValueError(f"Unknown --postprocess: {args.postprocess}")

    if args.paper:
        # Paper reports MSE and PCC; keep metric definitions consistent.
        metrics = {
            "mse": mse(yhat, y),
        }
    else:
        metrics = {
            "mse_scaled": mse(yhat_scaled, y_scaled),
            "mse_physical": mse(yhat, y),
        }
    pcc_s, pcc_d = pcc_shallow_deep(yhat, y, shallow_frac=0.2)
    metrics["pcc_shallow"] = pcc_s
    metrics["pcc_deep"] = pcc_d

    # Noise robustness (evaluate on noisy seismic while keeping same true impedance)
    rng = np.random.default_rng(args.seed)
    noise_levels = [35, 25, 15, 5]
    table = []
    # Build a loader that we can feed noisy seismic
    x_test = seismic[test_idx].copy()
    y_test = impedance[test_idx].copy()
    for snr in noise_levels:
        x_noisy = add_gaussian_noise_snr(x_test, snr_db=snr, rng=rng).astype(np.float32)
        ds = SeismicImpedanceDataset(x_noisy, y_test, np.arange(x_noisy.shape[0]), seismic_scaler, imp_scaler)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        yhat_s, y_s = predict(model, dl, device=torch.device(args.device))
        yhat_n = imp_scaler.inverse_transform(yhat_s)
        y_n = imp_scaler.inverse_transform(y_s)
        pcc_s_n, pcc_d_n = pcc_shallow_deep(yhat_n, y_n, shallow_frac=0.2)
        row = {
            "snr_db": snr,
            **({"mse": mse(yhat_n, y_n)} if args.paper else {"mse_scaled": mse(yhat_s, y_s), "mse_physical": mse(yhat_n, y_n)}),
            "pcc_shallow": pcc_s_n,
            "pcc_deep": pcc_d_n,
        }
        table.append(row)

    out_dir = run_dir/"results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir/"noise_table.json").write_text(json.dumps(table, indent=2), encoding="utf-8")

    if args.paper:
        if tuple(seismic.shape) != paper_shape:
            deviations.append(
                f"Non-paper dataset shape: observed {tuple(seismic.shape)}, paper uses {paper_shape}. This evaluation is a sanity/engineering check, not a strict reproduction."
            )
        note = {
            "paper_mode": True,
            "metric_definitions": {
                "mse": "MSE on physical-scale impedance (after inverse scaling)",
                "pcc_shallow": "PCC on shallow interval (first 1/5 samples)",
                "pcc_deep": "PCC on deep interval (last 4/5 samples)",
                "noise": "Gaussian noise added to seismic with SNR 35/25/15/5 dB",
            },
            "assumptions": [
                "Uses the same scalers as training (loaded from run_dir/scalers.json).",
                "Example traces default to validation split local indices 650 and 1250, as described in the paper.",
            ],
            "deviations_from_paper": deviations,
        }
        (out_dir/"eval_paper_record.json").write_text(json.dumps(note, indent=2), encoding="utf-8")

    # Plot example traces (paper: validation set #650 and #1250)
    if args.plot_split == "val":
        plot_idx = val_idx
        plot_tag = "val"
    else:
        plot_idx = test_idx
        plot_tag = "test"
    ds_plot = SeismicImpedanceDataset(seismic, impedance, plot_idx, seismic_scaler, imp_scaler)
    dl_plot = DataLoader(ds_plot, batch_size=args.batch_size, shuffle=False, num_workers=0)
    yhat_plot_scaled, y_plot_scaled = predict(model, dl_plot, device=torch.device(args.device))
    yhat_plot = imp_scaler.inverse_transform(yhat_plot_scaled)
    y_plot = imp_scaler.inverse_transform(y_plot_scaled)

    t = np.arange(y_plot.shape[1])
    for one_based in args.plot_local_indices:
        local = int(one_based) - 1
        if 0 <= local < y_plot.shape[0]:
            plot_trace_compare(
                out_dir / f"trace_{plot_tag}_{one_based}.png",
                t,
                y_plot[local],
                yhat_plot[local],
                title=f"Trace {one_based} ({plot_tag})",
            )

    # Plot a small section (first 512 traces of the test set)
    n_show = min(512, yhat.shape[0])
    plot_section(out_dir/"pred_section.png", yhat[:n_show], title="Predicted impedance section (subset)")

    print("[OK] metrics:", metrics)
    print("[OK] noise_table saved:", out_dir/"noise_table.json")
    print("[OK] figures saved to:", out_dir)

if __name__ == "__main__":
    main()
