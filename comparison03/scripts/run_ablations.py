from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Experiment:
    name: str
    imp_loss: str
    grad_loss_weight: float
    k_large: int | None = None


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _load_metrics(metrics_json_path: Path) -> dict[str, Any]:
    with metrics_json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/marmousi2_2721_like_l101.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=100, help="Use a smaller number for quick ablation.")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=1100.0)
    ap.add_argument("--beta", type=float, default=550.0)
    ap.add_argument("--lambda_gp", type=float, default=10.0)
    ap.add_argument("--k_large", type=int, default=299)
    ap.add_argument("--k_small", type=int, default=3)
    ap.add_argument("--base_ch_g", type=int, default=16)
    ap.add_argument("--base_ch_d", type=int, default=8)
    ap.add_argument("--wavelet_freq", type=float, default=30.0)
    ap.add_argument("--wavelet_dt", type=float, default=0.001)
    ap.add_argument("--wavelet_dur", type=float, default=0.128)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--loss_in_physical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When normalize=true, compute Li/Ls in physical units (recommended)",
    )
    ap.add_argument("--runs_root", default=None, help="Default: runs/ablations_<timestamp>")

    ap.add_argument("--infer_batch_size", type=int, default=64)
    ap.add_argument("--time_ms_max", type=float, default=2350.0)
    ap.add_argument("--trace_ids", nargs="*", type=int, default=[299, 2299, 599, 1699])

    ap.add_argument("--post_clip", nargs=2, type=float, default=None, metavar=("P_LO", "P_HI"))
    ap.add_argument("--post_median_k", type=int, default=0)

    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = Path(args.runs_root or f"runs/ablations_{ts}")
    runs_root.mkdir(parents=True, exist_ok=True)

    exps = [
        Experiment("A_norm_mse", "mse", 0.0),
        Experiment("B_norm_l1", "l1", 0.0),
        Experiment("C_norm_l1_grad", "l1", 5.0),
        # For 470 samples, k_large=299 is very smoothing; try a smaller kernel.
        Experiment("D_norm_l1_grad_k99", "l1", 5.0, k_large=99),
    ]

    results: list[dict[str, Any]] = []

    for exp in exps:
        run_dir = runs_root / exp.name
        cfg_path = run_dir / "config.yaml"

        cfg = {
            "run_dir": str(run_dir).replace("\\", "/"),
            "seed": 1234,
            "device": args.device,
            "dataset": {"path": args.dataset, "normalize": bool(args.normalize)},
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "n_critic": int(args.n_critic),
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "lambda_gp": float(args.lambda_gp),
            "k_large": int(exp.k_large if exp.k_large is not None else args.k_large),
            "k_small": int(args.k_small),
            "base_ch_g": int(args.base_ch_g),
            "base_ch_d": int(args.base_ch_d),
            "wavelet_freq": float(args.wavelet_freq),
            "wavelet_dt": float(args.wavelet_dt),
            "wavelet_dur": float(args.wavelet_dur),
            # Reduce checkpoint spam during ablation.
            "save_every": int(args.epochs),
            "amp": bool(args.amp),
            # knobs
            "imp_loss": exp.imp_loss,
            "grad_loss_weight": float(exp.grad_loss_weight),
            "loss_in_physical": bool(args.loss_in_physical),
        }

        _write_yaml(cfg_path, cfg)

        # Train
        _run([sys.executable, "train.py", "--config", str(cfg_path)])

        # Infer
        ckpt = run_dir / "checkpoints" / "best.pt"
        pred_npz = run_dir / "pred_test.npz"
        infer_cmd = [
            sys.executable,
            "infer.py",
            "--dataset",
            args.dataset,
            "--ckpt",
            str(ckpt),
            "--split",
            "test",
            "--out",
            str(pred_npz),
            "--batch_size",
            str(args.infer_batch_size),
        ]
        if args.post_clip is not None:
            infer_cmd += ["--clip_percentiles", str(args.post_clip[0]), str(args.post_clip[1])]
        if int(args.post_median_k) > 1:
            infer_cmd += ["--median_k", str(int(args.post_median_k))]
        _run(infer_cmd)

        # Plot 4 traces
        out_png = run_dir / "traces_compare.png"
        plot_cmd = [
            sys.executable,
            "scripts/plot_traces_compare.py",
            "--pred",
            str(pred_npz),
            "--out",
            str(out_png),
            "--one_based",
            "--time_ms_max",
            str(float(args.time_ms_max)),
            "--trace_ids",
            *[str(t) for t in args.trace_ids],
            "--ckpt",
            str(ckpt),
        ]
        _run(plot_cmd)

        metrics = _load_metrics(run_dir / "pred_test_metrics.json")
        row = {
            "name": exp.name,
            "imp_loss": exp.imp_loss,
            "grad_loss_weight": exp.grad_loss_weight,
            "k_large": cfg["k_large"],
            "loss_in_physical": cfg["loss_in_physical"],
            "metrics": metrics.get("metrics"),
            "metrics_raw": metrics.get("metrics_raw"),
            "metrics_phys": metrics.get("metrics_phys"),
            "metrics_phys_raw": metrics.get("metrics_phys_raw"),
            "postprocess": metrics.get("postprocess"),
            "run_dir": str(run_dir).replace("\\", "/"),
            "plot": str(out_png).replace("\\", "/"),
        }
        results.append(row)

    # Write summary
    summary_json = runs_root / "summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump({"results": results}, f, indent=2)

    summary_csv = runs_root / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "name",
            "imp_loss",
            "grad_loss_weight",
            "k_large",
            "loss_in_physical",
            "pcc",
            "r2",
            "mse",
            "pcc_phys",
            "r2_phys",
            "mse_phys",
            "run_dir",
            "plot",
        ])
        for r in results:
            m = r.get("metrics") or {}
            mp = r.get("metrics_phys") or {}
            w.writerow([
                r["name"],
                r["imp_loss"],
                r["grad_loss_weight"],
                r.get("k_large"),
                r.get("loss_in_physical"),
                m.get("pcc"),
                m.get("r2"),
                m.get("mse"),
                mp.get("pcc"),
                mp.get("r2"),
                mp.get("mse"),
                r["run_dir"],
                r["plot"],
            ])

    print("\nSaved summary:")
    print("-", str(summary_json))
    print("-", str(summary_csv))


if __name__ == "__main__":
    main()
