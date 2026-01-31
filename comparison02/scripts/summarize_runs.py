"""Summarize metrics for each run under ./runs.

Outputs (per run):
- Loss: best/last train & val from results/loss_curve.json (if present)
- PCC: pcc_shallow, pcc_deep from results/metrics.json (if present)
- R2: computed on test split from results/true_impedance_all.npy & pred_impedance_all.npy

Usage:
  python scripts/summarize_runs.py
  python scripts/summarize_runs.py --runs_dir runs

This is intentionally lightweight and does not re-run evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _best_last(loss_curve: dict[str, Any], key: str) -> tuple[float | None, float | None]:
    arr = loss_curve.get(key)
    if not isinstance(arr, list) or len(arr) == 0:
        return None, None
    try:
        values = [float(v) for v in arr]
    except Exception:
        return None, None
    return float(min(values)), float(values[-1])


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if ss_tot == 0.0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _find_test_indices(split: dict[str, Any]) -> np.ndarray | None:
    for key in ("test_idx", "test_indices", "test"):
        if key in split:
            idx = split[key]
            try:
                return np.asarray(idx, dtype=np.int64)
            except Exception:
                return None
    # Some formats nest under "split" or similar; try shallow search
    for k, v in split.items():
        if isinstance(v, dict):
            out = _find_test_indices(v)
            if out is not None:
                return out
    return None


def _format_float(x: Any, digits: int = 6) -> str:
    if x is None:
        return "-"
    try:
        xf = float(x)
    except Exception:
        return "-"
    if np.isnan(xf) or np.isinf(xf):
        return str(xf)
    return f"{xf:.{digits}f}"


def iter_runs(runs_dir: Path) -> Iterable[Path]:
    if not runs_dir.exists():
        return []
    return sorted([p for p in runs_dir.iterdir() if p.is_dir()])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs", help="Directory containing run subfolders")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)

    rows: list[dict[str, Any]] = []
    for run_path in iter_runs(runs_dir):
        results = run_path / "results"
        metrics_p = results / "metrics.json"
        loss_p = results / "loss_curve.json"
        split_p = run_path / "split.json"
        true_p = results / "true_impedance_all.npy"
        pred_p = results / "pred_impedance_all.npy"

        if not metrics_p.exists():
            continue

        metrics = _read_json(metrics_p)
        loss_curve = _read_json(loss_p) if loss_p.exists() else {}

        train_best, train_last = _best_last(loss_curve, "train")
        val_best, val_last = _best_last(loss_curve, "val")

        r2_test = None
        if split_p.exists() and true_p.exists() and pred_p.exists():
            split = _read_json(split_p)
            test_idx = _find_test_indices(split)
            if test_idx is not None:
                y_true_all = np.squeeze(np.load(true_p))
                y_pred_all = np.squeeze(np.load(pred_p))
                if y_true_all.shape == y_pred_all.shape and y_true_all.ndim == 2:
                    y_true = y_true_all[test_idx]
                    y_pred = y_pred_all[test_idx]
                    r2_test = _r2_score(y_true, y_pred)

        rows.append(
            {
                "run": run_path.name,
                "mse": metrics.get("mse"),
                "pcc_shallow": metrics.get("pcc_shallow"),
                "pcc_deep": metrics.get("pcc_deep"),
                "train_best": train_best,
                "train_last": train_last,
                "val_best": val_best,
                "val_last": val_last,
                "r2_test": r2_test,
            }
        )

    # Print a compact table
    headers = [
        "run",
        "train_best",
        "train_last",
        "val_best",
        "val_last",
        "mse",
        "pcc_shallow",
        "pcc_deep",
        "r2_test",
    ]

    print("\nRun metrics summary\n")
    print("\t".join(headers))
    for r in rows:
        print(
            "\t".join(
                [
                    str(r.get("run", "-")),
                    _format_float(r.get("train_best")),
                    _format_float(r.get("train_last")),
                    _format_float(r.get("val_best")),
                    _format_float(r.get("val_last")),
                    _format_float(r.get("mse"), digits=4),
                    _format_float(r.get("pcc_shallow"), digits=6),
                    _format_float(r.get("pcc_deep"), digits=6),
                    _format_float(r.get("r2_test"), digits=6),
                ]
            )
        )

    # Also emit a JSON next to runs_dir for easy ingestion
    out_json = runs_dir / "runs_summary.json"
    out_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
