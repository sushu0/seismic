#!/usr/bin/env python
"""Validate SEG-Y files against the paper's Marmousi2 synthetic experiment requirements.

Checks:
- sampling interval dt (from SEG-Y headers)
- trace count and samples per trace
- value range and NaN/Inf ratio
- optional: compare against expected paper spec (13601x2800, dt=1ms)

Writes a JSON report for reproducibility.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read_dt_us(path: Path) -> int:
    import segyio

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        dt_us = int(f.bin.get(segyio.BinField.Interval, 0) or 0)
        if dt_us <= 0:
            dt_us = int(f.header[0].get(segyio.TraceField.TRACE_SAMPLE_INTERVAL, 0) or 0)
    return dt_us


def _read_traces(path: Path) -> np.ndarray:
    import segyio

    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        arr = np.asarray(segyio.tools.collect(f.trace[:]), dtype=np.float64)
    return arr


def _stats(arr: np.ndarray) -> dict:
    finite = np.isfinite(arr)
    out: dict = {
        "shape": list(arr.shape),
        "finite_ratio": float(finite.mean()),
        "nan_ratio": float(np.isnan(arr).mean()),
        "inf_ratio": float(np.isinf(arr).mean()),
    }
    if finite.any():
        a = arr[finite]
        out.update(
            {
                "min": float(a.min()),
                "max": float(a.max()),
                "mean": float(a.mean()),
                "p01": float(np.percentile(a, 1)),
                "p99": float(np.percentile(a, 99)),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vp_segy", type=str, required=True)
    ap.add_argument("--rho_segy", type=str, default="")
    ap.add_argument("--out", type=str, default="segy_report.json")
    ap.add_argument("--paper", action="store_true", help="Check against paper spec: dt=1ms and shape 13601x2800")
    args = ap.parse_args()

    vp_path = Path(args.vp_segy)
    rho_path = Path(args.rho_segy) if args.rho_segy else None

    report: dict = {"vp": str(vp_path)}

    dt_us = _read_dt_us(vp_path)
    report["dt_us"] = int(dt_us)
    report["dt_s"] = float(dt_us) / 1e6 if dt_us > 0 else None

    vp = _read_traces(vp_path)
    report["vp_stats"] = _stats(vp)

    if rho_path is not None:
        report["rho"] = str(rho_path)
        dt2_us = _read_dt_us(rho_path)
        report["rho_dt_us"] = int(dt2_us)
        rho = _read_traces(rho_path)
        report["rho_stats"] = _stats(rho)

        if vp.shape == rho.shape and np.isfinite(vp).any() and np.isfinite(rho).any():
            imp = vp * rho
            report["impedance_stats"] = _stats(imp)

    if args.paper:
        expected_shape = (13601, 2800)
        expected_dt_s = 0.001
        report["paper_expected_shape"] = list(expected_shape)
        report["paper_expected_dt_s"] = expected_dt_s
        report["paper_shape_ok"] = tuple(vp.shape) == expected_shape
        report["paper_dt_ok"] = (report["dt_s"] is not None) and (abs(report["dt_s"] - expected_dt_s) < 1e-6)

    out = Path(args.out)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")
    print(json.dumps({k: report[k] for k in report if k in ['dt_s','vp_stats','paper_shape_ok','paper_dt_ok']}, indent=2))


if __name__ == "__main__":
    main()
