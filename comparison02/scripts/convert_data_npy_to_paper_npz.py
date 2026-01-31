#!/usr/bin/env python
"""Convert a Marmousi2-style data.npy (pickled dict) into an npz compatible with this repo.

This is NOT a strict paper reproduction.
It is a deterministic format-conversion utility to make the provided data runnable with
paper-aligned settings while explicitly recording deviations.

Expected input (as discovered in this workspace):
- data.npy is a pickled dict with keys:
  - seismic: (N, 1, T_s) or (N, T_s)
  - acoustic_impedance: (N, 1, T_z) or (N, T_z)

Output npz:
- seismic: (N_out, target_nt)
- impedance: (N_out, target_nt)
- meta: json string
- conversion: json string (explicit deviations/assumptions)
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add project root

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from fcrsn_cw.physics.forward import make_synthetic_pair_from_impedance


def _load_pickled_dict(npy_path: Path) -> dict[str, Any]:
    obj = np.load(npy_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
    if not isinstance(obj, dict):
        raise ValueError("Input .npy must be a pickled dict.")
    return obj


def _squeeze_channel(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3 and x.shape[1] == 1:
        return x[:, 0, :]
    return x


def _ensure_2d(name: str, x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected {name} to be 2D after squeeze; got shape={x.shape} ndim={x.ndim}")
    return x


def _resample_time(x: np.ndarray, target_nt: int) -> np.ndarray:
    """Deterministic time resampling using resample_poly with rational factor."""
    if x.shape[1] == target_nt:
        return x.astype(np.float32)

    src_nt = int(x.shape[1])
    # exact rational: target/src = (target/g) / (src/g)
    g = int(np.gcd(src_nt, target_nt))
    up = target_nt // g
    down = src_nt // g
    y = resample_poly(x.astype(np.float32), up=up, down=down, axis=1).astype(np.float32)
    # clip/pad to exact length
    if y.shape[1] > target_nt:
        y = y[:, :target_nt]
    elif y.shape[1] < target_nt:
        pad = target_nt - y.shape[1]
        y = np.pad(y, ((0, 0), (0, pad)), mode="edge").astype(np.float32)
    return y


def _resize_traces(
    x: np.ndarray,
    target_n: int,
    mode: str,
) -> np.ndarray:
    if x.shape[0] == target_n:
        return x
    if target_n <= 0:
        raise ValueError("target_n must be positive")
    if mode == "repeat":
        idx = np.arange(target_n) % x.shape[0]
        return x[idx]
    if mode == "pad_edge":
        if x.shape[0] > target_n:
            return x[:target_n]
        pad = target_n - x.shape[0]
        return np.concatenate([x, np.repeat(x[-1:, :], pad, axis=0)], axis=0)
    raise ValueError(f"Unknown trace resize mode: {mode}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_npy", type=str, default="data.npy")
    ap.add_argument("--out_npz", type=str, default="data_paper_format.npz")
    ap.add_argument("--key_seismic", type=str, default="seismic")
    ap.add_argument("--key_impedance", type=str, default="acoustic_impedance")

    ap.add_argument("--target_nt", type=int, default=2800, help="Paper uses 2800 samples.")
    ap.add_argument("--dt_s", type=float, default=0.001, help="Paper uses dt=0.001s.")

    ap.add_argument(
        "--seismic_source",
        type=str,
        default="regenerate",
        choices=["regenerate", "provided_resample"],
        help=(
            "How to produce seismic for output. "
            "'regenerate' uses paper forward model from impedance (reflectivity + Ricker convolution). "
            "'provided_resample' time-resamples the provided seismic." 
        ),
    )

    ap.add_argument("--f0", type=float, default=30.0)
    ap.add_argument("--phase", type=float, default=0.0)
    ap.add_argument("--wavelet_nt", type=int, default=299)

    ap.add_argument(
        "--target_n_traces",
        type=int,
        default=0,
        help=(
            "Optional: force number of traces (paper uses 13601). "
            "WARNING: this creates synthetic duplication/padding and is NOT a strict paper reproduction."
        ),
    )
    ap.add_argument(
        "--trace_resize_mode",
        type=str,
        default="repeat",
        choices=["repeat", "pad_edge"],
        help="How to expand/shrink traces when --target_n_traces is set.",
    )

    args = ap.parse_args()

    input_npy = Path(args.input_npy)
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    obj = _load_pickled_dict(input_npy)
    if args.key_seismic not in obj:
        raise ValueError(f"Missing key '{args.key_seismic}'. Available keys: {sorted(list(obj.keys()))}")
    if args.key_impedance not in obj:
        raise ValueError(f"Missing key '{args.key_impedance}'. Available keys: {sorted(list(obj.keys()))}")

    seismic_in = _ensure_2d("seismic", _squeeze_channel(np.asarray(obj[args.key_seismic])))
    imp_in = _ensure_2d("impedance", _squeeze_channel(np.asarray(obj[args.key_impedance])))

    if seismic_in.shape[0] != imp_in.shape[0]:
        raise ValueError(f"Trace count mismatch: seismic {seismic_in.shape} vs impedance {imp_in.shape}")

    conversion_devs: list[str] = []

    # Align impedance to target_nt
    imp = _resample_time(imp_in, target_nt=int(args.target_nt))
    if imp.shape[1] != imp_in.shape[1]:
        conversion_devs.append(
            f"Time-resampled acoustic_impedance from nt={imp_in.shape[1]} to target_nt={args.target_nt} using resample_poly."
        )

    if args.seismic_source == "provided_resample":
        seismic = _resample_time(seismic_in, target_nt=int(args.target_nt))
        if seismic.shape[1] != seismic_in.shape[1]:
            conversion_devs.append(
                f"Time-resampled provided seismic from nt={seismic_in.shape[1]} to target_nt={args.target_nt} using resample_poly."
            )
    else:
        seismic, _ = make_synthetic_pair_from_impedance(
            impedance=imp,
            f0_hz=float(args.f0),
            dt_s=float(args.dt_s),
            wavelet_nt=int(args.wavelet_nt),
            phase_deg=float(args.phase),
        )
        conversion_devs.append(
            "Re-generated seismic from (resampled) impedance using reflectivity + Ricker convolution per paper forward model settings."
        )

    # Optional trace resizing (NOT paper)
    if int(args.target_n_traces) > 0 and int(args.target_n_traces) != seismic.shape[0]:
        target_n = int(args.target_n_traces)
        seismic = _resize_traces(seismic, target_n=target_n, mode=str(args.trace_resize_mode))
        imp = _resize_traces(imp, target_n=target_n, mode=str(args.trace_resize_mode))
        conversion_devs.append(
            f"Adjusted number of traces from N={seismic_in.shape[0]} to N={target_n} via trace_resize_mode='{args.trace_resize_mode}'."
        )

    meta = {
        "source": "converted_from_npy",
        "input_npy": str(input_npy),
        "dt_s": float(args.dt_s),
        "target_nt": int(args.target_nt),
        "seismic_source": str(args.seismic_source),
        "f0_hz": float(args.f0),
        "phase_deg": float(args.phase),
        "wavelet_nt": int(args.wavelet_nt),
    }
    conversion = {
        "input": {
            "seismic_shape": tuple(seismic_in.shape),
            "impedance_shape": tuple(imp_in.shape),
        },
        "output": {
            "seismic_shape": tuple(seismic.shape),
            "impedance_shape": tuple(imp.shape),
        },
        "deviations_from_paper": [
            "This conversion enforces paper-like array shapes/dt for executability, but does not create the true paper dataset.",
            *conversion_devs,
        ],
    }

    np.savez_compressed(
        out_npz,
        seismic=seismic.astype(np.float32),
        impedance=imp.astype(np.float32),
        meta=json.dumps(meta),
        conversion=json.dumps(conversion),
    )

    record_path = out_npz.with_suffix(".conversion.json")
    record_path.write_text(json.dumps(conversion, indent=2), encoding="utf-8")

    print("[OK] wrote", out_npz)
    print("[OK] wrote", record_path)
    print("[OK] output shapes:", conversion["output"])


if __name__ == "__main__":
    main()
