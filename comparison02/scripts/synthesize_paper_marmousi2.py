#!/usr/bin/env python
"""Paper-locked Marmousi2 synthetic data synthesis.

This script follows the synthesis flow described in the paper section shown by the user:
1) Acoustic impedance: Z = ρ * v, with constant density ρ and P-wave velocity v.
2) Reflectivity series:
     r_i = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i)
3) Convolution of reflectivity with a Ricker wavelet (30 Hz, phase 0°) to obtain synthetic seismic.

Hard requirements from the paper excerpt:
- 13601 traces
- 2800 samples per trace
- dt = 1 ms

Notes:
- The paper text in the provided image states a Ricker wavelet time window “-0.4~1 ms”.
  This script implements that window literally by default via --wavelet_tmin_ms/--wavelet_tmax_ms.
  If you need a different window that matches the original paper/PDF more precisely, override via CLI.

Outputs:
- .npz with keys: seismic, impedance, meta
- .json record with all parameters and any implementation details.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add project root

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve


def ricker_window(f0_hz: float, dt_s: float, tmin_s: float, tmax_s: float) -> np.ndarray:
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if tmax_s < tmin_s:
        raise ValueError("tmax_s must be >= tmin_s")
    nt = int(round((tmax_s - tmin_s) / dt_s)) + 1
    # Ensure at least 2 samples to make convolution meaningful.
    if nt < 2:
        nt = 2
        tmax_s = tmin_s + dt_s
    t = tmin_s + np.arange(nt, dtype=np.float64) * dt_s
    a = (np.pi * f0_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return w.astype(np.float32)


def impedance_to_reflectivity(Z: np.ndarray) -> np.ndarray:
    """Compute reflectivity along time axis.

    Z: shape (N, T)
    returns r: shape (N, T)

    Implementation detail (must be recorded): for the last sample where Z_{i+1} is undefined,
    we copy the previous reflectivity value so that r has the same length as Z.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (n_traces, n_samples)")
    num = Z[:, 1:] - Z[:, :-1]
    den = Z[:, 1:] + Z[:, :-1] + 1e-12
    r = (num / den).astype(np.float32)
    r_last = r[:, -1:]
    return np.concatenate([r, r_last], axis=1)


def reflectivity_to_seismic(r: np.ndarray, wavelet: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=np.float32)
    if r.ndim != 2:
        raise ValueError("r must be 2D")
    w = np.asarray(wavelet, dtype=np.float32)
    out = np.empty_like(r, dtype=np.float32)
    for i in range(r.shape[0]):
        out[i] = fftconvolve(r[i], w, mode="same").astype(np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vp_npy", type=str, required=True, help="Marmousi2 P-wave velocity v as .npy array (shape 13601x2800).")
    ap.add_argument("--out", type=str, default="data/marmousi2_paper_synth.npz")

    # Paper-required geometry
    ap.add_argument("--n_traces", type=int, default=13601)
    ap.add_argument("--n_samples", type=int, default=2800)
    ap.add_argument("--dt_s", type=float, default=0.001)

    ap.add_argument(
        "--vp_unit",
        type=str,
        default="auto",
        choices=["auto", "m/s", "km/s"],
        help=(
            "Unit of input vp values. The paper's impedance unit is (km/s) * (g/cm^3). "
            "If vp is in m/s, it will be converted to km/s by /1000 before computing Z=ρv. "
            "'auto' uses a minimal heuristic: if median(vp) > 50, treat as m/s; else km/s."
        ),
    )

    # Impedance
    ap.add_argument(
        "--rho_const",
        type=float,
        default=1.0,
        help=(
            "Constant density ρ in g/cm^3 used in Z=ρv. The paper states ρ is constant but does not disclose its numeric value. "
            "Minimal default: 1.0 g/cm^3. Override if you have a specific ρ."
        ),
    )

    # Wavelet parameters per excerpt
    ap.add_argument("--f0_hz", type=float, default=30.0)
    ap.add_argument("--phase_deg", type=float, default=0.0, help="Paper excerpt specifies phase 0°. Only 0° is allowed in strict mode.")
    ap.add_argument(
        "--wavelet_tmin_ms",
        type=float,
        default=-0.4,
        help="Wavelet time window start in ms (paper excerpt shows “-0.4~1 ms”).",
    )
    ap.add_argument(
        "--wavelet_tmax_ms",
        type=float,
        default=1.0,
        help="Wavelet time window end in ms (paper excerpt shows “-0.4~1 ms”).",
    )

    ap.add_argument(
        "--strict",
        action="store_true",
        help="Enforce paper hard requirements: shape 13601x2800, dt=0.001s, phase=0°.",
    )

    args = ap.parse_args()

    vp = np.load(args.vp_npy).astype(np.float32)
    if vp.ndim != 2:
        raise ValueError(f"vp_npy must be 2D. Got shape={vp.shape}")

    expected = (int(args.n_traces), int(args.n_samples))
    if args.strict:
        if vp.shape != expected:
            raise ValueError(f"Strict mode: vp shape must be {expected}, got {vp.shape}")
        if abs(float(args.dt_s) - 0.001) > 1e-12:
            raise ValueError(f"Strict mode: dt_s must be 0.001, got {args.dt_s}")
        if abs(float(args.phase_deg)) > 1e-12:
            raise ValueError(f"Strict mode: phase_deg must be 0, got {args.phase_deg}")

    # Step 1: Z = ρ v (ρ constant)
    rho = float(args.rho_const)

    vp_median = float(np.median(vp))
    if args.vp_unit == "auto":
        vp_unit = "m/s" if vp_median > 50.0 else "km/s"
        vp_unit_source = "auto_heuristic"
    else:
        vp_unit = str(args.vp_unit)
        vp_unit_source = "user_provided"

    if vp_unit == "m/s":
        v_km_s = (vp / 1000.0).astype(np.float32)
    elif vp_unit == "km/s":
        v_km_s = vp.astype(np.float32)
    else:
        raise ValueError(f"Unexpected vp_unit: {vp_unit}")

    # Impedance unit: (km/s) * (g/cm^3)
    Z = (v_km_s * rho).astype(np.float32)

    # Step 2: reflectivity
    r = impedance_to_reflectivity(Z)

    # Step 3: Ricker wavelet and convolution
    w = ricker_window(
        f0_hz=float(args.f0_hz),
        dt_s=float(args.dt_s),
        tmin_s=float(args.wavelet_tmin_ms) * 1e-3,
        tmax_s=float(args.wavelet_tmax_ms) * 1e-3,
    )

    # phase rotation: paper excerpt is 0°, so we do nothing unless user sets non-zero (non-strict)
    if abs(float(args.phase_deg)) > 1e-12:
        # Minimal implementation: constant phase rotation via analytic signal.
        # Kept inline to avoid changing library behavior outside paper mode.
        from scipy.signal import hilbert

        phase = np.deg2rad(float(args.phase_deg))
        analytic = hilbert(w.astype(np.float64))
        w = np.real(analytic * np.exp(1j * phase)).astype(np.float32)

    seismic = reflectivity_to_seismic(r, w)

    meta = {
        "source": "paper_synthesis",
        "vp_npy": str(args.vp_npy),
        "rho_const": rho,
        "rho_const_source": ("default_assumption" if ("--rho_const" not in sys.argv) else "user_provided"),
        "units": {
            "vp_input": vp_unit,
            "vp_unit_source": vp_unit_source,
            "impedance": "(km/s)*(g/cm^3)",
            "rho_const": "g/cm^3",
        },
        "vp_median": vp_median,
        "n_traces": int(vp.shape[0]),
        "n_samples": int(vp.shape[1]),
        "dt_s": float(args.dt_s),
        "reflectivity_formula": "r_i=(Z_{i+1}-Z_i)/(Z_{i+1}+Z_i)",
        "wavelet": {
            "type": "ricker",
            "f0_hz": float(args.f0_hz),
            "phase_deg": float(args.phase_deg),
            "tmin_ms": float(args.wavelet_tmin_ms),
            "tmax_ms": float(args.wavelet_tmax_ms),
            "nt": int(w.shape[0]),
        },
        "implementation_details": {
            "reflectivity_last_sample": "copied from previous sample so r has same length as Z",
            "convolution": "fftconvolve per trace, mode='same'",
        },
        "strict": bool(args.strict),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, seismic=seismic.astype(np.float32), impedance=Z.astype(np.float32), meta=json.dumps(meta))

    record = out.with_suffix(".record.json")
    record.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote {out} seismic={seismic.shape} impedance={Z.shape}")
    print(f"[OK] wrote {record}")


if __name__ == "__main__":
    main()
