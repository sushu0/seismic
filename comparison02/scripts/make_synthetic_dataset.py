#!/usr/bin/env python
"""Generate a synthetic seismic-impedance dataset following the paper's forward modeling logic.

Paper: impedance Z(t) -> reflectivity r_i=(Z_{i+1}-Z_i)/(Z_{i+1}+Z_i) -> convolve with a Ricker wavelet (30Hz, 0 phase) fileciteturn2file0L228-L245

Since Marmousi2 is not bundled, this script can:
1) Load a user-provided impedance section (.npy, shape [n_traces, n_samples])
2) Or generate a 'marmousi-like' layered impedance section procedurally.

Outputs: data/<name>.npz with keys: seismic, impedance, meta
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add project root
import argparse
import json
from pathlib import Path
import numpy as np

from fcrsn_cw.physics.forward import make_synthetic_pair_from_impedance

def generate_layered_impedance(n_traces: int, n_samples: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Generate random layered structure per trace with lateral continuity
    base = np.zeros((n_traces, n_samples), dtype=np.float32)
    # Create a few horizons with smooth lateral variations
    n_layers = 25
    horizon = np.linspace(0, n_samples - 1, n_layers).astype(int)
    horizon = horizon + rng.integers(-8, 9, size=(n_traces, n_layers))
    horizon = np.clip(horizon, 0, n_samples - 1)
    horizon.sort(axis=1)
    # Impedance values per layer (in arbitrary units)
    layer_vals = rng.uniform(2.0, 6.0, size=(n_traces, n_layers)).astype(np.float32)
    for i in range(n_traces):
        h = horizon[i]
        zvals = layer_vals[i]
        prev = 0
        for j, hh in enumerate(h):
            base[i, prev:hh] = zvals[j]
            prev = hh
        base[i, prev:] = zvals[-1]
    # Add smooth vertical trend + correlated noise
    trend = np.linspace(0.0, 1.0, n_samples, dtype=np.float32)[None, :]
    base = base * (1.0 + 0.15 * trend)
    # Lateral smoothing for continuity
    for t in range(n_samples):
        base[:, t] = np.convolve(base[:, t], np.ones(9)/9.0, mode="same")
    # Small random texture
    base += rng.normal(0.0, 0.05, size=base.shape).astype(np.float32)
    base = np.clip(base, 1.0, None)
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/synth_marmousi_like.npz")
    ap.add_argument("--impedance_npy", type=str, default="", help="Optional .npy impedance section [n_traces,n_samples]")
    ap.add_argument("--n_traces", type=int, default=13601)
    ap.add_argument("--n_samples", type=int, default=2800)
    ap.add_argument("--dt", type=float, default=0.001, help="Sampling interval in seconds (paper uses 1ms)")
    ap.add_argument("--f0", type=float, default=30.0, help="Ricker central frequency (Hz)")
    ap.add_argument("--phase", type=float, default=0.0, help="Phase rotation in degrees")
    ap.add_argument("--wavelet_nt", type=int, default=299, help="Wavelet length in samples (default aligns with paper's kernel size)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.impedance_npy:
        Z = np.load(args.impedance_npy).astype(np.float32)
        assert Z.ndim == 2, "impedance_npy must be a 2D array [n_traces,n_samples]"
        n_traces, n_samples = Z.shape
    else:
        Z = generate_layered_impedance(args.n_traces, args.n_samples, seed=args.seed)
        n_traces, n_samples = Z.shape

    seismic, impedance = make_synthetic_pair_from_impedance(
        impedance=Z,
        f0_hz=args.f0,
        dt_s=args.dt,
        wavelet_nt=args.wavelet_nt,
        phase_deg=args.phase,
    )

    meta = {
        "n_traces": int(n_traces),
        "n_samples": int(n_samples),
        "dt_s": float(args.dt),
        "wavelet_f0_hz": float(args.f0),
        "wavelet_phase_deg": float(args.phase),
        "wavelet_nt": int(args.wavelet_nt),
        "seed": int(args.seed),
        "note": "Synthetic dataset generated following FCRSN-CW paper forward modeling.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, seismic=seismic, impedance=impedance, meta=json.dumps(meta))
    print(f"[OK] Saved: {out}  seismic={seismic.shape} impedance={impedance.shape}")

if __name__ == "__main__":
    main()
