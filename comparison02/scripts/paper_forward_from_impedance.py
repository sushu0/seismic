#!/usr/bin/env python
"""Paper forward modeling from impedance Z(t) -> reflectivity r(t) -> synthetic seismic s(t).

Implements exactly the flow described in the provided image:
- Input: per-trace impedance Z(t) with shape (13601, 2800), dt=1ms
- Reflectivity:
    r_i = (Z_{i+1} - Z_i) / (Z_{i+1} + Z_i)
- Wavelet: Ricker, dominant frequency 30 Hz, phase 0°, time window [-0.4, 1] ms
- Convolution:
    s(t) = r(t) * w(t)

This script focuses on data processing + reproducibility artifacts:
- Shape validation
- Explicit boundary handling for r
- Explicit convolution alignment (zero-time alignment)
- Saves Z, r, s and a JSON record
- Produces small sample visualizations

No extra modeling assumptions are introduced beyond what is required to implement the above.
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
import matplotlib.pyplot as plt


def ricker_window(f0_hz: float, dt_s: float, tmin_s: float, tmax_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (t, w) where t spans [tmin_s, tmax_s] inclusive with step dt_s."""
    if dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if tmax_s < tmin_s:
        raise ValueError("tmax_s must be >= tmin_s")
    nt = int(round((tmax_s - tmin_s) / dt_s)) + 1
    if nt < 2:
        nt = 2
    t = tmin_s + np.arange(nt, dtype=np.float64) * dt_s
    a = (np.pi * f0_hz * t) ** 2
    w = (1.0 - 2.0 * a) * np.exp(-a)
    return t.astype(np.float64), w.astype(np.float32)


def impedance_to_reflectivity(Z: np.ndarray) -> np.ndarray:
    """Compute reflectivity r with same shape as Z.

    Boundary handling (explicit): the last reflectivity sample is copied from r[-2]
    so that r has length T and uses only the paper formula where defined.
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2:
        raise ValueError("Z must be 2D (n_traces, n_samples)")
    num = Z[:, 1:] - Z[:, :-1]
    den = Z[:, 1:] + Z[:, :-1] + 1e-12
    r = (num / den).astype(np.float32)  # shape (N, T-1)
    r_last = r[:, -1:]
    return np.concatenate([r, r_last], axis=1)  # shape (N, T)


def convolve_zero_time_aligned(r: np.ndarray, t_w: np.ndarray, w: np.ndarray, dt_s: float) -> np.ndarray:
    """Convolve r with wavelet w using explicit zero-time alignment.

    Alignment rule:
    - Wavelet samples correspond to times t_w[k] (in seconds).
    - Let k0 be the index where t_w[k0] == 0 (or closest).
    - Define seismic s[i] = sum_j r[j] * w[k0 + i - j].
      This equals taking the FULL convolution and slicing:
        full = conv(r, w)
        s = full[k0 : k0 + T]

    This makes the output s have the same time indexing as input r.
    """
    r = np.asarray(r, dtype=np.float32)
    if r.ndim != 2:
        raise ValueError("r must be 2D")
    if w.ndim != 1:
        raise ValueError("w must be 1D")

    # choose k0 as the closest to 0
    k0 = int(np.argmin(np.abs(t_w)))
    if abs(float(t_w[k0])) > 0.5 * dt_s:
        # still proceed, but record via caller
        pass

    T = r.shape[1]
    out = np.empty_like(r, dtype=np.float32)
    for i in range(r.shape[0]):
        full = fftconvolve(r[i], w.astype(np.float32), mode="full").astype(np.float32)
        out[i] = full[k0 : k0 + T]
    return out


def save_trace_plot(out_path: Path, t_idx: np.ndarray, z: np.ndarray, r: np.ndarray, s: np.ndarray, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 4))
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(z, t_idx)
    ax1.set_title("Z")
    ax1.set_xlabel("Impedance")
    ax1.set_ylabel("Sample")
    ax1.invert_yaxis()

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(r, t_idx)
    ax2.set_title("r")
    ax2.set_xlabel("Reflectivity")
    ax2.invert_yaxis()

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(s, t_idx)
    ax3.set_title("s")
    ax3.set_xlabel("Seismic")
    ax3.invert_yaxis()

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_section_plot(out_path: Path, arr: np.ndarray, title: str, cmap: str = "viridis") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.imshow(arr.T, aspect="auto", origin="upper", cmap=cmap)
    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--impedance_npy", type=str, required=True, help="Impedance Z as .npy (shape [N,T]).")
    ap.add_argument("--out_npz", type=str, default="data/paper_forward_output.npz")
    ap.add_argument("--out_dir", type=str, default="runs/paper_forward")

    # Paper parameters (as per image)
    ap.add_argument("--n_traces", type=int, default=13601)
    ap.add_argument("--n_samples", type=int, default=2800)
    ap.add_argument("--dt_s", type=float, default=0.001)
    ap.add_argument("--f0_hz", type=float, default=30.0)
    ap.add_argument("--phase_deg", type=float, default=0.0)
    ap.add_argument("--wavelet_tmin_ms", type=float, default=-0.4)
    ap.add_argument("--wavelet_tmax_ms", type=float, default=1.0)

    ap.add_argument("--strict", action="store_true", help="Enforce shape=(13601,2800), dt=0.001, phase=0.")

    # Visualization
    ap.add_argument("--viz_traces", type=int, nargs="*", default=[0, 650, 1250], help="Trace indices to plot (0-based).")
    ap.add_argument("--viz_section_traces", type=int, default=256, help="How many traces to show in section plots.")

    args = ap.parse_args()

    Z = np.load(args.impedance_npy).astype(np.float32)
    if Z.ndim != 2:
        raise ValueError(f"impedance_npy must be 2D. Got shape={Z.shape}")

    expected = (int(args.n_traces), int(args.n_samples))
    if args.strict:
        if tuple(Z.shape) != expected:
            raise ValueError(f"Strict mode: expected Z shape {expected}, got {tuple(Z.shape)}")
        if abs(float(args.dt_s) - 0.001) > 1e-12:
            raise ValueError(f"Strict mode: dt_s must be 0.001, got {args.dt_s}")
        if abs(float(args.phase_deg)) > 1e-12:
            raise ValueError(f"Strict mode: phase_deg must be 0, got {args.phase_deg}")

    # Step 1: reflectivity
    r = impedance_to_reflectivity(Z)

    # Step 2: wavelet (time window) and convolution
    t_w, w = ricker_window(
        f0_hz=float(args.f0_hz),
        dt_s=float(args.dt_s),
        tmin_s=float(args.wavelet_tmin_ms) * 1e-3,
        tmax_s=float(args.wavelet_tmax_ms) * 1e-3,
    )

    # phase rotation only if requested (non-strict)
    phase_note = "0 (no rotation)"
    if abs(float(args.phase_deg)) > 1e-12:
        from scipy.signal import hilbert

        phase = np.deg2rad(float(args.phase_deg))
        analytic = hilbert(w.astype(np.float64))
        w = np.real(analytic * np.exp(1j * phase)).astype(np.float32)
        phase_note = f"rotated by {args.phase_deg} deg via analytic signal"

    s = convolve_zero_time_aligned(r, t_w=t_w, w=w, dt_s=float(args.dt_s))

    # Save outputs
    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "input": {
            "impedance_npy": str(args.impedance_npy),
            "shape": list(Z.shape),
        },
        "paper_params": {
            "dt_s": float(args.dt_s),
            "f0_hz": float(args.f0_hz),
            "phase_deg": float(args.phase_deg),
            "wavelet_tmin_ms": float(args.wavelet_tmin_ms),
            "wavelet_tmax_ms": float(args.wavelet_tmax_ms),
        },
        "reflectivity": {
            "formula": "r_i=(Z_{i+1}-Z_i)/(Z_{i+1}+Z_i)",
            "boundary": "r[T-1] is copied from r[T-2] so r has length T",
        },
        "wavelet": {
            "type": "ricker",
            "t_ms": [float(t_w[0] * 1e3), float(t_w[-1] * 1e3)],
            "nt": int(w.shape[0]),
            "phase": phase_note,
            "zero_time_alignment": "use k0 = argmin(|t_w|); s = full_conv[r,w][k0 : k0+T]",
        },
        "outputs": {
            "r_shape": list(r.shape),
            "s_shape": list(s.shape),
            "out_npz": str(out_npz),
        },
        "strict": bool(args.strict),
    }

    np.savez_compressed(out_npz, impedance=Z, reflectivity=r, seismic=s, meta=json.dumps(record, ensure_ascii=False))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "paper_forward_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # Visualizations
    t_idx = np.arange(Z.shape[1])
    for i in [int(x) for x in args.viz_traces]:
        if 0 <= i < Z.shape[0]:
            save_trace_plot(out_dir / f"trace_{i}.png", t_idx, Z[i], r[i], s[i], title=f"Trace {i}")

    n_show = min(int(args.viz_section_traces), Z.shape[0])
    save_section_plot(out_dir / "Z_section.png", Z[:n_show], title=f"Z section (first {n_show} traces)")
    save_section_plot(out_dir / "r_section.png", r[:n_show], title=f"r section (first {n_show} traces)")
    save_section_plot(out_dir / "s_section.png", s[:n_show], title=f"s section (first {n_show} traces)")

    print(f"[OK] wrote {out_npz}")
    print(f"[OK] wrote {out_dir / 'paper_forward_record.json'}")
    print(f"[OK] shapes: Z={Z.shape} r={r.shape} s={s.shape}")


if __name__ == "__main__":
    main()
