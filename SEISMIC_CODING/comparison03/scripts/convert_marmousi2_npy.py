from __future__ import annotations

import os
import argparse
import numpy as np


def _squeeze_to_2d(a: np.ndarray, name: str) -> np.ndarray:
    """Convert input to shape [N, T].

    Supports [N, 1, T] and [N, T].
    """
    if a.ndim == 3 and a.shape[1] == 1:
        return a[:, 0, :]
    if a.ndim == 2:
        return a
    raise ValueError(f"{name} must have shape [N,T] or [N,1,T], got {a.shape}")


def _resample_linear_1d(x: np.ndarray, out_len: int) -> np.ndarray:
    """Linear resample along last axis for a batch array [N, T]."""
    if x.shape[1] == out_len:
        return x
    in_len = x.shape[1]
    xp = np.linspace(0.0, 1.0, in_len, dtype=np.float64)
    xq = np.linspace(0.0, 1.0, out_len, dtype=np.float64)
    out = np.empty((x.shape[0], out_len), dtype=np.float32)
    for i in range(x.shape[0]):
        out[i] = np.interp(xq, xp, x[i].astype(np.float64)).astype(np.float32)
    return out


def _downsample_by_factor_mean(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsample [N,T] to [N,T/factor] by block mean."""
    if factor <= 1:
        return x.astype(np.float32)
    n, t = x.shape
    if t % factor != 0:
        raise ValueError(f"T={t} not divisible by factor={factor}")
    out_t = t // factor
    return x.reshape(n, out_t, factor).mean(axis=2).astype(np.float32)


def _split_indices(n: int, n_labeled: int, n_val: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    labeled = idx[:n_labeled]
    val = idx[n_labeled : n_labeled + n_val]
    rest = idx[n_labeled + n_val :]
    return labeled, val, rest


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert Marmousi2 root data.npy into ss_gan npz format")
    ap.add_argument("--in", dest="in_path", default="data.npy", help="input .npy (dict with keys seismic/acoustic_impedance)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_labeled", type=int, default=101)
    ap.add_argument("--n_val", type=int, default=1350)
    ap.add_argument(
        "--target_T",
        type=int,
        default=None,
        help="optional: resample impedance to match seismic length (or to a fixed length)",
    )
    ap.add_argument(
        "--imp_resample",
        choices=["auto", "mean", "linear"],
        default="auto",
        help="how to align impedance length to target_T; auto uses mean downsample when divisible else linear",
    )
    args = ap.parse_args()

    obj = np.load(args.in_path, allow_pickle=True)
    if not (isinstance(obj, np.ndarray) and obj.shape == () and isinstance(obj.item(), dict)):
        raise ValueError("Expected a pickled dict saved in .npy (0-d object array)")
    d = obj.item()
    if "seismic" not in d or "acoustic_impedance" not in d:
        raise KeyError("data.npy dict must contain keys: 'seismic', 'acoustic_impedance'")

    seismic = _squeeze_to_2d(np.asarray(d["seismic"]), "seismic").astype(np.float32)
    imp = _squeeze_to_2d(np.asarray(d["acoustic_impedance"]), "acoustic_impedance").astype(np.float32)

    n = seismic.shape[0]
    if imp.shape[0] != n:
        raise ValueError(f"N mismatch: seismic N={n}, impedance N={imp.shape[0]}")

    # If lengths differ, default to aligning impedance onto seismic sample count.
    target_T = args.target_T
    if target_T is None:
        target_T = int(seismic.shape[1])

    imp_align = {"target_T": int(target_T)}
    if imp.shape[1] != target_T:
        if args.imp_resample in ("auto", "mean") and (imp.shape[1] % target_T == 0):
            factor = int(imp.shape[1] // target_T)
            imp = _downsample_by_factor_mean(imp, factor)
            imp_align.update({"method": "block_mean", "factor": factor})
        else:
            imp = _resample_linear_1d(imp, target_T)
            imp_align.update({"method": "linear_interp", "factor": None})

    if seismic.shape[1] != target_T:
        seismic = _resample_linear_1d(seismic, target_T)

    li, vi, ui = _split_indices(n, args.n_labeled, args.n_val, args.seed)

    x_l, y_l = seismic[li], imp[li]
    x_v, y_v = seismic[vi], imp[vi]
    x_u = seismic[ui]

    # For this project's evaluation split, test uses all samples.
    x_test, y_test = seismic, imp

    stats = {
        "x_mean": float(x_l.mean()),
        "x_std": float(x_l.std() + 1e-12),
        "y_mean": float(y_l.mean()),
        "y_std": float(y_l.std() + 1e-12),
        "preset": "marmousi2_root_npy",
        "T": int(target_T),
        "seed": int(args.seed),
        "split": {
            "n_total": int(n),
            "n_labeled": int(args.n_labeled),
            "n_val": int(args.n_val),
            "n_unlabeled": int(len(ui)),
        },
        "source": {
            "in": os.path.basename(args.in_path),
            "seismic_shape": list(np.asarray(d["seismic"]).shape),
            "impedance_shape": list(np.asarray(d["acoustic_impedance"]).shape),
            "resample": {
                "target_T": int(target_T),
                **imp_align,
                "note": "Impedance aligned to seismic length when needed.",
            },
        },
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        x_labeled=x_l.astype(np.float32),
        y_labeled=y_l.astype(np.float32),
        x_unlabeled=x_u.astype(np.float32),
        x_val=x_v.astype(np.float32),
        y_val=y_v.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        stats=stats,
    )
    print("Saved", args.out)
    print("Stats", stats)


if __name__ == "__main__":
    main()
