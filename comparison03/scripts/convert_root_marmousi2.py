from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ConvertConfig:
    in_path: str
    out_path: str
    seed: int = 1234
    strict_paper: bool = True

    # Paper: Marmousi2 total=13601, labeled=101 (uniform), val=1350 (random), rest unlabeled; test=all.
    n_total_paper: int = 13601
    n_labeled_paper: int = 101
    n_val_paper: int = 1350

    # Optional overrides (mainly for non-strict experiments)
    n_labeled_override: int | None = None
    n_val_override: int | None = None


def _avg_pool_downsample_last_dim(x: np.ndarray, factor: int) -> np.ndarray:
    """Downsample last dimension by integer factor using mean pooling."""
    if x.shape[-1] % factor != 0:
        raise ValueError(f"Cannot avg-pool: length {x.shape[-1]} not divisible by factor {factor}")
    new_len = x.shape[-1] // factor
    x2 = x.reshape(*x.shape[:-1], new_len, factor)
    return x2.mean(axis=-1)


def _split_indices_paper(n_total: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implements the paper split policy for Marmousi2.

    - labeled: uniformly select 101 traces over the whole section
    - val: randomly select 1350 traces from remaining
    - unlabeled: all remaining
    """
    # Uniform selection across the full range.
    labeled = np.linspace(0, n_total - 1, 101, dtype=np.int64)

    remaining = np.ones(n_total, dtype=bool)
    remaining[labeled] = False
    remaining_idx = np.nonzero(remaining)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(remaining_idx)

    val = remaining_idx[:1350]
    unlabeled = remaining_idx[1350:]

    return labeled, val, unlabeled


def _split_indices_like_paper(n_total: int, n_labeled: int, n_val: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Paper-like split for arbitrary n_total.

    - labeled: uniformly across section
    - val: random from remaining
    - unlabeled: rest
    """
    labeled = np.linspace(0, n_total - 1, n_labeled, dtype=np.int64)
    remaining = np.ones(n_total, dtype=bool)
    remaining[labeled] = False
    remaining_idx = np.nonzero(remaining)[0]

    rng = np.random.default_rng(seed)
    rng.shuffle(remaining_idx)

    val = remaining_idx[:n_val]
    unlabeled = remaining_idx[n_val:]
    return labeled, val, unlabeled


def convert(cfg: ConvertConfig) -> None:
    obj = np.load(cfg.in_path, allow_pickle=True).item()
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {cfg.in_path}, got {type(obj)}")

    if "seismic" not in obj or "acoustic_impedance" not in obj:
        raise KeyError("data.npy must contain keys: 'seismic' and 'acoustic_impedance'")

    seis = np.asarray(obj["seismic"], dtype=np.float32)
    imp = np.asarray(obj["acoustic_impedance"], dtype=np.float32)

    # Source file stores (N, 1, T). Repo NPZ format stores (N, T).
    if seis.ndim != 3 or imp.ndim != 3:
        raise ValueError(f"Expected 3D arrays (N,1,T). Got seismic {seis.shape}, impedance {imp.shape}")
    if seis.shape[0] != imp.shape[0]:
        raise ValueError(f"N mismatch: seismic {seis.shape[0]} vs impedance {imp.shape[0]}")
    if seis.shape[1] != 1 or imp.shape[1] != 1:
        raise ValueError(f"Expected channel dim=1. Got seismic {seis.shape}, impedance {imp.shape}")

    seis = seis[:, 0, :]
    imp = imp[:, 0, :]

    n_total = int(seis.shape[0])
    t_seis = int(seis.shape[-1])
    t_imp = int(imp.shape[-1])

    # Paper strictness checks.
    if cfg.strict_paper and n_total != cfg.n_total_paper:
        raise ValueError(
            "Strict paper reproduction requires Marmousi2 total traces = "
            f"{cfg.n_total_paper}, but data has {n_total}. "
            "Please provide the full Marmousi2 dataset used in the paper or disable --strict_paper."
        )

    # Align impedance length to seismic length.
    if t_imp == t_seis:
        imp_aligned = imp
    elif t_imp % t_seis == 0:
        factor = t_imp // t_seis
        imp_aligned = _avg_pool_downsample_last_dim(imp, factor)
    else:
        if cfg.strict_paper:
            raise ValueError(
                f"Strict paper reproduction cannot align impedance length {t_imp} to seismic length {t_seis} "
                "without introducing a non-paper interpolation rule."
            )
        # Fallback (non-strict): linear interpolation on last dimension.
        x_old = np.linspace(0.0, 1.0, t_imp, dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, t_seis, dtype=np.float32)
        imp_aligned = np.empty((n_total, t_seis), dtype=np.float32)
        for i in range(n_total):
            imp_aligned[i] = np.interp(x_new, x_old, imp[i]).astype(np.float32)

    # Split.
    if cfg.strict_paper:
        labeled_idx, val_idx, unlabeled_idx = _split_indices_paper(n_total, cfg.seed)
        n_labeled = cfg.n_labeled_paper
        n_val = cfg.n_val_paper
    else:
        # Non-strict: preserve the paper ratios and selection policy.
        n_labeled = max(1, int(round(n_total * (cfg.n_labeled_paper / cfg.n_total_paper))))
        n_val = max(1, int(round(n_total * (cfg.n_val_paper / cfg.n_total_paper))))
        if cfg.n_labeled_override is not None:
            n_labeled = int(cfg.n_labeled_override)
        if cfg.n_val_override is not None:
            n_val = int(cfg.n_val_override)
        if n_labeled + n_val >= n_total:
            raise ValueError(f"Invalid split: n_total={n_total}, n_labeled={n_labeled}, n_val={n_val}")
        labeled_idx, val_idx, unlabeled_idx = _split_indices_like_paper(n_total, n_labeled, n_val, cfg.seed)

    x_l, y_l = seis[labeled_idx], imp_aligned[labeled_idx]
    x_v, y_v = seis[val_idx], imp_aligned[val_idx]
    x_u = seis[unlabeled_idx]

    x_test, y_test = seis, imp_aligned

    stats = {
        "preset": "marmousi2_root",
        "seed": int(cfg.seed),
        "split": {
            "n_total": int(n_total),
            "n_labeled": int(n_labeled),
            "n_val": int(n_val),
            "n_unlabeled": int(len(unlabeled_idx)),
        },
        "length": {"seismic": int(t_seis), "impedance_raw": int(t_imp), "impedance_aligned": int(imp_aligned.shape[-1])},
        "note": "Converted from root data.npy. Strict mode enforces paper counts and avoids interpolation rules.",
    }

    os.makedirs(os.path.dirname(cfg.out_path) or ".", exist_ok=True)
    np.savez_compressed(
        cfg.out_path,
        x_labeled=x_l.astype(np.float32),
        y_labeled=y_l.astype(np.float32),
        x_unlabeled=x_u.astype(np.float32),
        x_val=x_v.astype(np.float32),
        y_val=y_v.astype(np.float32),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.float32),
        stats=stats,
    )

    print("Saved", cfg.out_path)
    print("Stats", stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data.npy")
    ap.add_argument("--out", dest="out_path", default="data/marmousi2_paper.npz")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--strict_paper", action="store_true", help="enforce paper dataset size/split rules; default: True")
    ap.add_argument("--no_strict_paper", action="store_true", help="allow adapting to different dataset sizes")
    ap.add_argument("--n_labeled", type=int, default=None, help="Override labeled trace count (non-strict only)")
    ap.add_argument("--n_val", type=int, default=None, help="Override val trace count (non-strict only)")
    args = ap.parse_args()

    strict = True
    if args.no_strict_paper:
        strict = False
    if args.strict_paper:
        strict = True

    convert(
        ConvertConfig(
            in_path=args.in_path,
            out_path=args.out_path,
            seed=args.seed,
            strict_paper=strict,
            n_labeled_override=args.n_labeled,
            n_val_override=args.n_val,
        )
    )


if __name__ == "__main__":
    main()
