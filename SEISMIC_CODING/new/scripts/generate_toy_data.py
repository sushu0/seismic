from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def ricker(f0: float, dt: float, length: float):
    t = np.arange(-length/2, length/2 + dt, dt, dtype=np.float64)
    pi2 = (np.pi**2)
    w = (1.0 - 2*pi2*(f0**2)*(t**2)) * np.exp(-pi2*(f0**2)*(t**2))
    return w.astype(np.float32)

def reflectivity(imp: np.ndarray, eps: float = 1e-6):
    imp_prev = np.roll(imp, 1, axis=-1)
    r = (imp - imp_prev) / (imp + imp_prev + eps)
    r[..., 0] = 0.0
    return r

def conv_same(x: np.ndarray, k: np.ndarray):
    pad = (len(k) - 1) // 2
    xpad = np.pad(x, ((0,0),(pad,pad)), mode="edge")
    y = np.array([np.convolve(xpad[i], k, mode="valid") for i in range(x.shape[0])], dtype=np.float32)
    if y.shape[1] != x.shape[1]:
        y = y[:, :x.shape[1]]
    return y

def make_impedance(n: int, T: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    imp = np.zeros((n, T), dtype=np.float32)
    for i in range(n):
        # piecewise layers + smooth variations
        n_layers = rng.integers(3, 7)
        boundaries = np.sort(rng.choice(np.arange(50, T-50), size=n_layers-1, replace=False))
        levels = rng.uniform(4000, 12000, size=n_layers).astype(np.float32)
        y = np.zeros(T, dtype=np.float32)
        start = 0
        for j, b in enumerate(list(boundaries) + [T]):
            y[start:b] = levels[j]
            start = b
        # add gentle trend + small random smoothness
        trend = np.linspace(0, rng.uniform(-800, 800), T).astype(np.float32)
        y = y + trend
        # smooth
        y = np.convolve(y, np.ones(9)/9, mode="same").astype(np.float32)
        imp[i] = y
    return imp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--dt", type=float, default=0.002)
    ap.add_argument("--f0", type=float, default=30.0)
    ap.add_argument("--wlen", type=float, default=0.128)
    ap.add_argument("--n_train", type=int, default=64)
    ap.add_argument("--n_val", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=16)
    ap.add_argument("--n_unlabeled", type=int, default=128)
    ap.add_argument("--noise_std", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    w = ricker(args.f0, args.dt, args.wlen)
    imp_train = make_impedance(args.n_train, args.T, seed=args.seed)
    imp_val   = make_impedance(args.n_val, args.T, seed=args.seed+1)
    imp_test  = make_impedance(args.n_test, args.T, seed=args.seed+2)

    def make_seis(imp, seed):
        rng = np.random.default_rng(seed)
        r = reflectivity(imp)
        s = conv_same(r, w)
        s = s + args.noise_std * rng.standard_normal(s.shape).astype(np.float32)
        return s

    seis_train = make_seis(imp_train, args.seed+10)
    seis_val   = make_seis(imp_val,   args.seed+11)
    seis_test  = make_seis(imp_test,  args.seed+12)

    # unlabeled: seismic only, from different impedance distribution
    imp_u = make_impedance(args.n_unlabeled, args.T, seed=args.seed+3)
    seis_u = make_seis(imp_u, args.seed+13)

    np.save(out / "train_labeled_seis.npy", seis_train)
    np.save(out / "train_labeled_imp.npy",  imp_train)
    np.save(out / "val_seis.npy", seis_val)
    np.save(out / "val_imp.npy",  imp_val)
    np.save(out / "test_seis.npy", seis_test)
    np.save(out / "test_imp.npy",  imp_test)
    np.save(out / "train_unlabeled_seis.npy", seis_u)

    print(f"[OK] Wrote toy dataset to {out}")

if __name__ == "__main__":
    main()
