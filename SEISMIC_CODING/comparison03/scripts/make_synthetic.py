from __future__ import annotations
import os, argparse
import numpy as np

def ricker(freq_hz: float, dt: float, duration_s: float) -> np.ndarray:
    n = int(duration_s / dt)
    if n % 2 == 0:
        n += 1
    t = np.linspace(-(n//2)*dt, (n//2)*dt, n, dtype=np.float32)
    pi2 = (np.pi * freq_hz)**2
    w = (1.0 - 2.0*pi2*t**2) * np.exp(-pi2*t**2)
    w = w / (np.sqrt(np.sum(w**2)) + 1e-12)
    return w.astype(np.float32)

def imp_to_reflectivity(imp: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = (imp[:,1:] - imp[:,:-1]) / (imp[:,1:] + imp[:,:-1] + eps)
    r = np.pad(r, ((0,0),(1,0)), mode="constant", constant_values=0.0)
    return r.astype(np.float32)

def conv_same(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    K = w.shape[0]
    pad = K//2
    xpad = np.pad(x, ((0,0),(pad,pad)), mode="constant")
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[1]):
        out[:,i] = np.sum(xpad[:, i:i+K] * w[None,:], axis=1)
    return out.astype(np.float32)

def add_gaussian_noise_by_snr_db(x: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add zero-mean Gaussian noise to achieve a target SNR (dB).

    SNR(dB) = 10 * log10(P_signal / P_noise)
    """
    rng = np.random.default_rng(seed)
    sig_power = float(np.mean(x.astype(np.float64) ** 2))
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    noise_power = sig_power / (snr_linear + 1e-12)
    noise_std = float(np.sqrt(max(noise_power, 0.0)))
    noise = rng.normal(0.0, noise_std, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)

def make_impedance_section(n_traces: int, T: int, seed: int, preset: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if preset == "toy":
        base = rng.uniform(3000, 6000, size=(n_traces,1)).astype(np.float32)
        imp = np.tile(base, (1, T))
        for tr in range(n_traces):
            n_layers = rng.integers(6, 12)
            boundaries = np.sort(rng.integers(10, T-10, size=n_layers))
            val = base[tr,0]
            for b in boundaries:
                val *= rng.uniform(0.85, 1.15)
                imp[tr, b:] = val
        kernel = np.ones(7, dtype=np.float32) / 7.0
        imp = np.apply_along_axis(lambda a: np.convolve(a, kernel, mode="same"), 1, imp).astype(np.float32)
        return imp
    if preset == "marmousi2":
        x = np.linspace(0, 1, n_traces, dtype=np.float32)[:,None]
        t = np.linspace(0, 1, T, dtype=np.float32)[None,:]
        imp = (3500 + 2500*t + 800*np.sin(2*np.pi*(t*3 + x*1.5))).astype(np.float32)
        beds = 300*np.sin(2*np.pi*(t*20 + x*5)) + 200*np.cos(2*np.pi*(t*35 - x*2))
        imp += beds.astype(np.float32)
        shift = (x[:,0] > 0.55).astype(np.int32) * int(T*0.03)
        for i in range(n_traces):
            if shift[i] > 0:
                imp[i] = np.roll(imp[i], shift[i])
        imp += rng.normal(0, 80, size=imp.shape).astype(np.float32)
        return np.clip(imp, 1500, 9000)
    if preset == "valve":
        x = np.linspace(0, 1, n_traces, dtype=np.float32)[:,None]
        t = np.linspace(0, 1, T, dtype=np.float32)[None,:]
        imp = (4000 + 2200*t + 400*np.sin(2*np.pi*(t*2 + x*2))).astype(np.float32)
        imp += rng.normal(0, 60, size=imp.shape).astype(np.float32)
        return np.clip(imp, 2000, 9000)
    raise ValueError("preset must be toy/marmousi2/valve")

def split_indices(n: int, n_labeled: int, n_val: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    labeled = idx[:n_labeled]
    val = idx[n_labeled:n_labeled+n_val]
    rest = idx[n_labeled+n_val:]
    return labeled, val, rest

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", choices=["toy","marmousi2","valve"], default="toy")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--n_traces", type=int, default=1000)
    ap.add_argument("--T", type=int, default=512)
    ap.add_argument("--freq", type=float, default=30.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--dur", type=float, default=0.128)
    ap.add_argument("--n_labeled", type=int, default=None)
    ap.add_argument("--n_val", type=int, default=None)
    ap.add_argument("--snr_db", type=float, default=None, help="add Gaussian noise with target SNR in dB (e.g. 25, 15, 5)")
    args = ap.parse_args()

    if args.preset == "marmousi2":
        n = 13601 if args.n_traces == 1000 else args.n_traces
        T = 512 if args.T == 512 else args.T
        n_l = 101 if args.n_labeled is None else args.n_labeled
        n_v = 1350 if args.n_val is None else args.n_val
    elif args.preset == "valve":
        n = 1300 if args.n_traces == 1000 else args.n_traces
        T = 160 if args.T == 512 else args.T
        n_l = 750 if args.n_labeled is None else args.n_labeled
        n_v = 275 if args.n_val is None else args.n_val
        args.dur = 0.064
    else:
        n = args.n_traces
        T = args.T
        n_l = 100 if args.n_labeled is None else args.n_labeled
        n_v = max(100, int(0.1*n)) if args.n_val is None else args.n_val

    w = ricker(args.freq, args.dt, args.dur)
    imp = make_impedance_section(n, T, args.seed, args.preset)
    r = imp_to_reflectivity(imp)
    seis = conv_same(r, w)

    if args.snr_db is not None:
        seis = add_gaussian_noise_by_snr_db(seis, args.snr_db, seed=args.seed + 999)

    li, vi, ui = split_indices(n, n_l, n_v, args.seed)

    x_l, y_l = seis[li], imp[li]
    x_v, y_v = seis[vi], imp[vi]
    x_u = seis[ui]
    x_test, y_test = seis, imp

    stats = {
        "x_mean": float(x_l.mean()), "x_std": float(x_l.std() + 1e-12),
        "y_mean": float(y_l.mean()), "y_std": float(y_l.std() + 1e-12),
        "preset": args.preset, "T": int(T), "seed": int(args.seed),
        "wavelet": {"freq": args.freq, "dt": args.dt, "dur": args.dur},
        "noise": {"snr_db": None if args.snr_db is None else float(args.snr_db)},
        "split": {"n_total": int(n), "n_labeled": int(n_l), "n_val": int(n_v), "n_unlabeled": int(len(ui))}
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out,
        x_labeled=x_l.astype(np.float32), y_labeled=y_l.astype(np.float32),
        x_unlabeled=x_u.astype(np.float32),
        x_val=x_v.astype(np.float32), y_val=y_v.astype(np.float32),
        x_test=x_test.astype(np.float32), y_test=y_test.astype(np.float32),
        stats=stats
    )
    print("Saved", args.out)
    print("Stats", stats)

if __name__ == "__main__":
    main()
