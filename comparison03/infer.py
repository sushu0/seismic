from __future__ import annotations
import os, argparse, json
import numpy as np
import torch


def _median_filter_1d_per_trace(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    if k % 2 == 0:
        raise ValueError("median_k must be odd")
    pad = k // 2
    try:
        from numpy.lib.stride_tricks import sliding_window_view
    except Exception as e:
        raise RuntimeError("Your NumPy is missing sliding_window_view; cannot run median filter") from e

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    win = sliding_window_view(x_pad, window_shape=k, axis=2)
    return np.median(win, axis=-1)


def _clip_percentiles(x: np.ndarray, p_lo: float, p_hi: float) -> tuple[np.ndarray, float, float]:
    if not (0.0 <= p_lo < p_hi <= 100.0):
        raise ValueError("clip_percentiles must satisfy 0 <= lo < hi <= 100")
    lo, hi = np.percentile(x.astype(np.float64), [p_lo, p_hi])
    return np.clip(x, lo, hi), float(lo), float(hi)

from ss_gan.models import UNet1D
from ss_gan.data import NPZDatasetConfig, make_loader
from ss_gan.utils import pcc, r2, mse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--median_k", type=int, default=0, help="Optional median filter kernel (odd). 0 disables.")
    ap.add_argument(
        "--clip_percentiles",
        nargs=2,
        type=float,
        default=None,
        metavar=("P_LO", "P_HI"),
        help="Optional percentile clipping, e.g. 0.5 99.5",
    )
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["cfg"]
    stats = ckpt.get("stats", {})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = cfg.get("device", device)
    dev = torch.device(device)

    G = UNet1D(1, 1, int(cfg.get("base_ch_g", 16)), int(cfg.get("k_large", 299)), int(cfg.get("k_small", 3))).to(dev)
    G.load_state_dict(ckpt["G"])
    G.eval()

    normalize = bool(cfg.get("normalize", True))
    loader = make_loader(NPZDatasetConfig(args.dataset, args.split, normalize),
                         args.batch_size, False, 0, stats)

    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(dev)
            y = batch["y"].to(dev)
            p = G(x)
            preds.append(p.cpu().numpy())
            trues.append(y.cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    metrics_raw = {"pcc": pcc(true, pred), "r2": r2(true, pred), "mse": mse(true, pred)}

    post = {
        "median_k": int(args.median_k),
        "clip_percentiles": None,
        "clip_values": None,
    }

    pred_out = pred
    if args.clip_percentiles is not None:
        p_lo, p_hi = float(args.clip_percentiles[0]), float(args.clip_percentiles[1])
        pred_out, lo, hi = _clip_percentiles(pred_out, p_lo, p_hi)
        post["clip_percentiles"] = [p_lo, p_hi]
        post["clip_values"] = [lo, hi]

    if int(args.median_k) > 1:
        pred_out = _median_filter_1d_per_trace(pred_out, int(args.median_k))

    metrics = {"pcc": pcc(true, pred_out), "r2": r2(true, pred_out), "mse": mse(true, pred_out)}

    metrics_phys = None
    metrics_phys_raw = None
    if normalize and ("y_mean" in stats) and ("y_std" in stats):
        y_mean, y_std = float(stats["y_mean"]), float(stats["y_std"])
        pred_phys = pred_out * y_std + y_mean
        true_phys = true * y_std + y_mean
        metrics_phys = {"pcc": pcc(true_phys, pred_phys), "r2": r2(true_phys, pred_phys), "mse": mse(true_phys, pred_phys)}
        pred_phys_raw = pred * y_std + y_mean
        metrics_phys_raw = {"pcc": pcc(true_phys, pred_phys_raw), "r2": r2(true_phys, pred_phys_raw), "mse": mse(true_phys, pred_phys_raw)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(
        args.out,
        pred=pred_out,
        pred_raw=pred,
        true=true,
        metrics=metrics,
        metrics_raw=metrics_raw,
        metrics_phys=metrics_phys,
        metrics_phys_raw=metrics_phys_raw,
        postprocess=post,
    )
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w", encoding="utf-8") as f:
        payload = {
            "metrics": metrics,
            "metrics_raw": metrics_raw,
            "metrics_phys": metrics_phys,
            "metrics_phys_raw": metrics_phys_raw,
            "postprocess": post,
        }
        json.dump(payload, f, indent=2)

    print("Metrics (raw):", metrics_raw)
    if post["clip_percentiles"] is not None or int(post["median_k"]) > 1:
        print("Metrics (postprocessed):", metrics)
    else:
        print("Metrics:", metrics)
    if metrics_phys is not None:
        print("Metrics (physical):", metrics_phys)

if __name__ == "__main__":
    main()
