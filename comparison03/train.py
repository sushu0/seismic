from __future__ import annotations
import os, argparse, json
import yaml
import torch
import numpy as np

from ss_gan.utils import seed_everything
from ss_gan.data import NPZDatasetConfig, make_loader
from ss_gan.trainer import TrainConfig, train

def compute_stats(npz_path: str) -> dict:
    z = np.load(npz_path, allow_pickle=True)
    x = z["x_labeled"].astype("float32")
    y = z["y_labeled"].astype("float32")
    return {
        "x_mean": float(x.mean()),
        "x_std": float(x.std() + 1e-12),
        "y_mean": float(y.mean()),
        "y_std": float(y.std() + 1e-12),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    seed = int(y.get("seed", 1234))
    deterministic = bool(y.get("deterministic", True))
    benchmark = bool(y.get("benchmark", False))
    tf32 = bool(y.get("tf32", False))
    seed_everything(seed, deterministic=deterministic, benchmark=benchmark, tf32=tf32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = y.get("device", device)
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] device=cuda requested but CUDA not available; falling back to cpu")
        device = "cpu"

    if device == "cuda":
        name0 = torch.cuda.get_device_name(0)
        print(f"[device] cuda:0 ({name0}), torch={torch.__version__}, cudnn={torch.backends.cudnn.version()}, deterministic={deterministic}, benchmark={benchmark}, tf32={tf32}")
    else:
        print(f"[device] cpu, torch={torch.__version__}, deterministic={deterministic}")

    run_dir = y.get("run_dir", "runs/exp")
    os.makedirs(run_dir, exist_ok=True)

    dataset_path = y["dataset"]["path"]
    stats = compute_stats(dataset_path)
    with open(os.path.join(run_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    bs = int(y.get("batch_size", 10))
    nw = int(y.get("num_workers", 0))

    normalize = bool(y.get("dataset", {}).get("normalize", True))
    labeled = make_loader(NPZDatasetConfig(dataset_path, "labeled", normalize), bs, True, nw, stats)
    unlabeled = make_loader(NPZDatasetConfig(dataset_path, "unlabeled", normalize), bs, True, nw, stats)
    val = make_loader(NPZDatasetConfig(dataset_path, "val", normalize), bs, False, nw, stats)

    cfg = TrainConfig(
        run_dir=run_dir,
        seed=seed,
        device=device,
        epochs=int(y.get("epochs", 50)),
        batch_size=bs,
        lr=float(y.get("lr", 1e-3)),
        n_critic=int(y.get("n_critic", 5)),
        alpha=float(y.get("alpha", 1100.0)),
        beta=float(y.get("beta", 550.0)),
        lambda_gp=float(y.get("lambda_gp", 10.0)),
        k_large=int(y.get("k_large", 299)),
        k_small=int(y.get("k_small", 3)),
        base_ch_g=int(y.get("base_ch_g", 16)),
        base_ch_d=int(y.get("base_ch_d", 8)),
        wavelet_freq=float(y.get("wavelet_freq", 30.0)),
        wavelet_dt=float(y.get("wavelet_dt", 0.001)),
        wavelet_dur=float(y.get("wavelet_dur", 0.128)),
        save_every=int(y.get("save_every", 1)),
        amp=bool(y.get("amp", False)),
        normalize=normalize,
        imp_loss=str(y.get("imp_loss", "mse")),
        huber_delta=float(y.get("huber_delta", 1.0)),
        grad_loss_weight=float(y.get("grad_loss_weight", 0.0)),
        loss_in_physical=bool(y.get("loss_in_physical", False)),
        warmup_epochs=int(y.get("warmup_epochs", 5)),
        use_ema=bool(y.get("use_ema", True)),
        ema_decay=float(y.get("ema_decay", 0.999)),
    )

    train(cfg, labeled, unlabeled, val, stats=stats)

if __name__ == "__main__":
    main()
