from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import re
import torch
from torch.utils.data import DataLoader

from seisinv.utils.seed import seed_everything
from seisinv.utils.logger import SimpleLogger
from seisinv.utils.wavelet import wavelet_tensor
from seisinv.data.dataset import build_datasets, NormConfig
from seisinv.losses.physics import ForwardModel

from seisinv.models.baselines import UNet1D, TCN1D, CNNBiLSTM
from seisinv.models.ms_physformer import MSPhysFormer
from seisinv.trainer.train import train_one_experiment

def collate_fn_with_none(batch):
    """Custom collate function that handles None values for unlabeled data."""
    seis = torch.stack([item["seis"] for item in batch])
    imp = [item["imp"] for item in batch]
    if imp[0] is None:
        return {"seis": seis, "imp": None}
    imp = torch.stack(imp)
    return {"seis": seis, "imp": imp}

def parse_override_kv(pairs):
    """Parse --override a.b.c=value into nested dict updates."""
    updates = {}
    for p in pairs or []:
        if "=" not in p:
            raise ValueError(f"Invalid override: {p}")
        k, v = p.split("=", 1)
        # cast
        if v.lower() in ("true", "false"):
            val = v.lower() == "true"
        else:
            try:
                if "." in v:
                    val = float(v)
                else:
                    val = int(v)
            except ValueError:
                val = v
        updates[k] = val
    return updates

def apply_overrides(cfg: dict, overrides: dict):
    for k, v in overrides.items():
        keys = k.split(".")
        d = cfg
        for kk in keys[:-1]:
            if kk not in d or not isinstance(d[kk], dict):
                d[kk] = {}
            d = d[kk]
        d[keys[-1]] = v


def cast_numeric_strings(obj):
    """Recursively cast numeric-like strings (incl. scientific notation) into int/float."""
    if isinstance(obj, dict):
        return {k: cast_numeric_strings(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [cast_numeric_strings(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        # match int/float/scientific
        if re.fullmatch(r"[+-]?\d+", s):
            try:
                return int(s)
            except Exception:
                return obj
        if re.fullmatch(r"[+-]?(\d+\.\d*|\d*\.\d+|\d+)([eE][+-]?\d+)?", s):
            try:
                return float(s)
            except Exception:
                return obj
    return obj

def build_model(cfg: dict):
    name = cfg["model"]["name"]
    if name == "unet1d":
        return UNet1D(in_ch=1, base=cfg["model"]["base"], depth=cfg["model"]["depth"])
    if name == "tcn1d":
        return TCN1D(in_ch=1, ch=cfg["model"]["ch"], layers=cfg["model"]["layers"], dropout=cfg["model"]["dropout"])
    if name == "cnn_bilstm":
        return CNNBiLSTM(in_ch=1, cnn_ch=cfg["model"]["cnn_ch"], lstm_hidden=cfg["model"]["lstm_hidden"], lstm_layers=cfg["model"]["lstm_layers"])
    if name == "ms_physformer":
        return MSPhysFormer(
            in_ch=1, base=cfg["model"]["base"], depth=cfg["model"]["depth"],
            nhead=cfg["model"]["nhead"], tf_dim_mult=cfg["model"]["tf_dim_mult"], tf_layers=cfg["model"]["tf_layers"]
        )
    raise ValueError(f"Unknown model name: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    ap.add_argument("--override", nargs="*", default=None, help="Override config, e.g. data.data_root=/path train.epochs=50")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    apply_overrides(cfg, parse_override_kv(args.override))
    cfg = cast_numeric_strings(cfg)

    # YAML sometimes parses scientific notation like '1e-6' as string; normalize to float
    for path in [("data","norm","eps"), ("physics","eps")]:
        d = cfg
        for k in path[:-1]:
            d = d.get(k, {})
        if path[-1] in d and isinstance(d[path[-1]], str):
            d[path[-1]] = float(d[path[-1]])

    seed_everything(cfg["seed"], deterministic=cfg["deterministic"])

    device = torch.device("cuda" if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    out_dir = Path(cfg["output"]["root"]) / cfg["output"]["exp_name"]
    logger = SimpleLogger(out_dir)
    logger.log(f"Using device: {device}")
    logger.log(f"Config: {cfg}")

    norm = NormConfig(**cfg["data"]["norm"])
    train_ds, val_ds, test_ds, unl_ds, stats = build_datasets(cfg["data"]["data_root"], norm=norm)
    logger.save_json(stats, "norm_stats.json")

    train_loader_l = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, drop_last=True)
    val_loader     = DataLoader(val_ds, batch_size=cfg["train"]["batch_size_eval"], shuffle=False, num_workers=0)
    test_loader    = DataLoader(test_ds, batch_size=cfg["train"]["batch_size_eval"], shuffle=False, num_workers=0)

    train_loader_u = None
    if unl_ds is not None and cfg["train"]["use_unlabeled"]:
        train_loader_u = DataLoader(unl_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=0, drop_last=True, collate_fn=collate_fn_with_none)

    # forward model
    w = wavelet_tensor(
        f0=cfg["physics"]["wavelet_f0_hz"],
        dt=cfg["physics"]["dt_s"],
        length=cfg["physics"]["wavelet_length_s"],
        device=device,
    )
    fm = ForwardModel(wavelet=w, eps=cfg["physics"]["eps"]).to(device)

    model = build_model(cfg).to(device)
    teacher = None
    if cfg["train"]["use_teacher"]:
        teacher = build_model(cfg).to(device)
        teacher.load_state_dict(model.state_dict())

    test_metrics = train_one_experiment(
        model=model,
        teacher=teacher,
        train_loader_l=train_loader_l,
        train_loader_u=train_loader_u,
        val_loader=val_loader,
        test_loader=test_loader,
        out_dir=out_dir,
        device=device,
        fm=fm,
        cfg=cfg,
        logger=logger,
    )
    logger.log(f"Done. test_metrics={test_metrics}")

if __name__ == "__main__":
    main()
