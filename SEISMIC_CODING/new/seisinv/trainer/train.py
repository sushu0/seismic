from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from seisinv.utils.metrics import summarize_metrics
from seisinv.utils.plotting import save_trace_plot, save_section
from seisinv.losses.physics import PhysicsLoss, ForwardModel
from seisinv.losses.frequency import STFTMagLoss

@dataclass
class TrainState:
    best_val_mse: float = float("inf")
    best_epoch: int = -1

def ema_update(teacher: nn.Module, student: nn.Module, ema: float):
    with torch.no_grad():
        for t, s in zip(teacher.parameters(), student.parameters()):
            t.data.mul_(ema).add_(s.data, alpha=1.0-ema)

def strong_aug(x: torch.Tensor, noise_std: float = 0.02, amp_jitter: float = 0.1, time_shift: int = 8):
    """Simple strong augmentation for seismic traces: noise + amplitude jitter + small time shift."""
    if noise_std > 0:
        x = x + noise_std * torch.randn_like(x)
    if amp_jitter > 0:
        scale = (1.0 + amp_jitter * (2*torch.rand(x.shape[0], 1, 1, device=x.device)-1.0))
        x = x * scale
    if time_shift > 0:
        shift = int(torch.randint(low=-time_shift, high=time_shift+1, size=(1,)).item())
        x = torch.roll(x, shifts=shift, dims=-1)
    return x

def deep_supervision_loss(pred_full: torch.Tensor, pred_ms: list[torch.Tensor], y_true: torch.Tensor, sup_loss_fn: nn.Module):
    loss = sup_loss_fn(pred_full, y_true)
    # pred_ms are pooled outputs; match targets by pooling y_true similarly
    weights = [0.25, 0.35, 0.4]  # heuristic weights
    pools = [16, 8, 4]
    for w, p, ph in zip(weights, pools, pred_ms):
        yt = torch.nn.functional.avg_pool1d(y_true, kernel_size=p, stride=p)
        # ph and yt should match length; crop if needed
        m = min(ph.shape[-1], yt.shape[-1])
        loss = loss + w * sup_loss_fn(ph[..., :m], yt[..., :m])
    return loss

def train_one_experiment(
    model: nn.Module,
    teacher: nn.Module | None,
    train_loader_l: DataLoader,
    train_loader_u: DataLoader | None,
    val_loader: DataLoader,
    test_loader: DataLoader,
    out_dir: Path,
    device: torch.device,
    fm: ForwardModel,
    cfg: dict,
    logger,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # losses
    sup_loss_name = cfg["train"]["sup_loss"]
    if sup_loss_name == "smoothl1":
        sup_loss_fn = nn.SmoothL1Loss(beta=1.0)
    elif sup_loss_name == "mse":
        sup_loss_fn = nn.MSELoss()
    else:
        raise ValueError(f"Unknown sup_loss: {sup_loss_name}")

    phys_loss_fn = PhysicsLoss(fm, mode=cfg["train"]["phys_loss_mode"])
    freq_loss_fn = STFTMagLoss(
        n_fft=cfg["train"]["stft_n_fft"],
        hop_length=cfg["train"]["stft_hop"],
        win_length=cfg["train"]["stft_win"],
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["train"]["epochs"])

    lam_phys = cfg["train"]["lambda_phys"]
    lam_freq = cfg["train"]["lambda_freq"]
    lam_cons = cfg["train"]["lambda_cons"]
    use_teacher = (teacher is not None) and (train_loader_u is not None) and (lam_cons > 0)

    state = TrainState()
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        if teacher is not None:
            teacher.eval()

        losses = {"sup": 0.0, "phys": 0.0, "freq": 0.0, "cons": 0.0, "total": 0.0}
        n_batches = 0

        u_iter = iter(train_loader_u) if train_loader_u is not None else None

        for batch in tqdm(train_loader_l, desc=f"epoch {epoch}/{cfg['train']['epochs']}"):
            x = batch["seis"].to(device)  # [B,1,T]
            y = batch["imp"].to(device)   # [B,1,T]

            # supervised forward
            if cfg["model"]["name"] == "ms_physformer":
                yhat, yms = model(x)
                l_sup = deep_supervision_loss(yhat, yms, y, sup_loss_fn)
            else:
                yhat = model(x)
                l_sup = sup_loss_fn(yhat, y)

            # physics & frequency (on labeled)
            s_hat = fm(yhat)
            l_phys = phys_loss_fn(yhat, x) if lam_phys > 0 else torch.tensor(0.0, device=device)
            l_freq = freq_loss_fn(s_hat, x) if lam_freq > 0 else torch.tensor(0.0, device=device)

            l_cons = torch.tensor(0.0, device=device)
            if train_loader_u is not None:
                try:
                    batch_u = next(u_iter)
                except StopIteration:
                    u_iter = iter(train_loader_u)
                    batch_u = next(u_iter)

                xu = batch_u["seis"].to(device)
                xu_s = strong_aug(xu, noise_std=cfg["train"]["aug_noise"], amp_jitter=cfg["train"]["aug_amp"], time_shift=cfg["train"]["aug_shift"])
                if cfg["model"]["name"] == "ms_physformer":
                    yu_s, _ = model(xu_s)
                else:
                    yu_s = model(xu_s)

                # unlabeled physics & frequency (self-supervised)
                l_phys_u = phys_loss_fn(yu_s, xu) if lam_phys > 0 else torch.tensor(0.0, device=device)
                s_u_hat = fm(yu_s)
                l_freq_u = freq_loss_fn(s_u_hat, xu) if lam_freq > 0 else torch.tensor(0.0, device=device)

                l_phys = l_phys + l_phys_u
                l_freq = l_freq + l_freq_u

                if use_teacher:
                    with torch.no_grad():
                        xu_t = strong_aug(xu, noise_std=cfg["train"]["aug_noise"], amp_jitter=cfg["train"]["aug_amp"], time_shift=cfg["train"]["aug_shift"])
                        if cfg["model"]["name"] == "ms_physformer":
                            yu_t, _ = teacher(xu_t)
                        else:
                            yu_t = teacher(xu_t)
                    l_cons = nn.functional.mse_loss(yu_s, yu_t)

            loss = l_sup + lam_phys * l_phys + lam_freq * l_freq + lam_cons * l_cons

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg["train"]["grad_clip"])
            opt.step()

            if use_teacher:
                ema_update(teacher, model, ema=cfg["train"]["ema"])

            losses["sup"] += float(l_sup.detach().cpu())
            losses["phys"] += float(l_phys.detach().cpu())
            losses["freq"] += float(l_freq.detach().cpu())
            losses["cons"] += float(l_cons.detach().cpu())
            losses["total"] += float(loss.detach().cpu())
            n_batches += 1

        scheduler.step()
        for k in losses:
            losses[k] /= max(1, n_batches)

        # validation
        val_metrics, val_mse = evaluate(model, val_loader, device)
        logger.log(f"Epoch {epoch}: train_total={losses['total']:.6f} val_mse={val_mse:.6f} val_pcc={val_metrics['PCC']:.4f} val_r2={val_metrics['R2']:.4f}")

        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in losses.items()},
            "val_MSE": val_metrics["MSE"],
            "val_PCC": val_metrics["PCC"],
            "val_R2": val_metrics["R2"],
            "lr": opt.param_groups[0]["lr"],
        })

        # checkpoint
        torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, ckpt_dir / "last.pt")
        if val_mse < state.best_val_mse:
            state.best_val_mse = val_mse
            state.best_epoch = epoch
            torch.save({"epoch": epoch, "model": model.state_dict(), "cfg": cfg}, ckpt_dir / "best.pt")

    # test using best
    best = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    test_metrics, _ = evaluate(model, test_loader, device)

    logger.save_json({"best_epoch": state.best_epoch, "best_val_mse": state.best_val_mse, "test_metrics": test_metrics}, "test_metrics.json")

    # produce plots on test set
    make_plots = bool(cfg.get('output', {}).get('make_plots', True))
    if make_plots:
        y_true, y_pred, s_obs, s_pred = predict_arrays(model, test_loader, device, fm)
        save_trace_plot(y_true, y_pred, out_dir / "pred_vs_true_traces.png", max_traces=6)
        # section view if enough traces
        if y_true.shape[0] >= 16:
            save_section(y_true[:256], out_dir / "true_imp_section.png", "True impedance (first 256 traces)")
            save_section(y_pred[:256], out_dir / "pred_imp_section.png", "Pred impedance (first 256 traces)")
        save_section(s_obs[:256], out_dir / "seis_obs_section.png", "Observed seismic (first 256 traces)")
        save_section(s_pred[:256], out_dir / "seis_recon_section.png", "Reconstructed seismic from pred impedance")

    # save history csv
    import csv
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    return test_metrics

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    ys = []
    yhats = []
    for batch in loader:
        x = batch["seis"].to(device)
        y = batch["imp"].to(device)
        out = model(x)
        if isinstance(out, tuple):
            yhat, _ = out
        else:
            yhat = out
        ys.append(y.detach().cpu().numpy())
        yhats.append(yhat.detach().cpu().numpy())
    y_true = np.concatenate(ys, axis=0)[:, 0, :]
    y_pred = np.concatenate(yhats, axis=0)[:, 0, :]
    metrics = summarize_metrics(y_true, y_pred)
    return metrics, metrics["MSE"]

@torch.no_grad()
def predict_arrays(model: nn.Module, loader: DataLoader, device: torch.device, fm: ForwardModel):
    model.eval()
    ys = []
    yhats = []
    seis = []
    seis_hat = []
    for batch in loader:
        x = batch["seis"].to(device)
        y = batch["imp"].to(device)
        out = model(x)
        if isinstance(out, tuple):
            yhat, _ = out
        else:
            yhat = out
        s_hat = fm(yhat)

        ys.append(y.detach().cpu().numpy())
        yhats.append(yhat.detach().cpu().numpy())
        seis.append(x.detach().cpu().numpy())
        seis_hat.append(s_hat.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)[:, 0, :]
    y_pred = np.concatenate(yhats, axis=0)[:, 0, :]
    s_obs = np.concatenate(seis, axis=0)[:, 0, :]
    s_pred = np.concatenate(seis_hat, axis=0)[:, 0, :]
    return y_true, y_pred, s_obs, s_pred
