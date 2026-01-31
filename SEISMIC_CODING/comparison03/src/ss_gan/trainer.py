from __future__ import annotations
import os, json, time
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

from torch.amp import autocast, GradScaler

from .models import UNet1D, Critic1D
from .forward import RickerWavelet, forward_seismic_from_impedance
from .losses import gradient_penalty
from .utils import pcc, r2, mse

class EMA:
    """Exponential Moving Average for model parameters"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

@dataclass
class TrainConfig:
    run_dir: str
    seed: int = 1234
    device: str = "cuda"
    epochs: int = 50
    batch_size: int = 10
    lr: float = 1e-3
    n_critic: int = 5
    alpha: float = 1100.0
    beta: float = 550.0
    lambda_gp: float = 10.0
    k_large: int = 299
    k_small: int = 3
    base_ch_g: int = 16
    base_ch_d: int = 8
    wavelet_freq: float = 30.0
    wavelet_dt: float = 0.001
    wavelet_dur: float = 0.128
    save_every: int = 1
    amp: bool = False
    normalize: bool = True

    # Supervised impedance loss.
    imp_loss: str = "mse"  # mse|l1|huber
    huber_delta: float = 1.0
    grad_loss_weight: float = 0.0  # optional: match first-derivative to reduce over-smoothing

    # If dataset.normalize is enabled, optionally compute Li/Ls in physical units
    # (denormalized by stats) to keep alpha/beta comparable to the paper setting.
    loss_in_physical: bool = False
    
    # Advanced training options
    warmup_epochs: int = 5
    use_ema: bool = True
    ema_decay: float = 0.999


def _loss_impedance(pred: torch.Tensor, target: torch.Tensor, loss_type: str, huber_delta: float) -> torch.Tensor:
    lt = (loss_type or "mse").lower()
    if lt == "mse":
        return F.mse_loss(pred, target)
    if lt == "l1":
        return F.l1_loss(pred, target)
    if lt == "huber":
        return F.smooth_l1_loss(pred, target, beta=float(huber_delta))
    raise ValueError(f"Unknown imp_loss: {loss_type}")


def _loss_first_derivative_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred/target: [B,1,T]
    dp = pred[..., 1:] - pred[..., :-1]
    dt = target[..., 1:] - target[..., :-1]
    return F.l1_loss(dp, dt)

def _eval(G: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    G.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            p = G(x)
            ys.append(y.cpu().numpy()); ps.append(p.cpu().numpy())
    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    return {"pcc": pcc(y, p), "r2": r2(y, p), "mse": mse(y, p)}

def train(cfg: TrainConfig, labeled_loader, unlabeled_loader, val_loader, stats: Dict[str, Any] | None = None) -> None:
    os.makedirs(cfg.run_dir, exist_ok=True)
    ckpt_dir = os.path.join(cfg.run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device(cfg.device)
    G = UNet1D(1, 1, cfg.base_ch_g, cfg.k_large, cfg.k_small).to(device)
    D = Critic1D(2, cfg.base_ch_d, cfg.k_large, cfg.k_small).to(device)

    if device.type == "cuda":
        print(f"[train] using cuda:0 ({torch.cuda.get_device_name(0)}), amp={cfg.amp}, k_large={cfg.k_large}, batch_size={cfg.batch_size}")
    else:
        print(f"[train] using {cfg.device}, amp={cfg.amp}, k_large={cfg.k_large}, batch_size={cfg.batch_size}")

    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=(0.5, 0.9))

    # Learning rate schedulers with warmup
    def lr_lambda(epoch):
        if epoch < cfg.warmup_epochs:
            return (epoch + 1) / cfg.warmup_epochs
        else:
            progress = (epoch - cfg.warmup_epochs) / (cfg.epochs - cfg.warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    scheduler_d = torch.optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)
    
    # EMA for generator
    ema_g = EMA(G, decay=cfg.ema_decay) if cfg.use_ema else None

    use_amp = bool(cfg.amp) and (device.type == "cuda")
    scaler_g = GradScaler("cuda", enabled=use_amp)
    scaler_d = GradScaler("cuda", enabled=use_amp)

    wavelet = RickerWavelet(cfg.wavelet_freq, cfg.wavelet_dt, cfg.wavelet_dur, cfg.device).tensor()

    stats = stats or {}
    x_mean = float(stats.get("x_mean", 0.0))
    x_std = float(stats.get("x_std", 1.0))
    y_mean = float(stats.get("y_mean", 0.0))
    y_std = float(stats.get("y_std", 1.0))

    unl_it = iter(unlabeled_loader)
    history = {"train": [], "val": []}
    best = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        G.train(); D.train()
        pbar = tqdm(labeled_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

        n_steps = 0
        sum_ld = 0.0
        sum_lg = 0.0
        sum_li = 0.0
        sum_ls = 0.0

        for batch_l in pbar:
            x_l = batch_l["x"].to(device)
            y_l = batch_l["y"].to(device)

            # (Stage 1) Train D with labeled data while keeping G fixed.
            loss_d = torch.tensor(0.0, device=device)
            with torch.no_grad():
                y_fake_l = G(x_l)
            for _ in range(cfg.n_critic):
                with autocast(device_type="cuda", enabled=use_amp):
                    d_real = D(torch.cat([x_l, y_l], dim=1)).mean()
                    d_fake = D(torch.cat([x_l, y_fake_l], dim=1)).mean()
                    gp = gradient_penalty(D, x_l, y_l, y_fake_l, cfg.lambda_gp)
                    loss_d = -d_real + d_fake + gp

                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(loss_d).backward()
                scaler_d.unscale_(opt_d)
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                scaler_d.step(opt_d)
                scaler_d.update()

            # (Stage 2) Train G with labeled + unlabeled; keep D fixed.
            try:
                batch_u = next(unl_it)
            except StopIteration:
                unl_it = iter(unlabeled_loader)
                batch_u = next(unl_it)
            x_u = batch_u["x"].to(device)

            y_pred_l = G(x_l)
            with autocast(device_type="cuda", enabled=use_amp):
                # Supervised impedance loss
                if cfg.normalize and bool(cfg.loss_in_physical) and ("y_std" in stats) and ("y_mean" in stats):
                    y_pred_l_phys = y_pred_l * y_std + y_mean
                    y_l_phys = y_l * y_std + y_mean
                    loss_i = _loss_impedance(y_pred_l_phys, y_l_phys, cfg.imp_loss, cfg.huber_delta)
                    loss_grad = torch.tensor(0.0, device=device)
                    if float(cfg.grad_loss_weight) > 0.0:
                        loss_grad = _loss_first_derivative_l1(y_pred_l_phys, y_l_phys)
                else:
                    loss_i = _loss_impedance(y_pred_l, y_l, cfg.imp_loss, cfg.huber_delta)
                    loss_grad = torch.tensor(0.0, device=device)
                    if float(cfg.grad_loss_weight) > 0.0:
                        loss_grad = _loss_first_derivative_l1(y_pred_l, y_l)
                loss_adv = -D(torch.cat([x_l, y_pred_l], dim=1)).mean()

                y_pred_u = G(x_u)
                if cfg.normalize and ("x_std" in stats) and ("y_std" in stats):
                    y_u_phys = y_pred_u * y_std + y_mean
                    seis_u_phys = forward_seismic_from_impedance(y_u_phys, wavelet)
                    if bool(cfg.loss_in_physical):
                        x_u_phys = x_u * x_std + x_mean
                        loss_s = F.mse_loss(seis_u_phys, x_u_phys)
                    else:
                        seis_pred_u = (seis_u_phys - x_mean) / (x_std + 1e-12)
                        loss_s = F.mse_loss(seis_pred_u, x_u)
                else:
                    seis_pred_u = forward_seismic_from_impedance(y_pred_u, wavelet)
                    loss_s = F.mse_loss(seis_pred_u, x_u)

                loss_g = loss_adv + cfg.alpha * loss_i + cfg.beta * loss_s + float(cfg.grad_loss_weight) * loss_grad

            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(loss_g).backward()
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            scaler_g.step(opt_g)
            scaler_g.update()

            pbar.set_postfix(Ld=f"{float(loss_d):.2f}",
                             Lg=f"{float(loss_g):.2f}",
                             Li=f"{float(loss_i):.4f}",
                             Ls=f"{float(loss_s):.4f}",
                             Adv=f"{float(loss_adv):.2f}")

            n_steps += 1
            sum_ld += float(loss_d.detach().cpu())
            sum_lg += float(loss_g.detach().cpu())
            sum_li += float(loss_i.detach().cpu())
            sum_ls += float(loss_s.detach().cpu())

        # Update EMA
        if ema_g is not None:
            ema_g.update()
        
        # Step schedulers
        scheduler_g.step()
        scheduler_d.step()

        if n_steps > 0:
            history["train"].append(
                {
                    "epoch": epoch,
                    "Ld": sum_ld / n_steps,
                    "Lg": sum_lg / n_steps,
                    "Li": sum_li / n_steps,
                    "Ls": sum_ls / n_steps,
                }
            )

        # Validate with EMA if enabled
        if ema_g is not None:
            ema_g.apply_shadow()
        val = _eval(G, val_loader, device)
        if ema_g is not None:
            ema_g.restore()
            
        history["val"].append({"epoch": epoch, **val, "time_sec": time.time()-t0})
        with open(os.path.join(cfg.run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        payload = {"G": G.state_dict(), "D": D.state_dict(), "cfg": cfg.__dict__, "stats": stats or {}}
        if ema_g is not None:
            payload["G_ema"] = ema_g.shadow
        if val["mse"] < best:
            best = val["mse"]
            torch.save(payload, os.path.join(ckpt_dir, "best.pt"))
        if epoch % cfg.save_every == 0:
            torch.save(payload, os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt"))
