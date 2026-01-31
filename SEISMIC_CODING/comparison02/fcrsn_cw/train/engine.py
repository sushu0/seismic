from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

@dataclass
class TrainState:
    epoch: int
    best_val_loss: float

def save_checkpoint(path: str | Path, model, optimizer, state: TrainState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": {"epoch": state.epoch, "best_val_loss": state.best_val_loss},
    }, path)

def load_checkpoint(path: str | Path, model, optimizer=None) -> TrainState:
    ckpt = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    s = ckpt.get("state", {})
    return TrainState(epoch=int(s.get("epoch", 0)), best_val_loss=float(s.get("best_val_loss", float("inf"))))

def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)  # MSE; paper uses 1/(2N) sum (..)^2. Constant factor doesn't change optimum. fileciteturn2file0L251-L269
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def eval_one_epoch(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = torch.mean((pred - y) ** 2)
        losses.append(loss.item())
    return float(np.mean(losses))


def _time_grad(x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, T)
    return x[..., 1:] - x[..., :-1]


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    grad_weight: float = 0.0,
) -> torch.Tensor:
    lt = str(loss_type).lower()
    if lt == "mse":
        base = torch.mean((pred - target) ** 2)
    elif lt == "huber":
        base = torch.mean(torch.nn.functional.huber_loss(pred, target, delta=float(huber_delta), reduction="none"))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use mse|huber")

    gw = float(grad_weight)
    if gw <= 0:
        return base
    gp = _time_grad(pred)
    gt = _time_grad(target)
    grad = torch.mean((gp - gt) ** 2)
    return base + gw * grad


def train_one_epoch_cfg(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    grad_weight: float = 0.0,
    train_snr_db: float | None = None,
    noise_seed: int = 42,
) -> float:
    model.train()
    losses = []
    gen = torch.Generator(device=device)
    gen.manual_seed(int(noise_seed))
    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device)
        y = y.to(device)

        if train_snr_db is not None:
            # Add per-trace Gaussian noise to input to reach target SNR.
            snr_db = float(train_snr_db)
            # power over time axis
            power = torch.mean(x * x, dim=-1, keepdim=True).clamp_min(1e-12)
            noise_power = power / (10.0 ** (snr_db / 10.0))
            noise = torch.randn(x.shape, device=device, generator=gen) * torch.sqrt(noise_power)
            x = x + noise

        pred = model(x)
        loss = compute_loss(pred, y, loss_type=loss_type, huber_delta=huber_delta, grad_weight=grad_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def eval_one_epoch_cfg(
    model,
    loader: DataLoader,
    device: torch.device,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
    grad_weight: float = 0.0,
) -> float:
    model.eval()
    losses = []
    for x, y in tqdm(loader, desc="val", leave=False):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = compute_loss(pred, y, loss_type=loss_type, huber_delta=huber_delta, grad_weight=grad_weight)
        losses.append(loss.item())
    return float(np.mean(losses))
