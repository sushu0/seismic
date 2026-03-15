import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from scipy.signal import butter, filtfilt
from torch.utils.data import DataLoader, Dataset


DEFAULT_SGY = Path(r"d:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy")
DEFAULT_OUT = Path(r"d:\SEISMIC_CODING\new\sgy_inversion_v8")
V6_PRIOR_DIRS = [
    Path(r"d:\SEISMIC_CODING\new\results\01_20Hz_v6"),
    Path(r"d:\SEISMIC_CODING\new\results\01_30Hz_v6"),
    Path(r"d:\SEISMIC_CODING\new\results\01_40Hz_v6"),
]


@dataclass
class PriorSpec:
    name: str
    ckpt_path: Path
    norm_path: Path
    highpass_cutoff: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def uniform_indices(total: int, count: int) -> np.ndarray:
    count = min(int(count), int(total))
    if count <= 0:
        raise ValueError("count must be positive")
    if count == total:
        return np.arange(total, dtype=np.int64)
    idx = np.linspace(0, total - 1, count)
    return np.round(idx).astype(np.int64)


def read_tracecount_and_samples(sgy_path: Path) -> Tuple[int, int, np.ndarray, float]:
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as f:
        tracecount = f.tracecount
        samples = np.asarray(f.samples, dtype=np.float32)
        interval_us = segyio.tools.dt(f) or 2000
    return tracecount, len(samples), samples, float(interval_us) / 1e6


def read_sgy_traces(sgy_path: Path, indices: np.ndarray) -> np.ndarray:
    traces = []
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as f:
        for i in indices.tolist():
            traces.append(np.asarray(f.trace[int(i)], dtype=np.float32))
    return np.stack(traces, axis=0)


def highpass_filter(data: np.ndarray, cutoff_hz: float, fs_hz: float) -> np.ndarray:
    nyq = 0.5 * fs_hz
    cutoff = min(float(cutoff_hz), nyq * 0.95)
    b, a = butter(4, cutoff / nyq, btype="high")
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


def preprocess_v6_style(data: np.ndarray, cutoff_hz: float, fs_hz: float) -> np.ndarray:
    centered = data - data.mean(axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True) + 1e-6
    seismic = centered / scale
    seismic_hf = highpass_filter(centered, cutoff_hz=cutoff_hz, fs_hz=fs_hz) / scale
    return np.stack([seismic.astype(np.float32), seismic_hf.astype(np.float32)], axis=1)


def preprocess_v8_channels(
    seismic: np.ndarray,
    base_prior_log: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
) -> np.ndarray:
    centered = seismic - seismic.mean(axis=1, keepdims=True)
    scale = np.std(centered, axis=1, keepdims=True) + 1e-6
    seismic_n = centered / scale
    seismic_hp = highpass_filter(centered, cutoff_hz=cutoff_hz, fs_hz=fs_hz) / scale
    prior_center = base_prior_log - base_prior_log.mean(axis=1, keepdims=True)
    prior_scale = np.std(prior_center, axis=1, keepdims=True) + 1e-6
    prior_n = prior_center / prior_scale
    return np.stack(
        [seismic_n.astype(np.float32), seismic_hp.astype(np.float32), prior_n.astype(np.float32)],
        axis=1,
    )


def ricker(f0: float, dt: float, length: float) -> np.ndarray:
    t = np.arange(-length / 2.0, length / 2.0 + dt, dt, dtype=np.float64)
    p2 = np.pi ** 2
    wavelet = (1.0 - 2.0 * p2 * (f0 ** 2) * (t ** 2)) * np.exp(-p2 * (f0 ** 2) * (t ** 2))
    return wavelet.astype(np.float32)


def per_trace_corr(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_v = pred - pred.mean(dim=-1, keepdim=True)
    target_v = target - target.mean(dim=-1, keepdim=True)
    num = (pred_v * target_v).sum(dim=-1)
    den = (pred_v.norm(dim=-1) * target_v.norm(dim=-1)).clamp(min=1e-8)
    return num / den


def pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (1.0 - per_trace_corr(pred, target)).mean()


def multiscale_pearson_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    losses = [pearson_loss(pred, target)]
    for stride in (2, 4):
        losses.append(pearson_loss(pred[..., ::stride], target[..., ::stride]))
    return sum(losses) / len(losses)


def stft_logmag_l1(pred: torch.Tensor, target: torch.Tensor, n_fft: int = 256, hop: int = 64) -> torch.Tensor:
    p = pred.squeeze(1)
    t = target.squeeze(1)
    p = p / (p.std(dim=-1, keepdim=True) + 1e-6)
    t = t / (t.std(dim=-1, keepdim=True) + 1e-6)
    win = torch.hann_window(n_fft, device=pred.device)
    p_stft = torch.stft(p, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    t_stft = torch.stft(t, n_fft=n_fft, hop_length=hop, window=win, return_complex=True)
    return F.l1_loss(torch.log(torch.abs(p_stft) + 1e-6), torch.log(torch.abs(t_stft) + 1e-6))


def gradient_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_n = pred / (pred.std(dim=-1, keepdim=True) + 1e-6)
    target_n = target / (target.std(dim=-1, keepdim=True) + 1e-6)
    return F.l1_loss(pred_n[..., 1:] - pred_n[..., :-1], target_n[..., 1:] - target_n[..., :-1])


def _normalize_edge_map(edge: torch.Tensor) -> torch.Tensor:
    edge = edge / (edge.amax(dim=-1, keepdim=True) + 1e-6)
    return edge


def seismic_event_guide(seismic_hp: torch.Tensor) -> torch.Tensor:
    guide = torch.abs(seismic_hp)
    guide = F.avg_pool1d(guide, kernel_size=11, stride=1, padding=5)
    guide = guide / (guide.amax(dim=-1, keepdim=True) + 1e-6)
    return guide


def structure_guided_losses(
    log_imp: torch.Tensor,
    seismic_hp: torch.Tensor,
    base_log: torch.Tensor,
    edge_gain_ratio: float,
    variance_floor_ratio: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    imp_edge = torch.abs(log_imp[..., 1:] - log_imp[..., :-1])
    imp_edge = F.avg_pool1d(imp_edge, kernel_size=7, stride=1, padding=3)
    imp_edge = _normalize_edge_map(imp_edge)

    seis_edge = torch.abs(seismic_hp[..., 1:] - seismic_hp[..., :-1])
    seis_edge = F.avg_pool1d(seis_edge, kernel_size=9, stride=1, padding=4)
    seis_edge = _normalize_edge_map(seis_edge)

    edge_corr = (1.0 - per_trace_corr(imp_edge, seis_edge)).mean()
    edge_l1 = F.l1_loss(imp_edge, seis_edge)
    background_penalty = ((1.0 - seis_edge) * imp_edge).mean()

    guide_full = seismic_event_guide(seismic_hp)
    guide_edge = 0.5 * (guide_full[..., 1:] + guide_full[..., :-1])
    base_edge = torch.abs(base_log[..., 1:] - base_log[..., :-1])
    base_edge = F.avg_pool1d(base_edge, kernel_size=7, stride=1, padding=3)
    pred_edge_energy = (guide_edge * imp_edge).sum(dim=-1) / (guide_edge.sum(dim=-1) + 1e-6)
    base_edge_energy = (guide_edge * base_edge).sum(dim=-1) / (guide_edge.sum(dim=-1) + 1e-6)
    edge_gain = F.relu(edge_gain_ratio * base_edge_energy - pred_edge_energy).mean()

    impedance = torch.exp(log_imp)
    base_impedance = torch.exp(base_log)
    pred_std = impedance.std(dim=-1)
    base_std = base_impedance.std(dim=-1).detach()
    variance_floor = (F.relu(variance_floor_ratio * base_std - pred_std) / (base_std + 1e-6)).mean()
    return edge_corr, edge_l1, background_penalty, edge_gain, variance_floor, guide_full


class FixedGaussian1D(nn.Module):
    def __init__(self, kernel_size: int = 9, sigma: float = 1.5):
        super().__init__()
        radius = kernel_size // 2
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel.view(1, 1, -1))
        self.pad = radius

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv1d(x, self.kernel, padding=self.pad)


def weighted_lowpass_prior_loss(
    log_imp: torch.Tensor,
    base_log: torch.Tensor,
    weight_map: torch.Tensor,
    smoother: FixedGaussian1D,
) -> torch.Tensor:
    lp_pred = smoother(log_imp)
    lp_base = smoother(base_log)
    return (torch.abs(lp_pred - lp_base) * weight_map).mean()


def delta_tv_loss(delta: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(delta[..., 1:] - delta[..., :-1]))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).unsqueeze(-1)


class DilatedBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilations: Tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        branch_channels = out_ch // len(dilations)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_ch, branch_channels, 3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(branch_channels),
                    nn.GELU(),
                )
                for dilation in dilations
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv1d(branch_channels * len(dilations), out_ch, 1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([branch(x) for branch in self.branches], dim=1)
        out = self.fuse(out)
        out = self.se(out)
        return out + self.skip(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU(),
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.se(out)
        return out + self.skip(x)


class InversionNet(nn.Module):
    def __init__(self, in_ch: int = 3, base: int = 48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3),
            nn.BatchNorm1d(base),
            nn.GELU(),
        )
        self.ms = DilatedBlock(base, base)
        self.e1 = ResBlock(base, base * 2)
        self.p1 = nn.MaxPool1d(2)
        self.e2 = ResBlock(base * 2, base * 4)
        self.p2 = nn.MaxPool1d(2)
        self.neck = nn.Sequential(DilatedBlock(base * 4, base * 8), ResBlock(base * 8, base * 8))
        self.u2 = nn.ConvTranspose1d(base * 8, base * 4, 2, 2)
        self.d2 = ResBlock(base * 8, base * 4)
        self.u1 = nn.ConvTranspose1d(base * 4, base * 2, 2, 2)
        self.d1 = ResBlock(base * 4, base * 2)
        self.refine = nn.Sequential(
            ResBlock(base * 2 + base, base * 2),
            nn.Conv1d(base * 2, base, 3, padding=1),
            nn.GELU(),
        )
        self.out = nn.Conv1d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.ms(self.stem(x))
        e1 = self.e1(x0)
        e2 = self.e2(self.p1(e1))
        bottleneck = self.neck(self.p2(e2))
        d2 = self.u2(bottleneck)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode="linear", align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], dim=1))
        d1 = self.u1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode="linear", align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], dim=1))
        if d1.shape[-1] != x0.shape[-1]:
            d1 = F.interpolate(d1, size=x0.shape[-1], mode="linear", align_corners=False)
        return self.out(self.refine(torch.cat([d1, x0], dim=1)))


class ForwardModel(nn.Module):
    def __init__(self, wavelet: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        self.register_buffer("wavelet", wavelet.clone().float().view(1, 1, -1))
        self.eps = eps

    def forward(self, impedance: torch.Tensor) -> torch.Tensor:
        prev = torch.roll(impedance, shifts=1, dims=-1)
        reflectivity = (impedance - prev) / (impedance + prev + self.eps)
        reflectivity[..., 0] = 0.0
        pad = (self.wavelet.shape[-1] - 1) // 2
        synth = F.conv1d(reflectivity, self.wavelet, padding=pad)
        return synth[..., : impedance.shape[-1]]


class FullTraceDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        seismic: np.ndarray,
        base_prior_log: np.ndarray,
        prior_weight: np.ndarray,
        trace_indices: np.ndarray,
    ):
        self.features = torch.from_numpy(features[trace_indices].astype(np.float32))
        self.seismic = torch.from_numpy(seismic[trace_indices, None, :].astype(np.float32))
        self.base_prior_log = torch.from_numpy(base_prior_log[trace_indices, None, :].astype(np.float32))
        self.prior_weight = torch.from_numpy(prior_weight[trace_indices, None, :].astype(np.float32))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.seismic[index],
            self.base_prior_log[index],
            self.prior_weight[index],
        )


def load_prior_specs() -> List[PriorSpec]:
    specs: List[PriorSpec] = []
    for result_dir in V6_PRIOR_DIRS:
        norm_path = result_dir / "norm_stats.json"
        ckpt_path = result_dir / "checkpoints" / "best.pt"
        with norm_path.open("r", encoding="utf-8") as f:
            norm = json.load(f)
        specs.append(
            PriorSpec(
                name=result_dir.name,
                ckpt_path=ckpt_path,
                norm_path=norm_path,
                highpass_cutoff=float(norm["highpass_cutoff"]),
            )
        )
    return specs


def run_supervised_prior_models(
    prior_specs: List[PriorSpec],
    seismic: np.ndarray,
    fs_hz: float,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, dict]]:
    predictions = []
    metadata: Dict[str, dict] = {}
    for spec in prior_specs:
        with spec.norm_path.open("r", encoding="utf-8") as f:
            norm = json.load(f)
        features = preprocess_v6_style(seismic, cutoff_hz=spec.highpass_cutoff, fs_hz=fs_hz)
        model = InversionNet(in_ch=2, base=48).to(device)
        payload = torch.load(spec.ckpt_path, map_location=device)
        state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = torch.from_numpy(features[start : start + batch_size]).to(device)
                out = model(batch).squeeze(1).cpu().numpy()
                preds.append(out)
        pred_norm = np.concatenate(preds, axis=0)
        imp = pred_norm * float(norm["imp_std"]) + float(norm["imp_mean"])
        imp = np.clip(imp, 1e5, None)
        predictions.append(np.log(imp).astype(np.float32))
        metadata[spec.name] = {
            "highpass_cutoff": spec.highpass_cutoff,
            "imp_mean": float(norm["imp_mean"]),
            "imp_std": float(norm["imp_std"]),
        }
    stacked = np.stack(predictions, axis=0)
    return stacked, metadata


def smooth_log_prior(base_prior_log: np.ndarray) -> np.ndarray:
    return gaussian_filter(base_prior_log, sigma=(1.0, 4.0)).astype(np.float32)


def prior_weight_map(prior_uncertainty: np.ndarray) -> np.ndarray:
    denom = np.maximum(prior_uncertainty, np.percentile(prior_uncertainty, 5))
    raw = np.median(denom) / (denom + 1e-6)
    return np.clip(raw, 0.25, 1.0).astype(np.float32)


def init_from_30hz_v6(model: InversionNet, device: torch.device) -> None:
    ckpt_path = Path(r"d:\SEISMIC_CODING\new\results\01_30Hz_v6\checkpoints\best.pt")
    payload = torch.load(ckpt_path, map_location=device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model_state = model.state_dict()
    for key, tensor in state_dict.items():
        if key not in model_state:
            continue
        if key == "stem.0.weight" and tensor.shape[1] == 2 and model_state[key].shape[1] == 3:
            init = model_state[key].clone()
            init[:, :2, :] = tensor
            init[:, 2:3, :] = 0.0
            model_state[key] = init
            continue
        if model_state[key].shape == tensor.shape:
            model_state[key] = tensor
    model.load_state_dict(model_state, strict=True)


def checkpoint_payload(
    model: InversionNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val: float,
    history: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "history": history,
    }


def train_one_epoch(
    model: InversionNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    forward_model: ForwardModel,
    delta_smoother: FixedGaussian1D,
    lowpass_smoother: FixedGaussian1D,
    epoch: int,
    schedule_epoch: int,
    grad_clip: float,
    structure_weight: float,
    structure_l1_weight: float,
    structure_bg_weight: float,
    structure_gain_weight: float,
    variance_floor_weight: float,
    edge_gain_ratio: float,
    variance_floor_ratio: float,
    prior_relax: float,
    delta_relax: float,
) -> dict:
    model.train()
    stats = {
        "total": 0.0,
        "corr": 0.0,
        "stft": 0.0,
        "grad": 0.0,
        "prior": 0.0,
        "tv": 0.0,
        "delta": 0.0,
        "struct_corr": 0.0,
        "struct_l1": 0.0,
        "struct_bg": 0.0,
        "struct_gain": 0.0,
        "var_floor": 0.0,
    }
    w_phys = min(1.0, float(epoch) / 10.0)
    w_struct = min(1.0, float(schedule_epoch) / 12.0)
    for features, seismic, base_log, prior_weight in loader:
        features = features.to(device, non_blocking=True)
        seismic = seismic.to(device, non_blocking=True)
        base_log = base_log.to(device, non_blocking=True)
        prior_weight = prior_weight.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            raw_delta = model(features)
            smooth_delta = delta_smoother(raw_delta)
            bounded_delta = 0.35 * torch.tanh(smooth_delta)
            log_pred = base_log + bounded_delta
            impedance = torch.exp(log_pred)
            synth = forward_model(impedance)
            event_guide = seismic_event_guide(features[:, 1:2, :])
            loss_corr = multiscale_pearson_loss(synth, seismic)
            loss_stft = stft_logmag_l1(synth, seismic)
            loss_grad = gradient_consistency_loss(synth, seismic)
            relaxed_prior_weight = prior_weight * torch.clamp(1.0 - prior_relax * event_guide, min=0.25)
            loss_prior = weighted_lowpass_prior_loss(log_pred, base_log, relaxed_prior_weight, lowpass_smoother)
            loss_tv = delta_tv_loss(bounded_delta)
            delta_region_weight = torch.clamp(1.0 - delta_relax * event_guide, min=0.20)
            loss_delta = torch.mean((bounded_delta ** 2) * delta_region_weight)
            (
                loss_struct_corr,
                loss_struct_l1,
                loss_struct_bg,
                loss_struct_gain,
                loss_var_floor,
                _,
            ) = structure_guided_losses(
                log_pred,
                features[:, 1:2, :],
                base_log,
                edge_gain_ratio=edge_gain_ratio,
                variance_floor_ratio=variance_floor_ratio,
            )
            total = (
                w_phys * (1.0 * loss_corr + 0.25 * loss_stft + 0.25 * loss_grad)
                + 0.20 * loss_prior
                + w_struct * (
                    structure_weight * loss_struct_corr
                    + structure_l1_weight * loss_struct_l1
                    + structure_bg_weight * loss_struct_bg
                    + structure_gain_weight * loss_struct_gain
                    + variance_floor_weight * loss_var_floor
                )
                + 0.02 * loss_tv
                + 0.02 * loss_delta
            )
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        stats["total"] += float(total.item())
        stats["corr"] += float(loss_corr.item())
        stats["stft"] += float(loss_stft.item())
        stats["grad"] += float(loss_grad.item())
        stats["prior"] += float(loss_prior.item())
        stats["tv"] += float(loss_tv.item())
        stats["delta"] += float(loss_delta.item())
        stats["struct_corr"] += float(loss_struct_corr.item())
        stats["struct_l1"] += float(loss_struct_l1.item())
        stats["struct_bg"] += float(loss_struct_bg.item())
        stats["struct_gain"] += float(loss_struct_gain.item())
        stats["var_floor"] += float(loss_var_floor.item())
    for key in stats:
        stats[key] /= max(len(loader), 1)
    stats["w_phys"] = w_phys
    stats["w_struct"] = w_struct
    return stats


@torch.no_grad()
def validate_one_epoch(
    model: InversionNet,
    loader: DataLoader,
    device: torch.device,
    forward_model: ForwardModel,
    delta_smoother: FixedGaussian1D,
    lowpass_smoother: FixedGaussian1D,
    epoch: int,
    schedule_epoch: int,
    structure_weight: float,
    structure_l1_weight: float,
    structure_bg_weight: float,
    structure_gain_weight: float,
    variance_floor_weight: float,
    edge_gain_ratio: float,
    variance_floor_ratio: float,
    prior_relax: float,
    delta_relax: float,
) -> dict:
    model.eval()
    stats = {
        "total": 0.0,
        "corr": 0.0,
        "stft": 0.0,
        "grad": 0.0,
        "prior": 0.0,
        "tv": 0.0,
        "delta": 0.0,
        "struct_corr": 0.0,
        "struct_l1": 0.0,
        "struct_bg": 0.0,
        "struct_gain": 0.0,
        "var_floor": 0.0,
        "pcc": 0.0,
    }
    w_phys = min(1.0, float(epoch) / 10.0)
    w_struct = min(1.0, float(schedule_epoch) / 12.0)
    for features, seismic, base_log, prior_weight in loader:
        features = features.to(device, non_blocking=True)
        seismic = seismic.to(device, non_blocking=True)
        base_log = base_log.to(device, non_blocking=True)
        prior_weight = prior_weight.to(device, non_blocking=True)
        raw_delta = model(features)
        smooth_delta = delta_smoother(raw_delta)
        bounded_delta = 0.35 * torch.tanh(smooth_delta)
        log_pred = base_log + bounded_delta
        impedance = torch.exp(log_pred)
        synth = forward_model(impedance)
        event_guide = seismic_event_guide(features[:, 1:2, :])
        loss_corr = multiscale_pearson_loss(synth, seismic)
        loss_stft = stft_logmag_l1(synth, seismic)
        loss_grad = gradient_consistency_loss(synth, seismic)
        relaxed_prior_weight = prior_weight * torch.clamp(1.0 - prior_relax * event_guide, min=0.25)
        loss_prior = weighted_lowpass_prior_loss(log_pred, base_log, relaxed_prior_weight, lowpass_smoother)
        loss_tv = delta_tv_loss(bounded_delta)
        delta_region_weight = torch.clamp(1.0 - delta_relax * event_guide, min=0.20)
        loss_delta = torch.mean((bounded_delta ** 2) * delta_region_weight)
        (
            loss_struct_corr,
            loss_struct_l1,
            loss_struct_bg,
            loss_struct_gain,
            loss_var_floor,
            _,
        ) = structure_guided_losses(
            log_pred,
            features[:, 1:2, :],
            base_log,
            edge_gain_ratio=edge_gain_ratio,
            variance_floor_ratio=variance_floor_ratio,
        )
        total = (
            w_phys * (1.0 * loss_corr + 0.25 * loss_stft + 0.25 * loss_grad)
            + 0.20 * loss_prior
            + w_struct * (
                structure_weight * loss_struct_corr
                + structure_l1_weight * loss_struct_l1
                + structure_bg_weight * loss_struct_bg
                + structure_gain_weight * loss_struct_gain
                + variance_floor_weight * loss_var_floor
            )
            + 0.02 * loss_tv
            + 0.02 * loss_delta
        )
        stats["total"] += float(total.item())
        stats["corr"] += float(loss_corr.item())
        stats["stft"] += float(loss_stft.item())
        stats["grad"] += float(loss_grad.item())
        stats["prior"] += float(loss_prior.item())
        stats["tv"] += float(loss_tv.item())
        stats["delta"] += float(loss_delta.item())
        stats["struct_corr"] += float(loss_struct_corr.item())
        stats["struct_l1"] += float(loss_struct_l1.item())
        stats["struct_bg"] += float(loss_struct_bg.item())
        stats["struct_gain"] += float(loss_struct_gain.item())
        stats["var_floor"] += float(loss_var_floor.item())
        stats["pcc"] += float(per_trace_corr(synth, seismic).mean().item())
    for key in stats:
        stats[key] /= max(len(loader), 1)
    stats["w_phys"] = w_phys
    stats["w_struct"] = w_struct
    return stats


@torch.no_grad()
def infer_log_impedance(
    model: InversionNet,
    features: np.ndarray,
    base_prior_log: np.ndarray,
    batch_size: int,
    device: torch.device,
    delta_smoother: FixedGaussian1D,
) -> np.ndarray:
    model.eval()
    outputs: List[np.ndarray] = []
    for start in range(0, features.shape[0], batch_size):
        batch_feat = torch.from_numpy(features[start : start + batch_size]).to(device)
        batch_base = torch.from_numpy(base_prior_log[start : start + batch_size, None, :]).to(device)
        raw_delta = model(batch_feat)
        bounded_delta = 0.35 * torch.tanh(delta_smoother(raw_delta))
        log_pred = batch_base + bounded_delta
        outputs.append(log_pred.squeeze(1).cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def postprocess_impedance(raw_impedance: np.ndarray) -> Tuple[np.ndarray, str]:
    trend = gaussian_filter(raw_impedance, sigma=(2, 8))
    detail = raw_impedance - gaussian_filter(raw_impedance, sigma=(0, 2))
    detail = gaussian_filter(detail, sigma=(1, 1))
    final_plan = trend + 0.4 * detail
    if float(final_plan.std()) < 0.85 * float(raw_impedance.std()):
        variance_safe = 0.90 * raw_impedance + 0.10 * gaussian_filter(raw_impedance, sigma=(1, 3))
        return variance_safe.astype(np.float32), "variance_safe_blend"
    return final_plan.astype(np.float32), "planned_detail_blend"


def soft_display_image(section: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], float]:
    lo = float(np.percentile(section, 0.5))
    hi = float(np.percentile(section, 99.5))
    center = 0.5 * (lo + hi)
    half_span = max(0.5 * (hi - lo), 1e-6)
    normalized = (section - center) / half_span
    soft_mapped = center + half_span * np.tanh(normalized)
    boundary_eps = 1e-3 * (hi - lo + 1e-6)
    clipped_ratio = float(np.mean((soft_mapped <= lo + boundary_eps) | (soft_mapped >= hi - boundary_eps)))
    return soft_mapped.astype(np.float32), (lo, hi), clipped_ratio


def mean_trace_corr_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred_v = pred - pred.mean(axis=1, keepdims=True)
    target_v = target - target.mean(axis=1, keepdims=True)
    num = np.sum(pred_v * target_v, axis=1)
    den = np.linalg.norm(pred_v, axis=1) * np.linalg.norm(target_v, axis=1)
    corr = num / np.clip(den, 1e-8, None)
    return float(np.mean(corr))


def structure_alignment_score(section: np.ndarray, observed: np.ndarray, fs_hz: float, cutoff_hz: float) -> float:
    log_imp = np.log(np.clip(section, 1e5, None))
    imp_grad = np.abs(np.diff(log_imp, axis=1))
    seismic_center = observed - observed.mean(axis=1, keepdims=True)
    seismic_hp = highpass_filter(seismic_center, cutoff_hz=cutoff_hz, fs_hz=fs_hz)
    seismic_grad = np.abs(np.diff(gaussian_filter(seismic_hp, sigma=(0, 0.8)), axis=1))
    imp_norm = (imp_grad - imp_grad.mean()) / (imp_grad.std() + 1e-6)
    seismic_norm = (seismic_grad - seismic_grad.mean()) / (seismic_grad.std() + 1e-6)
    return float(np.mean(imp_norm * seismic_norm))


def generate_postprocess_candidates(raw_impedance: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    candidates: List[Tuple[str, np.ndarray]] = []
    candidates.append(("raw_identity", raw_impedance.astype(np.float32)))

    planned, planned_name = postprocess_impedance(raw_impedance)
    candidates.append((planned_name, planned.astype(np.float32)))

    trend = gaussian_filter(raw_impedance, sigma=(1, 5))
    detail = raw_impedance - trend
    for alpha in (0.91, 0.93, 0.95):
        candidates.append((f"mild_detail_a{alpha:.2f}", (trend + alpha * detail).astype(np.float32)))
    return candidates


def postprocess_selection_score(metrics: dict) -> float:
    score = float(metrics["final_trace_pcc"])
    mean_val = float(metrics["final_impedance_mean"])
    std_val = float(metrics["final_impedance_std"])
    mae_ratio = float(metrics["mae_ratio_vs_prior"])
    edge_align = float(metrics["edge_alignment"])
    if 6.7e6 <= mean_val <= 8.3e6:
        score += 0.04
    else:
        score -= min(abs(mean_val - 7.5e6) / 7.5e6, 1.0) * 0.15
    if 3.0e5 <= std_val <= 1.3e6:
        score += 0.05
    elif std_val < 3.0e5:
        score -= min((3.0e5 - std_val) / 3.0e5, 1.0) * 0.12
    else:
        score -= min((std_val - 1.3e6) / 1.3e6, 1.0) * 0.08
    if 0.03 <= mae_ratio <= 0.20:
        score += 0.04
    elif mae_ratio < 0.03:
        score -= min((0.03 - mae_ratio) / 0.03, 1.0) * 0.08
    else:
        score -= min((mae_ratio - 0.20) / 0.20, 1.0) * 0.06
    score += 0.15 * edge_align
    return score


@torch.no_grad()
def evaluate_log_candidate(
    candidate_name: str,
    infer_log: np.ndarray,
    checkpoint_path: str,
    checkpoint_epoch: int,
    infer_base_log: np.ndarray,
    infer_seismic: np.ndarray,
    fs_hz: float,
    highpass_cutoff: float,
    device: torch.device,
    forward_model: ForwardModel,
) -> dict:
    imp_raw = np.exp(infer_log).astype(np.float32)
    base_imp = np.exp(infer_base_log).astype(np.float32)
    synth_raw = forward_model(torch.from_numpy(imp_raw[:, None, :]).to(device)).squeeze(1).cpu().numpy().astype(np.float32)
    raw_trace_pcc = mean_trace_corr_np(synth_raw, infer_seismic)
    postprocess_candidates = generate_postprocess_candidates(imp_raw)
    postprocess_metrics = []
    best_post = None
    for post_name, imp_final in postprocess_candidates:
        synth_final = forward_model(torch.from_numpy(imp_final[:, None, :]).to(device)).squeeze(1).cpu().numpy().astype(np.float32)
        metrics = {
            "postprocess_name": post_name,
            "final_trace_pcc": mean_trace_corr_np(synth_final, infer_seismic),
            "final_impedance_mean": float(imp_final.mean()),
            "final_impedance_std": float(imp_final.std()),
            "mae_ratio_vs_prior": float(np.mean(np.abs(imp_final - base_imp)) / (np.mean(base_imp) + 1e-6)),
            "edge_alignment": structure_alignment_score(imp_final, infer_seismic, fs_hz, highpass_cutoff),
        }
        metrics["postprocess_score"] = postprocess_selection_score(metrics)
        postprocess_metrics.append(metrics)
        if best_post is None or metrics["postprocess_score"] > best_post["metrics"]["postprocess_score"]:
            best_post = {"metrics": metrics, "imp_final": imp_final, "synth_final": synth_final}
    if best_post is None:
        raise RuntimeError("No postprocess candidate evaluated")
    metrics = {
        "checkpoint_name": candidate_name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_epoch": checkpoint_epoch,
        "postprocess_mode": best_post["metrics"]["postprocess_name"],
        "raw_trace_pcc": raw_trace_pcc,
        "mean_trace_pcc": best_post["metrics"]["final_trace_pcc"],
        "raw_impedance_mean": float(imp_raw.mean()),
        "raw_impedance_std": float(imp_raw.std()),
        "final_impedance_mean": best_post["metrics"]["final_impedance_mean"],
        "final_impedance_std": best_post["metrics"]["final_impedance_std"],
        "mae_ratio_vs_prior": best_post["metrics"]["mae_ratio_vs_prior"],
        "edge_alignment": best_post["metrics"]["edge_alignment"],
        "postprocess_candidates": postprocess_metrics,
    }
    return {
        "metrics": metrics,
        "infer_log": infer_log,
        "imp_raw": imp_raw,
        "imp_final": best_post["imp_final"],
        "synth": best_post["synth_final"],
        "synth_raw": synth_raw,
    }


@torch.no_grad()
def evaluate_candidate_checkpoint(
    ckpt_name: str,
    ckpt_path: Path,
    model: InversionNet,
    infer_features: np.ndarray,
    infer_base_log: np.ndarray,
    infer_seismic: np.ndarray,
    infer_batch_size: int,
    fs_hz: float,
    highpass_cutoff: float,
    device: torch.device,
    delta_smoother: FixedGaussian1D,
    forward_model: ForwardModel,
) -> dict:
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"])
    infer_log = infer_log_impedance(model, infer_features, infer_base_log, infer_batch_size, device, delta_smoother)
    return evaluate_log_candidate(
        candidate_name=ckpt_name,
        infer_log=infer_log,
        checkpoint_path=str(ckpt_path),
        checkpoint_epoch=int(payload.get("epoch", -1)),
        infer_base_log=infer_base_log,
        infer_seismic=infer_seismic,
        fs_hz=fs_hz,
        highpass_cutoff=highpass_cutoff,
        device=device,
        forward_model=forward_model,
    )


def candidate_selection_score(metrics: dict) -> float:
    score = float(metrics["mean_trace_pcc"])
    mean_val = float(metrics["final_impedance_mean"])
    std_val = float(metrics["final_impedance_std"])
    mae_ratio = float(metrics["mae_ratio_vs_prior"])
    if 6.7e6 <= mean_val <= 8.3e6:
        score += 0.05
    else:
        score -= min(abs(mean_val - 7.5e6) / 7.5e6, 1.0) * 0.20
    if 3.0e5 <= std_val <= 1.3e6:
        score += 0.05
    elif std_val < 3.0e5:
        score -= min((3.0e5 - std_val) / 3.0e5, 1.0) * 0.20
    else:
        score -= min((std_val - 1.3e6) / 1.3e6, 1.0) * 0.10
    if 0.03 <= mae_ratio <= 0.20:
        score += 0.05
    elif mae_ratio < 0.03:
        score -= min((0.03 - mae_ratio) / 0.03, 1.0) * 0.15
    else:
        score -= min((mae_ratio - 0.20) / 0.20, 1.0) * 0.10
    score += 0.12 * float(metrics.get("edge_alignment", 0.0))
    return score


def plot_section(
    image: np.ndarray,
    title: str,
    out_path: Path,
    value_range: Optional[Tuple[float, float]] = None,
    cmap: str = "turbo",
) -> None:
    plt.figure(figsize=(14, 7))
    if value_range is None:
        plt.imshow(image, aspect="auto", cmap=cmap, origin="upper")
    else:
        plt.imshow(image, aspect="auto", cmap=cmap, origin="upper", vmin=value_range[0], vmax=value_range[1])
    plt.title(title)
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_training(history: dict, out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_total"]) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history["train_total"], label="train_total")
    plt.plot(epochs, history["val_total"], label="val_total")
    plt.plot(epochs, history["val_pcc"], label="val_pcc")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("V8 Training History")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_prior_vs_final(base_prior: np.ndarray, final_imp: np.ndarray, out_path: Path) -> None:
    base_disp, vr, _ = soft_display_image(base_prior)
    final_disp, _, _ = soft_display_image(final_imp)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(base_disp, aspect="auto", cmap="turbo", origin="upper", vmin=vr[0], vmax=vr[1])
    axes[0].set_title("Base Prior")
    axes[1].imshow(final_disp, aspect="auto", cmap="turbo", origin="upper", vmin=vr[0], vmax=vr[1])
    axes[1].set_title("V8 Final")
    for ax in axes:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_uncertainty(uncertainty: np.ndarray, out_path: Path) -> None:
    plt.figure(figsize=(14, 7))
    plt.imshow(uncertainty, aspect="auto", cmap="magma", origin="upper")
    plt.title("Prior Uncertainty (log impedance std)")
    plt.xlabel("Trace")
    plt.ylabel("Sample")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def load_baseline_sections(infer_count: int) -> Dict[str, np.ndarray]:
    baselines: Dict[str, np.ndarray] = {}
    if infer_count != 5000:
        return baselines
    baseline_candidates = {
        "v6": Path(r"d:\SEISMIC_CODING\new\sgy_inversion_v6\impedance_pred.npy"),
        "v7": Path(r"d:\SEISMIC_CODING\new\sgy_inversion_v7\impedance_pred.npy"),
    }
    for name, path in baseline_candidates.items():
        if path.exists():
            baselines[name] = np.load(path)
    return baselines


def plot_comparison(
    observed: np.ndarray,
    synthetic: np.ndarray,
    final_imp: np.ndarray,
    out_path: Path,
) -> None:
    residual = observed - synthetic
    seis_lim = float(np.percentile(np.abs(np.concatenate([observed, synthetic], axis=1)), 99.0))
    resid_lim = float(np.percentile(np.abs(residual), 99.0))
    imp_disp, imp_vr, _ = soft_display_image(final_imp)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    axes[0, 0].imshow(observed, aspect="auto", cmap="gray", origin="upper", vmin=-seis_lim, vmax=seis_lim)
    axes[0, 0].set_title("Original Seismic")
    axes[0, 1].imshow(synthetic, aspect="auto", cmap="gray", origin="upper", vmin=-seis_lim, vmax=seis_lim)
    axes[0, 1].set_title("Synthetic From Final Impedance")
    axes[1, 0].imshow(residual, aspect="auto", cmap="RdBu_r", origin="upper", vmin=-resid_lim, vmax=resid_lim)
    axes[1, 0].set_title("Seismic Residual")
    axes[1, 1].imshow(imp_disp, aspect="auto", cmap="turbo", origin="upper", vmin=imp_vr[0], vmax=imp_vr[1])
    axes[1, 1].set_title("Final Impedance")
    for ax in axes.flat:
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_baseline_comparison(
    observed: np.ndarray,
    baselines: Dict[str, np.ndarray],
    final_imp: np.ndarray,
    out_path: Path,
) -> None:
    panels = [("Original Seismic", observed, "gray")]
    if "v6" in baselines:
        panels.append(("V6 Baseline", baselines["v6"], "turbo"))
    if "v7" in baselines:
        panels.append(("V7 Baseline", baselines["v7"], "turbo"))
    panels.append(("V8 Final", final_imp, "turbo"))
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 7), constrained_layout=True)
    if cols == 1:
        axes = [axes]
    imp_refs = [section for _, section, cmap in panels if cmap != "gray"]
    vmin = min(float(np.percentile(s, 0.5)) for s in imp_refs)
    vmax = max(float(np.percentile(s, 99.5)) for s in imp_refs)
    for ax, (title, section, cmap) in zip(axes, panels):
        if cmap == "gray":
            lim = float(np.percentile(np.abs(section), 99.0))
            ax.imshow(section, aspect="auto", cmap=cmap, origin="upper", vmin=-lim, vmax=lim)
        else:
            ax.imshow(section, aspect="auto", cmap=cmap, origin="upper", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Trace")
        ax.set_ylabel("Sample")
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_trace_comparison(
    observed: np.ndarray,
    synth: np.ndarray,
    base_prior: np.ndarray,
    final_imp: np.ndarray,
    out_path: Path,
) -> None:
    trace_ids = [observed.shape[0] // 4, observed.shape[0] // 2, (observed.shape[0] * 3) // 4]
    fig, axes = plt.subplots(len(trace_ids), 2, figsize=(14, 4 * len(trace_ids)), constrained_layout=True)
    if len(trace_ids) == 1:
        axes = np.array([axes])
    for row, trace_id in enumerate(trace_ids):
        axes[row, 0].plot(observed[trace_id], label="original")
        axes[row, 0].plot(synth[trace_id], label="synthetic")
        axes[row, 0].set_title(f"Trace {trace_id}: seismic")
        axes[row, 0].legend()
        axes[row, 0].grid(True, alpha=0.25)
        axes[row, 1].plot(base_prior[trace_id], label="base_prior")
        axes[row, 1].plot(final_imp[trace_id], label="v8_final")
        axes[row, 1].set_title(f"Trace {trace_id}: impedance")
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.25)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def save_json(payload: dict, out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="0908 SGY inversion v8: supervised prior + self-supervised adaptation")
    parser.add_argument("--sgy", type=Path, default=DEFAULT_SGY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-traces", type=int, default=12000)
    parser.add_argument("--infer-traces", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--highpass-cutoff", type=float, default=12.0)
    parser.add_argument("--wavelet-f0", type=float, default=25.0)
    parser.add_argument("--prior-batch-size", type=int, default=32)
    parser.add_argument("--infer-batch-size", type=int, default=16)
    parser.add_argument("--structure-weight", type=float, default=0.08)
    parser.add_argument("--structure-l1-weight", type=float, default=0.04)
    parser.add_argument("--structure-bg-weight", type=float, default=0.02)
    parser.add_argument("--structure-gain-weight", type=float, default=0.05)
    parser.add_argument("--variance-floor-weight", type=float, default=0.06)
    parser.add_argument("--edge-gain-ratio", type=float, default=1.06)
    parser.add_argument("--variance-floor-ratio", type=float, default=0.68)
    parser.add_argument("--prior-relax", type=float, default=0.45)
    parser.add_argument("--delta-relax", type=float, default=0.55)
    parser.add_argument("--reset-history-on-resume", action="store_true")
    parser.add_argument("--reset-optim-on-resume", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), args.out / "run_config.json")
    seed_everything(args.seed)
    device = torch.device(args.device)
    tracecount, nsamples, sample_axis, dt = read_tracecount_and_samples(args.sgy)
    fs_hz = 1.0 / dt

    print(f"Device: {device}")
    print(f"SGY: {args.sgy}")
    print(f"Tracecount: {tracecount}, samples: {nsamples}, dt: {dt:.6f}s")

    train_trace_ids = uniform_indices(tracecount, args.train_traces)
    train_seismic = read_sgy_traces(args.sgy, train_trace_ids)
    prior_specs = load_prior_specs()
    prior_stack, prior_meta = run_supervised_prior_models(
        prior_specs=prior_specs,
        seismic=train_seismic,
        fs_hz=fs_hz,
        batch_size=args.prior_batch_size,
        device=device,
    )
    base_prior_log = np.median(prior_stack, axis=0).astype(np.float32)
    prior_uncertainty = np.std(prior_stack, axis=0).astype(np.float32)
    base_prior_log = smooth_log_prior(base_prior_log)
    prior_weight = prior_weight_map(prior_uncertainty)
    features = preprocess_v8_channels(train_seismic, base_prior_log, args.highpass_cutoff, fs_hz)

    n_train = int(round(0.9 * len(train_trace_ids)))
    train_idx = np.arange(n_train, dtype=np.int64)
    val_idx = np.arange(n_train, len(train_trace_ids), dtype=np.int64)
    train_ds = FullTraceDataset(features, train_seismic, base_prior_log, prior_weight, train_idx)
    val_ds = FullTraceDataset(features, train_seismic, base_prior_log, prior_weight, val_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    model = InversionNet(in_ch=3, base=48).to(device)
    init_from_30hz_v6(model, device)
    wavelet = torch.from_numpy(ricker(args.wavelet_f0, dt, length=0.128)).to(device)
    forward_model = ForwardModel(wavelet=wavelet).to(device)
    delta_smoother = FixedGaussian1D(kernel_size=9, sigma=1.5).to(device)
    lowpass_smoother = FixedGaussian1D(kernel_size=41, sigma=7.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = {"train_total": [], "val_total": [], "val_pcc": []}
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    start_epoch = 1
    best_val = float("inf")
    best_pcc = float("-inf")
    patience_counter = 0

    if args.resume is not None and args.resume.exists():
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"])
        if not args.reset_optim_on_resume:
            optimizer.load_state_dict(payload["optim"])
            scheduler.load_state_dict(payload["sched"])
        start_epoch = int(payload["epoch"]) + 1
        best_val = float(payload["best_val"])
        history = payload.get("history", history)
        if history.get("val_pcc"):
            best_pcc = max(float(v) for v in history["val_pcc"])
        if args.reset_history_on_resume:
            best_val = float("inf")
            best_pcc = float("-inf")
            history = {"train_total": [], "val_total": [], "val_pcc": []}
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        schedule_epoch = epoch - start_epoch + 1 if args.reset_history_on_resume else epoch
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            forward_model=forward_model,
            delta_smoother=delta_smoother,
            lowpass_smoother=lowpass_smoother,
            epoch=epoch,
            schedule_epoch=schedule_epoch,
            grad_clip=1.0,
            structure_weight=args.structure_weight,
            structure_l1_weight=args.structure_l1_weight,
            structure_bg_weight=args.structure_bg_weight,
            structure_gain_weight=args.structure_gain_weight,
            variance_floor_weight=args.variance_floor_weight,
            edge_gain_ratio=args.edge_gain_ratio,
            variance_floor_ratio=args.variance_floor_ratio,
            prior_relax=args.prior_relax,
            delta_relax=args.delta_relax,
        )
        val_stats = validate_one_epoch(
            model=model,
            loader=val_loader,
            device=device,
            forward_model=forward_model,
            delta_smoother=delta_smoother,
            lowpass_smoother=lowpass_smoother,
            epoch=epoch,
            schedule_epoch=schedule_epoch,
            structure_weight=args.structure_weight,
            structure_l1_weight=args.structure_l1_weight,
            structure_bg_weight=args.structure_bg_weight,
            structure_gain_weight=args.structure_gain_weight,
            variance_floor_weight=args.variance_floor_weight,
            edge_gain_ratio=args.edge_gain_ratio,
            variance_floor_ratio=args.variance_floor_ratio,
            prior_relax=args.prior_relax,
            delta_relax=args.delta_relax,
        )
        scheduler.step()
        history["train_total"].append(train_stats["total"])
        history["val_total"].append(val_stats["total"])
        history["val_pcc"].append(val_stats["pcc"])
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_stats['total']:.4f} | val={val_stats['total']:.4f} | "
            f"pcc={val_stats['pcc']:.4f} | struct={val_stats['struct_corr']:.4f} | "
            f"gain={val_stats['struct_gain']:.4f} | var={val_stats['var_floor']:.4f} | "
            f"w_phys={val_stats['w_phys']:.2f} | w_struct={val_stats['w_struct']:.2f}"
        )
        latest_path = ckpt_dir / "latest.pt"
        torch.save(checkpoint_payload(model, optimizer, scheduler, epoch, best_val, history), latest_path)
        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            patience_counter = 0
            torch.save(checkpoint_payload(model, optimizer, scheduler, epoch, best_val, history), ckpt_dir / "best.pt")
            torch.save(checkpoint_payload(model, optimizer, scheduler, epoch, best_val, history), ckpt_dir / "best_loss.pt")
            if val_stats["pcc"] > best_pcc:
                best_pcc = val_stats["pcc"]
                torch.save(checkpoint_payload(model, optimizer, scheduler, epoch, best_val, history), ckpt_dir / "best_pcc.pt")
        else:
            if val_stats["pcc"] > best_pcc:
                best_pcc = val_stats["pcc"]
                torch.save(checkpoint_payload(model, optimizer, scheduler, epoch, best_val, history), ckpt_dir / "best_pcc.pt")
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    infer_trace_ids = uniform_indices(tracecount, args.infer_traces)
    infer_seismic = read_sgy_traces(args.sgy, infer_trace_ids)
    infer_prior_stack, _ = run_supervised_prior_models(
        prior_specs=prior_specs,
        seismic=infer_seismic,
        fs_hz=fs_hz,
        batch_size=args.prior_batch_size,
        device=device,
    )
    infer_base_log = smooth_log_prior(np.median(infer_prior_stack, axis=0).astype(np.float32))
    infer_uncertainty = np.std(infer_prior_stack, axis=0).astype(np.float32)
    infer_features = preprocess_v8_channels(infer_seismic, infer_base_log, args.highpass_cutoff, fs_hz)
    candidate_paths = []
    candidate_name_map = {
        "best_loss": ckpt_dir / "best_loss.pt",
        "best_pcc": ckpt_dir / "best_pcc.pt",
        "latest": ckpt_dir / "latest.pt",
    }
    if not candidate_name_map["best_loss"].exists() and (ckpt_dir / "best.pt").exists():
        candidate_name_map["best_loss"] = ckpt_dir / "best.pt"
    for ckpt_name, ckpt_path in candidate_name_map.items():
        if ckpt_path.exists():
            candidate_paths.append((ckpt_name, ckpt_path))
    if not candidate_paths and args.resume is not None and args.resume.exists():
        candidate_paths.append(("resume", args.resume))

    candidate_results = []
    for ckpt_name, ckpt_path in candidate_paths:
        result = evaluate_candidate_checkpoint(
            ckpt_name=ckpt_name,
            ckpt_path=ckpt_path,
            model=model,
            infer_features=infer_features,
            infer_base_log=infer_base_log,
            infer_seismic=infer_seismic,
            infer_batch_size=args.infer_batch_size,
            fs_hz=fs_hz,
            highpass_cutoff=args.highpass_cutoff,
            device=device,
            delta_smoother=delta_smoother,
            forward_model=forward_model,
        )
        result["metrics"]["selection_score"] = candidate_selection_score(result["metrics"])
        candidate_results.append(result)

    checkpoint_result_map = {item["metrics"]["checkpoint_name"]: item for item in candidate_results}
    if "best_loss" in checkpoint_result_map and "best_pcc" in checkpoint_result_map:
        blend_source_a = checkpoint_result_map["best_loss"]["infer_log"]
        blend_source_b = checkpoint_result_map["best_pcc"]["infer_log"]
        for alpha in (0.15, 0.18, 0.20, 0.22, 0.24):
            blend_log = ((1.0 - alpha) * blend_source_a + alpha * blend_source_b).astype(np.float32)
            blend_result = evaluate_log_candidate(
                candidate_name=f"blend_loss_pcc_a{alpha:.2f}",
                infer_log=blend_log,
                checkpoint_path="blend(best_loss,best_pcc)",
                checkpoint_epoch=-1,
                infer_base_log=infer_base_log,
                infer_seismic=infer_seismic,
                fs_hz=fs_hz,
                highpass_cutoff=args.highpass_cutoff,
                device=device,
                forward_model=forward_model,
            )
            blend_result["metrics"]["selection_score"] = candidate_selection_score(blend_result["metrics"])
            candidate_results.append(blend_result)

    if not candidate_results:
        raise RuntimeError("No candidate checkpoints available for inference")

    selected = max(candidate_results, key=lambda item: item["metrics"]["selection_score"])
    selected_metrics = selected["metrics"]
    imp_raw = selected["imp_raw"]
    imp_final = selected["imp_final"]
    synth = selected["synth"]
    synth_raw = selected["synth_raw"]
    postprocess_mode = selected_metrics["postprocess_mode"]

    np.save(args.out / "base_prior.npy", np.exp(infer_base_log).astype(np.float32))
    np.save(args.out / "prior_uncertainty.npy", infer_uncertainty)
    np.save(args.out / "impedance_pred_raw.npy", imp_raw)
    np.save(args.out / "impedance_pred_final.npy", imp_final)
    np.save(args.out / "synth_seismic.npy", synth)
    np.save(args.out / "synth_seismic_raw.npy", synth_raw)
    np.save(args.out / "synth_seismic_final.npy", synth)

    plot_training(history, args.out / "training_loss.png")
    disp_final, vr, clipped_ratio = soft_display_image(imp_final)
    plot_section(disp_final, "V8 Final Impedance", args.out / "final_section.png", value_range=vr)
    plot_prior_vs_final(np.exp(infer_base_log), imp_final, args.out / "prior_vs_final.png")
    plot_uncertainty(infer_uncertainty, args.out / "uncertainty_map.png")
    baselines = load_baseline_sections(args.infer_traces)
    plot_comparison(infer_seismic, synth, imp_final, args.out / "comparison.png")
    plot_baseline_comparison(infer_seismic, baselines, imp_final, args.out / "baseline_comparison.png")
    plot_trace_comparison(infer_seismic, synth, np.exp(infer_base_log), imp_final, args.out / "trace_comparison.png")

    _, _, clipped_ratio = soft_display_image(imp_final)
    summary = {
        "tracecount_total": tracecount,
        "train_traces": int(args.train_traces),
        "infer_traces": int(args.infer_traces),
        "nsamples": nsamples,
        "dt_seconds": dt,
        "selected_checkpoint": selected_metrics["checkpoint_name"],
        "selected_checkpoint_path": selected_metrics["checkpoint_path"],
        "selected_epoch": selected_metrics["checkpoint_epoch"],
        "postprocess_mode": postprocess_mode,
        "prior_models": prior_meta,
        "candidate_metrics": [item["metrics"] for item in candidate_results],
        "raw_trace_pcc": selected_metrics.get("raw_trace_pcc"),
        "mean_trace_pcc": selected_metrics["mean_trace_pcc"],
        "raw_impedance_mean": selected_metrics["raw_impedance_mean"],
        "raw_impedance_std": selected_metrics["raw_impedance_std"],
        "final_impedance_mean": selected_metrics["final_impedance_mean"],
        "final_impedance_std": selected_metrics["final_impedance_std"],
        "mae_ratio_vs_prior": selected_metrics["mae_ratio_vs_prior"],
        "edge_alignment": selected_metrics.get("edge_alignment"),
        "display_clip_ratio": clipped_ratio,
    }
    save_json(summary, args.out / "run_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
