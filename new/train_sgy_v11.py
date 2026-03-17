import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn as nn
import torch.nn.functional as F

from train_sgy_v8 import (
    DEFAULT_SGY,
    FixedGaussian1D,
    FullTraceDataset,
    InversionNet,
    delta_tv_loss,
    generate_postprocess_candidates,
    gradient_consistency_loss,
    highpass_filter,
    load_baseline_sections,
    load_prior_specs,
    mean_trace_corr_np,
    multiscale_pearson_loss,
    per_trace_corr,
    preprocess_v8_channels,
    prior_weight_map,
    read_sgy_traces,
    read_tracecount_and_samples,
    run_supervised_prior_models,
    save_json,
    seed_everything,
    seismic_event_guide,
    smooth_log_prior,
    soft_display_image,
    stft_logmag_l1,
    structure_alignment_score,
    structure_guided_losses,
    uniform_indices,
    weighted_lowpass_prior_loss,
    init_from_30hz_v6,
)


DEFAULT_OUT = Path(r"d:\SEISMIC_CODING\new\sgy_inversion_v11")
DEFAULT_INIT_CHECKPOINT = Path(r"d:\SEISMIC_CODING\new\sgy_inversion_v10\checkpoints\latest.pt")
DEFAULT_WAVELET_LENGTH = 0.128
DEFAULT_AMP_HUBER_DELTA = 0.75


class AdjacentTraceBlockDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        features: np.ndarray,
        seismic: np.ndarray,
        base_prior_log: np.ndarray,
        prior_weight: np.ndarray,
        block_indices: np.ndarray,
    ):
        self.features = torch.from_numpy(features[block_indices].astype(np.float32))
        self.seismic = torch.from_numpy(seismic[block_indices, :, None, :].astype(np.float32))
        self.base_prior_log = torch.from_numpy(base_prior_log[block_indices, :, None, :].astype(np.float32))
        self.prior_weight = torch.from_numpy(prior_weight[block_indices, :, None, :].astype(np.float32))

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.seismic[index],
            self.base_prior_log[index],
            self.prior_weight[index],
        )


def contiguous_block_trace_ids(total_traces: int, selected_traces: int, group_size: int) -> Tuple[np.ndarray, np.ndarray]:
    group_size = max(int(group_size), 2)
    block_count = max(int(selected_traces) // group_size, 1)
    max_block_start = max(int(total_traces) - group_size, 0)
    block_slots = max_block_start // group_size + 1
    block_starts = uniform_indices(block_slots, block_count) * group_size
    block_starts = np.clip(block_starts, 0, max_block_start).astype(np.int64)
    trace_ids = np.concatenate(
        [np.arange(start, start + group_size, dtype=np.int64) for start in block_starts],
        axis=0,
    )
    return block_starts, trace_ids


def parse_lncc_windows(raw: str) -> Tuple[int, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("lncc windows must not be empty")
    parsed = tuple(int(item) for item in values)
    if any(v <= 1 or v % 2 == 0 for v in parsed):
        raise ValueError("lncc windows must be positive odd integers greater than 1")
    return parsed


def ricker_torch(f0: torch.Tensor, dt: float, length: float, device: torch.device) -> torch.Tensor:
    t = torch.arange(-length / 2.0, length / 2.0 + dt, dt, dtype=torch.float32, device=device)
    p2 = math.pi ** 2
    return (1.0 - 2.0 * p2 * (f0 ** 2) * (t ** 2)) * torch.exp(-p2 * (f0 ** 2) * (t ** 2))


def phase_rotate_wavelet(wavelet: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    length = wavelet.shape[0]
    freq = torch.fft.fftfreq(length, d=1.0, device=wavelet.device)
    sign = torch.sign(freq)
    real = torch.cos(phi).expand_as(sign)
    imag = torch.sin(phi).expand_as(sign) * sign
    multiplier = torch.complex(real, imag)
    multiplier[freq == 0] = torch.complex(
        torch.ones((), device=wavelet.device, dtype=wavelet.dtype),
        torch.zeros((), device=wavelet.device, dtype=wavelet.dtype),
    )
    if length % 2 == 0:
        multiplier[length // 2] = torch.complex(
            torch.ones((), device=wavelet.device, dtype=wavelet.dtype),
            torch.zeros((), device=wavelet.device, dtype=wavelet.dtype),
        )
    return torch.fft.ifft(torch.fft.fft(wavelet) * multiplier).real.to(dtype=wavelet.dtype)


class LearnableWavelet(nn.Module):
    def __init__(self, f0: float, dt: float, length: float = DEFAULT_WAVELET_LENGTH):
        super().__init__()
        self.base_f0 = float(f0)
        self.dt = float(dt)
        self.length = float(length)
        self.phi = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.log_f_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)
        self.gain = nn.Parameter(torch.ones(1, dtype=torch.float32), requires_grad=False)
        init_wavelet = ricker_torch(
            torch.tensor(self.base_f0, dtype=torch.float32),
            dt=self.dt,
            length=self.length,
            device=torch.device("cpu"),
        )
        self.register_buffer("initial_wavelet", init_wavelet.view(1, 1, -1))

    def forward(self) -> torch.Tensor:
        phi = self.phi.float()
        log_f_scale = self.log_f_scale.float()
        gain = self.gain.float()
        f0 = torch.tensor(self.base_f0, dtype=torch.float32, device=self.phi.device) * torch.exp(log_f_scale)
        wavelet = ricker_torch(f0.squeeze(0), dt=self.dt, length=self.length, device=self.phi.device)
        wavelet = phase_rotate_wavelet(wavelet, phi.squeeze(0))
        wavelet = gain.squeeze(0) * wavelet
        return wavelet.view(1, 1, -1)

    def summary(self) -> dict:
        phi_deg = float(torch.rad2deg(self.phi.detach()).cpu().item())
        f_scale = float(torch.exp(self.log_f_scale.detach()).cpu().item())
        gain = float(self.gain.detach().cpu().item())
        return {
            "phi_rad": float(self.phi.detach().cpu().item()),
            "phi_deg": phi_deg,
            "f_scale": f_scale,
            "effective_f0_hz": float(self.base_f0 * f_scale),
            "gain": gain,
            "length_seconds": self.length,
            "length_samples": int(self.initial_wavelet.shape[-1]),
        }


class DynamicForwardModel(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        impedance: torch.Tensor,
        wavelet: torch.Tensor,
        return_reflectivity: bool = False,
    ):
        prev = torch.roll(impedance, shifts=1, dims=-1)
        reflectivity = (impedance - prev) / (impedance + prev + self.eps)
        reflectivity[..., 0] = 0.0
        pad = (wavelet.shape[-1] - 1) // 2
        synth = F.conv1d(reflectivity, wavelet, padding=pad)
        synth = synth[..., : impedance.shape[-1]]
        if return_reflectivity:
            return synth, reflectivity
        return synth


def local_ncc_1d(pred: torch.Tensor, target: torch.Tensor, window: int) -> torch.Tensor:
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        pred_f = pred.float()
        target_f = target.float()
        kernel = torch.ones(1, 1, window, device=pred.device, dtype=torch.float32) / float(window)
        mu_pred = F.conv1d(pred_f, kernel, padding=window // 2)
        mu_target = F.conv1d(target_f, kernel, padding=window // 2)
        pred2 = F.conv1d(pred_f * pred_f, kernel, padding=window // 2)
        target2 = F.conv1d(target_f * target_f, kernel, padding=window // 2)
        pred_target = F.conv1d(pred_f * target_f, kernel, padding=window // 2)
        var_pred = torch.clamp(pred2 - mu_pred * mu_pred, min=1e-6)
        var_target = torch.clamp(target2 - mu_target * mu_target, min=1e-6)
        cov = pred_target - mu_pred * mu_target
        return cov / torch.sqrt(var_pred * var_target + 1e-6)


def multiscale_lncc_loss(pred: torch.Tensor, target: torch.Tensor, windows: Sequence[int]) -> torch.Tensor:
    losses = [1.0 - local_ncc_1d(pred, target, window).mean() for window in windows]
    return torch.stack(losses).mean()


def optimal_trace_scale(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    num = (pred * target).sum(dim=-1, keepdim=True)
    den = torch.clamp((pred * pred).sum(dim=-1, keepdim=True), min=1e-6)
    return num / den


def scale_invariant_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = DEFAULT_AMP_HUBER_DELTA,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scale = optimal_trace_scale(pred, target)
    aligned = scale * pred
    target_scale = torch.sqrt(torch.mean(target * target, dim=-1, keepdim=True) + 1e-6)
    return F.huber_loss(aligned / target_scale, target / target_scale, delta=delta), aligned


def stft_logmag_l1_centered(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 256,
    hop: int = 64,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=pred.device.type, enabled=False):
        pred_f = pred.float().squeeze(1)
        target_f = target.float().squeeze(1)
        pred_centered = pred_f - pred_f.mean(dim=-1, keepdim=True)
        target_centered = target_f - target_f.mean(dim=-1, keepdim=True)
        window = torch.hann_window(n_fft, device=pred.device, dtype=torch.float32)
        pred_stft = torch.stft(pred_centered, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        target_stft = torch.stft(target_centered, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        return F.l1_loss(torch.log(torch.abs(pred_stft) + 1e-6), torch.log(torch.abs(target_stft) + 1e-6))


def stft_logmag_l1_scale_invariant(
    pred_aligned: torch.Tensor,
    target: torch.Tensor,
    n_fft: int = 256,
    hop: int = 64,
) -> torch.Tensor:
    with torch.amp.autocast(device_type=pred_aligned.device.type, enabled=False):
        pred_f = pred_aligned.float().squeeze(1)
        target_f = target.float().squeeze(1)
        target_scale = torch.sqrt(torch.mean(target_f * target_f, dim=-1, keepdim=True) + 1e-6)
        pred_norm = pred_f / target_scale
        target_norm = target_f / target_scale
        pred_centered = pred_norm - pred_norm.mean(dim=-1, keepdim=True)
        target_centered = target_norm - target_norm.mean(dim=-1, keepdim=True)
        window = torch.hann_window(n_fft, device=pred_aligned.device, dtype=torch.float32)
        pred_stft = torch.stft(pred_centered, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        target_stft = torch.stft(target_centered, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
        return F.l1_loss(torch.log(torch.abs(pred_stft) + 1e-6), torch.log(torch.abs(target_stft) + 1e-6))


def seismic_display_map_torch(seismic: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=seismic.device.type, enabled=False):
        seismic_f = seismic.float()
        scale = torch.quantile(torch.abs(seismic_f).flatten(1), 0.99, dim=1, keepdim=True).view(-1, 1, 1)
        return torch.tanh(seismic_f / (scale + 1e-6))


def impedance_display_map_torch(log_impedance: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=log_impedance.device.type, enabled=False):
        log_f = log_impedance.float()
        centered = log_f - log_f.mean(dim=-1, keepdim=True)
        scale = torch.quantile(torch.abs(centered).flatten(1), 0.99, dim=1, keepdim=True).view(-1, 1, 1)
        return torch.tanh(centered / (scale + 1e-6))


def display_similarity_losses(
    log_pred: torch.Tensor,
    seismic: torch.Tensor,
    event_guide: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seismic_disp = seismic_display_map_torch(seismic)
    imp_disp = impedance_display_map_torch(log_pred)
    weight = 0.35 + 0.65 * event_guide.detach()
    loss_l1 = torch.mean(weight * torch.abs(seismic_disp - imp_disp))
    grad_weight = weight[..., 1:]
    loss_grad = torch.mean(grad_weight * torch.abs(torch.diff(seismic_disp, dim=-1) - torch.diff(imp_disp, dim=-1)))
    w = weight.float()
    obs_mean = torch.sum(w * seismic_disp, dim=-1, keepdim=True) / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)
    imp_mean = torch.sum(w * imp_disp, dim=-1, keepdim=True) / (torch.sum(w, dim=-1, keepdim=True) + 1e-6)
    obs_center = seismic_disp - obs_mean
    imp_center = imp_disp - imp_mean
    cov = torch.sum(w * obs_center * imp_center, dim=-1)
    obs_var = torch.sum(w * obs_center.square(), dim=-1)
    imp_var = torch.sum(w * imp_center.square(), dim=-1)
    corr = cov / torch.sqrt(obs_var * imp_var + 1e-6)
    loss_corr = torch.mean(1.0 - corr.clamp(-1.0, 1.0))
    return loss_l1, loss_grad, loss_corr


def residual_guide(observed: torch.Tensor, synth_aligned: torch.Tensor) -> torch.Tensor:
    guide = torch.abs(observed - synth_aligned)
    guide = F.avg_pool1d(guide, kernel_size=11, stride=1, padding=5)
    return guide / (guide.amax(dim=-1, keepdim=True) + 1e-6)


def reflectivity_precision_loss(reflectivity: torch.Tensor, event_mask: torch.Tensor) -> torch.Tensor:
    return ((1.0 - event_mask) * torch.abs(reflectivity)).mean()


def compute_reflectivity_from_log(log_impedance: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    impedance = torch.exp(log_impedance)
    prev = torch.roll(impedance, shifts=1, dims=-1)
    reflectivity = (impedance - prev) / (impedance + prev + eps)
    reflectivity[..., 0] = 0.0
    return reflectivity


def ramp_weight(epoch: int, start: int, duration: int) -> float:
    if duration <= 0:
        return 1.0 if epoch >= start else 0.0
    return float(max(0.0, min(1.0, (float(epoch) - float(start)) / float(duration))))


def lateral_delta_block_loss(
    bounded_delta_blocks: torch.Tensor,
    seismic_hp_blocks: torch.Tensor,
    event_mask_blocks: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    delta_dx = bounded_delta_blocks[:, 1:] - bounded_delta_blocks[:, :-1]
    seismic_dx = torch.abs(seismic_hp_blocks[:, 1:] - seismic_hp_blocks[:, :-1])
    seismic_dx = seismic_dx / (seismic_dx.amax(dim=(1, 2, 3), keepdim=True) + 1e-6)
    weight = torch.exp(-beta * seismic_dx)
    weight = weight * (1.0 - 0.35 * 0.5 * (event_mask_blocks[:, 1:] + event_mask_blocks[:, :-1]))
    return torch.mean(torch.abs(delta_dx) * weight)


def reflectivity_floor_block_loss(
    reflectivity_blocks: torch.Tensor,
    base_log_blocks: torch.Tensor,
    event_mask_blocks: torch.Tensor,
    floor_ratio: float,
) -> torch.Tensor:
    base_reflectivity = compute_reflectivity_from_log(base_log_blocks)
    pred_energy = (event_mask_blocks * torch.abs(reflectivity_blocks)).sum(dim=-1) / (event_mask_blocks.sum(dim=-1) + 1e-6)
    base_energy = (event_mask_blocks * torch.abs(base_reflectivity)).sum(dim=-1) / (event_mask_blocks.sum(dim=-1) + 1e-6)
    return F.relu(floor_ratio * base_energy - pred_energy).mean()


def cross_gradient_block_loss(
    log_pred_blocks: torch.Tensor,
    seismic_hp_blocks: torch.Tensor,
    event_mask_blocks: torch.Tensor,
    beta: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    if log_pred_blocks.shape[1] < 2 or log_pred_blocks.shape[-1] < 2:
        return torch.zeros((), device=log_pred_blocks.device, dtype=log_pred_blocks.dtype)
    with torch.amp.autocast(device_type=log_pred_blocks.device.type, enabled=False):
        log_pred = log_pred_blocks.float().squeeze(2)
        seismic_hp = seismic_hp_blocks.float().squeeze(2)
        event_mask = event_mask_blocks.float().squeeze(2)

        dZ_dx = torch.diff(log_pred, dim=1)
        dZ_dx = torch.cat([dZ_dx, dZ_dx[:, -1:, :]], dim=1)
        dZ_dt = torch.diff(log_pred, dim=2)
        dZ_dt = torch.cat([dZ_dt, dZ_dt[:, :, -1:]], dim=2)

        dS_dx = torch.diff(seismic_hp, dim=1)
        dS_dx = torch.cat([dS_dx, dS_dx[:, -1:, :]], dim=1)
        dS_dt = torch.diff(seismic_hp, dim=2)
        dS_dt = torch.cat([dS_dt, dS_dt[:, :, -1:]], dim=2)

        cg = dZ_dx * dS_dt - dZ_dt * dS_dx

        event_norm = event_mask / (event_mask.amax(dim=(1, 2), keepdim=True) + eps)
        event_norm = event_norm.detach()

        abs_dS_dx = torch.abs(dS_dx).detach()
        scale = torch.quantile(abs_dS_dx.flatten(1), 0.95, dim=1, keepdim=True).view(-1, 1, 1)
        w_fault_relax = torch.exp(-beta * abs_dS_dx / (scale + eps))
        weight = event_norm * w_fault_relax
        return torch.mean(weight * torch.abs(cg))


def wavelet_regularization(current_wavelet: torch.Tensor, initial_wavelet: torch.Tensor) -> torch.Tensor:
    with torch.amp.autocast(device_type=current_wavelet.device.type, enabled=False):
        current = current_wavelet.float().squeeze(0).squeeze(0)
        initial = initial_wavelet.float().squeeze(0).squeeze(0).to(device=current.device, dtype=current.dtype)
        energy_loss = (current.norm(p=2) - initial.norm(p=2)).pow(2)
        current_mag = torch.abs(torch.fft.rfft(current))
        initial_mag = torch.abs(torch.fft.rfft(initial))
        current_mag = current_mag / (current_mag.amax() + 1e-6)
        initial_mag = initial_mag / (initial_mag.amax() + 1e-6)
        return energy_loss + F.mse_loss(current_mag, initial_mag)


def load_checkpoint_state(checkpoint_path: Path, device: torch.device) -> dict:
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        payload = {"model": payload}
    return payload


def load_model_weights(model: InversionNet, checkpoint_path: Path, device: torch.device) -> dict:
    payload = load_checkpoint_state(checkpoint_path, device)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state_dict, strict=True)
    return payload


def checkpoint_payload(
    model: InversionNet,
    wavelet_module: LearnableWavelet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    best_val: float,
    best_pcc: float,
    history: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "wavelet": wavelet_module.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "best_pcc": best_pcc,
        "history": history,
    }


def set_wavelet_trainability(
    wavelet_module: LearnableWavelet,
    epoch: int,
    freeze_epochs: int,
    experiment_mode: str,
) -> bool:
    train_phase = experiment_mode in {"core_v11", "core_v11_tuned", "core_v12_tuned", "core_v12_refine"} and epoch > freeze_epochs
    wavelet_module.phi.requires_grad_(train_phase)
    wavelet_module.log_f_scale.requires_grad_(False)
    wavelet_module.gain.requires_grad_(False)
    return train_phase


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
        outputs.append(log_pred.squeeze(1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def compute_v11_losses(
    model: InversionNet,
    wavelet_module: LearnableWavelet,
    forward_model: DynamicForwardModel,
    delta_smoother: FixedGaussian1D,
    lowpass_smoother: FixedGaussian1D,
    features: torch.Tensor,
    seismic: torch.Tensor,
    base_log: torch.Tensor,
    prior_weight: torch.Tensor,
    epoch: int,
    schedule_epoch: int,
    args,
):
    raw_delta = model(features)
    smooth_delta = delta_smoother(raw_delta)
    bounded_delta = 0.35 * torch.tanh(smooth_delta)
    log_pred = base_log + bounded_delta
    impedance = torch.exp(log_pred)
    wavelet = wavelet_module()
    synth, reflectivity = forward_model(impedance, wavelet, return_reflectivity=True)

    event_guide = seismic_event_guide(features[:, 1:2, :])
    amp_loss, synth_aligned = scale_invariant_huber_loss(synth, seismic)
    resid_guide = residual_guide(seismic, synth_aligned)
    relaxed_prior_weight = prior_weight
    relaxed_prior_weight = relaxed_prior_weight * torch.clamp(1.0 - args.prior_relax * event_guide, min=0.25)
    relaxed_prior_weight = relaxed_prior_weight * torch.clamp(1.0 - args.residual_relax * resid_guide, min=0.25)

    loss_prior = weighted_lowpass_prior_loss(log_pred, base_log, relaxed_prior_weight, lowpass_smoother)
    loss_tv = delta_tv_loss(bounded_delta)
    delta_region_weight = torch.clamp(1.0 - args.delta_relax * event_guide, min=0.20)
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
        edge_gain_ratio=args.edge_gain_ratio,
        variance_floor_ratio=args.variance_floor_ratio,
    )
    loss_refl_fp = reflectivity_precision_loss(reflectivity, event_guide)
    loss_wavelet_reg = wavelet_regularization(wavelet, wavelet_module.initial_wavelet)
    loss_grad = gradient_consistency_loss(synth.float(), seismic.float())
    loss_stft_legacy = stft_logmag_l1(synth.float(), seismic.float())
    loss_display_l1, loss_display_grad, loss_display_corr = display_similarity_losses(log_pred, seismic, event_guide)

    data_warmup_epochs = getattr(args, "data_warmup_epochs", 10)
    struct_warmup_epochs = getattr(args, "struct_warmup_epochs", 12)
    display_warmup_epochs = getattr(args, "display_warmup_epochs", 6)
    w_data = min(1.0, float(epoch) / float(max(data_warmup_epochs, 1)))
    w_struct = min(1.0, float(schedule_epoch) / float(max(struct_warmup_epochs, 1)))
    w_display = min(1.0, float(schedule_epoch) / float(max(display_warmup_epochs, 1)))
    if args.experiment_mode == "amp_only":
        loss_corr = multiscale_pearson_loss(synth, seismic)
        total = (
            w_data * (1.0 * loss_corr + args.amp_loss_weight * amp_loss + 0.25 * loss_stft_legacy + 0.25 * loss_grad)
            + 0.20 * loss_prior
            + w_struct * (
                args.structure_weight * loss_struct_corr
                + args.structure_l1_weight * loss_struct_l1
                + args.structure_bg_weight * loss_struct_bg
                + args.structure_gain_weight * loss_struct_gain
                + args.variance_floor_weight * loss_var_floor
            )
            + 0.02 * loss_tv
            + 0.02 * loss_delta
        )
        stats = {
            "corr": loss_corr,
            "lncc": torch.zeros_like(loss_struct_corr),
            "amp": amp_loss,
            "stft": loss_stft_legacy,
            "grad": loss_grad,
            "prior": loss_prior,
            "tv": loss_tv,
            "delta": loss_delta,
            "struct_corr": loss_struct_corr,
            "struct_l1": loss_struct_l1,
            "struct_bg": loss_struct_bg,
            "struct_gain": loss_struct_gain,
            "var_floor": loss_var_floor,
            "refl_fp": torch.zeros_like(loss_struct_corr),
            "wavelet_reg": torch.zeros_like(loss_struct_corr),
            "w_data": torch.tensor(w_data, device=features.device),
            "w_struct": torch.tensor(w_struct, device=features.device),
            "display_l1": torch.zeros_like(loss_struct_corr),
            "display_grad": torch.zeros_like(loss_struct_corr),
            "display_corr": torch.zeros_like(loss_struct_corr),
            "w_display": torch.zeros_like(loss_struct_corr),
        }
    elif args.experiment_mode == "core_v11":
        loss_lncc = multiscale_lncc_loss(synth, seismic, args.lncc_windows)
        loss_stft = stft_logmag_l1_centered(synth, seismic)
        total = (
            w_data * (1.0 * loss_lncc + args.amp_loss_weight * amp_loss + args.stft_weight * loss_stft)
            + 0.20 * loss_prior
            + w_struct * (
                0.04 * loss_struct_corr
                + 0.02 * loss_struct_l1
                + 0.02 * loss_struct_bg
                + 0.05 * loss_struct_gain
                + 0.06 * loss_var_floor
                + args.reflectivity_fp_weight * loss_refl_fp
            )
            + 0.02 * loss_tv
            + 0.02 * loss_delta
            + 0.01 * loss_wavelet_reg
        )
        stats = {
            "corr": torch.zeros_like(loss_struct_corr),
            "lncc": loss_lncc,
            "amp": amp_loss,
            "stft": loss_stft,
            "grad": loss_grad,
            "prior": loss_prior,
            "tv": loss_tv,
            "delta": loss_delta,
            "struct_corr": loss_struct_corr,
            "struct_l1": loss_struct_l1,
            "struct_bg": loss_struct_bg,
            "struct_gain": loss_struct_gain,
            "var_floor": loss_var_floor,
            "refl_fp": loss_refl_fp,
            "wavelet_reg": loss_wavelet_reg,
            "w_data": torch.tensor(w_data, device=features.device),
            "w_struct": torch.tensor(w_struct, device=features.device),
            "display_l1": torch.zeros_like(loss_struct_corr),
            "display_grad": torch.zeros_like(loss_struct_corr),
            "display_corr": torch.zeros_like(loss_struct_corr),
            "w_display": torch.zeros_like(loss_struct_corr),
        }
    elif args.experiment_mode == "core_v11_similarity":
        loss_corr = multiscale_pearson_loss(synth, seismic)
        loss_lncc = multiscale_lncc_loss(synth, seismic, args.lncc_windows)
        loss_stft = stft_logmag_l1_scale_invariant(synth_aligned, seismic)
        total = (
            w_data * (
                args.tuned_pcc_weight * loss_corr
                + args.tuned_lncc_weight * loss_lncc
                + args.amp_loss_weight * amp_loss
                + args.tuned_stft_weight * loss_stft
            )
            + 0.20 * loss_prior
            + w_struct * (
                args.tuned_structure_corr_weight * loss_struct_corr
                + args.tuned_structure_l1_weight * loss_struct_l1
                + args.tuned_structure_bg_weight * loss_struct_bg
                + args.tuned_structure_gain_weight * loss_struct_gain
                + args.tuned_variance_floor_weight * loss_var_floor
                + args.tuned_reflectivity_fp_weight * loss_refl_fp
            )
            + w_display * (
                args.display_l1_weight * loss_display_l1
                + args.display_grad_weight * loss_display_grad
                + args.display_corr_weight * loss_display_corr
            )
            + 0.02 * loss_tv
            + 0.02 * loss_delta
            + args.tuned_wavelet_reg_weight * loss_wavelet_reg
        )
        stats = {
            "corr": loss_corr,
            "lncc": loss_lncc,
            "amp": amp_loss,
            "stft": loss_stft,
            "grad": loss_grad,
            "prior": loss_prior,
            "tv": loss_tv,
            "delta": loss_delta,
            "struct_corr": loss_struct_corr,
            "struct_l1": loss_struct_l1,
            "struct_bg": loss_struct_bg,
            "struct_gain": loss_struct_gain,
            "var_floor": loss_var_floor,
            "refl_fp": loss_refl_fp,
            "wavelet_reg": loss_wavelet_reg,
            "w_data": torch.tensor(w_data, device=features.device),
            "w_struct": torch.tensor(w_struct, device=features.device),
            "display_l1": loss_display_l1,
            "display_grad": loss_display_grad,
            "display_corr": loss_display_corr,
            "w_display": torch.tensor(w_display, device=features.device),
        }
    else:
        loss_corr = multiscale_pearson_loss(synth, seismic)
        loss_lncc = multiscale_lncc_loss(synth, seismic, args.lncc_windows)
        loss_stft = stft_logmag_l1_scale_invariant(synth_aligned, seismic)
        total = (
            w_data * (
                args.tuned_pcc_weight * loss_corr
                + args.tuned_lncc_weight * loss_lncc
                + args.amp_loss_weight * amp_loss
                + args.tuned_stft_weight * loss_stft
            )
            + 0.20 * loss_prior
            + w_struct * (
                args.tuned_structure_corr_weight * loss_struct_corr
                + args.tuned_structure_l1_weight * loss_struct_l1
                + args.tuned_structure_bg_weight * loss_struct_bg
                + args.tuned_structure_gain_weight * loss_struct_gain
                + args.tuned_variance_floor_weight * loss_var_floor
                + args.tuned_reflectivity_fp_weight * loss_refl_fp
            )
            + 0.02 * loss_tv
            + 0.02 * loss_delta
            + args.tuned_wavelet_reg_weight * loss_wavelet_reg
        )
        stats = {
            "corr": loss_corr,
            "lncc": loss_lncc,
            "amp": amp_loss,
            "stft": loss_stft,
            "grad": loss_grad,
            "prior": loss_prior,
            "tv": loss_tv,
            "delta": loss_delta,
            "struct_corr": loss_struct_corr,
            "struct_l1": loss_struct_l1,
            "struct_bg": loss_struct_bg,
            "struct_gain": loss_struct_gain,
            "var_floor": loss_var_floor,
            "refl_fp": loss_refl_fp,
            "wavelet_reg": loss_wavelet_reg,
            "w_data": torch.tensor(w_data, device=features.device),
            "w_struct": torch.tensor(w_struct, device=features.device),
            "display_l1": torch.zeros_like(loss_struct_corr),
            "display_grad": torch.zeros_like(loss_struct_corr),
            "display_corr": torch.zeros_like(loss_struct_corr),
            "w_display": torch.zeros_like(loss_struct_corr),
        }
    stats["total"] = total
    aux = {
        "bounded_delta": bounded_delta,
        "base_log": base_log,
        "reflectivity": reflectivity,
        "event_guide": event_guide,
        "seismic_hp": features[:, 1:2, :],
        "log_pred": log_pred,
    }
    return total, stats, synth, wavelet, aux


def init_epoch_stats() -> dict:
    return {
        "total": 0.0,
        "corr": 0.0,
        "lncc": 0.0,
        "amp": 0.0,
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
        "refl_fp": 0.0,
        "lat_delta": 0.0,
        "refl_floor": 0.0,
        "cross_grad": 0.0,
        "wavelet_reg": 0.0,
        "display_l1": 0.0,
        "display_grad": 0.0,
        "display_corr": 0.0,
        "pcc": 0.0,
        "w_data": 0.0,
        "w_struct": 0.0,
        "w_lat": 0.0,
        "w_display": 0.0,
        "wavelet_phi_deg": 0.0,
    }


def train_one_epoch(
    model: InversionNet,
    wavelet_module: LearnableWavelet,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    device: torch.device,
    forward_model: DynamicForwardModel,
    delta_smoother: FixedGaussian1D,
    lowpass_smoother: FixedGaussian1D,
    epoch: int,
    schedule_epoch: int,
    grad_clip: float,
    args,
) -> dict:
    model.train()
    stats = init_epoch_stats()
    for features, seismic, base_log, prior_weight in loader:
        features = features.to(device, non_blocking=True)
        seismic = seismic.to(device, non_blocking=True)
        base_log = base_log.to(device, non_blocking=True)
        prior_weight = prior_weight.to(device, non_blocking=True)
        is_block_batch = features.ndim == 4
        if is_block_batch:
            batch_size, group_size, channels, nsamples = features.shape
            features_flat = features.reshape(batch_size * group_size, channels, nsamples)
            seismic_flat = seismic.reshape(batch_size * group_size, seismic.shape[2], seismic.shape[3])
            base_log_flat = base_log.reshape(batch_size * group_size, base_log.shape[2], base_log.shape[3])
            prior_weight_flat = prior_weight.reshape(batch_size * group_size, prior_weight.shape[2], prior_weight.shape[3])
        else:
            features_flat = features
            seismic_flat = seismic
            base_log_flat = base_log
            prior_weight_flat = prior_weight
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            total, batch_stats, synth, _, aux = compute_v11_losses(
                model=model,
                wavelet_module=wavelet_module,
                forward_model=forward_model,
                delta_smoother=delta_smoother,
                lowpass_smoother=lowpass_smoother,
                features=features_flat,
                seismic=seismic_flat,
                base_log=base_log_flat,
                prior_weight=prior_weight_flat,
                epoch=epoch,
                schedule_epoch=schedule_epoch,
                args=args,
            )
            if args.experiment_mode in {"core_v12_tuned", "core_v12_refine"} and is_block_batch:
                bounded_delta_blocks = aux["bounded_delta"].reshape(batch_size, group_size, 1, nsamples)
                reflectivity_blocks = aux["reflectivity"].reshape(batch_size, group_size, 1, nsamples)
                base_log_blocks = aux["base_log"].reshape(batch_size, group_size, 1, nsamples)
                seismic_hp_blocks = aux["seismic_hp"].reshape(batch_size, group_size, 1, nsamples)
                event_mask_blocks = aux["event_guide"].reshape(batch_size, group_size, 1, nsamples)
                loss_lat_delta = lateral_delta_block_loss(
                    bounded_delta_blocks,
                    seismic_hp_blocks,
                    event_mask_blocks,
                    beta=args.lateral_beta,
                )
                loss_refl_floor = reflectivity_floor_block_loss(
                    reflectivity_blocks,
                    base_log_blocks,
                    event_mask_blocks,
                    floor_ratio=args.reflectivity_floor_ratio,
                )
                if args.experiment_mode == "core_v12_refine":
                    w_lat = ramp_weight(schedule_epoch, args.lateral_warmup_start, args.lateral_warmup_epochs)
                    w_lat_tensor = torch.tensor(w_lat, device=features_flat.device, dtype=total.dtype)
                    log_pred_blocks = aux["log_pred"].reshape(batch_size, group_size, 1, nsamples)
                    loss_cross_grad = cross_gradient_block_loss(
                        log_pred_blocks,
                        seismic_hp_blocks,
                        event_mask_blocks,
                        beta=args.cross_grad_beta,
                    )
                    total = total + w_lat_tensor * (
                        args.lateral_delta_weight * loss_lat_delta
                        + args.reflectivity_floor_weight * loss_refl_floor
                        + args.cross_grad_weight * loss_cross_grad
                    )
                else:
                    w_lat_tensor = torch.ones((), device=features_flat.device, dtype=total.dtype)
                    loss_cross_grad = torch.zeros_like(total)
                    total = total + args.lateral_delta_weight * loss_lat_delta + args.reflectivity_floor_weight * loss_refl_floor
                batch_stats["lat_delta"] = loss_lat_delta
                batch_stats["refl_floor"] = loss_refl_floor
                batch_stats["cross_grad"] = loss_cross_grad
                batch_stats["w_lat"] = w_lat_tensor
                batch_stats["total"] = total
            else:
                zero_like = torch.zeros_like(batch_stats["total"])
                batch_stats["lat_delta"] = zero_like
                batch_stats["refl_floor"] = zero_like
                batch_stats["cross_grad"] = zero_like
                batch_stats["w_lat"] = zero_like
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if wavelet_module.phi.requires_grad:
            torch.nn.utils.clip_grad_norm_([wavelet_module.phi], 1.0)
        scaler.step(optimizer)
        scaler.update()

        stats["total"] += float(total.item())
        for key in (
            "corr",
            "lncc",
            "amp",
            "stft",
            "grad",
            "prior",
            "tv",
            "delta",
            "struct_corr",
            "struct_l1",
            "struct_bg",
            "struct_gain",
            "var_floor",
            "refl_fp",
            "lat_delta",
            "refl_floor",
            "cross_grad",
            "wavelet_reg",
            "display_l1",
            "display_grad",
            "display_corr",
            "w_data",
            "w_struct",
            "w_lat",
            "w_display",
        ):
            stats[key] += float(batch_stats[key].item())
        stats["pcc"] += float(per_trace_corr(synth, seismic_flat).mean().item())
        stats["wavelet_phi_deg"] += float(wavelet_module.summary()["phi_deg"])
    for key in stats:
        stats[key] /= max(len(loader), 1)
    return stats


@torch.no_grad()
def validate_one_epoch(
    model: InversionNet,
    wavelet_module: LearnableWavelet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    forward_model: DynamicForwardModel,
    delta_smoother: FixedGaussian1D,
    lowpass_smoother: FixedGaussian1D,
    epoch: int,
    schedule_epoch: int,
    args,
) -> dict:
    model.eval()
    stats = init_epoch_stats()
    for features, seismic, base_log, prior_weight in loader:
        features = features.to(device, non_blocking=True)
        seismic = seismic.to(device, non_blocking=True)
        base_log = base_log.to(device, non_blocking=True)
        prior_weight = prior_weight.to(device, non_blocking=True)
        is_block_batch = features.ndim == 4
        if is_block_batch:
            batch_size, group_size, channels, nsamples = features.shape
            features_flat = features.reshape(batch_size * group_size, channels, nsamples)
            seismic_flat = seismic.reshape(batch_size * group_size, seismic.shape[2], seismic.shape[3])
            base_log_flat = base_log.reshape(batch_size * group_size, base_log.shape[2], base_log.shape[3])
            prior_weight_flat = prior_weight.reshape(batch_size * group_size, prior_weight.shape[2], prior_weight.shape[3])
        else:
            features_flat = features
            seismic_flat = seismic
            base_log_flat = base_log
            prior_weight_flat = prior_weight
        total, batch_stats, synth, _, aux = compute_v11_losses(
            model=model,
            wavelet_module=wavelet_module,
            forward_model=forward_model,
            delta_smoother=delta_smoother,
            lowpass_smoother=lowpass_smoother,
            features=features_flat,
            seismic=seismic_flat,
            base_log=base_log_flat,
            prior_weight=prior_weight_flat,
            epoch=epoch,
            schedule_epoch=schedule_epoch,
            args=args,
        )
        if args.experiment_mode in {"core_v12_tuned", "core_v12_refine"} and is_block_batch:
            bounded_delta_blocks = aux["bounded_delta"].reshape(batch_size, group_size, 1, nsamples)
            reflectivity_blocks = aux["reflectivity"].reshape(batch_size, group_size, 1, nsamples)
            base_log_blocks = aux["base_log"].reshape(batch_size, group_size, 1, nsamples)
            seismic_hp_blocks = aux["seismic_hp"].reshape(batch_size, group_size, 1, nsamples)
            event_mask_blocks = aux["event_guide"].reshape(batch_size, group_size, 1, nsamples)
            loss_lat_delta = lateral_delta_block_loss(
                bounded_delta_blocks,
                seismic_hp_blocks,
                event_mask_blocks,
                beta=args.lateral_beta,
            )
            loss_refl_floor = reflectivity_floor_block_loss(
                reflectivity_blocks,
                base_log_blocks,
                event_mask_blocks,
                floor_ratio=args.reflectivity_floor_ratio,
            )
            if args.experiment_mode == "core_v12_refine":
                w_lat = ramp_weight(schedule_epoch, args.lateral_warmup_start, args.lateral_warmup_epochs)
                w_lat_tensor = torch.tensor(w_lat, device=features_flat.device, dtype=total.dtype)
                log_pred_blocks = aux["log_pred"].reshape(batch_size, group_size, 1, nsamples)
                loss_cross_grad = cross_gradient_block_loss(
                    log_pred_blocks,
                    seismic_hp_blocks,
                    event_mask_blocks,
                    beta=args.cross_grad_beta,
                )
                total = total + w_lat_tensor * (
                    args.lateral_delta_weight * loss_lat_delta
                    + args.reflectivity_floor_weight * loss_refl_floor
                    + args.cross_grad_weight * loss_cross_grad
                )
            else:
                w_lat_tensor = torch.ones((), device=features_flat.device, dtype=total.dtype)
                loss_cross_grad = torch.zeros_like(total)
                total = total + args.lateral_delta_weight * loss_lat_delta + args.reflectivity_floor_weight * loss_refl_floor
            batch_stats["lat_delta"] = loss_lat_delta
            batch_stats["refl_floor"] = loss_refl_floor
            batch_stats["cross_grad"] = loss_cross_grad
            batch_stats["w_lat"] = w_lat_tensor
            batch_stats["total"] = total
        else:
            zero_like = torch.zeros_like(batch_stats["total"])
            batch_stats["lat_delta"] = zero_like
            batch_stats["refl_floor"] = zero_like
            batch_stats["cross_grad"] = zero_like
            batch_stats["w_lat"] = zero_like
        stats["total"] += float(total.item())
        for key in (
            "corr",
            "lncc",
            "amp",
            "stft",
            "grad",
            "prior",
            "tv",
            "delta",
            "struct_corr",
            "struct_l1",
            "struct_bg",
            "struct_gain",
            "var_floor",
            "refl_fp",
            "lat_delta",
            "refl_floor",
            "cross_grad",
            "wavelet_reg",
            "display_l1",
            "display_grad",
            "display_corr",
            "w_data",
            "w_struct",
            "w_lat",
            "w_display",
        ):
            stats[key] += float(batch_stats[key].item())
        stats["pcc"] += float(per_trace_corr(synth, seismic_flat).mean().item())
        stats["wavelet_phi_deg"] += float(wavelet_module.summary()["phi_deg"])
    for key in stats:
        stats[key] /= max(len(loader), 1)
    return stats


def scale_invariant_metrics_np(pred: np.ndarray, target: np.ndarray) -> Tuple[float, float]:
    num = np.sum(pred * target, axis=1, keepdims=True)
    den = np.clip(np.sum(pred * pred, axis=1, keepdims=True), 1e-6, None)
    aligned = pred * (num / den)
    residual = target - aligned
    target_scale = np.sqrt(np.mean(target ** 2, axis=1, keepdims=True) + 1e-6)
    resid_l1_si = float(np.mean(np.abs(residual) / target_scale))
    resid_rms_ratio = float(np.sqrt(np.mean(residual ** 2)) / (np.sqrt(np.mean(target ** 2)) + 1e-6))
    return resid_l1_si, resid_rms_ratio


def lateral_tv_ratio(section: np.ndarray, base_section: np.ndarray) -> float:
    log_section = np.log(np.clip(section, 1e5, None))
    log_base = np.log(np.clip(base_section, 1e5, None))
    pred_lat = float(np.mean(np.abs(np.diff(log_section, axis=0))))
    base_lat = float(np.mean(np.abs(np.diff(log_base, axis=0))))
    return pred_lat / (base_lat + 1e-6)


def weighted_lateral_tv_ratio_np(
    section: np.ndarray,
    base_section: np.ndarray,
    observed: np.ndarray,
    fs_hz: float,
    cutoff_hz: float,
    beta: float,
) -> float:
    if section.shape[0] < 2 or base_section.shape[0] < 2:
        return 1.0
    seismic_center = observed - observed.mean(axis=1, keepdims=True)
    seismic_hp = highpass_filter(seismic_center, cutoff_hz=cutoff_hz, fs_hz=fs_hz)
    dx = np.abs(np.diff(seismic_hp, axis=0)).astype(np.float32)
    dx_scale = float(np.percentile(dx, 95.0)) + 1e-6
    weight = np.exp(-beta * dx / dx_scale).astype(np.float32)

    log_section = np.log(np.clip(section, 1e5, None))
    log_base = np.log(np.clip(base_section, 1e5, None))
    pred_dx = np.abs(np.diff(log_section, axis=0)).astype(np.float32)
    base_dx = np.abs(np.diff(log_base, axis=0)).astype(np.float32)

    pred_lat = float(np.sum(weight * pred_dx) / (np.sum(weight) + 1e-6))
    base_lat = float(np.sum(weight * base_dx) / (np.sum(weight) + 1e-6))
    if pred_lat < 1e-8 and base_lat < 1e-8:
        return 1.0
    return pred_lat / (base_lat + 1e-6)


def display_similarity_metrics_np(observed: np.ndarray, section: np.ndarray) -> Tuple[float, float, float]:
    obs_disp = shared_visual_map(observed, kind="seismic").astype(np.float64)
    sec_disp = shared_visual_map(section, kind="impedance").astype(np.float64)
    obs_flat = obs_disp.ravel()
    sec_flat = sec_disp.ravel()
    obs_std = float(np.std(obs_flat))
    sec_std = float(np.std(sec_flat))
    if obs_std < 1e-12 or sec_std < 1e-12:
        pearson = 0.0
    else:
        pearson = float(np.corrcoef(obs_flat, sec_flat)[0, 1])
        if not np.isfinite(pearson):
            pearson = 0.0
    cosine = float(np.dot(obs_flat, sec_flat) / ((np.linalg.norm(obs_flat) * np.linalg.norm(sec_flat)) + 1e-12))
    agreement = float(1.0 - np.mean(np.abs(obs_disp - sec_disp)) / 2.0)
    return pearson, cosine, agreement


def display_direct_match_impedance(
    observed: np.ndarray,
    base_impedance: np.ndarray,
    clip_value: float = 0.999999,
) -> np.ndarray:
    obs_disp = shared_visual_map(observed, kind="seismic").astype(np.float64)
    centered_log = np.arctanh(np.clip(obs_disp, -clip_value, clip_value))
    base_log = np.log(np.clip(base_impedance, 1e5, None)).astype(np.float64)
    base_median = np.median(base_log, axis=1, keepdims=True)
    return np.exp(base_median + centered_log).astype(np.float32)


def display_trace_bridge_impedance(
    observed: np.ndarray,
    raw_impedance: np.ndarray,
    base_impedance: np.ndarray,
    alpha_max: float = 0.41,
    power: float = 1.40,
) -> np.ndarray:
    raw_log = np.log(np.clip(raw_impedance, 1e5, None)).astype(np.float64)
    direct_log = np.log(
        np.clip(
            display_direct_match_impedance(
                observed=observed,
                base_impedance=base_impedance,
            ),
            1e5,
            None,
        )
    ).astype(np.float64)
    obs_disp = shared_visual_map(observed, kind="seismic").astype(np.float64)
    raw_disp = shared_visual_map(raw_impedance, kind="impedance").astype(np.float64)
    trace_diff = np.mean(np.abs(obs_disp - raw_disp), axis=1, keepdims=True)
    trace_scale = float(np.max(trace_diff)) + 1e-6
    alpha = alpha_max * np.power(trace_diff / trace_scale, power)
    alpha = np.clip(alpha, 0.0, alpha_max)
    return np.exp(raw_log + alpha * (direct_log - raw_log)).astype(np.float32)


def guided_lateral_delta_blend(
    raw_impedance: np.ndarray,
    base_impedance: np.ndarray,
    observed: np.ndarray,
    fs_hz: float,
    cutoff_hz: float,
    lam: float,
    gamma: float,
    iterations: int = 1,
) -> np.ndarray:
    log_raw = np.log(np.clip(raw_impedance, 1e5, None))
    log_base = np.log(np.clip(base_impedance, 1e5, None))
    delta = log_raw - log_base
    seismic_center = observed - observed.mean(axis=1, keepdims=True)
    seismic_hp = highpass_filter(seismic_center, cutoff_hz=cutoff_hz, fs_hz=fs_hz)
    seismic_hp = gaussian_filter(seismic_hp, sigma=(0.6, 1.0))
    lateral_diff = np.abs(np.diff(seismic_hp, axis=0))
    diff_scale = float(np.percentile(lateral_diff, 75.0)) + 1e-6
    sim = np.exp(-gamma * lateral_diff / diff_scale).astype(np.float32)
    event = np.abs(gaussian_filter(seismic_hp, sigma=(0.0, 1.0)))
    event_scale = float(np.percentile(event, 95.0)) + 1e-6
    event = np.clip(event / event_scale, 0.0, 1.0).astype(np.float32)
    for _ in range(max(int(iterations), 1)):
        left = np.concatenate([delta[:1], delta[:-1]], axis=0)
        right = np.concatenate([delta[1:], delta[-1:]], axis=0)
        left_w = np.concatenate([np.zeros_like(sim[:1]), sim], axis=0)
        right_w = np.concatenate([sim, np.zeros_like(sim[:1])], axis=0)
        relax = 1.0 - 0.35 * event
        left_w *= relax
        right_w *= relax
        delta = (delta + lam * (left_w * left + right_w * right)) / (1.0 + lam * (left_w + right_w) + 1e-6)
    return np.exp(log_base + delta).astype(np.float32)


def generate_extended_postprocess_candidates(
    raw_impedance: np.ndarray,
    base_impedance: np.ndarray,
    observed: np.ndarray,
    fs_hz: float,
    cutoff_hz: float,
) -> List[Tuple[str, np.ndarray]]:
    candidates = list(generate_postprocess_candidates(raw_impedance))
    for lam, gamma, iterations in ((0.12, 2.4, 1), (0.20, 2.8, 1), (0.18, 2.8, 2)):
        guided = guided_lateral_delta_blend(
            raw_impedance=raw_impedance,
            base_impedance=base_impedance,
            observed=observed,
            fs_hz=fs_hz,
            cutoff_hz=cutoff_hz,
            lam=lam,
            gamma=gamma,
            iterations=iterations,
        )
        candidates.append((f"guided_lat_l{lam:.2f}_g{gamma:.1f}_i{iterations}", guided))
    return candidates


def wavelet_penalty(phi_deg: float, f_scale: float) -> float:
    return 1.0 if abs(phi_deg) > 60.0 or f_scale < 0.85 or f_scale > 1.15 else 0.0


def candidate_selection_score(metrics: dict) -> float:
    s_pcc = float(metrics["mean_trace_pcc"])
    s_res = math.exp(-2.5 * float(metrics["resid_l1_si"]))
    std_imp = max(float(metrics["final_impedance_std"]), 1e-6)
    std_base = max(float(metrics["base_impedance_std"]), 1e-6)
    s_var = math.exp(-abs(math.log(std_imp / std_base)))
    s_dev = max(0.0, min(1.0, (float(metrics["mae_ratio_vs_prior"]) - 0.03) / 0.17))
    lat_metric = float(metrics.get("lat_tv_ratio_weighted", metrics["lat_tv_ratio"]))
    s_lat = math.exp(-abs(math.log(max(lat_metric, 1e-6))))
    return float(
        0.35 * s_pcc
        + 0.20 * s_res
        + 0.15 * float(metrics["edge_alignment"])
        + 0.10 * s_var
        + 0.10 * s_dev
        + 0.10 * s_lat
        - 0.10 * float(metrics["wavelet_bad"])
    )


def candidate_score_components(metrics: dict) -> dict:
    std_imp = max(float(metrics["final_impedance_std"]), 1e-6)
    std_base = max(float(metrics["base_impedance_std"]), 1e-6)
    lat_metric = float(metrics.get("lat_tv_ratio_weighted", metrics["lat_tv_ratio"]))
    return {
        "S_pcc": float(metrics["mean_trace_pcc"]),
        "S_res": math.exp(-2.5 * float(metrics["resid_l1_si"])),
        "S_edge": float(metrics["edge_alignment"]),
        "S_var": math.exp(-abs(math.log(std_imp / std_base))),
        "S_dev": max(0.0, min(1.0, (float(metrics["mae_ratio_vs_prior"]) - 0.03) / 0.17)),
        "S_lat": math.exp(-abs(math.log(max(lat_metric, 1e-6)))),
    }


def display_selection_score(
    metrics: dict,
    agreement_weight: float = 0.35,
    cosine_weight: float = 0.15,
    pearson_weight: float = 0.10,
    pcc_weight: float = 0.20,
    resid_weight: float = 0.15,
    edge_weight: float = 0.10,
) -> float:
    s_display_agreement = float(metrics["display_agreement"])
    s_display_cosine = max(0.0, min(1.0, 0.5 * (float(metrics.get("display_cosine", 0.0)) + 1.0)))
    pearson = float(metrics.get("display_pearson", 0.0))
    s_display_pearson = max(0.0, min(1.0, 0.5 * (pearson + 1.0))) if math.isfinite(pearson) else 0.0
    return float(
        agreement_weight * s_display_agreement
        + cosine_weight * s_display_cosine
        + pearson_weight * s_display_pearson
        + pcc_weight * float(metrics["mean_trace_pcc"])
        + resid_weight * math.exp(-2.5 * float(metrics["resid_l1_si"]))
        + edge_weight * float(metrics["edge_alignment"])
    )


def summarize_wavelet_tensor(wavelet_module: LearnableWavelet, wavelet: torch.Tensor) -> dict:
    summary = wavelet_module.summary()
    wavelet_np = wavelet.detach().cpu().numpy().reshape(-1)
    summary["energy_l2"] = float(np.linalg.norm(wavelet_np))
    summary["max_abs"] = float(np.max(np.abs(wavelet_np)))
    summary["wavelet_bad"] = wavelet_penalty(summary["phi_deg"], summary["f_scale"])
    return summary


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
    forward_model: DynamicForwardModel,
    wavelet_module: LearnableWavelet,
    lateral_score_beta: float,
    selection_mode: str,
    display_score_weights: dict,
) -> dict:
    imp_raw = np.exp(infer_log).astype(np.float32)
    base_imp = np.exp(infer_base_log).astype(np.float32)
    wavelet = wavelet_module().detach()
    wavelet_summary = summarize_wavelet_tensor(wavelet_module, wavelet)
    synth_raw = forward_model(
        torch.from_numpy(imp_raw[:, None, :]).to(device),
        wavelet.to(device),
    ).squeeze(1).cpu().numpy().astype(np.float32)
    raw_trace_pcc = mean_trace_corr_np(synth_raw, infer_seismic)
    postprocess_metrics: List[dict] = []
    best_post = None
    candidate_list = list(
        generate_extended_postprocess_candidates(
        raw_impedance=imp_raw,
        base_impedance=base_imp,
        observed=infer_seismic,
        fs_hz=fs_hz,
        cutoff_hz=highpass_cutoff,
        )
    )
    if selection_mode == "display":
        candidate_list.append(
            (
                "display_trace_bridge_am0.41_p1.40",
                display_trace_bridge_impedance(
                    observed=infer_seismic,
                    raw_impedance=imp_raw,
                    base_impedance=base_imp,
                    alpha_max=0.41,
                    power=1.40,
                ),
            )
        )
        candidate_list.append(
            (
                "display_direct_match",
                display_direct_match_impedance(
                    observed=infer_seismic,
                    base_impedance=base_imp,
                ),
            )
        )
    for post_name, imp_final in candidate_list:
        synth_final = forward_model(
            torch.from_numpy(imp_final[:, None, :]).to(device),
            wavelet.to(device),
        ).squeeze(1).cpu().numpy().astype(np.float32)
        resid_l1_si, resid_rms_ratio = scale_invariant_metrics_np(synth_final, infer_seismic)
        display_pearson, display_cosine, display_agreement = display_similarity_metrics_np(infer_seismic, imp_final)
        metrics = {
            "postprocess_name": post_name,
            "final_trace_pcc": mean_trace_corr_np(synth_final, infer_seismic),
            "final_impedance_mean": float(imp_final.mean()),
            "final_impedance_std": float(imp_final.std()),
            "base_impedance_std": float(base_imp.std()),
            "mae_ratio_vs_prior": float(np.mean(np.abs(imp_final - base_imp)) / (np.mean(base_imp) + 1e-6)),
            "edge_alignment": structure_alignment_score(imp_final, infer_seismic, fs_hz, highpass_cutoff),
            "resid_l1_si": resid_l1_si,
            "resid_rms_ratio": resid_rms_ratio,
            "lat_tv_ratio": lateral_tv_ratio(imp_final, base_imp),
            "lat_tv_ratio_weighted": weighted_lateral_tv_ratio_np(
                imp_final,
                base_imp,
                infer_seismic,
                fs_hz=fs_hz,
                cutoff_hz=highpass_cutoff,
                beta=lateral_score_beta,
            ),
            "display_pearson": display_pearson,
            "display_cosine": display_cosine,
            "display_agreement": display_agreement,
            "wavelet_phi_deg": float(wavelet_summary["phi_deg"]),
            "wavelet_f_scale": float(wavelet_summary["f_scale"]),
            "wavelet_bad": float(wavelet_summary["wavelet_bad"]),
        }
        metrics["selection_score"] = candidate_selection_score(
            {
                "mean_trace_pcc": metrics["final_trace_pcc"],
                "final_impedance_std": metrics["final_impedance_std"],
                "base_impedance_std": metrics["base_impedance_std"],
                "mae_ratio_vs_prior": metrics["mae_ratio_vs_prior"],
                "edge_alignment": metrics["edge_alignment"],
                "resid_l1_si": metrics["resid_l1_si"],
                "lat_tv_ratio": metrics["lat_tv_ratio"],
                "lat_tv_ratio_weighted": metrics["lat_tv_ratio_weighted"],
                "wavelet_bad": metrics["wavelet_bad"],
            }
        )
        metrics["display_selection_score"] = display_selection_score(
            {
                "display_agreement": metrics["display_agreement"],
                "mean_trace_pcc": metrics["final_trace_pcc"],
                "resid_l1_si": metrics["resid_l1_si"],
                "edge_alignment": metrics["edge_alignment"],
                "display_cosine": metrics["display_cosine"],
                "display_pearson": metrics["display_pearson"],
            }
            ,
            **display_score_weights,
        )
        postprocess_metrics.append(metrics)
        current_score = metrics["display_selection_score"] if selection_mode == "display" else metrics["selection_score"]
        best_score = (
            best_post["metrics"]["display_selection_score"]
            if best_post is not None and selection_mode == "display"
            else (best_post["metrics"]["selection_score"] if best_post is not None else None)
        )
        if best_post is None or current_score > best_score:
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
        "base_impedance_std": best_post["metrics"]["base_impedance_std"],
        "mae_ratio_vs_prior": best_post["metrics"]["mae_ratio_vs_prior"],
        "edge_alignment": best_post["metrics"]["edge_alignment"],
        "resid_l1_si": best_post["metrics"]["resid_l1_si"],
        "resid_rms_ratio": best_post["metrics"]["resid_rms_ratio"],
        "lat_tv_ratio": best_post["metrics"]["lat_tv_ratio"],
        "lat_tv_ratio_weighted": best_post["metrics"]["lat_tv_ratio_weighted"],
        "display_pearson": best_post["metrics"]["display_pearson"],
        "display_cosine": best_post["metrics"]["display_cosine"],
        "display_agreement": best_post["metrics"]["display_agreement"],
        "wavelet_phi_deg": best_post["metrics"]["wavelet_phi_deg"],
        "wavelet_f_scale": best_post["metrics"]["wavelet_f_scale"],
        "wavelet_bad": best_post["metrics"]["wavelet_bad"],
        "postprocess_candidates": postprocess_metrics,
    }
    metrics["score_components"] = candidate_score_components(metrics)
    metrics["selection_score"] = candidate_selection_score(metrics)
    metrics["display_selection_score"] = display_selection_score(metrics, **display_score_weights)
    return {
        "metrics": metrics,
        "infer_log": infer_log,
        "imp_raw": imp_raw,
        "imp_final": best_post["imp_final"],
        "synth": best_post["synth_final"],
        "synth_raw": synth_raw,
        "wavelet_np": wavelet.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "wavelet_summary": wavelet_summary,
    }


@torch.no_grad()
def evaluate_candidate_checkpoint(
    ckpt_name: str,
    ckpt_path: Path,
    model: InversionNet,
    wavelet_module: LearnableWavelet,
    infer_features: np.ndarray,
    infer_base_log: np.ndarray,
    infer_seismic: np.ndarray,
    infer_batch_size: int,
    fs_hz: float,
    highpass_cutoff: float,
    device: torch.device,
    delta_smoother: FixedGaussian1D,
    forward_model: DynamicForwardModel,
    lateral_score_beta: float,
    selection_mode: str,
    display_score_weights: dict,
) -> dict:
    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model"], strict=True)
    if "wavelet" in payload:
        wavelet_module.load_state_dict(payload["wavelet"], strict=True)
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
        wavelet_module=wavelet_module,
        lateral_score_beta=lateral_score_beta,
        selection_mode=selection_mode,
        display_score_weights=display_score_weights,
    )


def plot_training_history(history: dict, out_path: Path) -> None:
    epochs = np.arange(1, len(history["train_total"]) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(epochs, history["train_total"], label="train_total")
    ax1.plot(epochs, history["val_total"], label="val_total")
    ax1.plot(epochs, history["val_pcc"], label="val_pcc")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / PCC")
    ax1.set_title("Training History")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper left")
    if history.get("wavelet_phi_deg"):
        ax2 = ax1.twinx()
        ax2.plot(epochs, history["wavelet_phi_deg"], color="tab:red", label="phi_deg", alpha=0.7)
        ax2.set_ylabel("Wavelet Phase (deg)")
        ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def section_extent(section: np.ndarray, dt: float) -> List[float]:
    n_traces, n_samples = section.shape
    return [0.0, float(n_traces - 1), float(n_samples * dt * 1000.0), 0.0]


def shared_visual_map(section: np.ndarray, kind: str) -> np.ndarray:
    if kind == "seismic":
        centered = section.astype(np.float32)
        scale = float(np.percentile(np.abs(centered), 99.0)) + 1e-6
        mapped = np.tanh(centered / scale)
        return mapped.astype(np.float32)
    if kind == "impedance":
        log_section = np.log(np.clip(section, 1e5, None)).astype(np.float32)
        centered = log_section - np.median(log_section)
        scale = float(np.percentile(np.abs(centered), 99.0)) + 1e-6
        mapped = np.tanh(centered / scale)
        return mapped.astype(np.float32)
    raise ValueError(f"Unsupported visual map kind: {kind}")


def show_section(
    ax,
    section: np.ndarray,
    dt: float,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    ax.imshow(
        section.T,
        aspect="auto",
        cmap=cmap,
        origin="upper",
        extent=section_extent(section, dt),
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("Trace")
    ax.set_ylabel("Time (ms)")


def plot_original_vs_inversion(observed: np.ndarray, final_imp: np.ndarray, dt: float, out_path: Path) -> None:
    shared_cmap = "RdBu_r"
    obs_disp = shared_visual_map(observed, kind="seismic")
    imp_disp = shared_visual_map(final_imp, kind="impedance")
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    show_section(axes[0], obs_disp, dt, "Original Seismic (shared cmap)", shared_cmap, -1.0, 1.0)
    show_section(axes[1], imp_disp, dt, "Inverted Impedance (shared cmap)", shared_cmap, -1.0, 1.0)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_forward_diagnostic(
    observed: np.ndarray,
    synthetic: np.ndarray,
    final_imp: np.ndarray,
    dt: float,
    out_path: Path,
) -> None:
    residual = observed - synthetic
    shared_cmap = "RdBu_r"
    obs_disp = shared_visual_map(observed, kind="seismic")
    synth_disp = shared_visual_map(synthetic, kind="seismic")
    resid_disp = shared_visual_map(residual, kind="seismic")
    imp_disp = shared_visual_map(final_imp, kind="impedance")
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), constrained_layout=True)
    show_section(axes[0], obs_disp, dt, "Original Seismic (shared cmap)", shared_cmap, -1.0, 1.0)
    show_section(axes[1], synth_disp, dt, "Synthetic Seismic (shared cmap)", shared_cmap, -1.0, 1.0)
    show_section(axes[2], resid_disp, dt, "Residual (shared cmap)", shared_cmap, -1.0, 1.0)
    show_section(axes[3], imp_disp, dt, "Inverted Impedance (shared cmap)", shared_cmap, -1.0, 1.0)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_prior_vs_final(base_prior: np.ndarray, final_imp: np.ndarray, dt: float, out_path: Path) -> None:
    base_disp, vr, _ = soft_display_image(base_prior)
    final_disp, _, _ = soft_display_image(final_imp)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    show_section(axes[0], base_disp, dt, "Base Prior", "turbo", vr[0], vr[1])
    show_section(axes[1], final_disp, dt, "Selected Final", "turbo", vr[0], vr[1])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_baseline_comparison(
    observed: np.ndarray,
    baselines: Dict[str, np.ndarray],
    final_imp: np.ndarray,
    dt: float,
    out_path: Path,
) -> None:
    panels = [("Original Seismic", observed, "gray")]
    if "v6" in baselines:
        panels.append(("V6 Baseline", baselines["v6"], "turbo"))
    if "v7" in baselines:
        panels.append(("V7 Baseline", baselines["v7"], "turbo"))
    panels.append(("Selected Final", final_imp, "turbo"))
    cols = len(panels)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 7), constrained_layout=True)
    if cols == 1:
        axes = [axes]
    imp_refs = [section for _, section, cmap in panels if cmap != "gray"]
    vmin = min(float(np.percentile(section, 0.5)) for section in imp_refs)
    vmax = max(float(np.percentile(section, 99.5)) for section in imp_refs)
    for ax, (title, section, cmap) in zip(axes, panels):
        if cmap == "gray":
            lim = float(np.percentile(np.abs(section), 99.0))
            show_section(ax, section, dt, title, "gray", -lim, lim)
        else:
            disp_section, _, _ = soft_display_image(section)
            show_section(ax, disp_section, dt, title, "turbo", vmin, vmax)
    fig.savefig(out_path, dpi=180)
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
        axes[row, 1].plot(final_imp[trace_id], label="selected_final")
        axes[row, 1].set_title(f"Trace {trace_id}: impedance")
        axes[row, 1].legend()
        axes[row, 1].grid(True, alpha=0.25)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_wavelet(initial_wavelet: np.ndarray, final_wavelet: np.ndarray, dt: float, out_path: Path) -> None:
    time_axis = np.arange(initial_wavelet.shape[0]) * dt * 1000.0
    freqs = np.fft.rfftfreq(initial_wavelet.shape[0], d=dt)
    init_spec = np.abs(np.fft.rfft(initial_wavelet))
    final_spec = np.abs(np.fft.rfft(final_wavelet))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].plot(time_axis, initial_wavelet, label="initial")
    axes[0].plot(time_axis, final_wavelet, label="selected")
    axes[0].set_title("Wavelet Time Domain")
    axes[0].set_xlabel("Time (ms)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(freqs, init_spec, label="initial")
    axes[1].plot(freqs, final_spec, label="selected")
    axes[1].set_xlim(0.0, min(80.0, float(freqs.max())))
    axes[1].set_title("Wavelet Amplitude Spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_residual_spectrum(observed: np.ndarray, synthetic: np.ndarray, dt: float, out_path: Path) -> None:
    num = np.sum(synthetic * observed, axis=1, keepdims=True)
    den = np.clip(np.sum(synthetic * synthetic, axis=1, keepdims=True), 1e-6, None)
    scaled = synthetic * (num / den)
    residual = observed - scaled
    freqs = np.fft.rfftfreq(observed.shape[1], d=dt)
    obs_spec = np.mean(np.abs(np.fft.rfft(observed, axis=1)), axis=0)
    synth_spec = np.mean(np.abs(np.fft.rfft(scaled, axis=1)), axis=0)
    resid_spec = np.mean(np.abs(np.fft.rfft(residual, axis=1)), axis=0)
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, obs_spec, label="observed")
    plt.plot(freqs, synth_spec, label="synthetic (scaled)")
    plt.plot(freqs, resid_spec, label="residual")
    plt.xlim(0.0, min(80.0, float(freqs.max())))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Mean amplitude")
    plt.title("Residual Spectrum")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_uncertainty_time_trace(uncertainty: np.ndarray, dt: float, out_path: Path) -> None:
    disp, vr, _ = soft_display_image(uncertainty)
    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    show_section(ax, disp, dt, "Prior Uncertainty", "turbo", vr[0], vr[1])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="0908 SGY inversion v11/v12: learnable wavelet + LNCC + structure-aware adaptation")
    parser.add_argument("--sgy", type=Path, default=DEFAULT_SGY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--train-traces", type=int, default=12000)
    parser.add_argument("--infer-traces", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--highpass-cutoff", type=float, default=12.0)
    parser.add_argument("--wavelet-f0", type=float, default=25.0)
    parser.add_argument("--wavelet-phase-lr", type=float, default=3e-5)
    parser.add_argument("--wavelet-freeze-epochs", type=int, default=10)
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
    parser.add_argument("--residual-relax", type=float, default=0.35)
    parser.add_argument("--delta-relax", type=float, default=0.55)
    parser.add_argument("--lncc-windows", type=str, default="81,161")
    parser.add_argument("--amp-loss-weight", type=float, default=0.25)
    parser.add_argument("--stft-weight", type=float, default=0.15)
    parser.add_argument("--reflectivity-fp-weight", type=float, default=0.10)
    parser.add_argument("--data-warmup-epochs", type=int, default=10)
    parser.add_argument("--struct-warmup-epochs", type=int, default=12)
    parser.add_argument("--tuned-pcc-weight", type=float, default=0.45)
    parser.add_argument("--tuned-lncc-weight", type=float, default=0.55)
    parser.add_argument("--tuned-stft-weight", type=float, default=0.03)
    parser.add_argument("--tuned-structure-corr-weight", type=float, default=0.02)
    parser.add_argument("--tuned-structure-l1-weight", type=float, default=0.01)
    parser.add_argument("--tuned-structure-bg-weight", type=float, default=0.01)
    parser.add_argument("--tuned-structure-gain-weight", type=float, default=0.02)
    parser.add_argument("--tuned-variance-floor-weight", type=float, default=0.04)
    parser.add_argument("--tuned-reflectivity-fp-weight", type=float, default=0.03)
    parser.add_argument("--tuned-wavelet-reg-weight", type=float, default=0.005)
    parser.add_argument("--lateral-group-size", type=int, default=5)
    parser.add_argument("--lateral-delta-weight", type=float, default=0.02)
    parser.add_argument("--lateral-beta", type=float, default=3.0)
    parser.add_argument("--reflectivity-floor-weight", type=float, default=0.02)
    parser.add_argument("--reflectivity-floor-ratio", type=float, default=0.90)
    parser.add_argument("--cross-grad-weight", type=float, default=0.0)
    parser.add_argument("--cross-grad-beta", type=float, default=3.0)
    parser.add_argument("--lateral-warmup-start", type=int, default=3)
    parser.add_argument("--lateral-warmup-epochs", type=int, default=6)
    parser.add_argument("--display-l1-weight", type=float, default=0.0)
    parser.add_argument("--display-grad-weight", type=float, default=0.0)
    parser.add_argument("--display-corr-weight", type=float, default=0.0)
    parser.add_argument("--display-warmup-epochs", type=int, default=6)
    parser.add_argument("--display-score-agreement-weight", type=float, default=0.35)
    parser.add_argument("--display-score-cosine-weight", type=float, default=0.15)
    parser.add_argument("--display-score-pearson-weight", type=float, default=0.10)
    parser.add_argument("--display-score-pcc-weight", type=float, default=0.20)
    parser.add_argument("--display-score-resid-weight", type=float, default=0.15)
    parser.add_argument("--display-score-edge-weight", type=float, default=0.10)
    parser.add_argument("--experiment-mode", type=str, choices=["core_v11", "core_v11_tuned", "core_v11_similarity", "core_v12_tuned", "core_v12_refine", "amp_only"], default="core_v11")
    parser.add_argument("--reset-history-on-resume", action="store_true")
    parser.add_argument("--reset-optim-on-resume", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.lncc_windows = parse_lncc_windows(args.lncc_windows)
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(vars(args), args.out / "run_config.json")
    seed_everything(args.seed)
    device = torch.device(args.device)
    tracecount, nsamples, _, dt = read_tracecount_and_samples(args.sgy)
    fs_hz = 1.0 / dt

    print(f"Device: {device}")
    print(f"SGY: {args.sgy}")
    print(f"Tracecount: {tracecount}, samples: {nsamples}, dt: {dt:.6f}s")
    print(f"Experiment mode: {args.experiment_mode}")

    actual_train_traces = int(args.train_traces)
    block_starts = None
    if args.experiment_mode in {"core_v12_tuned", "core_v12_refine"}:
        block_starts, train_trace_ids = contiguous_block_trace_ids(
            total_traces=tracecount,
            selected_traces=args.train_traces,
            group_size=args.lateral_group_size,
        )
        actual_train_traces = int(train_trace_ids.size)
        print(
            f"Using adjacent trace blocks: {len(block_starts)} blocks x "
            f"{args.lateral_group_size} traces = {actual_train_traces} traces"
        )
    else:
        train_trace_ids = uniform_indices(tracecount, args.train_traces)
        actual_train_traces = int(train_trace_ids.size)
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

    if args.experiment_mode in {"core_v12_tuned", "core_v12_refine"}:
        group_size = int(args.lateral_group_size)
        block_count = features.shape[0] // group_size
        features = features.reshape(block_count, group_size, features.shape[1], features.shape[2])
        train_seismic = train_seismic.reshape(block_count, group_size, train_seismic.shape[1])
        base_prior_log = base_prior_log.reshape(block_count, group_size, base_prior_log.shape[1])
        prior_weight = prior_weight.reshape(block_count, group_size, prior_weight.shape[1])
        n_train = int(round(0.9 * block_count))
        train_idx = np.arange(n_train, dtype=np.int64)
        val_idx = np.arange(n_train, block_count, dtype=np.int64)
        train_ds = AdjacentTraceBlockDataset(features, train_seismic, base_prior_log, prior_weight, train_idx)
        val_ds = AdjacentTraceBlockDataset(features, train_seismic, base_prior_log, prior_weight, val_idx)
    else:
        n_train = int(round(0.9 * len(train_trace_ids)))
        train_idx = np.arange(n_train, dtype=np.int64)
        val_idx = np.arange(n_train, len(train_trace_ids), dtype=np.int64)
        train_ds = FullTraceDataset(features, train_seismic, base_prior_log, prior_weight, train_idx)
        val_ds = FullTraceDataset(features, train_seismic, base_prior_log, prior_weight, val_idx)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = InversionNet(in_ch=3, base=48).to(device)
    init_payload = None
    if args.init_checkpoint is not None and args.init_checkpoint.exists():
        init_payload = load_model_weights(model, args.init_checkpoint, device)
        print(f"Initialized model weights from {args.init_checkpoint}")
    else:
        init_from_30hz_v6(model, device)
        print("Initialized model weights from 01_30Hz_v6 fallback")
    wavelet_module = LearnableWavelet(args.wavelet_f0, dt, length=DEFAULT_WAVELET_LENGTH).to(device)
    if init_payload is not None and "wavelet" in init_payload:
        wavelet_module.load_state_dict(init_payload["wavelet"], strict=True)
        print(f"Initialized wavelet state from {args.init_checkpoint}")
    forward_model = DynamicForwardModel().to(device)
    delta_smoother = FixedGaussian1D(kernel_size=9, sigma=1.5).to(device)
    lowpass_smoother = FixedGaussian1D(kernel_size=41, sigma=7.0).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": list(wavelet_module.parameters()), "lr": args.wavelet_phase_lr, "weight_decay": 0.0},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    history = {"train_total": [], "val_total": [], "val_pcc": [], "wavelet_phi_deg": []}
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    start_epoch = 1
    best_val = float("inf")
    best_pcc = float("-inf")
    patience_counter = 0

    if args.resume is not None and args.resume.exists():
        payload = torch.load(args.resume, map_location=device)
        model.load_state_dict(payload["model"], strict=True)
        if "wavelet" in payload:
            wavelet_module.load_state_dict(payload["wavelet"], strict=True)
        if not args.reset_optim_on_resume:
            optimizer.load_state_dict(payload["optim"])
            scheduler.load_state_dict(payload["sched"])
        start_epoch = int(payload["epoch"]) + 1
        best_val = float(payload.get("best_val", best_val))
        best_pcc = float(payload.get("best_pcc", best_pcc))
        history = payload.get("history", history)
        if history.get("val_pcc"):
            best_pcc = max(best_pcc, max(float(v) for v in history["val_pcc"]))
        if args.reset_history_on_resume:
            best_val = float("inf")
            best_pcc = float("-inf")
            history = {"train_total": [], "val_total": [], "val_pcc": [], "wavelet_phi_deg": []}
        print(f"Resumed from epoch {start_epoch - 1}")

    for epoch in range(start_epoch, args.epochs + 1):
        schedule_epoch = epoch - start_epoch + 1 if args.reset_history_on_resume else epoch
        wavelet_trainable = set_wavelet_trainability(
            wavelet_module=wavelet_module,
            epoch=epoch,
            freeze_epochs=args.wavelet_freeze_epochs,
            experiment_mode=args.experiment_mode,
        )
        train_stats = train_one_epoch(
            model=model,
            wavelet_module=wavelet_module,
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
            args=args,
        )
        val_stats = validate_one_epoch(
            model=model,
            wavelet_module=wavelet_module,
            loader=val_loader,
            device=device,
            forward_model=forward_model,
            delta_smoother=delta_smoother,
            lowpass_smoother=lowpass_smoother,
            epoch=epoch,
            schedule_epoch=schedule_epoch,
            args=args,
        )
        scheduler.step()
        history["train_total"].append(train_stats["total"])
        history["val_total"].append(val_stats["total"])
        history["val_pcc"].append(val_stats["pcc"])
        history["wavelet_phi_deg"].append(val_stats["wavelet_phi_deg"])
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_stats['total']:.4f} | val={val_stats['total']:.4f} | "
            f"pcc={val_stats['pcc']:.4f} | amp={val_stats['amp']:.4f} | "
            f"struct={val_stats['struct_corr']:.4f} | refl_fp={val_stats['refl_fp']:.4f} | "
            f"lat={val_stats['lat_delta']:.4f} | floor={val_stats['refl_floor']:.4f} | "
            f"cg={val_stats['cross_grad']:.4f} | w_lat={val_stats['w_lat']:.3f} | "
            f"phi={val_stats['wavelet_phi_deg']:.2f}deg | trainable={wavelet_trainable}"
        )
        payload = checkpoint_payload(model, wavelet_module, optimizer, scheduler, epoch, best_val, best_pcc, history)
        torch.save(payload, ckpt_dir / "latest.pt")
        if val_stats["total"] < best_val:
            best_val = val_stats["total"]
            patience_counter = 0
            payload = checkpoint_payload(model, wavelet_module, optimizer, scheduler, epoch, best_val, best_pcc, history)
            torch.save(payload, ckpt_dir / "best.pt")
            torch.save(payload, ckpt_dir / "best_loss.pt")
        else:
            patience_counter += 1
        if val_stats["pcc"] > best_pcc:
            best_pcc = val_stats["pcc"]
            payload = checkpoint_payload(model, wavelet_module, optimizer, scheduler, epoch, best_val, best_pcc, history)
            torch.save(payload, ckpt_dir / "best_pcc.pt")
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

    candidate_name_map = {
        "best_loss": ckpt_dir / "best_loss.pt",
        "best_pcc": ckpt_dir / "best_pcc.pt",
        "latest": ckpt_dir / "latest.pt",
    }
    if not candidate_name_map["best_loss"].exists() and (ckpt_dir / "best.pt").exists():
        candidate_name_map["best_loss"] = ckpt_dir / "best.pt"
    candidate_paths = [(name, path) for name, path in candidate_name_map.items() if path.exists()]
    if not candidate_paths and args.resume is not None and args.resume.exists():
        candidate_paths = [("resume", args.resume)]
    if not candidate_paths:
        raise RuntimeError("No candidate checkpoints available for inference")

    candidate_results = []
    selection_mode = "display" if args.experiment_mode == "core_v11_similarity" else "default"
    display_score_weights = {
        "agreement_weight": float(args.display_score_agreement_weight),
        "cosine_weight": float(args.display_score_cosine_weight),
        "pearson_weight": float(args.display_score_pearson_weight),
        "pcc_weight": float(args.display_score_pcc_weight),
        "resid_weight": float(args.display_score_resid_weight),
        "edge_weight": float(args.display_score_edge_weight),
    }
    for ckpt_name, ckpt_path in candidate_paths:
        candidate_results.append(
            evaluate_candidate_checkpoint(
                ckpt_name=ckpt_name,
                ckpt_path=ckpt_path,
                model=model,
                wavelet_module=wavelet_module,
                infer_features=infer_features,
                infer_base_log=infer_base_log,
                infer_seismic=infer_seismic,
                infer_batch_size=args.infer_batch_size,
                fs_hz=fs_hz,
                highpass_cutoff=args.highpass_cutoff,
                device=device,
                delta_smoother=delta_smoother,
                forward_model=forward_model,
                lateral_score_beta=args.cross_grad_beta,
                selection_mode=selection_mode,
                display_score_weights=display_score_weights,
            )
        )

    if selection_mode == "display":
        selected = max(candidate_results, key=lambda item: item["metrics"]["display_selection_score"])
    else:
        selected = max(candidate_results, key=lambda item: item["metrics"]["selection_score"])
    selected_metrics = selected["metrics"]
    imp_raw = selected["imp_raw"]
    imp_final = selected["imp_final"]
    synth = selected["synth"]
    _, _, clipped_ratio = soft_display_image(imp_final)

    np.save(args.out / "base_prior.npy", np.exp(infer_base_log).astype(np.float32))
    np.save(args.out / "prior_uncertainty.npy", infer_uncertainty)
    np.save(args.out / "impedance_pred_raw.npy", imp_raw)
    np.save(args.out / "impedance_pred_final.npy", imp_final)
    np.save(args.out / "synth_seismic.npy", synth)
    initial_wavelet_summary = {
        "phi_rad": 0.0,
        "phi_deg": 0.0,
        "f_scale": 1.0,
        "effective_f0_hz": float(args.wavelet_f0),
        "gain": 1.0,
        "length_seconds": DEFAULT_WAVELET_LENGTH,
        "length_samples": int(wavelet_module.initial_wavelet.shape[-1]),
        "energy_l2": float(np.linalg.norm(wavelet_module.initial_wavelet.cpu().numpy().reshape(-1))),
        "max_abs": float(np.max(np.abs(wavelet_module.initial_wavelet.cpu().numpy().reshape(-1)))),
        "wavelet_bad": 0.0,
    }
    save_json(
        {
            "initial_wavelet": initial_wavelet_summary,
            "selected_wavelet": selected["wavelet_summary"],
            "selected_checkpoint": selected_metrics["checkpoint_name"],
            "candidates": [
                {
                    "checkpoint_name": item["metrics"]["checkpoint_name"],
                    "wavelet_phi_deg": item["metrics"]["wavelet_phi_deg"],
                    "wavelet_f_scale": item["metrics"]["wavelet_f_scale"],
                    "wavelet_bad": item["metrics"]["wavelet_bad"],
                }
                for item in candidate_results
            ],
        },
        args.out / "wavelet_summary.json",
    )
    save_json([item["metrics"] for item in candidate_results], args.out / "candidate_metrics.json")

    plot_training_history(history, args.out / "training_loss.png")
    plot_original_vs_inversion(infer_seismic, imp_final, dt=dt, out_path=args.out / "comparison.png")
    plot_original_vs_inversion(infer_seismic, imp_final, dt=dt, out_path=args.out / "original_vs_inversion.png")
    plot_forward_diagnostic(infer_seismic, synth, imp_final, dt=dt, out_path=args.out / "forward_diagnostic.png")
    plot_trace_comparison(
        observed=infer_seismic,
        synth=synth,
        base_prior=np.exp(infer_base_log).astype(np.float32),
        final_imp=imp_final,
        out_path=args.out / "trace_comparison.png",
    )
    plot_prior_vs_final(np.exp(infer_base_log).astype(np.float32), imp_final, dt=dt, out_path=args.out / "prior_vs_final.png")
    plot_uncertainty_time_trace(infer_uncertainty, dt=dt, out_path=args.out / "uncertainty_map.png")
    plot_wavelet(
        wavelet_module.initial_wavelet.cpu().numpy().reshape(-1),
        selected["wavelet_np"],
        dt=dt,
        out_path=args.out / "wavelet.png",
    )
    plot_residual_spectrum(infer_seismic, synth, dt=dt, out_path=args.out / "residual_spectrum.png")
    baselines = load_baseline_sections(args.infer_traces)
    if baselines:
        plot_baseline_comparison(infer_seismic, baselines, imp_final, dt=dt, out_path=args.out / "baseline_comparison.png")

    summary = {
        "tracecount_total": tracecount,
        "train_traces": actual_train_traces,
        "infer_traces": args.infer_traces,
        "nsamples": nsamples,
        "dt_seconds": dt,
        "experiment_mode": args.experiment_mode,
        "lateral_group_size": int(args.lateral_group_size) if args.experiment_mode in {"core_v12_tuned", "core_v12_refine"} else 1,
        "selected_checkpoint": selected_metrics["checkpoint_name"],
        "selected_checkpoint_path": selected_metrics["checkpoint_path"],
        "selected_epoch": selected_metrics["checkpoint_epoch"],
        "postprocess_mode": selected_metrics["postprocess_mode"],
        "prior_models": prior_meta,
        "candidate_metrics": [item["metrics"] for item in candidate_results],
        "raw_trace_pcc": selected_metrics["raw_trace_pcc"],
        "mean_trace_pcc": selected_metrics["mean_trace_pcc"],
        "raw_impedance_mean": selected_metrics["raw_impedance_mean"],
        "raw_impedance_std": selected_metrics["raw_impedance_std"],
        "final_impedance_mean": selected_metrics["final_impedance_mean"],
        "final_impedance_std": selected_metrics["final_impedance_std"],
        "mae_ratio_vs_prior": selected_metrics["mae_ratio_vs_prior"],
        "edge_alignment": selected_metrics["edge_alignment"],
        "resid_l1_si": selected_metrics["resid_l1_si"],
        "resid_rms_ratio": selected_metrics["resid_rms_ratio"],
        "lat_tv_ratio": selected_metrics["lat_tv_ratio"],
        "lat_tv_ratio_weighted": selected_metrics["lat_tv_ratio_weighted"],
        "display_pearson": selected_metrics["display_pearson"],
        "display_cosine": selected_metrics["display_cosine"],
        "display_agreement": selected_metrics["display_agreement"],
        "wavelet_phi_deg": selected_metrics["wavelet_phi_deg"],
        "wavelet_f_scale": selected_metrics["wavelet_f_scale"],
        "wavelet_bad": selected_metrics["wavelet_bad"],
        "selection_score": selected_metrics["selection_score"],
        "display_selection_score": selected_metrics["display_selection_score"],
        "display_clip_ratio": clipped_ratio,
    }
    save_json(summary, args.out / "run_summary.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
