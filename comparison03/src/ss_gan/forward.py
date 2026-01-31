from __future__ import annotations
import torch
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class RickerWavelet:
    freq_hz: float = 30.0
    dt: float = 0.001
    duration_s: float = 0.128
    device: str = "cpu"

    def tensor(self) -> torch.Tensor:
        n = int(self.duration_s / self.dt)
        if n % 2 == 0:
            n += 1
        t = torch.linspace(-(n//2)*self.dt, (n//2)*self.dt, n, device=self.device)
        pi2 = (torch.pi * self.freq_hz)**2
        w = (1.0 - 2.0*pi2*t**2) * torch.exp(-pi2*t**2)
        w = w / (torch.sqrt(torch.sum(w**2)) + 1e-12)
        return w

def impedance_to_reflectivity(imp: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    imp_fwd = imp[..., 1:]
    imp_cur = imp[..., :-1]
    r = (imp_fwd - imp_cur) / (imp_fwd + imp_cur + eps)
    r = F.pad(r, (1, 0), mode="constant", value=0.0)
    return r

def convolve_seismic(r: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
    K = wavelet.numel()
    w = wavelet.view(1, 1, K)
    pad = K // 2
    return F.conv1d(r, w, padding=pad)

def forward_seismic_from_impedance(imp: torch.Tensor, wavelet: torch.Tensor) -> torch.Tensor:
    r = impedance_to_reflectivity(imp)
    return convolve_seismic(r, wavelet)
