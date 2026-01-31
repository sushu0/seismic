from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ForwardModel(nn.Module):
    """Differentiable convolutional forward model:
    impedance -> reflectivity -> seismic via wavelet convolution.
    reflectivity: r[t] = (I[t] - I[t-1]) / (I[t] + I[t-1] + eps); r[0]=0

    Wavelet convolution: s = r * w (same-length via padding).
    """
    def __init__(self, wavelet: torch.Tensor, eps: float = 1e-6):
        super().__init__()
        assert wavelet.ndim == 1, "wavelet must be 1D [K]"
        self.register_buffer("wavelet", wavelet.clone().detach().float())
        self.eps = eps

    def reflectivity(self, imp: torch.Tensor) -> torch.Tensor:
        # imp: [B,1,T]
        imp_prev = torch.roll(imp, shifts=1, dims=-1)
        num = imp - imp_prev
        den = imp + imp_prev + self.eps
        r = num / den
        r[..., 0] = 0.0
        return r

    def forward(self, imp: torch.Tensor) -> torch.Tensor:
        r = self.reflectivity(imp)
        # conv1d expects [B,C,T]; kernel [out_ch,in_ch,K]
        w = self.wavelet.view(1, 1, -1)
        pad = (w.shape[-1] - 1) // 2
        s = F.conv1d(r, w, padding=pad)
        # if kernel length even, adjust
        if s.shape[-1] != r.shape[-1]:
            s = s[..., :r.shape[-1]]
        return s

class PhysicsLoss(nn.Module):
    def __init__(self, forward_model: ForwardModel, mode: str = "mse"):
        super().__init__()
        self.fm = forward_model
        self.mode = mode
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, imp_pred: torch.Tensor, seis_obs: torch.Tensor) -> torch.Tensor:
        seis_pred = self.fm(imp_pred)
        if self.mode == "mse":
            return self.mse(seis_pred, seis_obs)
        if self.mode == "l1":
            return self.l1(seis_pred, seis_obs)
        raise ValueError(f"Unknown physics loss mode: {self.mode}")
