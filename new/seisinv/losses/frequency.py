from __future__ import annotations
import torch
import torch.nn as nn

class STFTMagLoss(nn.Module):
    def __init__(self, n_fft: int = 256, hop_length: int = 64, win_length: int | None = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.window = None
        self.mse = nn.MSELoss()

    def _get_window(self, device):
        if self.window is None or self.window.device != device:
            self.window = torch.hann_window(self.win_length, device=device)
        return self.window

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compare STFT magnitude of x and y. x,y: [B,1,T]"""
        assert x.ndim == 3 and y.ndim == 3
        x1 = x.squeeze(1)
        y1 = y.squeeze(1)
        w = self._get_window(x.device)
        X = torch.stft(x1, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length,
                       window=w, return_complex=True)
        Y = torch.stft(y1, n_fft=self.n_fft, hop_length=self.hop, win_length=self.win_length,
                       window=w, return_complex=True)
        return self.mse(torch.abs(X), torch.abs(Y))
