from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseSoftThreshold(nn.Module):
    """Soft-thresholding with channel-wise adaptive thresholds.

    Implements:
      tau_c = alpha_c * mean(|x_c|)  (mean over temporal dimension)
    where alpha_c is produced by a lightweight 'attention' subnetwork
    (global avg pool -> FC -> ReLU -> FC -> sigmoid).

    This follows the residual shrinkage idea referenced in the paper.
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        abs_x = torch.abs(x)
        gap = abs_x.mean(dim=2)  # (B, C)
        alpha = torch.sigmoid(self.fc2(F.relu(self.fc1(gap))))  # (B, C) in (0,1)
        tau = alpha * gap  # (B, C)
        tau = tau.unsqueeze(2)  # (B, C, 1)
        # Soft threshold: sign(x) * relu(|x|-tau)
        return torch.sign(x) * F.relu(abs_x - tau)
