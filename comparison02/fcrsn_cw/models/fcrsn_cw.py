from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from .shrinkage import ChannelWiseSoftThreshold

def same_padding(kernel_size: int) -> int:
    return kernel_size // 2

class RSBU_CW(nn.Module):
    """Residual Shrinkage Building Unit with Channel-Wise thresholds (RSBU-CW).

    Block pattern (1D):
      x -> BN -> ReLU -> Conv(k1) -> BN -> ReLU -> Conv(k2) -> Shrinkage -> +x -> ReLU

    Kernel sizes follow the paper: first conv 299, second conv 3. fileciteturn2file0L208-L214
    """
    def __init__(self, channels: int, k1: int = 299, k2: int = 3):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=k1, stride=1, padding=same_padding(k1), bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=k2, stride=1, padding=same_padding(k2), bias=False)
        self.shrink = ChannelWiseSoftThreshold(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(x))
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.shrink(out)
        out = out + identity
        out = F.relu(out)
        return out

class FCRSN_CW(nn.Module):
    """Fully Convolutional Residual Shrinkage Network with Channel-Wise thresholds.

    Architecture (paper Fig.2):
      Seismic (1xT) ->
        Conv1d(1->16, k=299) -> BN -> ReLU ->
        RSBU-CW (32) -> RSBU-CW (64) -> RSBU-CW (64) ->
        Conv1d(64->1, k=3) -> ReLU -> Impedance (1xT)

    Note: the paper describes residual blocks with output channels 32, 64, 64 and
    first/last conv settings. fileciteturn2file0L215-L223
    """
    def __init__(
        self,
        k_first: int = 299,
        k_last: int = 3,
        k_res1: int = 299,
        k_res2: int = 3,
        last_relu: bool = True,
        output_activation: str | None = None,
    ):
        super().__init__()
        self.conv_in = nn.Conv1d(1, 16, kernel_size=k_first, stride=1, padding=same_padding(k_first), bias=False)
        self.bn_in = nn.BatchNorm1d(16)

        # channel expansion to match the paper: 16 -> 32 -> 64 -> 64
        self.proj1 = nn.Conv1d(16, 32, kernel_size=1, bias=False)
        self.block1 = RSBU_CW(32, k1=k_res1, k2=k_res2)

        self.proj2 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.block2 = RSBU_CW(64, k1=k_res1, k2=k_res2)
        self.block3 = RSBU_CW(64, k1=k_res1, k2=k_res2)

        self.conv_out = nn.Conv1d(64, 1, kernel_size=k_last, stride=1, padding=same_padding(k_last), bias=True)
        self.last_relu = last_relu
        self.output_activation = output_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_in(self.conv_in(x)))
        out = self.proj1(out)
        out = self.block1(out)
        out = self.proj2(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.conv_out(out)
        if self.output_activation is None:
            if self.last_relu:
                out = F.relu(out)
            return out

        act = str(self.output_activation).lower()
        if act in ("none", "identity", "linear"):
            return out
        if act == "relu":
            return F.relu(out)
        if act == "sigmoid":
            return torch.sigmoid(out)
        raise ValueError(f"Unknown output_activation: {self.output_activation}. Use one of: none|relu|sigmoid")
        return out
