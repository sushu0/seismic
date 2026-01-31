from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Baseline 1: UNet1D
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32, depth: int = 4, out_ch: int = 1):
        super().__init__()
        self.depth = depth
        chs = [base*(2**i) for i in range(depth)]
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(ConvBlock(prev, c))
            self.pool.append(nn.MaxPool1d(2))
            prev = c

        self.bottleneck = ConvBlock(prev, prev*2)
        prev = prev*2

        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose1d(prev, c, kernel_size=2, stride=2))
            self.dec.append(ConvBlock(prev, c))
            prev = c

        self.out = nn.Conv1d(prev, out_ch, kernel_size=1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = self.pool[i](x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.up[i](x)
            skip = skips[-(i+1)]
            if x.shape[-1] != skip.shape[-1]:
                # pad/crop to match
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = self.dec[i](x)
        return self.out(x)

# -------------------------
# Baseline 2: TCN1D (dilated residual)
# -------------------------
class TCNBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, k: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.act(self.bn1(self.conv1(x)))
        y = self.drop(y)
        y = self.act(self.bn2(self.conv2(y)))
        return x + y

class TCN1D(nn.Module):
    def __init__(self, in_ch: int = 1, ch: int = 64, layers: int = 6, out_ch: int = 1, dropout: float = 0.1):
        super().__init__()
        self.inproj = nn.Conv1d(in_ch, ch, kernel_size=1)
        blocks = []
        for i in range(layers):
            blocks.append(TCNBlock(ch, dilation=2**i, dropout=dropout))
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Conv1d(ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.inproj(x)
        x = self.blocks(x)
        return self.out(x)

# -------------------------
# Optional: CNN-BiLSTM (trace-wise)
# -------------------------
class CNNBiLSTM(nn.Module):
    def __init__(self, in_ch: int = 1, cnn_ch: int = 32, lstm_hidden: int = 64, lstm_layers: int = 3, out_ch: int = 1):
        super().__init__()
        # local feature extraction (multi-dilation conv stack)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_ch, cnn_ch, 3, padding=1, dilation=1),
            nn.BatchNorm1d(cnn_ch),
            nn.GELU(),
            nn.Conv1d(cnn_ch, cnn_ch, 3, padding=2, dilation=2),
            nn.BatchNorm1d(cnn_ch),
            nn.GELU(),
            nn.Conv1d(cnn_ch, cnn_ch, 3, padding=4, dilation=4),
            nn.BatchNorm1d(cnn_ch),
            nn.GELU(),
        )
        self.bilstm = nn.LSTM(
            input_size=cnn_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.reg_lstm = nn.LSTM(
            input_size=2*lstm_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(lstm_hidden, out_ch)

    def forward(self, x):
        # x: [B,1,T] -> cnn: [B,C,T] -> transpose to [B,T,C]
        feat = self.cnn(x).transpose(1, 2)
        y, _ = self.bilstm(feat)
        y, _ = self.reg_lstm(y)
        y = self.fc(y)  # [B,T,1]
        return y.transpose(1, 2)  # [B,1,T]
