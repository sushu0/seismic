from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention1D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Conv1d(channels, channels // 8, 1)
        self.key = nn.Conv1d(channels, channels // 8, 1)
        self.value = nn.Conv1d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, L = x.shape
        q = self.query(x).view(B, -1, L).permute(0, 2, 1)  # [B, L, C//8]
        k = self.key(x).view(B, -1, L)  # [B, C//8, L]
        attn = torch.softmax(torch.bmm(q, k), dim=-1)  # [B, L, L]
        v = self.value(x).view(B, C, L)  # [B, C, L]
        out = torch.bmm(v, attn.permute(0, 2, 1)).view(B, C, L)
        return self.gamma * out + x

def conv_block(in_ch: int, out_ch: int, k_large: int = 299, k_small: int = 3) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=k_large, padding=k_large//2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv1d(out_ch, out_ch, kernel_size=k_small, padding=k_small//2),
        nn.LeakyReLU(0.2, inplace=True),
    )

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_large: int = 299, k_small: int = 3):
        super().__init__()
        self.block = conv_block(in_ch, out_ch, k_large, k_small)
        self.pool = nn.AvgPool1d(2)

    def forward(self, x):
        x = self.block(x)
        return x, self.pool(x)

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k_large: int = 299, k_small: int = 3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.block = conv_block(in_ch, out_ch, k_large, k_small)

    def forward(self, x, skip):
        x = self.up(x)
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            if diff > 0:
                x = F.pad(x, (diff//2, diff-diff//2))
            else:
                diff = -diff
                x = x[..., diff//2 : x.size(-1)-(diff-diff//2)]
        x = torch.cat([skip, x], dim=1)
        return self.block(x)

class UNet1D(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 1, base_ch: int = 32, k_large: int = 31, k_small: int = 3):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16
        self.in_block = conv_block(in_ch, c1, k_large, k_small)
        self.down1 = Down(c1, c2, k_large, k_small)
        self.down2 = Down(c2, c3, k_large, k_small)
        self.down3 = Down(c3, c4, k_large, k_small)
        self.down4 = Down(c4, c5, k_large, k_small)
        self.bottleneck = conv_block(c5, c5, k_large, k_small)
        self.attn = SelfAttention1D(c5)
        self.up4 = Up(c5 + c5, c4, k_large, k_small)
        self.up3 = Up(c4 + c4, c3, k_large, k_small)
        self.up2 = Up(c3 + c3, c2, k_large, k_small)
        self.up1 = Up(c2 + c2, c1, k_large, k_small)
        self.out_conv = nn.Conv1d(c1, out_ch, kernel_size=1)
        self.out_residual = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x0 = self.in_block(x)
        s1, x1 = self.down1(x0)
        s2, x2 = self.down2(x1)
        s3, x3 = self.down3(x2)
        s4, x4 = self.down4(x3)
        b = self.bottleneck(x4)
        b = self.attn(b)
        u4 = self.up4(b, s4)
        u3 = self.up3(u4, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s1)
        out_main = self.out_conv(u1)
        out_res = self.out_residual(x)
        return out_main + 0.1 * out_res

class ResBlock1D(nn.Module):
    def __init__(self, ch: int, k_large: int = 299, k_small: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k_large, padding=k_large//2)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k_small, padding=k_small//2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        h = self.act(self.conv1(x))
        h = self.conv2(h)
        return self.act(x + h)

class Critic1D(nn.Module):
    def __init__(self, in_ch: int = 2, base_ch: int = 16, k_large: int = 31, k_small: int = 3):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv_in = nn.Conv1d(in_ch, base_ch, kernel_size=k_large, padding=k_large//2)

        self.stage1 = nn.Sequential(
            nn.Conv1d(base_ch, base_ch*2, kernel_size=k_small, stride=2, padding=k_small//2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(base_ch*2, k_large, k_small),
        )
        self.stage2 = nn.Sequential(
            nn.Conv1d(base_ch*2, base_ch*4, kernel_size=k_small, stride=2, padding=k_small//2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(base_ch*4, k_large, k_small),
        )
        self.stage3 = nn.Sequential(
            nn.Conv1d(base_ch*4, base_ch*8, kernel_size=k_small, stride=2, padding=k_small//2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(base_ch*8, k_large, k_small),
        )
        self.stage4 = nn.Sequential(
            nn.Conv1d(base_ch*8, base_ch*16, kernel_size=k_small, stride=2, padding=k_small//2),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock1D(base_ch*16, k_large, k_small),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(base_ch*16, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        h = self.act(self.conv_in(x))
        h = self.stage1(h)
        h = self.stage2(h)
        h = self.stage3(h)
        h = self.stage4(h)
        h = self.gap(h).squeeze(-1)
        h = self.act(self.fc1(h))
        return self.fc2(h)
