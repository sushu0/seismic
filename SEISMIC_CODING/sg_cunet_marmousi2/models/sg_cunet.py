import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=7):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class SG_CUnet(nn.Module):
    def __init__(self, base=32):
        super().__init__()
        self.inc  = DoubleConv(1, base)
        self.d1   = Down(base, base*2)
        self.d2   = Down(base*2, base*4)
        self.d3   = Down(base*4, base*8)
        self.bot  = DoubleConv(base*8, base*16)
        self.u3   = Up(base*16, base*8)
        self.u2   = Up(base*8, base*4)
        self.u1   = Up(base*4, base*2)
        self.u0   = Up(base*2, base)
        self.fuse = nn.Conv2d(base, base, 7, padding=3)
        self.head_imp  = nn.Conv2d(base, 1, 1)
        self.head_refl = nn.Conv2d(base, 1, 1)
        self.log_vars  = nn.Parameter(torch.zeros(4))  # Kendall weights
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.d1(x1)
        x3 = self.d2(x2)
        x4 = self.d3(x3)
        b  = self.bot(x4)
        u3 = self.u3(b, x4)
        u2 = self.u2(u3, x3)
        u1 = self.u1(u2, x2)
        u0 = self.u0(u1, x1)
        f  = self.fuse(u0)
        imp  = self.head_imp(f)
        refl = self.head_refl(f)
        return imp, refl, self.log_vars
