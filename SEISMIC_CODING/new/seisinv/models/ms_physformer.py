from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvBlock(nn.Module):
    """Depthwise-separable conv block for efficiency."""
    def __init__(self, c_in: int, c_out: int, k: int = 3):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_in, kernel_size=k, padding=p, groups=c_in),
            nn.Conv1d(c_in, c_out, kernel_size=1),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=p, groups=c_out),
            nn.Conv1d(c_out, c_out, kernel_size=1),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)

class MSPhysFormer(nn.Module):
    """Multi-Scale U-Net with Transformer bottleneck and deep supervision heads."""
    def __init__(
        self,
        in_ch: int = 1,
        base: int = 48,
        depth: int = 4,
        nhead: int = 4,
        tf_dim_mult: int = 2,
        tf_layers: int = 2,
        out_ch: int = 1,
    ):
        super().__init__()
        self.depth = depth
        chs = [base*(2**i) for i in range(depth)]  # encoder channels
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()

        prev = in_ch
        for c in chs:
            self.enc.append(DWConvBlock(prev, c))
            self.down.append(nn.MaxPool1d(2))
            prev = c

        bott_ch = prev * tf_dim_mult
        self.bottleneck = DWConvBlock(prev, bott_ch)

        # Transformer bottleneck on downsampled sequence
        self.tf_in = nn.Conv1d(bott_ch, bott_ch, kernel_size=1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=bott_ch,
            nhead=nhead,
            dim_feedforward=bott_ch*2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=tf_layers)
        self.tf_out = nn.Conv1d(bott_ch, bott_ch, kernel_size=1)

        # Decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        dec_chs = list(reversed(chs))
        prev = bott_ch
        for c in dec_chs:
            self.up.append(nn.ConvTranspose1d(prev, c, kernel_size=2, stride=2))
            self.dec.append(DWConvBlock(c * 2, c))  # c*2 because we concat skip
            prev = c

        self.out = nn.Conv1d(prev, out_ch, kernel_size=1)

        # Deep supervision heads (at 1/2, 1/4, 1/8 scales)
        self.ds_heads = nn.ModuleList([
            nn.Conv1d(chs[-1], out_ch, kernel_size=1),       # deepest encoder feature (1/16)
            nn.Conv1d(chs[-2], out_ch, kernel_size=1),       # 1/8
            nn.Conv1d(chs[-3], out_ch, kernel_size=1),       # 1/4
        ])

    def forward(self, x):
        # returns: final_pred, multi_scale_preds (list)
        skips = []
        feats = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            feats.append(x)
            x = self.down[i](x)

        x = self.bottleneck(x)
        x = self.tf_in(x)

        # transformer expects [B, L, C]
        xt = x.transpose(1, 2)
        xt = self.transformer(xt)
        x = xt.transpose(1, 2)

        x = self.tf_out(x)

        # decoder
        for i in range(self.depth):
            x = self.up[i](x)
            skip = skips[-(i+1)]
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = self.dec[i](x)

        y = self.out(x)

        # deep supervision preds from encoder feats
        # pick deepest(1/16), 1/8, 1/4
        ms = []
        if len(feats) >= 4:
            f16 = feats[-1]          # 1/1 before pool; but after pools it's 1/16 at bottleneck input
            f8  = feats[-2]
            f4  = feats[-3]
            ms.append(self.ds_heads[0](F.avg_pool1d(f16, kernel_size=16, stride=16)))  # approximate to 1/16
            ms.append(self.ds_heads[1](F.avg_pool1d(f8,  kernel_size=8,  stride=8)))
            ms.append(self.ds_heads[2](F.avg_pool1d(f4,  kernel_size=4,  stride=4)))
        return y, ms
