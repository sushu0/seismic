# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F

def _gaussian_kernel_1d(ksz, sigma, device, dtype):
    x = torch.arange(ksz, device=device, dtype=torch.float32) - ksz // 2
    k = torch.exp(-(x**2) / (2.0 * (sigma**2) + 1e-12))
    k = (k / (k.sum() + 1e-12)).to(dtype)
    kx = k.view(1, 1, 1, ksz)
    ky = k.view(1, 1, ksz, 1)
    return kx, ky

def _gaussian_blur_2d(img, sigma_pix):
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype
    ksz = int(max(3, round(6.0 * sigma_pix))) | 1
    kx, ky = _gaussian_kernel_1d(ksz, sigma_pix, device, dtype)
    out = F.conv2d(img, kx, padding=(0, ksz // 2), groups=1)
    out = F.conv2d(out, ky, padding=(ksz // 2, 0), groups=1)
    return out

def elastic_deform_triplet(Zp, Rp, Sp, alpha=900.0, sigma=70.0):
    """对 (Z, R, S) 同步施加弹性形变；输入输出均为 (1,T,X)"""
    device, dtype = Zp.device, Zp.dtype
    _, H, W = Zp.shape

    sigma_pix = max(1.0, sigma / 10.0)
    noise_dx = torch.randn(1, 1, H, W, device=device, dtype=dtype)
    noise_dy = torch.randn(1, 1, H, W, device=device, dtype=dtype)
    dx = _gaussian_blur_2d(noise_dx, sigma_pix)
    dy = _gaussian_blur_2d(noise_dy, sigma_pix)

    scale = alpha / float(max(H, W))
    dx = dx.squeeze(0).squeeze(0) * scale
    dy = dy.squeeze(0).squeeze(0) * scale

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, H, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, W, device=device, dtype=dtype),
        indexing='ij'
    )
    grid = torch.stack([
        grid_x + (dx / (W / 2.0)),
        grid_y + (dy / (H / 2.0)),
    ], dim=-1).unsqueeze(0)   # (1,H,W,2)

    def warp(x):
        return F.grid_sample(x.unsqueeze(0), grid, mode='bilinear',
                             padding_mode='border', align_corners=True).squeeze(0)

    return warp(Zp), warp(Rp), warp(Sp)
