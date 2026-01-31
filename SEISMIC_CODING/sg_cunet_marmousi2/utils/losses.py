# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target)**2 + self.eps**2))

def grad1d_loss(pred, target, dim=2):
    """时间轴一阶梯度一致性"""
    idx0 = torch.tensor([0], device=pred.device)
    dp = pred.diff(dim=dim, prepend=pred.index_select(dim, idx0))
    dt = target.diff(dim=dim, prepend=target.index_select(dim, idx0))
    return torch.mean(torch.abs(dp - dt))

def tv_loss(pred, dims=(2,3), w=(1.0, 0.3)):
    """各向异性 TV 正则，轻微平滑以抑制伪影，同时保边"""
    loss = 0.0
    if 2 in dims:
        loss = loss + w[0] * torch.mean(torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :]))
    if 3 in dims:
        loss = loss + w[1] * torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))
    return loss
