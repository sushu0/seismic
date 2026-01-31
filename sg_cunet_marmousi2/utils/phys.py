import numpy as np
import torch
import torch.nn.functional as F

def ricker_wavelet(freq=35.0, dt=0.001, nt=128):
    t = np.arange(-(nt//2), nt - nt//2) * dt
    y = (1.0 - 2.0*(np.pi*freq*t)**2) * np.exp(-(np.pi*freq*t)**2)
    return y.astype(np.float32)

def conv_time_axis(r, kernel):
    # r: (B,1,T,X), kernel: (1,1,L)
    B, C, T, X = r.shape
    r_flat = r.permute(0,3,1,2).reshape(B*X,1,T)
    pad = kernel.shape[-1]//2
    out = F.conv1d(F.pad(r_flat, (pad,pad)), kernel)
    out = out.squeeze(1)
    Tout = out.shape[-1]
    if Tout > T:
        out = out[:, :T]
    elif Tout < T:
        out = F.pad(out, (0, T - Tout))
    out = out.view(B, X, T).permute(0,2,1).unsqueeze(1)
    return out
