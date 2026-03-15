# -*- coding: utf-8 -*-
"""
地震反演 v7 — 扰动法 + 强制平滑 + 多尺度物理
=============================================
诊断 v5/v6 根本问题:
  - 阻抗输出与地震振幅高度相似 → 模型学到了"平凡解"
  - sigmoid自由映射解空间太大, 缺乏先验约束
  - TV权重(0.005)远不够阻止高频振荡
  - 真实阻抗是"块状/阶梯"的, 不应有地震频率的振荡

v7 核心设计:
  1. 扰动法: Z = Z_base × exp(tanh(smooth(net_output)) × 0.5)
     - 基础模型提供低频趋势 (类似初始模型)
     - 网络仅学习相对扰动 (幅度受限)
     - 自然保证阻抗为正且范围合理
  2. 强制高斯平滑(sigma=3): 消除地震频率振荡
  3. 3通道输入: 地震 + 高通滤波 + 低频基础模型
  4. 多尺度Pearson相关: 在原始/2x/4x三个尺度匹配
  5. 大幅增强TV (W=0.1) + 扰动L2惩罚 (W=0.1)
  6. 保留 InversionNet 架构 (SE + DilatedBlock)
"""
import json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import hilbert, butter, filtfilt
from pathlib import Path
import segyio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
P = lambda *a, **kw: print(*a, **kw, flush=True)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========================= 强制平滑层 =========================
class GaussianSmoothing(nn.Module):
    """固定高斯平滑: 消除网络输出的高频振荡"""
    def __init__(self, sigma=3.0, kernel_size=21):
        super().__init__()
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel = kernel / kernel.sum()
        self.register_buffer('kernel', kernel.view(1, 1, -1))
        self.pad = kernel_size // 2

    def forward(self, x):
        return F.conv1d(x, self.kernel, padding=self.pad)


# ========================= V6 InversionNet (3通道输入) =========================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(ch, max(ch // r, 4)), nn.ReLU(),
            nn.Linear(max(ch // r, 4), ch), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(x).unsqueeze(-1)


class DilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=(1, 2, 4, 8)):
        super().__init__()
        b_ch = out_ch // len(dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_ch, b_ch, 3, padding=d, dilation=d),
                nn.BatchNorm1d(b_ch), nn.GELU()
            ) for d in dilations
        ])
        self.fuse = nn.Sequential(
            nn.Conv1d(b_ch * len(dilations), out_ch, 1),
            nn.BatchNorm1d(out_ch), nn.GELU()
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        out = torch.cat([b(x) for b in self.branches], dim=1)
        out = self.fuse(out); out = self.se(out)
        return out + self.skip(x)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch, out_ch, 3, padding=1), nn.BatchNorm1d(out_ch), nn.GELU(),
        )
        self.se = SEBlock(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        return self.se(self.conv(x)) + self.skip(x)


class InversionNet(nn.Module):
    """3通道输入: 地震 + 高通滤波 + 低频基础模型"""
    def __init__(self, in_ch=3, base=48):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, base, 7, padding=3), nn.BatchNorm1d(base), nn.GELU()
        )
        self.ms = DilatedBlock(base, base)
        self.e1 = ResBlock(base, base * 2)
        self.p1 = nn.MaxPool1d(2)
        self.e2 = ResBlock(base * 2, base * 4)
        self.p2 = nn.MaxPool1d(2)
        self.neck = nn.Sequential(DilatedBlock(base * 4, base * 8), ResBlock(base * 8, base * 8))
        self.u2 = nn.ConvTranspose1d(base * 8, base * 4, 2, 2)
        self.d2 = ResBlock(base * 8, base * 4)
        self.u1 = nn.ConvTranspose1d(base * 4, base * 2, 2, 2)
        self.d1 = ResBlock(base * 4, base * 2)
        self.refine = nn.Sequential(
            ResBlock(base * 2 + base, base * 2),
            nn.Conv1d(base * 2, base, 3, padding=1), nn.GELU(),
        )
        # 输出: 扰动 (无激活, 由外部 tanh 限制)
        self.out = nn.Conv1d(base, 1, 1)

    def forward(self, x):
        x0 = self.ms(self.stem(x))
        e1 = self.e1(x0)
        e2 = self.e2(self.p1(e1))
        b = self.neck(self.p2(e2))
        d2 = self.u2(b)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, e2.shape[-1], mode='linear', align_corners=False)
        d2 = self.d2(torch.cat([d2, e2], 1))
        d1 = self.u1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, e1.shape[-1], mode='linear', align_corners=False)
        d1 = self.d1(torch.cat([d1, e1], 1))
        if d1.shape[-1] != x0.shape[-1]:
            d1 = F.interpolate(d1, x0.shape[-1], mode='linear', align_corners=False)
        return self.out(self.refine(torch.cat([d1, x0], 1)))


# ========================= 正演模型 =========================
class ForwardModel(nn.Module):
    def __init__(self, wavelet, eps=1e-6):
        super().__init__()
        self.register_buffer("wavelet", wavelet.clone().float())
        self.eps = eps

    def forward(self, imp):
        imp_prev = torch.roll(imp, 1, -1)
        r = (imp - imp_prev) / (imp + imp_prev + self.eps)
        r[..., 0] = 0.0
        w = self.wavelet.view(1, 1, -1)
        pad = (w.shape[-1] - 1) // 2
        s = F.conv1d(r, w, padding=pad)
        return s[..., :imp.shape[-1]]


def ricker(f0, dt, length):
    t = np.arange(-length / 2, length / 2 + dt, dt, dtype=np.float64)
    pi2 = np.pi ** 2
    return ((1 - 2 * pi2 * f0**2 * t**2) * np.exp(-pi2 * f0**2 * t**2)).astype(np.float32)


# ========================= 损失函数 =========================
def pearson_corr_loss(pred, target):
    vp = pred - pred.mean(dim=-1, keepdim=True)
    vt = target - target.mean(dim=-1, keepdim=True)
    num = (vp * vt).sum(dim=-1)
    den = (vp.norm(dim=-1) * vt.norm(dim=-1)).clamp(min=1e-8)
    return (1.0 - num / den).mean()


def multiscale_pearson_loss(pred, target, scales=(1, 2, 4)):
    """多尺度Pearson相关: 在原始/2x/4x尺度都匹配"""
    total = 0.0
    weights = [1.0, 0.5, 0.25]
    for s, w in zip(scales, weights):
        if s == 1:
            total += w * pearson_corr_loss(pred, target)
        else:
            p_ds = F.avg_pool1d(pred, s)
            t_ds = F.avg_pool1d(target, s)
            total += w * pearson_corr_loss(p_ds, t_ds)
    return total / sum(weights)


def stft_spectral_loss(pred, target, n_fft=256, hop=64):
    p_n = pred / (pred.std(dim=-1, keepdim=True) + 1e-8)
    t_n = target / (target.std(dim=-1, keepdim=True) + 1e-8)
    p1, t1 = p_n.squeeze(1), t_n.squeeze(1)
    win = torch.hann_window(n_fft, device=pred.device)
    P_stft = torch.stft(p1, n_fft, hop, window=win, return_complex=True)
    T_stft = torch.stft(t1, n_fft, hop, window=win, return_complex=True)
    lp = torch.log(torch.abs(P_stft) + 1e-8)
    lt = torch.log(torch.abs(T_stft) + 1e-8)
    return F.l1_loss(lp, lt)


def gradient_consistency_loss(synth, target):
    s_n = synth / (synth.std(dim=-1, keepdim=True) + 1e-8)
    t_n = target / (target.std(dim=-1, keepdim=True) + 1e-8)
    return F.l1_loss(s_n[..., 1:] - s_n[..., :-1], t_n[..., 1:] - t_n[..., :-1])


# ========================= 高通滤波 =========================
def highpass_filter(data, cutoff_hz, fs=500.0):
    nyq = 0.5 * fs
    if cutoff_hz >= nyq:
        cutoff_hz = nyq * 0.9
    b, a = butter(4, cutoff_hz / nyq, btype='high')
    return filtfilt(b, a, data, axis=-1).astype(np.float32)


# ========================= 数据集 =========================
class TraceDS(Dataset):
    def __init__(self, seis, seis_hf, lf_model, aug=False):
        self.s = torch.from_numpy(seis.astype(np.float32))
        self.s_hf = torch.from_numpy(seis_hf.astype(np.float32))
        self.lf = torch.from_numpy(lf_model.astype(np.float32))
        self.aug = aug

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        s, sh, lf = self.s[idx], self.s_hf[idx], self.lf[idx]
        # 3通道: [地震, 高通, 低频模型]
        x = torch.stack([s, sh, lf], dim=0)
        d = {"x": x, "lf": lf.unsqueeze(0)}
        if self.aug and torch.rand(1).item() < 0.3:
            scale = 0.95 + 0.1 * torch.rand(1).item()
            # 只增强地震通道, 不增强低频模型
            d["x"][0] = d["x"][0] * scale
            d["x"][1] = d["x"][1] * scale
        return d


# ========================= 低频基础模型 =========================
def build_base_model(seismic_raw):
    """构建低频基础模型 (初始阻抗模型) — 在log域更平滑"""
    n_traces, n_samples = seismic_raw.shape
    P(f"  构建低频基础模型: {n_traces} x {n_samples}")

    trace_max = np.abs(seismic_raw).max(axis=1, keepdims=True) + 1e-10
    seis_norm = seismic_raw / trace_max

    # 包络
    envelope = np.zeros_like(seis_norm)
    for i in range(n_traces):
        analytic = hilbert(seis_norm[i])
        envelope[i] = np.abs(analytic)

    # 超低通
    lowpass = np.zeros_like(seis_norm)
    for i in range(n_traces):
        lowpass[i] = gaussian_filter1d(seis_norm[i], sigma=80)

    # 深度趋势 (在log-impedance空间线性增加)
    depth = np.linspace(0, 1, n_samples).reshape(1, -1)
    depth_trend = 0.35 + 0.30 * depth  # [0.35, 0.65] 在 [0,1] 空间

    # 组合
    lf = depth_trend + 0.03 * lowpass + 0.02 * envelope
    lf = gaussian_filter(lf, sigma=[8, 60])  # 更强平滑
    lf = np.clip(lf, 0.05, 0.95)

    P(f"  基础模型 [0,1] 范围: [{lf.min():.3f}, {lf.max():.3f}]")
    return lf


# ========================= 主流程 =========================
def main():
    SGY = Path(r'D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy')
    OUT = Path(r'D:\SEISMIC_CODING\new\sgy_inversion_v7')
    OUT.mkdir(parents=True, exist_ok=True)

    T_START, T_END, DT = 2500, 6000, 0.002
    FS = 1.0 / DT
    TRAIN_TRACES = 15000    # 增加训练道数
    TRACE_LEN = 1024
    STRIDE = 512
    BS = 48
    EPOCHS = 200
    LR = 3e-4
    INFER_TRACES = 5000
    WAVELET_F0 = 25.0
    HIGHPASS_CUTOFF = 10.0
    PATIENCE = 40

    IMP_MIN = 3000.0
    IMP_MAX = 15000.0
    PERT_SCALE = 0.5        # log域扰动缩放 (exp(0.5)≈1.65x)
    SMOOTH_SIGMA = 3.0      # 高斯平滑sigma
    SMOOTH_KS = 21          # 平滑核大小

    # 损失权重
    W_CORR = 1.0       # 多尺度Pearson (主物理损失)
    W_STFT = 0.2       # STFT频谱
    W_GRAD = 0.2       # 梯度一致性
    W_TV = 0.10        # TV正则化 (v6: 0.005 → v7: 0.1, 20倍!)
    W_PERT = 0.10      # 扰动L2惩罚 (保持接近基础模型)

    PHYS_WARMUP = 10

    np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    P(f"{'=' * 60}")
    P(f" 地震反演训练 v7 (扰动法 + 强制平滑)")
    P(f"{'=' * 60}")
    P(f" 设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    P(f" 核心: Z = Z_base × exp(tanh(smooth(net)) × {PERT_SCALE})")
    P(f" 平滑: sigma={SMOOTH_SIGMA}, kernel={SMOOTH_KS}")
    P(f" 输入: 3通道 (地震 + 高通@{HIGHPASS_CUTOFF}Hz + 基础模型)")
    P(f" 损失: corr={W_CORR}(多尺度) stft={W_STFT} grad={W_GRAD} tv={W_TV} pert={W_PERT}")

    # ================== 步骤1: 加载 ==================
    P(f"\n[1/6] 加载SGY数据...")
    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        n_total = f.tracecount
        n_samples = len(f.samples)
        P(f"  总道数: {n_total}, 采样点: {n_samples}")
        idx = np.linspace(0, n_total - 1, TRAIN_TRACES, dtype=int)
        seismic_raw = np.stack([f.trace[int(i)] for i in idx], axis=0).astype(np.float32)
    P(f"  抽样 {TRAIN_TRACES} 道, range=[{seismic_raw.min():.2e}, {seismic_raw.max():.2e}]")

    # ================== 步骤2: 数据准备 ==================
    P(f"\n[2/6] 准备训练数据...")

    # 低频基础模型 (在 [0,1] 空间)
    base_01 = build_base_model(seismic_raw)
    # 转为真实阻抗 (用于日志)
    base_imp = base_01 * (IMP_MAX - IMP_MIN) + IMP_MIN
    P(f"  基础阻抗范围: [{base_imp.min():.0f}, {base_imp.max():.0f}]")

    # 归一化地震
    trace_rms = np.sqrt(np.mean(seismic_raw**2, axis=1, keepdims=True)) + 1e-10
    seis_rms = seismic_raw / trace_rms
    s_mean = float(seis_rms.mean())
    s_std = float(seis_rms.std() + 1e-8)
    seis_norm = (seis_rms - s_mean) / s_std

    # 高通滤波
    P(f"  高通滤波: cutoff={HIGHPASS_CUTOFF}Hz, fs={FS}Hz")
    seis_hf_raw = highpass_filter(seismic_raw, HIGHPASS_CUTOFF, fs=FS)
    hf_rms = np.sqrt(np.mean(seis_hf_raw**2, axis=1, keepdims=True)) + 1e-10
    seis_hf_rms = seis_hf_raw / hf_rms
    hf_mean = float(seis_hf_rms.mean())
    hf_std = float(seis_hf_rms.std() + 1e-8)
    seis_hf_norm = (seis_hf_rms - hf_mean) / hf_std

    # 归一化基础模型到 [-1, 1] 范围 (作为网络输入)
    base_input = base_01 * 2.0 - 1.0  # [0,1] → [-1,1]

    # 切片
    segs_s, segs_hf, segs_base = [], [], []
    for start in range(0, n_samples - TRACE_LEN + 1, STRIDE):
        segs_s.append(seis_norm[:, start:start + TRACE_LEN])
        segs_hf.append(seis_hf_norm[:, start:start + TRACE_LEN])
        segs_base.append(base_01[:, start:start + TRACE_LEN])
    seis_all = np.concatenate(segs_s, axis=0)
    hf_all = np.concatenate(segs_hf, axis=0)
    base_all = np.concatenate(segs_base, axis=0)
    P(f"  切片: {seis_all.shape}")

    n = seis_all.shape[0]
    perm = np.random.permutation(n)
    nt = int(n * 0.85)
    tr_s, tr_hf, tr_base = seis_all[perm[:nt]], hf_all[perm[:nt]], base_all[perm[:nt]]
    va_s, va_hf, va_base = seis_all[perm[nt:]], hf_all[perm[nt:]], base_all[perm[nt:]]
    P(f"  训练: {tr_s.shape[0]}, 验证: {va_s.shape[0]}")

    # 基础模型输入归一化: 使用 [0,1] → [-1,1] 映射
    tr_base_input = tr_base * 2.0 - 1.0
    va_base_input = va_base * 2.0 - 1.0

    stats = {
        'seis_mean': s_mean, 'seis_std': s_std,
        'hf_mean': hf_mean, 'hf_std': hf_std,
        'imp_min': IMP_MIN, 'imp_max': IMP_MAX,
        'trace_len': TRACE_LEN, 'pert_scale': PERT_SCALE,
        'smooth_sigma': SMOOTH_SIGMA,
        'rms_mean': float(trace_rms.mean()),
        'highpass_cutoff': HIGHPASS_CUTOFF,
    }
    with open(OUT / 'norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    tr_ld = DataLoader(TraceDS(tr_s, tr_hf, tr_base_input, aug=True), batch_size=BS, shuffle=True,
                       drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True)
    va_ld = DataLoader(TraceDS(va_s, va_hf, va_base_input), batch_size=BS * 2, shuffle=False,
                       pin_memory=True, num_workers=2, persistent_workers=True)

    # ================== 步骤3: 模型 ==================
    P(f"\n[3/6] 构建模型...")
    model = InversionNet(in_ch=3, base=48).to(device)
    smoother = GaussianSmoothing(sigma=SMOOTH_SIGMA, kernel_size=SMOOTH_KS).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    P(f"  InversionNet(3ch) 参数量: {n_params:,}")

    wav = torch.from_numpy(ricker(WAVELET_F0, DT, 0.128)).to(device)
    fm = ForwardModel(wav).to(device)
    P(f"  Ricker子波: f0={WAVELET_F0}Hz, 长度={len(wav)}点")
    P(f"  高斯平滑: sigma={SMOOTH_SIGMA}, kernel={SMOOTH_KS}")

    # ================== 步骤4: 训练 ==================
    P(f"\n[4/6] 训练 ({EPOCHS} epochs)...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    scaler = GradScaler()
    best_val = float('inf')
    best_epoch = 0
    t_losses, v_losses, corr_history = [], [], []
    ckpt_dir = OUT / 'checkpoints'; ckpt_dir.mkdir(exist_ok=True)
    no_improve = 0
    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        ep_start = time.time()
        model.train()
        ep_loss, ep_corr, ep_stft, ep_grad, ep_tv, ep_pert, nb = 0., 0., 0., 0., 0., 0., 0
        phys_w = min(1.0, ep / PHYS_WARMUP)

        for batch in tr_ld:
            x_in = batch['x'].to(device, non_blocking=True)       # [B, 3, T]
            lf_base = batch['lf'].to(device, non_blocking=True)   # [B, 1, T] 基础模型 [0,1]

            # 恢复基础模型到 [0,1] (输入是 [-1,1])
            lf_01 = (lf_base + 1.0) / 2.0

            sx = x_in[:, 0:1, :]  # 地震通道

            opt.zero_grad(set_to_none=True)

            with autocast():
                raw_out = model(x_in)  # [B, 1, T] — 原始扰动

            # 平滑和扰动计算在FP32
            with autocast(enabled=False):
                raw_f32 = raw_out.float()
                # 强制平滑 → 消除高频振荡
                pert_smooth = smoother(raw_f32)
                # tanh限制 + 缩放
                pert = PERT_SCALE * torch.tanh(pert_smooth)

                # Z = Z_base × exp(perturbation)
                # Z_base = lf_01 * (IMP_MAX - IMP_MIN) + IMP_MIN
                base_imp_t = lf_01.float() * (IMP_MAX - IMP_MIN) + IMP_MIN
                pred_imp = base_imp_t * torch.exp(pert)
                pred_imp = pred_imp.clamp(IMP_MIN * 0.5, IMP_MAX * 1.5)

                # 正演
                synth = fm(pred_imp)

                # === 多尺度Pearson相关 ===
                loss_corr = multiscale_pearson_loss(synth, sx.float())

                # === STFT ===
                loss_stft = stft_spectral_loss(synth, sx.float())

                # === 梯度一致性 ===
                loss_grad = gradient_consistency_loss(synth, sx.float())

                # === TV: 对扰动施加 (让阻抗变化平缓) ===
                loss_tv = torch.abs(torch.diff(pert, dim=-1)).mean()

                # === 扰动L2: 保持接近基础模型 ===
                loss_pert = (pert ** 2).mean()

                # 总损失
                total = (phys_w * (W_CORR * loss_corr + W_STFT * loss_stft + W_GRAD * loss_grad)
                         + W_TV * loss_tv
                         + W_PERT * loss_pert)

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            ep_loss += total.item()
            ep_corr += loss_corr.item()
            ep_stft += loss_stft.item()
            ep_grad += loss_grad.item()
            ep_tv += loss_tv.item()
            ep_pert += loss_pert.item()
            nb += 1

        sched.step()
        t_losses.append(ep_loss / nb)
        avg_corr = ep_corr / nb
        corr_history.append(1.0 - avg_corr)

        # 验证
        model.eval()
        vl, vcorr, nv = 0., 0., 0
        with torch.no_grad():
            for batch in va_ld:
                x_v = batch['x'].to(device, non_blocking=True)
                lf_v = batch['lf'].to(device, non_blocking=True)
                lf_01_v = (lf_v + 1.0) / 2.0
                sx_v = x_v[:, 0:1, :].float()

                raw_v = model(x_v).float()
                pert_v = PERT_SCALE * torch.tanh(smoother(raw_v))
                base_v = lf_01_v.float() * (IMP_MAX - IMP_MIN) + IMP_MIN
                imp_v = base_v * torch.exp(pert_v)
                imp_v = imp_v.clamp(IMP_MIN * 0.5, IMP_MAX * 1.5)
                synth_v = fm(imp_v)

                vl += pearson_corr_loss(synth_v, sx_v).item()
                vp = synth_v - synth_v.mean(dim=-1, keepdim=True)
                vt = sx_v - sx_v.mean(dim=-1, keepdim=True)
                c = ((vp * vt).sum(dim=-1) / (vp.norm(dim=-1) * vt.norm(dim=-1) + 1e-8)).mean()
                vcorr += c.item()
                nv += 1

        v_losses.append(vl / max(nv, 1))
        v_corr = vcorr / max(nv, 1)

        if v_losses[-1] < best_val:
            best_val = v_losses[-1]
            best_epoch = ep
            no_improve = 0
            torch.save({'epoch': ep, 'model': model.state_dict(),
                        'val_loss': best_val, 'stats': stats}, ckpt_dir / 'best.pt')
        else:
            no_improve += 1

        ep_time = time.time() - ep_start
        if ep % 5 == 0 or ep == 1:
            elapsed = time.time() - t0
            P(f"  Ep {ep:3d}/{EPOCHS} | Loss: {t_losses[-1]:.5f} "
              f"(corr={avg_corr:.4f} stft={ep_stft/nb:.4f} grad={ep_grad/nb:.4f} "
              f"tv={ep_tv/nb:.4f} pert={ep_pert/nb:.4f}) "
              f"| Val_corr: {v_losses[-1]:.4f} (PCC={v_corr:.4f}) "
              f"| {ep_time:.1f}s/ep | {elapsed:.0f}s")

        if ep % 50 == 0:
            torch.save({'epoch': ep, 'model': model.state_dict(),
                        'val_loss': v_losses[-1], 'stats': stats}, ckpt_dir / f'epoch_{ep}.pt')

        if no_improve >= PATIENCE:
            P(f"  Early stopping @ ep {ep} (best_val={best_val:.6f} @ ep {best_epoch})")
            break

    total_time = time.time() - t0
    torch.save({'epoch': ep, 'model': model.state_dict(),
                'val_loss': v_losses[-1], 'stats': stats}, ckpt_dir / 'last.pt')
    P(f"  训练完成! best corr_loss={best_val:.6f} @ ep {best_epoch}")
    P(f"  最终PCC={corr_history[-1]:.4f}, 耗时: {total_time:.1f}s ({total_time/60:.1f}min)")

    # 训练曲线
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(t_losses, 'b-', lw=1.5, label='Train loss')
    ax1.plot(v_losses, 'r-', lw=1.5, label='Val corr loss')
    ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3); ax1.set_yscale('log')
    ax1.set_title('v7 Training Loss', fontsize=14, fontweight='bold')
    ax2.plot(corr_history, 'g-', lw=1.5)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Train PCC (synth vs seismic)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3); ax2.set_ylim(-0.1, 1.0)
    plt.tight_layout()
    plt.savefig(OUT / 'training_loss.png', dpi=150); plt.close()

    # ================== 步骤5: 推理 ==================
    P(f"\n[5/6] 反演推理...")
    ckpt = torch.load(ckpt_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    P(f"  加载 ep={ckpt['epoch']}, val={ckpt['val_loss']:.6f}")

    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        idx2 = np.linspace(0, n_total - 1, INFER_TRACES, dtype=int)
        seis_infer_raw = np.stack([f.trace[int(i)] for i in idx2], axis=0).astype(np.float32)

    # 推理数据归一化
    rms_infer = np.sqrt(np.mean(seis_infer_raw**2, axis=1, keepdims=True)) + 1e-10
    seis_infer_norm = (seis_infer_raw / rms_infer - s_mean) / s_std

    seis_infer_hf = highpass_filter(seis_infer_raw, HIGHPASS_CUTOFF, fs=FS)
    hf_rms_infer = np.sqrt(np.mean(seis_infer_hf**2, axis=1, keepdims=True)) + 1e-10
    seis_infer_hf_norm = (seis_infer_hf / hf_rms_infer - hf_mean) / hf_std

    # 推理基础模型
    base_infer_01 = build_base_model(seis_infer_raw)
    base_infer_input = base_infer_01 * 2.0 - 1.0  # → [-1,1]

    # Hann窗滑动推理
    preds_pert = np.zeros((INFER_TRACES, n_samples), dtype=np.float32)
    counts = np.zeros((INFER_TRACES, n_samples), dtype=np.float32)
    stride_inf = TRACE_LEN // 2
    INF_BS = 256
    hann = np.hanning(TRACE_LEN).astype(np.float32)

    P(f"  推理 {INFER_TRACES} 道 (片段长={TRACE_LEN})...")
    t_infer = time.time()
    with torch.no_grad():
        starts = list(range(0, n_samples - TRACE_LEN + 1, stride_inf))
        if starts[-1] + TRACE_LEN < n_samples:
            starts.append(n_samples - TRACE_LEN)

        for start in starts:
            seg_s = seis_infer_norm[:, start:start + TRACE_LEN]
            seg_hf = seis_infer_hf_norm[:, start:start + TRACE_LEN]
            seg_base = base_infer_input[:, start:start + TRACE_LEN]

            for i in range(0, seg_s.shape[0], INF_BS):
                b_s = torch.from_numpy(seg_s[i:i + INF_BS]).unsqueeze(1).float()
                b_hf = torch.from_numpy(seg_hf[i:i + INF_BS]).unsqueeze(1).float()
                b_base = torch.from_numpy(seg_base[i:i + INF_BS]).unsqueeze(1).float()
                x = torch.cat([b_s, b_hf, b_base], dim=1).to(device, non_blocking=True)

                raw = model(x).float()
                pert_out = PERT_SCALE * torch.tanh(smoother(raw))
                p = pert_out[:, 0, :].cpu().numpy()

                p_w = p * hann[np.newaxis, :]
                preds_pert[i:i + INF_BS, start:start + TRACE_LEN] += p_w
                counts[i:i + INF_BS, start:start + TRACE_LEN] += hann[np.newaxis, :]

    counts = np.maximum(counts, 1e-8)
    pert_final = preds_pert / counts

    # 最终阻抗: Z = Z_base × exp(perturbation)
    base_infer_imp = base_infer_01 * (IMP_MAX - IMP_MIN) + IMP_MIN
    impedance = base_infer_imp * np.exp(pert_final)
    impedance = np.clip(impedance, IMP_MIN * 0.5, IMP_MAX * 1.5)

    P(f"  阻抗范围: [{impedance.min():.0f}, {impedance.max():.0f}]")
    P(f"  阻抗均值: {impedance.mean():.0f}, 标准差: {impedance.std():.0f}")
    P(f"  扰动范围: [{pert_final.min():.3f}, {pert_final.max():.3f}]")
    P(f"  推理耗时: {time.time() - t_infer:.1f}s")

    # 合成地震
    P("  生成合成地震 (正演验证)...")
    wav_np = wav.cpu().numpy()
    synth_full = np.zeros_like(impedance)
    for i in range(impedance.shape[0]):
        imp_t = impedance[i]
        r = np.zeros_like(imp_t)
        r[1:] = (imp_t[1:] - imp_t[:-1]) / (imp_t[1:] + imp_t[:-1] + 1e-6)
        synth_full[i] = np.convolve(r, wav_np, mode='same')

    corr_values = []
    for i in range(0, impedance.shape[0], max(1, impedance.shape[0] // 100)):
        vo = seis_infer_raw[i] - seis_infer_raw[i].mean()
        vs = synth_full[i] - synth_full[i].mean()
        c = np.dot(vo, vs) / (np.linalg.norm(vo) * np.linalg.norm(vs) + 1e-10)
        corr_values.append(c)
    mean_corr = np.mean(corr_values)
    P(f"  正演匹配 Pearson 相关: {mean_corr:.4f}")

    np.save(OUT / 'seismic_raw.npy', seis_infer_raw)
    np.save(OUT / 'impedance_pred.npy', impedance)
    np.save(OUT / 'synth_seismic.npy', synth_full)
    np.save(OUT / 'perturbation.npy', pert_final)
    np.save(OUT / 'base_model.npy', base_infer_imp)

    # ================== 步骤6: 可视化 ==================
    P(f"\n[6/6] 可视化...")
    step = max(1, INFER_TRACES // 3000)
    sd = seis_infer_raw[::step, :].T
    id_ = impedance[::step, :].T
    sy = synth_full[::step, :].T
    bd = base_infer_imp[::step, :].T
    pt = pert_final[::step, :].T

    colors = [
        (0.0, '#000080'), (0.12, '#0000FF'), (0.24, '#00BFFF'),
        (0.36, '#00FF7F'), (0.48, '#ADFF2F'), (0.60, '#FFFF00'),
        (0.72, '#FFA500'), (0.84, '#FF4500'), (1.0, '#8B0000'),
    ]
    cmap = LinearSegmentedColormap.from_list('rainbow', [(c[0], c[1]) for c in colors])
    ext = [0, n_total, T_END, T_START]

    # --- 原始地震 ---
    fig, ax = plt.subplots(figsize=(18, 10))
    vm = np.percentile(np.abs(sd), 99)
    ax.imshow(sd, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
              extent=ext, interpolation='bilinear')
    ax.set_xlabel('Trace', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Original Seismic', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT / 'seismic_original.png', dpi=200, bbox_inches='tight')
    plt.close(); P("  seismic_original.png")

    # --- 反演阻抗 ---
    fig, ax = plt.subplots(figsize=(18, 10))
    v1, v2 = np.percentile(id_, [2, 98])
    im = ax.imshow(id_, aspect='auto', cmap=cmap, vmin=v1, vmax=v2,
                   extent=ext, interpolation='bilinear')
    ax.set_xlabel('Trace', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title('Impedance Inversion v7 (Perturbation + Smoothing)', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label='Impedance (kg/m2s)')
    plt.tight_layout()
    plt.savefig(OUT / 'impedance_inversion.png', dpi=200, bbox_inches='tight')
    plt.close(); P("  impedance_inversion.png")

    # --- 四图对比 (原始/基础模型/反演阻抗/合成) ---
    fig, axes = plt.subplots(4, 1, figsize=(18, 22))

    axes[0].imshow(sd, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
                   extent=ext, interpolation='bilinear')
    axes[0].set_ylabel('Time (ms)', fontsize=13, fontweight='bold')
    axes[0].set_title('Original Seismic', fontsize=15, fontweight='bold')

    bv1, bv2 = np.percentile(bd, [2, 98])
    im_b = axes[1].imshow(bd, aspect='auto', cmap=cmap, vmin=bv1, vmax=bv2,
                           extent=ext, interpolation='bilinear')
    axes[1].set_ylabel('Time (ms)', fontsize=13, fontweight='bold')
    axes[1].set_title('Base Model (Low-frequency)', fontsize=15, fontweight='bold')
    plt.colorbar(im_b, ax=axes[1], pad=0.02, shrink=0.85, label='Impedance')

    im2 = axes[2].imshow(id_, aspect='auto', cmap=cmap, vmin=v1, vmax=v2,
                         extent=ext, interpolation='bilinear')
    axes[2].set_ylabel('Time (ms)', fontsize=13, fontweight='bold')
    axes[2].set_title('Impedance Inversion v7 (Base + Perturbation)', fontsize=15, fontweight='bold')
    plt.colorbar(im2, ax=axes[2], pad=0.02, shrink=0.85, label='Impedance')

    sy_scale = vm / (np.percentile(np.abs(sy), 99) + 1e-10)
    axes[3].imshow(sy * sy_scale, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
                   extent=ext, interpolation='bilinear')
    axes[3].set_xlabel('Trace', fontsize=13, fontweight='bold')
    axes[3].set_ylabel('Time (ms)', fontsize=13, fontweight='bold')
    axes[3].set_title(f'Synthetic Seismic (Forward, PCC={mean_corr:.4f})', fontsize=15, fontweight='bold')

    plt.suptitle('Seismic Inversion v7 (Perturbation + Forced Smoothing)\n'
                 '0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / 'comparison.png', dpi=200, bbox_inches='tight')
    plt.close(); P("  comparison.png")

    # --- 扰动可视化 ---
    fig, ax = plt.subplots(figsize=(18, 10))
    pv = np.percentile(np.abs(pt), 99)
    ax.imshow(pt, aspect='auto', cmap='RdBu_r', vmin=-pv, vmax=pv,
              extent=ext, interpolation='bilinear')
    ax.set_xlabel('Trace', fontsize=14, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=14, fontweight='bold')
    ax.set_title(f'Log-domain Perturbation (range: [{pert_final.min():.3f}, {pert_final.max():.3f}])',
                 fontsize=16, fontweight='bold')
    plt.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-pv, pv), cmap='RdBu_r'),
                 ax=ax, pad=0.02, shrink=0.85, label='ln(Z/Z_base)')
    plt.tight_layout()
    plt.savefig(OUT / 'perturbation.png', dpi=200, bbox_inches='tight')
    plt.close(); P("  perturbation.png")

    # --- 单道对比 ---
    trace_indices = [INFER_TRACES // 4, INFER_TRACES // 2, 3 * INFER_TRACES // 4]
    fig, axes = plt.subplots(len(trace_indices), 2, figsize=(16, 4 * len(trace_indices)))
    t_axis = np.linspace(T_START, T_END, n_samples)

    for row, ti in enumerate(trace_indices):
        obs_t = seis_infer_raw[ti]
        syn_t = synth_full[ti]
        obs_n = obs_t / (obs_t.std() + 1e-10)
        syn_n = syn_t / (syn_t.std() + 1e-10)
        vo = obs_t - obs_t.mean()
        vs = syn_t - syn_t.mean()
        pcc = np.dot(vo, vs) / (np.linalg.norm(vo) * np.linalg.norm(vs) + 1e-10)

        axes[row, 0].plot(t_axis, obs_n, 'b-', lw=0.6, alpha=0.8, label='Observed')
        axes[row, 0].plot(t_axis, syn_n, 'r-', lw=0.6, alpha=0.8, label='Synthetic')
        axes[row, 0].set_title(f'Trace #{ti} Waveform (PCC={pcc:.4f})', fontsize=12, fontweight='bold')
        axes[row, 0].legend(fontsize=9); axes[row, 0].grid(True, alpha=0.3)
        axes[row, 0].set_ylabel('Normalized Amplitude')

        axes[row, 1].plot(t_axis, base_infer_imp[ti], 'k--', lw=0.5, alpha=0.5, label='Base Model')
        axes[row, 1].plot(t_axis, impedance[ti], 'g-', lw=0.8, label='Inversion')
        axes[row, 1].set_title(f'Trace #{ti} Impedance', fontsize=12, fontweight='bold')
        axes[row, 1].grid(True, alpha=0.3); axes[row, 1].legend(fontsize=9)
        axes[row, 1].set_ylabel('Impedance (kg/m2s)')

    for ax in axes[-1]:
        ax.set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.savefig(OUT / 'trace_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(); P("  trace_comparison.png")

    # ================== 完成 ==================
    total_all = time.time() - t0
    P(f"\n{'=' * 60}")
    P(f" 完成! -> {OUT}")
    P(f" 扰动范围: [{pert_final.min():.3f}, {pert_final.max():.3f}]")
    P(f" 正演匹配 PCC = {mean_corr:.4f}")
    P(f" 耗时: {total_all:.1f}s ({total_all/60:.1f}min)")
    P(f"{'=' * 60}")
    for f_ in sorted(OUT.glob('*')):
        if f_.is_file():
            P(f"  {f_.name}")


if __name__ == '__main__':
    main()
