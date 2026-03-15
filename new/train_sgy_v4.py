# -*- coding: utf-8 -*-
"""
地震反演 v4 — 物理驱动 + GPU加速
核心改进:
  1. 物理驱动为主: 正演合成地震 ↔ 实际地震 作为主损失
  2. 低频模型仅作弱约束 (正则化), 不做监督标签
  3. 多尺度物理匹配: 时域 + 频域 + 包络匹配
  4. 自适应阻抗范围: 模型输出经sigmoid映射到合理区间
  5. 去掉错误的结构相似性损失
  6. 减轻后处理平滑
"""
import json, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.signal import hilbert
from pathlib import Path
import segyio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
P = lambda *a, **kw: print(*a, **kw, flush=True)

# ========================= GPU加速 =========================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ========================= 模型 =========================
class ResConvBlock(nn.Module):
    """残差卷积块 — 更好的梯度流动"""
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, 7, padding=3), nn.BatchNorm1d(co), nn.GELU(),
            nn.Conv1d(co, co, 7, padding=3), nn.BatchNorm1d(co))
        self.act = nn.GELU()
        self.skip = nn.Conv1d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class UNet1D(nn.Module):
    def __init__(self, in_ch=1, base=64, depth=4):
        super().__init__()
        self.depth = depth
        chs = [base * (2**i) for i in range(depth)]
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(ResConvBlock(prev, c))
            self.pool.append(nn.MaxPool1d(2))
            prev = c
        self.bottleneck = ResConvBlock(prev, prev * 2)
        prev *= 2
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose1d(prev, c, 2, 2))
            self.dec.append(ResConvBlock(prev, c))
            prev = c
        self.out = nn.Sequential(
            nn.Conv1d(prev, prev // 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(prev // 2, 1, 1),
            nn.Sigmoid()      # 输出 [0, 1], 后续映射到阻抗范围
        )

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x); skips.append(x); x = self.pool[i](x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.up[i](x)
            s = skips[-(i+1)]
            if x.shape[-1] != s.shape[-1]:
                x = F.pad(x, (0, s.shape[-1] - x.shape[-1])) if s.shape[-1] > x.shape[-1] else x[..., :s.shape[-1]]
            x = self.dec[i](torch.cat([x, s], 1))
        return self.out(x)


class ForwardModel(nn.Module):
    """正演模型: 阻抗 → 反射系数 → 合成地震"""
    def __init__(self, wavelet, eps=1e-6):
        super().__init__()
        self.register_buffer("wavelet", wavelet.clone().float())
        self.eps = eps

    def forward(self, imp):
        # 反射系数
        imp_shift = torch.roll(imp, 1, -1)
        r = (imp - imp_shift) / (imp + imp_shift + self.eps)
        r[..., 0] = 0.0
        # 卷积子波
        w = self.wavelet.view(1, 1, -1)
        synth = F.conv1d(r, w, padding=(w.shape[-1] - 1) // 2)
        return synth[..., :imp.shape[-1]]


def ricker(f0, dt, length):
    t = np.arange(-length / 2, length / 2 + dt, dt, dtype=np.float64)
    pi2 = np.pi ** 2
    return ((1 - 2 * pi2 * f0**2 * t**2) * np.exp(-pi2 * f0**2 * t**2)).astype(np.float32)


class TraceDS(Dataset):
    def __init__(self, seis, lf_imp=None):
        self.s = torch.from_numpy(seis.astype(np.float32))
        self.lf = torch.from_numpy(lf_imp.astype(np.float32)) if lf_imp is not None else None

    def __len__(self): return len(self.s)

    def __getitem__(self, idx):
        d = {"seis": self.s[idx].unsqueeze(0)}
        if self.lf is not None:
            d["lf_imp"] = self.lf[idx].unsqueeze(0)
        return d


# ========================= 低频模型 (仅做正则化) =========================
def build_low_freq_model(seismic_raw):
    """
    构建低频阻抗模型:
    - 包络提供相对变化
    - 大尺度低通提供平滑趋势
    - 仅作为正则化约束, 不作为监督标签
    """
    n_traces, n_samples = seismic_raw.shape
    P(f"  构建低频模型: {n_traces} 道 x {n_samples} 采样点")

    # 逐道归一化
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
        lowpass[i] = gaussian_filter1d(seis_norm[i], sigma=30)

    # 深度趋势  (较温和)
    depth = np.linspace(0, 1, n_samples).reshape(1, -1)
    depth_trend = 5500.0 + 3000.0 * depth

    # 组合 — 低频模型, 深度趋势为主
    lf_model = depth_trend + 500.0 * lowpass + 300.0 * envelope

    # 非常强的平滑 — 这只是低频约束
    lf_model = gaussian_filter(lf_model, sigma=[5, 30])
    lf_model = np.clip(lf_model, 3000, 15000)

    P(f"  低频模型范围: [{lf_model.min():.0f}, {lf_model.max():.0f}] kg/m²s")
    return lf_model


# ========================= 多尺度物理损失 =========================
def multiscale_physics_loss(synth, obs):
    """
    多尺度物理匹配损失:
    1. 归一化波形匹配 (时域)
    2. 包络匹配
    3. 多尺度匹配 (不同平滑尺度)
    """
    # 1. 归一化波形匹配
    obs_m = obs.mean(dim=-1, keepdim=True)
    obs_s = obs.std(dim=-1, keepdim=True) + 1e-8
    syn_m = synth.mean(dim=-1, keepdim=True)
    syn_s = synth.std(dim=-1, keepdim=True) + 1e-8
    obs_n = (obs - obs_m) / obs_s
    syn_n = (synth - syn_m) / syn_s

    # L1 + L2 时域匹配
    loss_wave = F.smooth_l1_loss(syn_n, obs_n) + 0.5 * F.mse_loss(syn_n, obs_n)

    # 2. 归一化互相关 (越大越好, 取负)
    ncc = (obs_n * syn_n).mean(dim=-1).mean()
    loss_ncc = 1.0 - ncc

    # 3. 包络匹配 — 用平滑近似包络
    # 对差的绝对值做平滑来近似包络差异
    diff_abs = torch.abs(syn_n - obs_n)
    loss_env = diff_abs.mean()

    return loss_wave + 0.3 * loss_ncc + 0.2 * loss_env


# ========================= 主流程 =========================
def main():
    SGY = Path(r'D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy')
    OUT = Path(r'D:\SEISMIC_CODING\new\sgy_inversion_v4')
    OUT.mkdir(parents=True, exist_ok=True)

    T_START, T_END, DT = 2500, 6000, 0.002
    TRAIN_TRACES = 10000
    TRACE_LEN = 512
    STRIDE = 256
    BS = 64
    EPOCHS = 150
    LR = 5e-4
    INFER_TRACES = 5000
    WAVELET_F0 = 25.0
    PATIENCE = 30

    # 阻抗映射范围 (sigmoid输出[0,1] → [IMP_MIN, IMP_MAX])
    IMP_MIN = 3000.0
    IMP_MAX = 15000.0

    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P(f"{'='*55}\n 地震反演训练 v4 (物理驱动+GPU加速)\n{'='*55}")
    P(f" 设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    P(f" 核心: 物理正演匹配为主 + 低频模型正则化")
    P(f" 阻抗范围: [{IMP_MIN}, {IMP_MAX}] kg/m²s")
    P(f" 模型: UNet1D(base=64, depth=4, ResBlock)")
    P(f" 训练: BS={BS}, Epochs={EPOCHS}, LR={LR}")

    # ================== 步骤1: 加载 ==================
    P(f"\n[1/6] 加载SGY数据...")
    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        n_total = f.tracecount
        n_samples = len(f.samples)
        P(f"  总道数: {n_total}, 采样点: {n_samples}, dt: {f.bin[segyio.BinField.Interval]}μs")
        idx = np.linspace(0, n_total - 1, TRAIN_TRACES, dtype=int)
        seismic_raw = np.stack([f.trace[int(i)] for i in idx], axis=0).astype(np.float32)
    P(f"  抽样 {TRAIN_TRACES} 道, range=[{seismic_raw.min():.2e}, {seismic_raw.max():.2e}]")

    # ================== 步骤2: 数据准备 ==================
    P(f"\n[2/6] 准备训练数据...")

    # 低频模型 (仅做正则化)
    lf_model = build_low_freq_model(seismic_raw)

    # 地震数据归一化: 逐道RMS归一化 + 全局标准化
    trace_rms = np.sqrt(np.mean(seismic_raw**2, axis=1, keepdims=True)) + 1e-10
    seis_rms = seismic_raw / trace_rms
    s_mean, s_std = seis_rms.mean(), seis_rms.std() + 1e-8
    seis_norm = (seis_rms - s_mean) / s_std

    # 低频模型归一化到 [0, 1] (与sigmoid输出对应)
    lf_01 = (lf_model - IMP_MIN) / (IMP_MAX - IMP_MIN)
    lf_01 = np.clip(lf_01, 0, 1)

    # 切片
    segs_s, segs_lf = [], []
    for start in range(0, n_samples - TRACE_LEN + 1, STRIDE):
        segs_s.append(seis_norm[:, start:start + TRACE_LEN])
        segs_lf.append(lf_01[:, start:start + TRACE_LEN])
    seis_all = np.concatenate(segs_s, axis=0)
    lf_all = np.concatenate(segs_lf, axis=0)
    P(f"  切片: {seis_all.shape}")

    n = seis_all.shape[0]
    perm = np.random.permutation(n)
    nt = int(n * 0.85)
    tr_s, tr_lf = seis_all[perm[:nt]], lf_all[perm[:nt]]
    va_s, va_lf = seis_all[perm[nt:]], lf_all[perm[nt:]]
    P(f"  训练: {tr_s.shape[0]}, 验证: {va_s.shape[0]}")

    stats = {
        'seis_rms_mean': float(trace_rms.mean()),
        'seis_mean': float(s_mean), 'seis_std': float(s_std),
        'imp_min': float(IMP_MIN), 'imp_max': float(IMP_MAX),
        'trace_len': TRACE_LEN,
    }
    with open(OUT / 'norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    tr_ld = DataLoader(TraceDS(tr_s, tr_lf), batch_size=BS, shuffle=True,
                       drop_last=True, pin_memory=True, num_workers=4, persistent_workers=True)
    va_ld = DataLoader(TraceDS(va_s, va_lf), batch_size=BS * 2, shuffle=False,
                       pin_memory=True, num_workers=2, persistent_workers=True)

    # ================== 步骤3: 模型 ==================
    P(f"\n[3/6] 构建模型...")
    model = UNet1D(in_ch=1, base=64, depth=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    P(f"  参数量: {n_params:,}")

    wav = torch.from_numpy(ricker(WAVELET_F0, DT, 0.128)).to(device)
    fm = ForwardModel(wav).to(device)
    P(f"  Ricker子波: f0={WAVELET_F0}Hz, 长度={len(wav)}点")

    # ================== 步骤4: 训练 ==================
    P(f"\n[4/6] 训练 ({EPOCHS} epochs, 物理驱动)...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()
    best_val = float('inf')
    t_losses, v_losses = [], []
    ckpt_dir = OUT / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    no_improve = 0
    best_epoch = 0

    # 损失权重
    W_PHYS = 1.0           # 物理损失 (主导)
    W_LF = 0.05            # 低频模型约束 (弱正则化)
    W_SMOOTH = 0.01        # 平滑正则化
    W_TV = 0.005            # TV正则化

    # 物理损失warmup: 逐渐增加物理约束
    PHYS_WARMUP = 15

    t0 = time.time()

    for ep in range(1, EPOCHS + 1):
        ep_start = time.time()
        model.train()
        ep_loss, ep_phys, ep_lf, ep_smooth, nb = 0., 0., 0., 0., 0
        phys_w = W_PHYS * min(1.0, ep / PHYS_WARMUP)

        for batch in tr_ld:
            sx = batch['seis'].to(device, non_blocking=True)   # [B, 1, L] 归一化地震
            lf = batch['lf_imp'].to(device, non_blocking=True) # [B, 1, L] 低频模型 [0,1]

            opt.zero_grad(set_to_none=True)

            with autocast():
                # 模型输出 [0,1] → 映射到真实阻抗
                pred_01 = model(sx)                              # [B, 1, L]  sigmoid已在模型内
                pred_imp = pred_01 * (IMP_MAX - IMP_MIN) + IMP_MIN  # [B, 1, L]  真实阻抗

                # === 物理损失 (主导) ===
                # 正演: 阻抗 → 合成地震
                synth = fm(pred_imp)                             # [B, 1, L]

                # 真实地震 (反归一化)
                obs = sx * s_std + s_mean                        # [B, 1, L]
                obs = obs * trace_rms.mean()                     # 恢复大致幅度

                loss_phys = multiscale_physics_loss(synth, obs)

                # === 低频模型约束 (弱正则化) ===
                loss_lf = F.smooth_l1_loss(pred_01, lf)

                # === 平滑正则化 ===
                diff1 = torch.diff(pred_01, dim=-1)
                loss_smooth = (diff1 ** 2).mean()

                # === TV正则化 ===
                loss_tv = torch.abs(diff1).mean()

                total = (phys_w * loss_phys
                         + W_LF * loss_lf
                         + W_SMOOTH * loss_smooth
                         + W_TV * loss_tv)

            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            ep_loss += total.item()
            ep_phys += loss_phys.item()
            ep_lf += loss_lf.item()
            ep_smooth += loss_smooth.item()
            nb += 1

        sched.step()
        t_losses.append(ep_loss / nb)

        # 验证: 用物理损失作为验证指标 (不是拟合伪标签的MSE)
        model.eval()
        vl, nv = 0., 0
        with torch.no_grad():
            for batch in va_ld:
                with autocast():
                    sx_v = batch['seis'].to(device, non_blocking=True)
                    pred_01_v = model(sx_v)
                    pred_imp_v = pred_01_v * (IMP_MAX - IMP_MIN) + IMP_MIN
                    synth_v = fm(pred_imp_v)
                    obs_v = sx_v * s_std + s_mean
                    obs_v = obs_v * trace_rms.mean()
                    vl += multiscale_physics_loss(synth_v, obs_v).item()
                nv += 1
        v_losses.append(vl / max(nv, 1))

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
              f"(phys={ep_phys/nb:.5f} lf={ep_lf/nb:.5f} sm={ep_smooth/nb:.6f}) "
              f"| Val_phys: {v_losses[-1]:.5f} | {ep_time:.1f}s/ep | 总{elapsed:.0f}s "
              f"| phys_w={phys_w:.3f} | LR={opt.param_groups[0]['lr']:.1e}")

        if no_improve >= PATIENCE:
            P(f"  Early stopping @ epoch {ep} (patience={PATIENCE}, best_val={best_val:.6f}, best_ep={best_epoch})")
            break

    total_time = time.time() - t0
    torch.save({'epoch': ep, 'model': model.state_dict(),
                'val_loss': v_losses[-1], 'stats': stats}, ckpt_dir / 'last.pt')
    P(f"  训练完成! best_val={best_val:.6f} @ epoch {best_epoch}, 总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")

    # 训练曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(t_losses, 'b-', lw=1.5, label='训练')
    ax.plot(v_losses, 'r-', lw=1.5, label='验证(物理)')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('训练损失曲线 v4 (物理驱动)', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(OUT / 'training_loss.png', dpi=150)
    plt.close()

    # ================== 步骤5: 推理 ==================
    P(f"\n[5/6] 反演推理...")
    ckpt = torch.load(ckpt_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    P(f"  加载最优模型 (epoch={ckpt['epoch']}, val={ckpt['val_loss']:.6f})")

    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        idx2 = np.linspace(0, n_total - 1, INFER_TRACES, dtype=int)
        seis_infer_raw = np.stack([f.trace[int(i)] for i in idx2], axis=0).astype(np.float32)

    rms_infer = np.sqrt(np.mean(seis_infer_raw**2, axis=1, keepdims=True)) + 1e-10
    seis_infer_rms = seis_infer_raw / rms_infer
    seis_infer_norm = (seis_infer_rms - s_mean) / s_std

    # Hann窗加权滑窗推理
    preds_01 = np.zeros((INFER_TRACES, n_samples), dtype=np.float32)
    counts = np.zeros((INFER_TRACES, n_samples), dtype=np.float32)
    stride_inf = TRACE_LEN // 2
    INF_BS = 512
    hann = np.hanning(TRACE_LEN).astype(np.float32)

    P(f"  推理 {INFER_TRACES} 道...")
    t_infer = time.time()
    with torch.no_grad():
        starts = list(range(0, n_samples - TRACE_LEN + 1, stride_inf))
        if starts[-1] + TRACE_LEN < n_samples:
            starts.append(n_samples - TRACE_LEN)

        for start in starts:
            end = start + TRACE_LEN
            seg = seis_infer_norm[:, start:end]
            for i in range(0, seg.shape[0], INF_BS):
                b = seg[i:i + INF_BS]
                x = torch.from_numpy(b).unsqueeze(1).float().to(device, non_blocking=True)
                with autocast():
                    p = model(x)[:, 0, :].float().cpu().numpy()
                p_w = p * hann[np.newaxis, :]
                preds_01[i:i + INF_BS, start:end] += p_w
                counts[i:i + INF_BS, start:end] += hann[np.newaxis, :]

    counts = np.maximum(counts, 1e-8)
    preds_01 /= counts

    # 映射到真实阻抗
    impedance = preds_01 * (IMP_MAX - IMP_MIN) + IMP_MIN

    # 轻度后处理 (v3用了sigma=[3,8], 这里减少到[1,3])
    impedance = gaussian_filter(impedance, sigma=[1, 3])

    P(f"  阻抗范围: [{impedance.min():.0f}, {impedance.max():.0f}]")
    P(f"  推理耗时: {time.time() - t_infer:.1f}s")
    np.save(OUT / 'seismic_raw.npy', seis_infer_raw)
    np.save(OUT / 'impedance_pred.npy', impedance)

    # ================== 步骤6: 可视化 ==================
    P(f"\n[6/6] 可视化...")

    step = max(1, INFER_TRACES // 3000)
    sd = seis_infer_raw[::step, :].T
    id_ = impedance[::step, :].T

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
    im = ax.imshow(sd, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
                   extent=ext, interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('原始地震剖面\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label='振幅')
    plt.tight_layout()
    plt.savefig(OUT / 'seismic_original.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("  seismic_original.png")

    # --- 反演阻抗 ---
    fig, ax = plt.subplots(figsize=(18, 10))
    v1, v2 = np.percentile(id_, [2, 98])
    im = ax.imshow(id_, aspect='auto', cmap=cmap, vmin=v1, vmax=v2,
                   extent=ext, interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('波阻抗反演剖面 v4 (物理驱动)\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85, label='波阻抗 (kg/m²s)')
    plt.tight_layout()
    plt.savefig(OUT / 'impedance_inversion.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("  impedance_inversion.png")

    # --- 合成地震对比 ---
    # 生成合成地震验证正演匹配质量
    P("  生成合成地震对比...")
    synth_full = np.zeros_like(impedance)
    wav_np = wav.cpu().numpy()
    for i in range(impedance.shape[0]):
        imp_t = impedance[i]
        r = np.zeros_like(imp_t)
        r[1:] = (imp_t[1:] - imp_t[:-1]) / (imp_t[1:] + imp_t[:-1] + 1e-6)
        synth_full[i] = np.convolve(r, wav_np, mode='same')

    sd_synth = synth_full[::step, :].T

    # --- 三图对比 ---
    fig, axes = plt.subplots(3, 1, figsize=(18, 18))

    im1 = axes[0].imshow(sd, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
                         extent=ext, interpolation='bilinear')
    axes[0].set_ylabel('时间 (ms)', fontsize=13, fontweight='bold')
    axes[0].set_title('原始地震剖面', fontsize=15, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], pad=0.02, shrink=0.85, label='振幅')

    im2 = axes[1].imshow(id_, aspect='auto', cmap=cmap, vmin=v1, vmax=v2,
                         extent=ext, interpolation='bilinear')
    axes[1].set_ylabel('时间 (ms)', fontsize=13, fontweight='bold')
    axes[1].set_title('反演波阻抗剖面 v4 (物理驱动)', fontsize=15, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], pad=0.02, shrink=0.85, label='波阻抗 (kg/m²s)')

    vm_s = np.percentile(np.abs(sd_synth), 99)
    im3 = axes[2].imshow(sd_synth, aspect='auto', cmap=cmap, vmin=-vm_s, vmax=vm_s,
                         extent=ext, interpolation='bilinear')
    axes[2].set_xlabel('道号 (Trace)', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('时间 (ms)', fontsize=13, fontweight='bold')
    axes[2].set_title('合成地震剖面 (正演验证)', fontsize=15, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], pad=0.02, shrink=0.85, label='振幅')

    plt.suptitle('地震反演对比 v4 (物理驱动+GPU加速)\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT / 'comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("  comparison.png")

    # --- 单道对比 (验证波形匹配) ---
    trace_idx = INFER_TRACES // 2
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    t_axis = np.linspace(T_START, T_END, n_samples)

    axes[0].plot(t_axis, seis_infer_raw[trace_idx], 'b-', lw=0.8, label='原始地震')
    axes[0].plot(t_axis, synth_full[trace_idx] * (np.abs(seis_infer_raw[trace_idx]).max() / (np.abs(synth_full[trace_idx]).max() + 1e-10)),
                 'r-', lw=0.8, alpha=0.7, label='合成地震(归一化)')
    axes[0].set_xlabel('时间 (ms)', fontsize=12)
    axes[0].set_ylabel('振幅', fontsize=12)
    axes[0].set_title(f'单道波形对比 (道 #{trace_idx})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_axis, impedance[trace_idx], 'g-', lw=1.0)
    axes[1].set_xlabel('时间 (ms)', fontsize=12)
    axes[1].set_ylabel('波阻抗 (kg/m²s)', fontsize=12)
    axes[1].set_title(f'反演阻抗曲线 (道 #{trace_idx})', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT / 'trace_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    P("  trace_comparison.png")

    P(f"\n{'='*55}\n 全部完成! -> {OUT}\n 总耗时: {time.time() - t0:.1f}s ({(time.time()-t0)/60:.1f}分钟)\n{'='*55}")
    for f_ in sorted(OUT.iterdir()):
        if f_.is_file():
            P(f"  {f_.name}")


if __name__ == '__main__':
    main()
