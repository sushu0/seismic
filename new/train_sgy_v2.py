# -*- coding: utf-8 -*-
"""
0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy 地震反演 v2
修复问题:
  1. 改用道积分+低频趋势的方式生成更合理的伪阻抗
  2. 逐道归一化保留横向变化
  3. 大幅加强物理约束（正演一致性为主损失）
  4. 增大模型容量
  5. 阻抗用jet/rainbow专业色图
"""
import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.signal import hilbert
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import segyio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
P = lambda *a, **kw: print(*a, **kw, flush=True)

# ========================= 模型 =========================
class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, 3, padding=1), nn.BatchNorm1d(co), nn.GELU(),
            nn.Conv1d(co, co, 3, padding=1), nn.BatchNorm1d(co), nn.GELU())
    def forward(self, x):
        return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_ch=1, base=48, depth=5):
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
        prev *= 2
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose1d(prev, c, 2, 2))
            self.dec.append(ConvBlock(prev, c))
            prev = c
        self.out = nn.Conv1d(prev, 1, 1)

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x); skips.append(x); x = self.pool[i](x)
        x = self.bottleneck(x)
        for i in range(self.depth):
            x = self.up[i](x); s = skips[-(i+1)]
            if x.shape[-1] != s.shape[-1]:
                x = F.pad(x, (0, s.shape[-1]-x.shape[-1])) if s.shape[-1]>x.shape[-1] else x[...,:s.shape[-1]]
            x = self.dec[i](torch.cat([x, s], 1))
        return self.out(x)


class ForwardModel(nn.Module):
    """阻抗 -> 反射系数 -> 合成地震"""
    def __init__(self, wavelet, eps=1e-6):
        super().__init__()
        self.register_buffer("wavelet", wavelet.clone().float())
        self.eps = eps
    def forward(self, imp):
        # imp: [B,1,T]  值域为正实数
        imp_prev = torch.roll(imp, 1, -1)
        r = (imp - imp_prev) / (imp + imp_prev + self.eps)
        r[..., 0] = 0.0
        w = self.wavelet.view(1,1,-1)
        s = F.conv1d(r, w, padding=(w.shape[-1]-1)//2)
        return s[...,:imp.shape[-1]]


def ricker(f0, dt, length):
    t = np.arange(-length/2, length/2+dt, dt, dtype=np.float64)
    pi2 = np.pi**2
    return ((1-2*pi2*f0**2*t**2)*np.exp(-pi2*f0**2*t**2)).astype(np.float32)


# ========================= 数据集 =========================
class TraceDS(Dataset):
    def __init__(self, seis, imp):
        self.s = seis.astype(np.float32)
        self.i = imp.astype(np.float32)
    def __len__(self): return len(self.s)
    def __getitem__(self, idx):
        return {"seis": torch.from_numpy(self.s[idx]).unsqueeze(0),
                "imp": torch.from_numpy(self.i[idx]).unsqueeze(0)}


# ========================= 核心: 伪阻抗生成 =========================
def generate_pseudo_impedance(seismic_raw):
    """
    用更合理的方法从地震数据生成伪阻抗:
    1. 逐道归一化
    2. 反射系数 ≈ 地震振幅 (去子波效应简化)
    3. 阻抗 = Z0 * exp(2 * cumsum(r))  (Fatti公式的简化)
    4. 叠加低频趋势模型
    """
    n_traces, n_samples = seismic_raw.shape
    P(f"  生成伪阻抗: {n_traces} 道 x {n_samples} 采样点")

    # 1. 逐道归一化到 [-1,1] 范围 (保留相对振幅关系)
    trace_max = np.abs(seismic_raw).max(axis=1, keepdims=True) + 1e-10
    seis_norm = seismic_raw / trace_max

    # 2. 带通滤波效果: 平滑反射系数序列
    r = np.zeros_like(seis_norm)
    for i in range(n_traces):
        r[i] = gaussian_filter(seis_norm[i], sigma=1.5)

    # 3. 缩放反射系数到合理范围 (典型值 |r| < 0.1)
    r_max = np.percentile(np.abs(r), 99)
    r = r / (r_max + 1e-10) * 0.08

    # 4. 阻抗 = Z0 * exp(2 * cumulative_sum(r))
    Z0 = 6000.0  # 基准阻抗 (典型沉积岩)
    log_imp = 2.0 * np.cumsum(r, axis=1)

    # 5. 添加低频线性趋势 (阻抗通常随深度增大)
    depth_trend = np.linspace(0, 0.3, n_samples).reshape(1, -1)
    log_imp = log_imp + depth_trend

    impedance = Z0 * np.exp(log_imp)

    # 6. 平滑以去除高频噪声
    for i in range(n_traces):
        impedance[i] = gaussian_filter(impedance[i], sigma=3)

    P(f"  伪阻抗范围: [{impedance.min():.0f}, {impedance.max():.0f}] kg/m2s")
    return impedance


# ========================= 主流程 =========================
def main():
    SGY = Path(r'D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy')
    OUT = Path(r'D:\SEISMIC_CODING\new\sgy_inversion_v2')
    OUT.mkdir(parents=True, exist_ok=True)

    T_START, T_END, DT = 2500, 6000, 0.002
    TRAIN_TRACES = 6000
    TRACE_LEN = 512
    STRIDE = 256
    BS, EPOCHS, LR = 64, 100, 5e-4
    INFER_TRACES = 5000
    WAVELET_F0 = 25.0

    np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    P(f"{'='*55}\n 地震反演训练 v2 (优化版)\n{'='*55}")
    P(f" 设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ""))

    # ================== 步骤1: 加载 ==================
    P(f"\n[1/6] 加载SGY数据...")
    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        n_total = f.tracecount
        n_samples = len(f.samples)
        P(f"  总道数: {n_total}, 采样点: {n_samples}, dt: {f.bin[segyio.BinField.Interval]}us")
        idx = np.linspace(0, n_total-1, TRAIN_TRACES, dtype=int)
        seismic_raw = np.stack([f.trace[int(i)] for i in idx], axis=0).astype(np.float32)
    P(f"  抽样 {TRAIN_TRACES} 道, range=[{seismic_raw.min():.2e}, {seismic_raw.max():.2e}]")

    # ================== 步骤2: 数据准备 ==================
    P(f"\n[2/6] 准备训练数据...")

    # 生成伪阻抗
    pseudo_imp = generate_pseudo_impedance(seismic_raw)

    # 逐道归一化地震数据
    trace_rms = np.sqrt(np.mean(seismic_raw**2, axis=1, keepdims=True)) + 1e-10
    seis_norm = seismic_raw / trace_rms  # 保幅归一化

    # 全局统计 (用于推理时)
    s_mean, s_std = seis_norm.mean(), seis_norm.std() + 1e-8
    seis_global = (seis_norm - s_mean) / s_std

    i_mean, i_std = pseudo_imp.mean(), pseudo_imp.std() + 1e-8
    imp_global = (pseudo_imp - i_mean) / i_std

    # 切片
    segs_s, segs_i = [], []
    for start in range(0, n_samples - TRACE_LEN + 1, STRIDE):
        segs_s.append(seis_global[:, start:start+TRACE_LEN])
        segs_i.append(imp_global[:, start:start+TRACE_LEN])
    seis_all = np.concatenate(segs_s, axis=0)
    imp_all = np.concatenate(segs_i, axis=0)
    P(f"  切片: {seis_all.shape}")

    # 划分
    n = seis_all.shape[0]
    perm = np.random.permutation(n)
    nt = int(n * 0.85)
    tr_s, tr_i = seis_all[perm[:nt]], imp_all[perm[:nt]]
    va_s, va_i = seis_all[perm[nt:]], imp_all[perm[nt:]]
    P(f"  训练: {tr_s.shape[0]}, 验证: {va_s.shape[0]}")

    stats = {
        'seis_rms_mean': float(trace_rms.mean()),
        'seis_mean': float(s_mean), 'seis_std': float(s_std),
        'imp_mean': float(i_mean), 'imp_std': float(i_std),
        'trace_len': TRACE_LEN, 'Z0': 6000.0,
    }
    with open(OUT/'norm_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    tr_ld = DataLoader(TraceDS(tr_s, tr_i), batch_size=BS, shuffle=True, drop_last=True, pin_memory=True)
    va_ld = DataLoader(TraceDS(va_s, va_i), batch_size=BS*2, shuffle=False, pin_memory=True)

    # ================== 步骤3: 模型 ==================
    P(f"\n[3/6] 构建模型...")
    model = UNet1D(in_ch=1, base=48, depth=5).to(device)
    P(f"  UNet1D(base=48, depth=5) 参数: {sum(p.numel() for p in model.parameters()):,}")

    wav = torch.from_numpy(ricker(WAVELET_F0, DT, 0.128)).to(device)
    fm = ForwardModel(wav).to(device)
    P(f"  Ricker子波: f0={WAVELET_F0}Hz, 长度={len(wav)}点")

    # ================== 步骤4: 训练 ==================
    P(f"\n[4/6] 训练 ({EPOCHS} epochs)...")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss()
    best_val, t_losses, v_losses = float('inf'), [], []
    ckpt_dir = OUT / 'checkpoints'; ckpt_dir.mkdir(exist_ok=True)

    # 物理约束权重: 从0.01线性增大到0.5 (预热策略)
    PHYS_MAX = 0.5
    PHYS_WARMUP = 20  # epoch

    for ep in range(1, EPOCHS+1):
        model.train()
        ep_loss, ep_sup, ep_phys, nb = 0., 0., 0., 0
        # 物理权重随epoch增大
        lam_phys = min(PHYS_MAX, PHYS_MAX * ep / PHYS_WARMUP)

        for batch in tr_ld:
            sx = batch['seis'].to(device)   # [B,1,T] 归一化地震
            iy = batch['imp'].to(device)    # [B,1,T] 归一化阻抗

            pred_norm = model(sx)  # 预测归一化阻抗

            # ---- 监督损失 ----
            loss_sup = sl1(pred_norm, iy) + 0.3 * mse(pred_norm, iy)

            # ---- 物理约束损失 ----
            # 反归一化到物理域阻抗 (正值)
            pred_phys = pred_norm * stats['imp_std'] + stats['imp_mean']
            pred_phys = F.softplus(pred_phys)  # 确保正值

            # 正演合成地震道
            seis_synth = fm(pred_phys)  # [B,1,T]

            # 逐sample归一化: 让合成和观测在相同尺度比较
            # 观测地震 (反归一化)
            seis_obs = sx * stats['seis_std'] + stats['seis_mean']
            # 逐批次归一化
            obs_std = seis_obs.std(dim=-1, keepdim=True) + 1e-8
            syn_std = seis_synth.std(dim=-1, keepdim=True) + 1e-8
            obs_n = (seis_obs - seis_obs.mean(dim=-1, keepdim=True)) / obs_std
            syn_n = (seis_synth - seis_synth.mean(dim=-1, keepdim=True)) / syn_std

            loss_phys = mse(syn_n, obs_n)

            # ---- 平滑正则 ----
            # 预测阻抗应该较平滑
            diff1 = torch.diff(pred_norm, dim=-1)
            loss_smooth = (diff1 ** 2).mean() * 0.01

            total = loss_sup + lam_phys * loss_phys + loss_smooth

            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

            ep_loss += total.item()
            ep_sup += loss_sup.item()
            ep_phys += loss_phys.item()
            nb += 1

        sched.step()
        t_losses.append(ep_loss/nb)

        # 验证
        model.eval(); vl, nv = 0., 0
        with torch.no_grad():
            for batch in va_ld:
                p = model(batch['seis'].to(device))
                vl += mse(p, batch['imp'].to(device)).item(); nv += 1
        v_losses.append(vl/max(nv,1))

        if v_losses[-1] < best_val:
            best_val = v_losses[-1]
            torch.save({'epoch':ep, 'model':model.state_dict(), 'val_loss':best_val, 'stats':stats}, ckpt_dir/'best.pt')

        if ep % 5 == 0 or ep == 1:
            P(f"  Ep {ep:3d}/{EPOCHS} | Loss: {t_losses[-1]:.5f} (sup={ep_sup/nb:.5f} phys={ep_phys/nb:.5f}) | Val: {v_losses[-1]:.5f} | phys_w={lam_phys:.3f} | LR={sched.get_last_lr()[0]:.1e}")

    torch.save({'epoch':EPOCHS, 'model':model.state_dict(), 'val_loss':v_losses[-1], 'stats':stats}, ckpt_dir/'last.pt')
    P(f"  训练完成! best_val={best_val:.6f}")

    # 训练曲线
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t_losses, 'b-', lw=1.5, label='训练'); ax.plot(v_losses, 'r-', lw=1.5, label='验证')
    ax.set_xlabel('Epoch', fontsize=13); ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('训练损失曲线', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3); ax.set_yscale('log')
    plt.tight_layout(); plt.savefig(OUT/'training_loss.png', dpi=150); plt.close()

    # ================== 步骤5: 推理 ==================
    P(f"\n[5/6] 反演推理...")
    ckpt = torch.load(ckpt_dir/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    P(f"  加载最优模型 (epoch={ckpt['epoch']}, val={ckpt['val_loss']:.6f})")

    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        idx2 = np.linspace(0, n_total-1, INFER_TRACES, dtype=int)
        seis_infer_raw = np.stack([f.trace[int(i)] for i in idx2], axis=0).astype(np.float32)

    # 逐道RMS归一化 + 全局zscore (与训练一致)
    rms_infer = np.sqrt(np.mean(seis_infer_raw**2, axis=1, keepdims=True)) + 1e-10
    seis_infer_rms = seis_infer_raw / rms_infer
    seis_infer_norm = (seis_infer_rms - stats['seis_mean']) / stats['seis_std']

    # 滑窗推理
    preds = np.zeros((INFER_TRACES, n_samples), dtype=np.float32)
    counts = np.zeros((INFER_TRACES, n_samples), dtype=np.float32)
    stride_inf = TRACE_LEN // 2
    INF_BS = 256

    P(f"  推理 {INFER_TRACES} 道...")
    with torch.no_grad():
        for start in range(0, n_samples - TRACE_LEN + 1, stride_inf):
            end = start + TRACE_LEN
            seg = seis_infer_norm[:, start:end]
            for i in range(0, seg.shape[0], INF_BS):
                b = seg[i:i+INF_BS]
                x = torch.from_numpy(b).unsqueeze(1).float().to(device)
                p = model(x)[:, 0, :].cpu().numpy()
                preds[i:i+INF_BS, start:end] += p
                counts[i:i+INF_BS, start:end] += 1.0
        # 最后窗口
        if (n_samples - TRACE_LEN) % stride_inf != 0:
            start = n_samples - TRACE_LEN
            seg = seis_infer_norm[:, start:start+TRACE_LEN]
            for i in range(0, seg.shape[0], INF_BS):
                b = seg[i:i+INF_BS]
                x = torch.from_numpy(b).unsqueeze(1).float().to(device)
                p = model(x)[:, 0, :].cpu().numpy()
                L = min(TRACE_LEN, n_samples - start)
                preds[i:i+INF_BS, start:start+L] += p[:, :L]
                counts[i:i+INF_BS, start:start+L] += 1.0

    counts = np.maximum(counts, 1.0)
    preds /= counts

    # 反归一化到物理域
    impedance = preds * stats['imp_std'] + stats['imp_mean']

    # 确保正值 + 平滑
    impedance = np.maximum(impedance, 1000.0)
    impedance = gaussian_filter(impedance, sigma=[2, 3])

    P(f"  阻抗范围: [{impedance.min():.0f}, {impedance.max():.0f}]")

    np.save(OUT/'seismic_raw.npy', seis_infer_raw)
    np.save(OUT/'impedance_pred.npy', impedance)

    # ================== 步骤6: 可视化 ==================
    P(f"\n[6/6] 可视化...")

    # 下采样
    step = max(1, INFER_TRACES // 3000)
    sd = seis_infer_raw[::step, :].T    # [samples, traces]
    id_ = impedance[::step, :].T

    # ------ 色图 ------
    # 地震: 红蓝色图
    seis_colors = [
        (0.0, '#00008B'), (0.2, '#0066FF'), (0.4, '#99CCFF'),
        (0.5, '#FFFFFF'),
        (0.6, '#FFAAAA'), (0.8, '#FF3300'), (1.0, '#8B0000')]
    seis_cmap = LinearSegmentedColormap.from_list('seis', [(c[0],c[1]) for c in seis_colors])

    # 阻抗: 鲜艳彩虹色图 (类似专业地震解释软件)
    imp_colors = [
        (0.0,  '#000080'),  # 深蓝 (低阻抗)
        (0.12, '#0000FF'),
        (0.24, '#00BFFF'),
        (0.36, '#00FF7F'),  # 绿
        (0.48, '#ADFF2F'),
        (0.60, '#FFFF00'),  # 黄
        (0.72, '#FFA500'),  # 橙
        (0.84, '#FF4500'),  # 红
        (1.0,  '#8B0000'),  # 深红 (高阻抗)
    ]
    imp_cmap = LinearSegmentedColormap.from_list('imp_rainbow', [(c[0],c[1]) for c in imp_colors])

    # ------ 图1: 原始地震剖面 ------
    fig, ax = plt.subplots(figsize=(18, 10))
    vm = np.percentile(np.abs(sd), 99)
    im = ax.imshow(sd, aspect='auto', cmap=seis_cmap, vmin=-vm, vmax=vm,
                   extent=[0, n_total, T_END, T_START], interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('地震剖面 - 原始数据\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('振幅', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT/'seismic_original.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("  seismic_original.png")

    # ------ 图2: 反演阻抗剖面 (彩虹色) ------
    fig, ax = plt.subplots(figsize=(18, 10))
    v1, v2 = np.percentile(id_, [2, 98])
    im = ax.imshow(id_, aspect='auto', cmap=imp_cmap, vmin=v1, vmax=v2,
                   extent=[0, n_total, T_END, T_START], interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('波阻抗反演剖面\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('波阻抗 (kg/m2*s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT/'impedance_inversion.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("  impedance_inversion.png")

    # ------ 图3: 对比图 ------
    fig, axes = plt.subplots(2, 1, figsize=(18, 14))

    im1 = axes[0].imshow(sd, aspect='auto', cmap=seis_cmap, vmin=-vm, vmax=vm,
                         extent=[0, n_total, T_END, T_START], interpolation='bilinear')
    axes[0].set_ylabel('时间 (ms)', fontsize=13, fontweight='bold')
    axes[0].set_title('原始地震剖面', fontsize=15, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], pad=0.02, shrink=0.85, label='振幅')

    im2 = axes[1].imshow(id_, aspect='auto', cmap=imp_cmap, vmin=v1, vmax=v2,
                         extent=[0, n_total, T_END, T_START], interpolation='bilinear')
    axes[1].set_xlabel('道号 (Trace)', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('时间 (ms)', fontsize=13, fontweight='bold')
    axes[1].set_title('反演波阻抗剖面', fontsize=15, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], pad=0.02, shrink=0.85, label='波阻抗 (kg/m2*s)')

    plt.suptitle('地震反演对比\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT/'comparison.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("  comparison.png")

    P(f"\n{'='*55}\n 全部完成! -> {OUT}\n{'='*55}")
    for f in sorted(OUT.iterdir()):
        if f.is_file(): P(f"  {f.name}")


if __name__ == '__main__':
    main()
