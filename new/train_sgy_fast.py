# -*- coding: utf-8 -*-
"""
针对 0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy 的地震反演训练与可视化
- 优化版: 更快的训练和推理
- 使用 CUDA, UNet1D + 物理约束
"""
import sys, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import segyio, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================== 模型 ==============================
class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(ci, co, 3, padding=1), nn.BatchNorm1d(co), nn.GELU(),
            nn.Conv1d(co, co, 3, padding=1), nn.BatchNorm1d(co), nn.GELU())
    def forward(self, x): return self.net(x)

class UNet1D(nn.Module):
    def __init__(self, in_ch=1, base=32, depth=4):
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
            x = self.up[i](x)
            s = skips[-(i+1)]
            if x.shape[-1] != s.shape[-1]:
                x = F.pad(x, (0, s.shape[-1]-x.shape[-1])) if s.shape[-1]>x.shape[-1] else x[...,:s.shape[-1]]
            x = self.dec[i](torch.cat([x, s], 1))
        return self.out(x)

class ForwardModel(nn.Module):
    def __init__(self, wavelet, eps=1e-6):
        super().__init__()
        self.register_buffer("wavelet", wavelet.clone().float())
        self.eps = eps
    def forward(self, imp):
        imp_prev = torch.roll(imp, 1, -1)
        r = (imp - imp_prev) / (imp + imp_prev + self.eps)
        r[..., 0] = 0.0
        w = self.wavelet.view(1,1,-1)
        return F.conv1d(r, w, padding=(w.shape[-1]-1)//2)[...,:imp.shape[-1]]

# ============================== 数据 ==============================
class TraceDS(Dataset):
    def __init__(self, seis, imp):
        self.s = seis.astype(np.float32)
        self.i = imp.astype(np.float32)
    def __len__(self): return len(self.s)
    def __getitem__(self, idx):
        return {"seis": torch.from_numpy(self.s[idx]).unsqueeze(0),
                "imp": torch.from_numpy(self.i[idx]).unsqueeze(0)}

def ricker(f0, dt, length):
    t = np.arange(-length/2, length/2+dt, dt, dtype=np.float64)
    pi2 = np.pi**2
    return ((1-2*pi2*f0**2*t**2)*np.exp(-pi2*f0**2*t**2)).astype(np.float32)

# ============================== 主流程 ==============================
def main():
    SGY = Path(r'D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy')
    OUT = Path(r'D:\SEISMIC_CODING\new\sgy_inversion_output')
    OUT.mkdir(parents=True, exist_ok=True)

    T_START, T_END, DT = 2500, 6000, 0.002
    TRAIN_TRACES = 5000   # 训练用道数
    TRACE_LEN = 512       # 切片长度
    STRIDE = 256
    BS, EPOCHS, LR = 64, 80, 1e-3
    INFER_TRACES = 5000   # 推理道数
    
    np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    P = lambda *a, **kw: print(*a, **kw, flush=True)
    P(f"{'='*50}\n地震反演训练系统\n{'='*50}")
    P(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type=='cuda' else ""))
    
    # ---- 步骤1: 加载数据 ----
    P(f"\n--- 步骤1: 加载SGY数据 ---")
    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        n_total = f.tracecount
        n_samples = len(f.samples)
        P(f"总道数: {n_total}, 采样点: {n_samples}")
        idx = np.linspace(0, n_total-1, TRAIN_TRACES, dtype=int)
        seismic = np.stack([f.trace[int(i)] for i in idx], axis=0).astype(np.float32)
    P(f"抽样 {TRAIN_TRACES} 道, 形状: {seismic.shape}")
    
    # ---- 步骤2: 准备训练数据 ----
    P(f"\n--- 步骤2: 准备训练数据 ---")
    # 切片
    segs = []
    for start in range(0, n_samples - TRACE_LEN + 1, STRIDE):
        segs.append(seismic[:, start:start+TRACE_LEN])
    seis_all = np.concatenate(segs, axis=0)
    P(f"切片后: {seis_all.shape}")
    
    # 归一化地震
    s_mean, s_std = seis_all.mean(), seis_all.std() + 1e-8
    seis_norm = (seis_all - s_mean) / s_std
    
    # 生成伪阻抗 (积分法 + 指数变换)
    imp_exp = np.exp(2.0 * np.cumsum(seis_norm * 0.01, axis=1))
    pseudo_imp = 5000.0 * imp_exp
    for i in range(pseudo_imp.shape[0]):
        pseudo_imp[i] = gaussian_filter(pseudo_imp[i], sigma=2)
    i_mean, i_std = pseudo_imp.mean(), pseudo_imp.std() + 1e-8
    imp_norm = (pseudo_imp - i_mean) / i_std
    
    # 划分
    n = seis_norm.shape[0]
    perm = np.random.permutation(n)
    nt = int(n * 0.85)
    tr_s, tr_i = seis_norm[perm[:nt]], imp_norm[perm[:nt]]
    va_s, va_i = seis_norm[perm[nt:]], imp_norm[perm[nt:]]
    P(f"训练: {tr_s.shape[0]} 道, 验证: {va_s.shape[0]} 道")
    
    stats = {'seis_mean': float(s_mean), 'seis_std': float(s_std),
             'imp_mean': float(i_mean), 'imp_std': float(i_std),
             'trace_len': TRACE_LEN}
    json.dump(stats, open(OUT/'norm_stats.json','w'), indent=2)
    
    tr_loader = DataLoader(TraceDS(tr_s, tr_i), batch_size=BS, shuffle=True, drop_last=True, pin_memory=True)
    va_loader = DataLoader(TraceDS(va_s, va_i), batch_size=BS*2, shuffle=False, pin_memory=True)
    
    # ---- 步骤3: 构建模型 ----
    P(f"\n--- 步骤3: 构建模型 ---")
    model = UNet1D(in_ch=1, base=32, depth=4).to(device)
    P(f"UNet1D参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    w = torch.from_numpy(ricker(25.0, DT, 0.128)).to(device)
    fm = ForwardModel(w).to(device)
    
    # ---- 步骤4: 训练 ----
    P(f"\n--- 步骤4: CUDA训练 ({EPOCHS} epochs) ---")
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
    mse = nn.MSELoss()
    sl1 = nn.SmoothL1Loss()
    best_val, t_losses, v_losses = float('inf'), [], []
    ckpt_dir = OUT / 'checkpoints'; ckpt_dir.mkdir(exist_ok=True)
    
    for ep in range(1, EPOCHS+1):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in tr_loader:
            sx = batch['seis'].to(device)
            iy = batch['imp'].to(device)
            pred = model(sx)
            loss = sl1(pred, iy) + 0.5*mse(pred, iy)
            # 物理约束
            imp_phys = pred * stats['imp_std'] + stats['imp_mean']
            ss = fm(imp_phys)
            ss_n = (ss - ss.mean()) / (ss.std() + 1e-8)
            sx_n = (sx - sx.mean()) / (sx.std() + 1e-8)
            loss = loss + 0.1 * mse(ss_n, sx_n)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item(); nb += 1
        sched.step()
        t_losses.append(ep_loss/nb)
        
        model.eval(); vl, nv = 0.0, 0
        with torch.no_grad():
            for batch in va_loader:
                pred = model(batch['seis'].to(device))
                vl += mse(pred, batch['imp'].to(device)).item(); nv += 1
        v_losses.append(vl/max(nv,1))
        
        if v_losses[-1] < best_val:
            best_val = v_losses[-1]
            torch.save({'epoch':ep, 'model':model.state_dict(), 'val_loss':best_val, 'stats':stats}, ckpt_dir/'best.pt')
        
        if ep % 5 == 0 or ep == 1:
            P(f"  Epoch {ep:3d}/{EPOCHS} | Train: {t_losses[-1]:.6f} | Val: {v_losses[-1]:.6f} | LR: {sched.get_last_lr()[0]:.2e}")
    
    torch.save({'epoch':EPOCHS, 'model':model.state_dict(), 'val_loss':v_losses[-1], 'stats':stats}, ckpt_dir/'last.pt')
    P(f"训练完成! 最优验证损失: {best_val:.6f}")
    
    # 训练曲线
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(t_losses, 'b-', label='训练损失'); ax.plot(v_losses, 'r-', label='验证损失')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.set_title('训练损失曲线')
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale('log')
    plt.tight_layout(); plt.savefig(OUT/'training_loss.png', dpi=150); plt.close()
    P("已保存: training_loss.png")
    
    # ---- 步骤5: 推理 ----
    P(f"\n--- 步骤5: 反演推理 ---")
    ckpt = torch.load(ckpt_dir/'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model']); model.eval()
    P(f"加载最优模型 (epoch={ckpt['epoch']})")
    
    with segyio.open(str(SGY), "r", ignore_geometry=True) as f:
        idx2 = np.linspace(0, n_total-1, INFER_TRACES, dtype=int)
        seis_infer = np.stack([f.trace[int(i)] for i in idx2], axis=0).astype(np.float32)
    
    seis_infer_norm = (seis_infer - stats['seis_mean']) / stats['seis_std']
    
    # 滑窗推理
    preds = np.zeros_like(seis_infer)
    counts = np.zeros_like(seis_infer)
    stride_inf = TRACE_LEN // 2
    
    P(f"推理 {INFER_TRACES} 道 (滑窗, trace_len={TRACE_LEN})...")
    with torch.no_grad():
        for start in range(0, n_samples - TRACE_LEN + 1, stride_inf):
            end = start + TRACE_LEN
            seg = seis_infer_norm[:, start:end]
            for i in range(0, seg.shape[0], 256):
                b = seg[i:i+256]
                x = torch.from_numpy(b).unsqueeze(1).float().to(device)
                p = model(x)[:, 0, :].cpu().numpy()
                preds[i:i+256, start:end] += p
                counts[i:i+256, start:end] += 1.0
        # 最后一个窗口
        start = n_samples - TRACE_LEN
        seg = seis_infer_norm[:, start:]
        for i in range(0, seg.shape[0], 256):
            b = seg[i:i+256]
            x = torch.from_numpy(b).unsqueeze(1).float().to(device)
            p = model(x)[:, 0, :].cpu().numpy()
            preds[i:i+256, start:] += p[:, :seis_infer.shape[1]-start]
            counts[i:i+256, start:] += 1.0
    
    counts = np.maximum(counts, 1.0)
    preds /= counts
    impedance = preds * stats['imp_std'] + stats['imp_mean']
    impedance = gaussian_filter(impedance, sigma=[3, 5])
    P(f"反演完成: range=[{impedance.min():.1f}, {impedance.max():.1f}]")
    
    np.save(OUT/'seismic.npy', seis_infer)
    np.save(OUT/'impedance_pred.npy', impedance)
    
    # ---- 步骤6: 可视化 ----
    P(f"\n--- 步骤6: 可视化 ---")
    
    # 色图 (与 seismic_original_single.png 一样的样式)
    colors = [
        (0.0, '#00008B'), (0.2, '#0066FF'), (0.4, '#99CCFF'),
        (0.5, '#FFFFFF'),
        (0.6, '#FFAAAA'), (0.8, '#FF3300'), (1.0, '#8B0000')]
    cmap = LinearSegmentedColormap.from_list('seismic_vivid', [(c[0],c[1]) for c in colors])
    
    # 下采样用于绘图
    step = max(1, INFER_TRACES // 3000)
    sd = seis_infer[::step,:].T
    id_ = impedance[::step,:].T
    
    # 图1: 原始地震剖面
    fig, ax = plt.subplots(figsize=(18, 10))
    vm = np.percentile(np.abs(sd), 99)
    im = ax.imshow(sd, aspect='auto', cmap=cmap, vmin=-vm, vmax=vm,
                   extent=[0, n_total, T_END, T_START], interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('地震剖面 - 原始数据\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('振幅', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT/'sgy_seismic_original.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("已保存: sgy_seismic_original.png")
    
    # 图2: 反演阻抗剖面 (同样式)
    fig, ax = plt.subplots(figsize=(18, 10))
    v1, v2 = np.percentile(id_, [1, 99])
    im = ax.imshow(id_, aspect='auto', cmap=cmap, vmin=v1, vmax=v2,
                   extent=[0, n_total, T_END, T_START], interpolation='bilinear')
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('波阻抗反演剖面\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', fontsize=16, fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('波阻抗 (kg/m²·s)', fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT/'sgy_impedance_inversion.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    P("已保存: sgy_impedance_inversion.png")
    
    P(f"\n{'='*50}\n全部完成! 结果: {OUT}\n{'='*50}")
    for f in sorted(OUT.iterdir()):
        if f.is_file(): P(f"  {f.name}")

if __name__ == '__main__':
    main()
