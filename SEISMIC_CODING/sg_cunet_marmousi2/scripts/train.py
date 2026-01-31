# -*- coding: utf-8 -*-
# —— 修复模块路径 ——
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x

from models.sg_cunet import SG_CUnet
from utils.data import normalize, denorm, PatchDataset
from utils.geo import elastic_deform_triplet
from utils.phys import ricker_wavelet, conv_time_axis
from utils.losses import CharbonnierLoss, grad1d_loss, tv_loss
import matplotlib.pyplot as plt
torch.backends.cudnn.benchmark = True

# ---------- helpers ----------
def make_aug(alpha, sigma):
    def _aug(Zp, Rp, Sp):
        seed = torch.randint(0, 10**9, (1,)).item()
        torch.manual_seed(seed)
        return elastic_deform_triplet(Zp, Rp, Sp, alpha, sigma)
    return _aug

def multitask_loss(losses, log_vars):
    """不确定性加权（Kendall & Gal）
       仅对 len(log_vars) 个损失项生效，其余项请在外面用固定权重加。"""
    total = 0.0
    n = min(len(losses), log_vars.shape[0])
    for i in range(n):
        total = total + torch.exp(-log_vars[i]) * losses[i] + log_vars[i]
    # 剩余的损失（如果有）由调用方自己加权；这里不处理
    return total

def load_ckpt_if_any(model, path):
    if path and os.path.isfile(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print(f"[RESUME] Loaded weights from: {path}")
    return model

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z_path', required=True)
    ap.add_argument('--s_path', required=True)
    ap.add_argument('--workdir', default='./exp_sg_cunet')

    # —— 4050 友好默认值（第一阶段） ——
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--accum_steps', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--base', type=int, default=24)
    ap.add_argument('--t_win', type=int, default=128)
    ap.add_argument('--x_win', type=int, default=16)
    ap.add_argument('--stride_t', type=int, default=64)
    ap.add_argument('--stride_x', type=int, default=8)
    ap.add_argument('--alpha', type=float, default=600.0)
    ap.add_argument('--sigma', type=float, default=50.0)
    ap.add_argument('--freq', type=float, default=35.0)
    ap.add_argument('--dt', type=float, default=0.005)
    ap.add_argument('--nt', type=int, default=128)
    ap.add_argument('--resume_from', type=str, default="")

    # TV 正则固定权重（因为 log_vars 只有 4 个）
    ap.add_argument('--tv_w_t', type=float, default=0.2, help='TV 时间向权重（未归一化 logZ 域）')
    ap.add_argument('--tv_w_x', type=float, default=0.05, help='TV 空间向权重（未归一化 logZ 域）')
    ap.add_argument('--tv_scale', type=float, default=0.05, help='TV 总系数，例如 0.05')

    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.workdir, exist_ok=True)

    # ---------- 数据 ----------
    Z = np.load(args.z_path).astype(np.float32)   # (T,X) 物理域阻抗
    S = np.load(args.s_path).astype(np.float32)   # (T,X) 合成地震
    assert Z.shape == S.shape

    # 在 log 阻抗域训练
    Z_log = np.log(np.clip(Z, 1.0, None))
    Z_norm, Zm, Zs = normalize(Z_log)             # z-score 规范化
    S_norm, Sm, Ss = normalize(S)
    np.save(os.path.join(args.workdir, 'norm_params.npy'),
            np.array([Zm, Zs, Sm, Ss], dtype=np.float32))

    train_ds = PatchDataset(Z_norm, S_norm, args.t_win, args.x_win,
                            args.stride_t, args.stride_x,
                            aug_fn=make_aug(args.alpha, args.sigma))
    val_ds   = PatchDataset(Z_norm, S_norm, args.t_win, args.x_win,
                            args.t_win, args.x_win, aug_fn=None)

    pin = bool(torch.cuda.is_available())
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=pin)
    val_ld   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=pin)

    # ---------- 模型 ----------
    model = SG_CUnet(base=args.base).to(device)
    model = load_ckpt_if_any(model, args.resume_from)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 以“参数更新次数”为步长的余弦退火（配合累积梯度）
    num_batches = max(1, len(train_ld))
    updates_per_epoch = (num_batches + args.accum_steps - 1) // args.accum_steps
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_updates = max(1, int(0.05 * total_updates))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, total_updates - warmup_updates), eta_min=1e-6
    )

    charbonnier = CharbonnierLoss()
    mse = nn.MSELoss()

    # 正演核
    wave = ricker_wavelet(args.freq, args.dt, args.nt)
    wave_t = torch.from_numpy(wave[::-1].copy()).to(device).view(1,1,-1)

    # 新版 AMP API
    scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

    tr_losses, va_losses, log_lines = [], [], []
    update_idx = 0

    for ep in range(1, args.epochs+1):
        model.train(); tr_acc = 0.0; n_tr = 0
        opt.zero_grad(set_to_none=True)

        for step, (s, z, r) in enumerate(tqdm(train_ld, desc=f"Epoch {ep}/{args.epochs}")):
            s = s.to(device)   # (B,1,T,W)
            z = z.to(device)   # (B,1,T,W) —— 规范化后的 logZ
            r = r.to(device)

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                z_pred, r_pred, logv = model(s)          # 预测：规范化 logZ 与 reflectivity

                # 1) logZ 重建（规范化域）
                L_imp = charbonnier(z_pred, z)

                # 2) 一阶梯度一致性（未规范化 logZ 域）
                z_log_pred = denorm(z_pred, Zm, Zs)
                z_log_true = denorm(z,      Zm, Zs)
                L_grad = grad1d_loss(z_log_pred, z_log_true, dim=2)

                # 3) Z -> r 的一致性
                z_phys  = torch.exp(z_log_pred).clamp(min=1.0)
                z_next  = torch.roll(z_phys, shifts=-1, dims=2)
                z_next[:,:, -1, :] = z_phys[:,:, -1, :]
                r_from_z = (z_next - z_phys) / (z_next + z_phys + 1e-8)
                r_from_z = (r_from_z - r_from_z.mean()) / (r_from_z.std()+1e-8)
                L_ref = mse(r_pred, r_from_z)

                # 4) 正演一致性
                r_phys = (r_pred - r_pred.mean())/(r_pred.std()+1e-8)
                s_hat  = conv_time_axis(r_phys, wave_t)
                s_hat  = (s_hat - s_hat.mean())/(s_hat.std()+1e-8)
                L_phys = mse(s_hat, s)

                # 5) TV 正则（未规范化 logZ 域），固定权重，不放入 log_vars
                L_tv = tv_loss(z_log_pred, dims=(2,3), w=(args.tv_w_t, args.tv_w_x))

                # —— 组合损失：前 4 项走不确定性加权；TV 走固定权重 ——
                loss_main = multitask_loss([L_imp, L_grad, L_ref, L_phys], logv)
                loss = loss_main + args.tv_scale * L_tv

            scaler.scale(loss / args.accum_steps).backward()

            # —— 累积到步或最后一批：更新一次 ——
            do_update = ((step + 1) % args.accum_steps == 0) or (step + 1 == num_batches)
            if do_update:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)

                # warmup & 调度按“更新次数”
                update_idx += 1
                if update_idx <= warmup_updates:
                    for pg in opt.param_groups:
                        pg['lr'] = args.lr * update_idx / float(max(1, warmup_updates))
                else:
                    scheduler.step()

            tr_acc += loss.item() * s.size(0); n_tr += s.size(0)

        tr_epoch = tr_acc / max(1, n_tr)

        # ---------- 验证（未规范化 logZ 的 MSE） ----------
        model.eval(); va_acc = 0.0; n_va = 0
        with torch.no_grad():
            for s, z, _ in val_ld:
                s, z = s.to(device), z.to(device)
                z_pred, _, _ = model(s)
                z_log_pred = denorm(z_pred, Zm, Zs)
                z_log_true = denorm(z,      Zm, Zs)
                va_acc += mse(z_log_pred, z_log_true).item() * s.size(0)
                n_va   += s.size(0)
        va_epoch = va_acc / max(1,n_va)

        tr_losses.append(tr_epoch); va_losses.append(va_epoch)
        line = f"Epoch {ep}/{args.epochs} train_total={tr_epoch:.6e} val_logZ_mse={va_epoch:.6e}"
        print(line); log_lines.append(line)

    # ---------- 保存 ----------
    with open(os.path.join(args.workdir, "train_log.txt"), "a", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    torch.save(model.state_dict(), os.path.join(args.workdir, "model_last.pth"))

    # 曲线
    plt.figure(figsize=(6,3))
    plt.plot(tr_losses, label="train_total"); plt.plot(va_losses, label="val_logZ_mse")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Curve")
    plt.tight_layout(); plt.savefig(os.path.join(args.workdir, "loss_curve.png"), dpi=150); plt.close()

if __name__ == "__main__":
    main()
