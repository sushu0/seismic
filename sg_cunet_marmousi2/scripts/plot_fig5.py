# -*- coding: utf-8 -*-
# —— 修复模块路径 ——
import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse, numpy as np, torch, matplotlib
import matplotlib.pyplot as plt
from models.sg_cunet import SG_CUnet
from utils.data import normalize, denorm
from utils.data import PatchDataset  # 仅取窗参数作为参考

# 尝试中文字体（Windows 通常有微软雅黑）
for f in (["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"]):
    if f in matplotlib.font_manager.get_font_names():
        matplotlib.rcParams['font.sans-serif'] = [f]
        break
matplotlib.rcParams['axes.unicode_minus'] = False

def tile_positions(L, win, stride):
    pos = [0]
    while pos[-1] + win < L:
        nxt = pos[-1] + stride
        if nxt + win >= L:
            pos.append(L - win)
            break
        pos.append(nxt)
    return sorted(set(pos))

def infer_full(model, S_norm, Zm, Zs, t_win, x_win, stride_t, stride_x, device, bsz=16):
    """滑窗推理并重叠平均，返回“未规范化 logZ”"""
    T, X = S_norm.shape
    tt = tile_positions(T, t_win, stride_t)
    xx = tile_positions(X, x_win, stride_x)
    out = np.zeros((T, X), dtype=np.float32)
    cnt = np.zeros((T, X), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for ti in tt:
            batch = []
            coords = []
            for xi in xx:
                patch = S_norm[ti:ti+t_win, xi:xi+x_win][None, None, ...]  # (1,1,t,x)
                batch.append(patch)
                coords.append((ti, xi))
                if len(batch) == bsz or xi == xx[-1]:
                    b = torch.from_numpy(np.concatenate(batch, 0)).to(device)
                    z_pred, _, _ = model(b)                   # 预测的是“规范化 logZ”
                    z_log = denorm(z_pred, Zm, Zs).cpu().numpy()  # 反规范化到 log 域
                    for k in range(z_log.shape[0]):
                        t0, x0 = coords[k]
                        out[t0:t0+t_win, x0:x0+x_win] += z_log[k,0]
                        cnt[t0:t0+t_win, x0:x0+x_win] += 1.0
                    batch, coords = [], []
    cnt[cnt == 0] = 1.0
    return out / cnt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z_path', required=True)
    ap.add_argument('--s_path', required=True)
    ap.add_argument('--workdir', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--trace_idx', type=int, default=1343)
    # 与训练保持一致的小窗/步长（避免边界误差）
    ap.add_argument('--t_win', type=int, default=128)
    ap.add_argument('--x_win', type=int, default=16)
    ap.add_argument('--stride_t', type=int, default=64)
    ap.add_argument('--stride_x', type=int, default=8)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.workdir, exist_ok=True)

    # 载入数据与归一化参数（与 train.py 同源）
    Z = np.load(args.z_path).astype(np.float32)
    S = np.load(args.s_path).astype(np.float32)
    Z_log = np.log(np.clip(Z, 1.0, None))
    # 训练时写在 workdir 的 norm_params
    Zm, Zs, Sm, Ss = np.load(os.path.join(args.workdir, 'norm_params.npy'))
    S_norm = (S - Sm) / (Ss + 1e-8)

    # 载入模型
    model = SG_CUnet(base=24).to(device)  # 与 run_all.py 的 base 保持一致
    sd = torch.load(args.model, map_location='cpu')
    model.load_state_dict(sd, strict=True)

    # 全图推理，得到“未规范化 logZ”
    z_log_pred = infer_full(model, S_norm, Zm, Zs, args.t_win, args.x_win,
                            args.stride_t, args.stride_x, device, bsz=16)
    # 转回物理域阻抗
    Z_pred = np.exp(z_log_pred).astype(np.float32)

    # 取指定道
    ix = max(0, min(Z.shape[1]-1, int(args.trace_idx)))
    gt  = Z[:, ix]       # 真实阻抗（物理域）
    prd = Z_pred[:, ix]  # 预测阻抗（物理域）

    # —— 仅两条线（真实/预测）——
    fig = plt.figure(figsize=(5,8))
    plt.plot(gt,  np.arange(len(gt)), color="#1f77b4", label="真实阻抗")
    plt.plot(prd, np.arange(len(prd)), color="#2ca02c", label="预测阻抗（SG-CUnet）")
    plt.gca().invert_yaxis()
    plt.xlabel("Impedance"); plt.ylabel("Time (ms)")
    plt.title(f"道 {ix} 真实/预测")
    plt.legend(loc="upper right", facecolor="white")
    out = os.path.join(args.workdir, f"fig5_trace{ix}_2lines.png")
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.close()
    print("Saved fig5 to", out)

if __name__ == "__main__":
    main()
