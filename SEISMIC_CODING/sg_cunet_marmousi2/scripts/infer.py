import os, sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os, argparse, numpy as np, torch
from models.sg_cunet import SG_CUnet
from utils.data import normalize, denorm
from utils.phys import ricker_wavelet, conv_time_axis

def infer_full(model, S_norm, t_win=128, x_win=16, device='cpu'):
    model.eval()
    T, X = S_norm.shape
    Z_pred_full = torch.zeros((T,X), device=device)
    W_full = torch.zeros_like(Z_pred_full)
    with torch.no_grad():
        for t0 in range(0, T - t_win + 1, t_win):
            for x0 in range(0, X - x_win + 1, x_win):
                s_blk = torch.from_numpy(S_norm[t0:t0+t_win, x0:x0+x_win]).float().to(device).unsqueeze(0).unsqueeze(0)
                z_pred, _, _ = model(s_blk)
                z_pred = z_pred.squeeze(0).squeeze(0)
                Z_pred_full[t0:t0+t_win, x0:x0+x_win] += z_pred
                W_full[t0:t0+t_win, x0:x0+x_win]     += 1.0
    Z_pred_full = Z_pred_full / (W_full + 1e-8)
    return Z_pred_full

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z_path', required=True)
    ap.add_argument('--s_path', required=True)
    ap.add_argument('--workdir', default='./exp_sg_cunet')
    ap.add_argument('--model', required=True)
    ap.add_argument('--t_win', type=int, default=128)
    ap.add_argument('--x_win', type=int, default=16)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Z = np.load(args.z_path).astype(np.float32)
    S = np.load(args.s_path).astype(np.float32)
    assert Z.shape == S.shape
    T,X = Z.shape

    # 与训练一致的规范化
    Zm,Zs,Sm,Ss = np.load(os.path.join(args.workdir,'norm_params.npy'))
    Z_norm = (Z - Zm)/Zs
    S_norm = (S - Sm)/Ss

    model = SG_CUnet(base=32).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    Z_pred_norm = infer_full(model, S_norm, args.t_win, args.x_win, device=device)  # (T,X)
    Z_pred = Z_pred_norm.cpu().numpy()*Zs + Zm
    np.save(os.path.join(args.workdir,'Z_pred.npy'), Z_pred)
    print("Saved:", os.path.join(args.workdir,'Z_pred.npy'))

if __name__ == "__main__":
    main()
