import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------- Config ----------------
OUT_DIR = "demo_out"   # 输出文件夹（相对路径）
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- utilities ----------------
def ricker_wavelet(freq=35, dt=0.004, nt=128):
    tmax = dt * (nt - 1)
    t = np.linspace(-tmax/2, tmax/2, nt)
    pi2, f2 = np.pi**2, freq**2
    w = (1 - 2*pi2*f2*t**2) * np.exp(-pi2*f2*t**2)
    return w.astype(np.float32)

def compute_reflection_from_impedance(imp):
    # imp: (N, T, W)
    Z_next = np.roll(imp, -1, axis=1)
    Z_next[:, -1, :] = imp[:, -1, :]  # 最后一时刻用自身填充
    return ((Z_next - imp) / (Z_next + imp + 1e-8)).astype(np.float32)

def forward_seismic(r, wave):
    # r: (N,T,W) ; wave: (L,)
    N, T, W = r.shape
    out = np.zeros_like(r, dtype=np.float32)
    for n in range(N):
        for w in range(W):
            out[n, :, w] = fftconvolve(r[n, :, w], wave, mode="same")
    return out

def generate_data(N=200, T=128, W=16, smooth=3):
    # 生成平滑随机阻抗并生成反射与合成地震
    raw = np.random.rand(N, T, W).astype(np.float32)
    for n in range(N):
        for w in range(W):
            raw[n, :, w] = gaussian_filter(raw[n, :, w], sigma=smooth)
    imp = 1500 + 3000 * (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    refl = compute_reflection_from_impedance(imp)
    wave = ricker_wavelet(nt=T)
    seis = forward_seismic(refl, wave)
    # 按样本归一化（零均值，单位方差）
    seis = (seis - seis.mean(axis=(1,2), keepdims=True)) / (seis.std(axis=(1,2), keepdims=True) + 1e-9)
    return seis, imp, refl, wave

# ---------------- Dataset ----------------
class SynDataset(Dataset):
    def __init__(self, seis, imp, refl):
        self.s = torch.from_numpy(seis).unsqueeze(1).float()  # (N,1,T,W)
        self.i = torch.from_numpy(imp).unsqueeze(1).float()
        self.r = torch.from_numpy(refl).unsqueeze(1).float()
    def __len__(self): return self.s.shape[0]
    def __getitem__(self, idx): return self.s[idx], self.i[idx], self.r[idx]

# ---------------- Model (Tiny SG-CUnet-like) ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=7):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, k, padding=p), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x): return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # padding if needed
        diffY = x2.size(2) - x1.size(2); diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class SG_CUnet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.inc = DoubleConv(1, base)
        self.d1 = Down(base, base*2); self.d2 = Down(base*2, base*4)
        self.bot = DoubleConv(base*4, base*8)
        self.u2 = Up(base*8, base*4); self.u1 = Up(base*4, base*2); self.up0 = Up(base*2, base)
        self.final = nn.Conv2d(base, base, 7, padding=3)
        self.head_imp = nn.Conv2d(base, 1, 1)
        self.head_ref = nn.Conv2d(base, 1, 1)
        self.log_vars = nn.Parameter(torch.zeros(4))  # 不确定性加权参数
    def forward(self, x):
        x1 = self.inc(x); x2 = self.d1(x1); x3 = self.d2(x2)
        b = self.bot(x3); u2 = self.u2(b, x3); u1 = self.u1(u2, x2); u0 = self.up0(u1, x1)
        f = self.final(u0)
        return self.head_imp(f), self.head_ref(f), self.log_vars

# ---------------- Loss helper ----------------
def multitask_loss(Ls, log_vars):
    total = 0.0
    for i, L in enumerate(Ls):
        total = total + (torch.exp(-log_vars[i]) * L + log_vars[i])
    return total

# ---------------- Training routine ----------------
def train_and_save(out_dir=OUT_DIR, epochs=12, batch_size=8, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    # generate small synthetic data (you can replace with your own .npy)
    seis, imp, refl, wave = generate_data(N=200, T=128, W=16, smooth=3)
    idx = np.arange(seis.shape[0]); np.random.shuffle(idx)
    tr_idx, te_idx = idx[:160], idx[160:]
    train_loader = DataLoader(SynDataset(seis[tr_idx], imp[tr_idx], refl[tr_idx]), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(SynDataset(seis[te_idx], imp[te_idx], refl[te_idx]), batch_size=batch_size, shuffle=False)

    model = SG_CUnet(base=16).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()
    wave_t = torch.from_numpy(wave).to(device).float()

    logs = []; train_losses = []; val_losses = []

    for ep in range(1, epochs+1):
        model.train(); train_loss_acc = 0.0
        for s, i, r in train_loader:
            s, i, r = s.to(device), i.to(device), r.to(device)
            opt.zero_grad()
            imp_p, refl_p, logv = model(s)  # shapes: (B,1,T,W)
            L_imp = mse(imp_p, i); L_refl = mse(refl_p, r)
            # reflection from predicted impedance
            imp_next = torch.roll(imp_p, shifts=-1, dims=2); imp_next[:, :, -1, :] = imp_p[:, :, -1, :]
            refl_from_imp = (imp_next - imp_p) / (imp_next + imp_p + 1e-8)
            L_calc = mse(refl_from_imp, refl_p)

            # forward modeling: conv1d along time for each trace
            # reshape refl_p (B,1,T,W) -> (B*W,1,T)
            r_flat = refl_p.squeeze(1).permute(0, 2, 1).reshape(-1, 1, refl_p.shape[2])
            k = wave_t.flip(0).view(1, 1, -1)  # kernel shape (1,1,L)
            pad = k.shape[-1] // 2
            out = F.conv1d(F.pad(r_flat, (pad, pad)), k)  # (B*W,1,T_out)
            out = out.squeeze(1)  # (B*W, T_out)
            T_out = out.shape[-1]; T = s.shape[2]
            # 自动对齐时间长度（裁剪或右补零）
            if T_out > T:
                out = out[:, :T]
            elif T_out < T:
                out = F.pad(out, (0, T - T_out))
            # reshape back to (B,1,T,W)
            B = s.shape[0]; W = s.shape[3]
            out = out.view(B, W, T)      # (B,W,T)
            out = out.permute(0, 2, 1).unsqueeze(1)  # (B,1,T,W)
            L_fwd = mse(out, s)

            loss = multitask_loss((L_imp, L_refl, L_calc, L_fwd), logv)
            loss.backward(); opt.step()
            train_loss_acc += loss.item() * s.size(0)

        train_loss_acc /= len(train_loader.dataset)

        # validation (impedance MSE as quick metric)
        model.eval(); val_acc = 0.0
        with torch.no_grad():
            for s, i, r in val_loader:
                s, i = s.to(device), i.to(device)
                imp_p, _, _ = model(s)
                val_acc += mse(imp_p, i).item() * s.size(0)
        val_acc /= len(val_loader.dataset)

        line = f"{datetime.now().isoformat()} Epoch {ep}/{epochs} train_loss={train_loss_acc:.6e} val_imp_mse={val_acc:.6e}"
        print(line)
        logs.append(line); train_losses.append(train_loss_acc); val_losses.append(val_acc)

    # save logs and model
    with open(os.path.join(out_dir, "train_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(logs))
    torch.save(model.state_dict(), os.path.join(out_dir, "model_last.pth"))

    # loss curve
    plt.figure(figsize=(6,3))
    plt.plot(train_losses, label="train_loss"); plt.plot(val_losses, label="val_imp_mse")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.title("Training Curve")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_curve.png"), dpi=150); plt.close()

    # save a visual comparison (first val sample)
    s, i, r = next(iter(val_loader))
    s, i = s[0:1].to(device), i[0:1].to(device)
    with torch.no_grad():
        imp_p, _, _ = model(s)
    fig, ax = plt.subplots(1, 3, figsize=(12,4))
    ax[0].imshow(s.cpu().numpy()[0,0], aspect="auto"); ax[0].set_title("Seismic (input)")
    ax[1].imshow(i.cpu().numpy()[0,0], aspect="auto"); ax[1].set_title("True Impedance")
    ax[2].imshow(imp_p.cpu().numpy()[0,0], aspect="auto"); ax[2].set_title("Predicted Impedance")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "pred_vs_true.png"), dpi=150); plt.close()

    print("Saved outputs to:", out_dir)
    return os.path.abspath(out_dir)

# ---------------- Run if script executed ----------------
if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    outp = train_and_save(out_dir=OUT_DIR, epochs=12, batch_size=8, device=dev)
    print("Finished. Outputs:", outp)
