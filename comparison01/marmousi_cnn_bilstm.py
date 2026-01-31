# marmousi_cnn_bilstm.py
import os
import json
import argparse
import logging
import random
from datetime import datetime
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from scipy.interpolate import CubicSpline

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


def set_global_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CNN-BiLSTM seismic impedance inversion (supervised + semi-supervised)",
    )
    p.add_argument("--data-root", type=str, default=r"D:\\SEISMIC_CODING\\comparison01")
    p.add_argument("--seed", type=int, default=0)
    det_group = p.add_mutually_exclusive_group()
    det_group.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    det_group.add_argument("--no-deterministic", dest="deterministic", action="store_false")

    # 数据划分：论文设置为从 2721 道中均匀选取 20/5/5
    # 若提供 *-count，则优先使用 count；否则使用 ratio。
    p.add_argument("--split-mode", type=str, default="uniform", choices=["uniform", "random"])
    p.add_argument("--train-count", type=int, default=20)
    p.add_argument("--val-count", type=int, default=5)
    p.add_argument("--test-count", type=int, default=5)
    p.add_argument("--train-ratio", type=float, default=0.6)
    p.add_argument("--val-ratio", type=float, default=0.2)

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs-supervised", type=int, default=300)
    # 论文：lr=0.005
    p.add_argument("--lr-supervised", type=float, default=0.005)

    semi_group = p.add_mutually_exclusive_group()
    semi_group.add_argument("--run-semi", dest="run_semi", action="store_true", default=True)
    semi_group.add_argument("--no-run-semi", dest="run_semi", action="store_false")

    freeze_group = p.add_mutually_exclusive_group()
    freeze_group.add_argument("--freeze-cnn", dest="freeze_cnn", action="store_true", default=True)
    freeze_group.add_argument("--no-freeze-cnn", dest="freeze_cnn", action="store_false")

    # 论文：N* = 10N；每道阻抗做 10 道增广
    p.add_argument("--augment-factor", type=int, default=10)
    p.add_argument("--n-aug-per-trace", type=int, default=10)

    p.add_argument("--mc-samples", type=int, default=10)
    p.add_argument("--mc-batch-size", type=int, default=32)
    # 降低阈值以提高伪标签利用率（原论文0.95，这里放宽到0.85）
    p.add_argument("--pseudo-conf-threshold", type=float, default=0.85)
    # 兼容旧实现：基于 ratio < alpha 的阈值（若显式设置，则优先使用）
    p.add_argument("--mc-alpha", type=float, default=None)

    p.add_argument("--pseudo-batch-size", type=int, default=16)

    p.add_argument("--epochs-semi", type=int, default=50)
    p.add_argument("--lr-semi", type=float, default=5e-5)
    p.add_argument("--lambda-pseudo", type=float, default=0.05)
    p.add_argument("--lambda-fwd", type=float, default=0.02)

    p.add_argument("--wavelet-f0", type=float, default=25.0)
    p.add_argument("--wavelet-dt", type=float, default=0.002)
    p.add_argument("--wavelet-nt", type=int, default=64)

    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--run-dir", type=str, default="")
    return p


def resolve_device(device_arg: str) -> str:
    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepare_run_dir(data_root: str, run_dir: str) -> Path:
    if run_dir:
        out = Path(run_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(data_root) / "runs" / ts
    out.mkdir(parents=True, exist_ok=True)
    return out


def setup_logging(log_path: Path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# -------------------------
# 1. CNN-BiLSTM 模型
# -------------------------
EPS = 1e-6


@dataclass
class NormParams:
    seis_min: float
    seis_max: float
    imp_min: float
    imp_max: float

    @property
    def seis_range(self):
        return self.seis_max - self.seis_min

    @property
    def imp_range(self):
        return self.imp_max - self.imp_min

    def to_dict(self):
        return {
            "seis_min": self.seis_min,
            "seis_max": self.seis_max,
            "imp_min": self.imp_min,
            "imp_max": self.imp_max,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            seis_min=float(data["seis_min"]),
            seis_max=float(data["seis_max"]),
            imp_min=float(data["imp_min"]),
            imp_max=float(data["imp_max"]),
        )


def minmax_norm(arr, min_val, max_val):
    return (arr - min_val) / (max_val - min_val + EPS)


def minmax_denorm(arr, min_val, max_val):
    return arr * (max_val - min_val + EPS) + min_val


def save_norm_params(path, params: NormParams):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(params.to_dict(), f, indent=2)


def load_norm_params(path) -> NormParams:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return NormParams.from_dict(data)


def compute_norm_params(seismic, impedance, train_idx):
    seismic_train = seismic[:, train_idx]
    impedance_train = impedance[:, train_idx]
    return NormParams(
        seis_min=float(seismic_train.min()),
        seis_max=float(seismic_train.max()),
        imp_min=float(impedance_train.min()),
        imp_max=float(impedance_train.max()),
    )


def normalize_seismic(arr, params: NormParams):
    return minmax_norm(arr, params.seis_min, params.seis_max)


def normalize_impedance(arr, params: NormParams):
    return minmax_norm(arr, params.imp_min, params.imp_max)


def denormalize_seismic(arr, params: NormParams):
    return minmax_denorm(arr, params.seis_min, params.seis_max)


def denormalize_impedance(arr, params: NormParams):
    return minmax_denorm(arr, params.imp_min, params.imp_max)


class CNNBiLSTM(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 1, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)  # [B, 64, T]
        x = x.transpose(1, 2)  # [B, T, 64]
        out, _ = self.bilstm(x)  # [B, T, 128]
        out = self.fc(out)  # [B, T, 1]
        return out.squeeze(-1)  # [B, T]


# -------------------------
# 2. 数据集
# -------------------------
class LabeledSeisDataset(Dataset):
    """只用原始有标签道"""
    def __init__(self, seismic, impedance, trace_idx):
        """
        seismic, impedance: np.array [T, Nx]
        trace_idx: 一维下标列表
        """
        self.seis = seismic[:, trace_idx]    # [T, N]
        self.imp = impedance[:, trace_idx]   # [T, N]

    def __len__(self):
        return self.seis.shape[1]

    def __getitem__(self, i):
        x = self.seis[:, i].astype(np.float32)   # [T]
        y = self.imp[:, i].astype(np.float32)    # [T]
        x = torch.from_numpy(x[None, :])         # [1, T]
        y = torch.from_numpy(y)
        return x, y


class AugmentedPseudoDataset(Dataset):
    """增广 + 伪标签数据集"""
    def __init__(self, seis_aug, imp_pseudo, mask):
        """
        seis_aug: [N_aug, T]  归一化后的地震
        imp_pseudo: [N_aug, T]  伪标签（归一化后的阻抗）
        mask: [N_aug, T]  高置信度掩膜 (0/1)
        """
        self.seis = seis_aug.astype(np.float32)
        self.imp_pseudo = imp_pseudo.astype(np.float32)
        self.mask = mask.astype(np.float32)

    def __len__(self):
        return self.seis.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(self.seis[i][None, :])      # [1, T]
        y = torch.from_numpy(self.imp_pseudo[i])         # [T]
        m = torch.from_numpy(self.mask[i])               # [T]
        return x, y, m


# -------------------------
# 3. 正演算子: Ricker + Conv
# -------------------------
def ricker(f0, dt, nt):
    t = np.arange(-nt // 2, nt // 2) * dt
    pi2 = (np.pi * f0) ** 2
    w = (1 - 2 * pi2 * t ** 2) * np.exp(-pi2 * t ** 2)
    return w.astype(np.float32)


def impedance_to_seismic_np(imp, wavelet):
    """Numpy 版: 单条阻抗 -> 单条地震"""
    # imp: [T]
    I1 = imp[1:]
    I0 = imp[:-1]
    r = (I1 - I0) / (I1 + I0 + 1e-8)  # [T-1]
    r = np.concatenate([[0.0], r])    # [T]
    s = np.convolve(r, wavelet, mode='same')
    return s.astype(np.float32)


class ForwardModel(nn.Module):
    """Torch 版可微正演：阻抗(物理量) -> 地震(物理量)，并强制输出长度与输入一致"""
    def __init__(self, wavelet):
        super().__init__()
        w = torch.tensor(wavelet, dtype=torch.float32)  # [K]
        self.register_buffer("w", w.view(1, 1, -1))     # [1,1,K]

    def forward(self, imp):
        # imp: [B, T] (物理量)
        B, T = imp.shape
        I1 = imp[:, 1:]
        I0 = imp[:, :-1]
        r = (I1 - I0) / (I1 + I0 + 1e-6)     # [B, T-1]
        r = F.pad(r, (1, 0))                 # [B, T]
        x = r.unsqueeze(1)                   # [B, 1, T]

        # 卷积正演
        s = F.conv1d(x, self.w, padding=self.w.shape[-1] // 2)  # [B,1,L]
        s = s.squeeze(1)  # [B, L]

        # ---- 关键：对齐长度，保证 L == T ----
        L = s.shape[1]
        if L > T:
            # 中心裁剪到 T 长度
            start = (L - T) // 2
            s = s[:, start:start + T]
        elif L < T:
            # 不足就左右补零到 T 长度（一般用不上）
            pad_left = (T - L) // 2
            pad_right = T - L - pad_left
            s = F.pad(s, (pad_left, pad_right))

        return s  # [B, T]


# -------------------------
# 4. 阻抗域数据增广
# -------------------------
def augment_impedance(trace, factor=10, n_aug=100):
    """
    trace: [T] 单条阻抗
    返回: [n_aug, T] 增广后的阻抗
    """
    T = len(trace)
    x = np.arange(T)
    x_fine = np.linspace(0, T - 1, T * factor)

    cs = CubicSpline(x, trace)
    z_fine = cs(x_fine)  # [T*factor]

    aug_traces = []
    for _ in range(n_aug):
        idx = np.sort(
            np.random.choice(len(x_fine), size=T, replace=False)
        )
        aug_traces.append(z_fine[idx])
    return np.stack(aug_traces).astype(np.float32)  # [n_aug, T]


def build_augmented_pairs(impedance, train_idx, wavelet,
                          factor=10, n_aug_per_trace=50):
    """
    对训练集的每条阻抗做增广，并正演得到地震
    返回:
      imp_aug_all: [N_aug, T]  物理量
      seis_aug_all: [N_aug, T] 物理量
    """
    imp_tr = impedance[:, train_idx]  # [T, Ntr]
    T, Ntr = imp_tr.shape

    imp_aug_list = []
    seis_aug_list = []
    for j in range(Ntr):
        base_imp = imp_tr[:, j]
        imp_aug = augment_impedance(base_imp, factor=factor,
                                    n_aug=n_aug_per_trace)  # [K,T]
        for k in range(imp_aug.shape[0]):
            s = impedance_to_seismic_np(imp_aug[k], wavelet)
            imp_aug_list.append(imp_aug[k])
            seis_aug_list.append(s)

    imp_aug_all = np.stack(imp_aug_list)   # [N_aug, T]
    seis_aug_all = np.stack(seis_aug_list) # [N_aug, T]
    return imp_aug_all, seis_aug_all


# -------------------------
# 5. MC Dropout 伪标签
# -------------------------
def mc_dropout_pseudo(model, seis_aug_norm, device,
                      num_samples=20, batch_size=32,
                      conf_threshold: float = 0.95,
                      alpha: float | None = None):
    """
    seis_aug_norm: np.array [N_aug, T] (已归一化)
    返回:
      mu_np: [N_aug, T]  伪标签（阻抗的均值预测）
      mask_np: [N_aug, T] 高置信度掩膜(0/1)
    """
    model.train()  # 关键：启用 dropout
    N, T = seis_aug_norm.shape
    seis_tensor = torch.from_numpy(seis_aug_norm).float().unsqueeze(1)  # [N,1,T]

    all_mu = []
    all_std = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            x = seis_tensor[i:i + batch_size].to(device)  # [B,1,T]
            preds = []
            for _ in range(num_samples):
                preds.append(model(x))  # [B,T]
            preds = torch.stack(preds, dim=0)  # [S,B,T]
            mu = preds.mean(dim=0)   # [B,T]
            std = preds.std(dim=0)   # [B,T]
            all_mu.append(mu.cpu())
            all_std.append(std.cpu())

    mu = torch.cat(all_mu, dim=0)   # [N,T]
    std = torch.cat(all_std, dim=0) # [N,T]

    ratio = 1.96 * std / (mu.abs() + 1e-6)
    if alpha is not None:
        # 旧逻辑：ratio 越小越可靠
        mask = (ratio < alpha).float()
        debug_key = f"alpha={alpha}"
    else:
        # 新逻辑：把不确定度映射到 [0,1] 的“置信度”，并按阈值筛选
        # ratio=0 -> confidence=1；ratio 越大 -> confidence 越小
        confidence = 1.0 / (1.0 + ratio)
        mask = (confidence >= conf_threshold).float()
        debug_key = f"conf_threshold={conf_threshold}"

    mu_np = mu.numpy()
    mask_np = mask.numpy()
    
    # 添加mask利用率调试信息
    mask_mean = mask_np.mean()
    print(f"Pseudo mask mean (有效比例): {mask_mean:.4f} ({debug_key})")
    if mask_mean < 0.05:
        print("  Warning: mask利用率过低，建议增大alpha值")
    elif mask_mean > 0.8:
        print("  Warning: mask利用率过高，伪标签质量可能参差不齐")
    
    return mu_np, mask_np


# -------------------------
# 6. 指标: PCC, R^2
# -------------------------
def pcc_np(pred, true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    pm = pred.mean()
    tm = true.mean()
    num = ((pred - pm) * (true - tm)).sum()
    den = np.sqrt(((pred - pm) ** 2).sum() * ((true - tm) ** 2).sum()) + 1e-8
    return float(num / den)


def r2_np(pred, true):
    pred = pred.reshape(-1)
    true = true.reshape(-1)
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum() + 1e-8
    return float(1 - ss_res / ss_tot)


def evaluate(model, loader, device, norm_params: NormParams):
    model.eval()
    all_pred = []
    all_true = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
    all_pred = np.concatenate(all_pred, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    pred_phys = denormalize_impedance(all_pred, norm_params)
    true_phys = denormalize_impedance(all_true, norm_params)
    pcc = pcc_np(pred_phys, true_phys)
    r2 = r2_np(pred_phys, true_phys)
    return pcc, r2


# -------------------------
# 7. 训练：纯监督 & 半监督
# -------------------------
def train_supervised(model, train_loader, val_loader,
                     device, num_epochs=200, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)
    # 论文反演评价与训练均强调 SmoothL1Loss 的鲁棒性
    criterion = nn.SmoothL1Loss()

    best_val = 1e9
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
        val_loss = float(np.mean(val_losses))
        print(f"[Supervised] Epoch {epoch:03d}, val loss = {val_loss:.5f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> New best val loss: {best_val:.5f}")

        scheduler.step(val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_semi_supervised(model,
                          labeled_loader,
                          pseudo_loader,
                          forward_model,
                          norm_params: NormParams,
                          device,
                          val_loader=None,
                          num_epochs=100,
                          lambda_pseudo=0.1,
                          lambda_fwd=0.05,
                          lr=5e-4):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    crit = nn.SmoothL1Loss()

    imp_min_t = torch.tensor(norm_params.imp_min, dtype=torch.float32, device=device)
    imp_range_t = torch.tensor(norm_params.imp_range + EPS, dtype=torch.float32, device=device)
    seis_min_t = torch.tensor(norm_params.seis_min, dtype=torch.float32, device=device)
    seis_range_t = torch.tensor(norm_params.seis_range + EPS, dtype=torch.float32, device=device)

    best_val = float('inf')
    best_state = None

    for epoch in range(1, num_epochs + 1):
        model.train()

        # 1) 有真实标签的监督部分
        for x, y in labeled_loader:
            x = x.to(device)      # 归一化地震
            y = y.to(device)      # 归一化阻抗

            pred = model(x)       # 归一化阻抗
            loss_sup = crit(pred, y)

            # 正演约束 (转回物理量)
            imp_phys = pred * imp_range_t + imp_min_t       # [B,T]
            seis_pred_phys = forward_model(imp_phys)       # [B,T]
            seis_true_phys = x.squeeze(1) * seis_range_t + seis_min_t
            loss_fwd = F.mse_loss(seis_pred_phys, seis_true_phys)

            loss = loss_sup + lambda_fwd * loss_fwd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 2) 伪标签部分（关闭正演损失，只保留伪标签损失）
        for x, y_pseudo, mask in pseudo_loader:
            x = x.to(device)
            y_pseudo = y_pseudo.to(device)
            mask = mask.to(device)

            pred = model(x)  # [B,T]
            diff = F.smooth_l1_loss(pred, y_pseudo, reduction='none')
            diff = diff * mask
            loss_pseudo = diff.sum() / (mask.sum() + 1e-6)

            # 关闭伪标签数据上的正演损失，避免过度约束
            loss = lambda_pseudo * loss_pseudo

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 验证（可选）
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(device)
                    y = y.to(device)
                    pred = model(x)
                    val_losses.append(crit(pred, y).item())
            val_loss = float(np.mean(val_losses))
            print(f"[Semi] Epoch {epoch:03d}, val loss = {val_loss:.5f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  -> New best val loss: {best_val:.5f}")
            scheduler.step(val_loss)
        else:
            print(f"[Semi] Epoch {epoch:03d} finished.")

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"[Semi] Loaded best checkpoint with val loss = {best_val:.5f}")
    return model


# -------------------------
# 8. 主流程: 串起来
# -------------------------
def main():
    args = build_argparser().parse_args()

    data_root = args.data_root
    run_dir = prepare_run_dir(data_root, args.run_dir)
    setup_logging(run_dir / "train.log")
    logger = logging.getLogger(__name__)
    logger.info("Run dir: %s", str(run_dir))
    logger.info("Args: %s", json.dumps(vars(args), ensure_ascii=False, indent=2))
    (run_dir / "config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    set_global_seed(args.seed, deterministic=args.deterministic)

    # ==== 数据路径 ====
    # 把 seismic.npy 和 impedance.npy 放在 data_root 目录下
    seis_path = os.path.join(data_root, "seismic.npy")
    imp_path = os.path.join(data_root, "impedance.npy")

    seismic = np.load(seis_path).astype(np.float32)    # [T, Nx]
    impedance = np.load(imp_path).astype(np.float32)   # [T, Nx]

    T, Nx = seismic.shape
    print("Data shape: ", seismic.shape)

    # 选择训练/验证/测试道：优先按论文的“均匀选取 20/5/5”策略
    np.random.seed(args.seed)
    use_count_split = (
        args.train_count is not None and args.val_count is not None and args.test_count is not None
        and args.train_count > 0 and args.val_count > 0 and args.test_count > 0
    )
    if use_count_split:
        total = int(args.train_count + args.val_count + args.test_count)
        total = max(3, min(total, Nx))
        if args.split_mode == "uniform":
            all_idx = np.unique(np.linspace(0, Nx - 1, total, dtype=int))
            # 若 unique 导致数量不足，补齐
            if len(all_idx) < total:
                remain = np.setdiff1d(np.arange(Nx), all_idx)
                need = total - len(all_idx)
                all_idx = np.concatenate([all_idx, remain[:need]])
        else:
            all_idx = np.arange(Nx)
            np.random.shuffle(all_idx)
            all_idx = all_idx[:total]
        train_end = min(int(args.train_count), len(all_idx))
        val_end = min(train_end + int(args.val_count), len(all_idx))
        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]
    else:
        all_idx = np.arange(Nx)
        np.random.shuffle(all_idx)
        train_ratio, val_ratio = args.train_ratio, args.val_ratio
        train_count = max(1, int(Nx * train_ratio))
        val_count = max(1, int(Nx * val_ratio))
        train_end = min(train_count, Nx)
        val_end = min(train_end + val_count, Nx)
        train_idx = all_idx[:train_end]
        val_idx = all_idx[train_end:val_end]
        test_idx = all_idx[val_end:]
    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    if len(test_idx) == 0:
        test_idx = val_idx[:1]

    norm_path = os.path.join(data_root, "norm_params.json")
    norm_params = compute_norm_params(seismic, impedance, train_idx)
    save_norm_params(norm_path, norm_params)
    print(f"Normalization params saved to {norm_path}")

    seismic_norm = normalize_seismic(seismic, norm_params)
    impedance_norm = normalize_impedance(impedance, norm_params)

    # 数据集 & DataLoader
    train_ds = LabeledSeisDataset(seismic_norm, impedance_norm, train_idx)
    val_ds = LabeledSeisDataset(seismic_norm, impedance_norm, val_idx)
    test_ds = LabeledSeisDataset(seismic_norm, impedance_norm, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)

    RUN_SEMI = args.run_semi  # 是否执行半监督消融阶段

    # 模型 & 正演算子
    model = CNNBiLSTM().to(device)
    wavelet = ricker(f0=args.wavelet_f0, dt=args.wavelet_dt, nt=args.wavelet_nt)
    forward_model = ForwardModel(wavelet).to(device)

    # -------- 阶段 1: 纯监督训练 --------
    model = train_supervised(
        model,
        train_loader,
        val_loader,
        device=device,
        num_epochs=args.epochs_supervised,
        lr=args.lr_supervised,
    )

    # 在测试集评估一下（简单算 SmoothL1 + PCC + R^2）
    pcc, r2 = evaluate(model, test_loader, device, norm_params)
    print(f"[Supervised] Test PCC = {pcc:.4f}, R2 = {r2:.4f}")

    sup_ckpt = os.path.join(data_root, "marmousi_cnn_bilstm_supervised.pth")
    torch.save(model.state_dict(), sup_ckpt)
    print(f"Supervised checkpoint saved to {sup_ckpt}")
    torch.save(model.state_dict(), run_dir / "marmousi_cnn_bilstm_supervised.pth")

    if RUN_SEMI:
        # -------- 阶段 2: 阻抗域增广 + MC Dropout 伪标签 --------
        print("Building augmented pairs ...")
        imp_aug_phys, seis_aug_phys = build_augmented_pairs(
            impedance,
            train_idx,
            wavelet,
            factor=args.augment_factor,
            n_aug_per_trace=args.n_aug_per_trace,
        )

        # 使用同一套最小-最大归一化
        seis_aug_norm = normalize_seismic(seis_aug_phys, norm_params)

        print("Running MC Dropout to build pseudo labels ...")
        mu_norm, mask = mc_dropout_pseudo(
            model,
            seis_aug_norm,
            device,
            num_samples=args.mc_samples,
            batch_size=args.mc_batch_size,
            conf_threshold=args.pseudo_conf_threshold,
            alpha=args.mc_alpha,
        )

        pseudo_ds = AugmentedPseudoDataset(
            seis_aug=seis_aug_norm,
            imp_pseudo=mu_norm,   # 归一化阻抗
            mask=mask
        )
        pseudo_loader = DataLoader(pseudo_ds, batch_size=args.pseudo_batch_size, shuffle=True)

        # -------- 阶段 3: 半监督训练（在已有模型基础上继续训） --------
        print("Start semi-supervised training ...")
        
        if args.freeze_cnn:
            # 冻结CNN层，只微调BiLSTM+FC层
            for name, param in model.named_parameters():
                if 'conv' in name or 'bn' in name:
                    param.requires_grad = False
            print("Frozen CNN layers, only fine-tuning BiLSTM + FC layers")
        
        semi_train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        model = train_semi_supervised(
            model,
            semi_train_loader,
            pseudo_loader,
            forward_model,
            norm_params,
            device=device,
            val_loader=val_loader,
            num_epochs=args.epochs_semi,
            lambda_pseudo=args.lambda_pseudo,
            lambda_fwd=args.lambda_fwd,
            lr=args.lr_semi,
        )

        # 半监督后的测试集评估
        pcc, r2 = evaluate(model, test_loader, device, norm_params)
        print(f"[Semi] Test PCC = {pcc:.4f}, R2 = {r2:.4f}")

        semi_ckpt = os.path.join(data_root, "marmousi_cnn_bilstm_semi.pth")
        torch.save(model.state_dict(), semi_ckpt)
        print(f"Semi-supervised checkpoint saved to {semi_ckpt}")
        torch.save(model.state_dict(), run_dir / "marmousi_cnn_bilstm_semi.pth")
    else:
        print("RUN_SEMI=False, skipping semi-supervised stage.")


if __name__ == "__main__":
    main()
