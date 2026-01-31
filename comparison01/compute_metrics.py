import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marmousi_cnn_bilstm import (
    CNNBiLSTM,
    NormParams,
    denormalize_impedance,
    load_norm_params,
    normalize_impedance,
    normalize_seismic,
    pcc_np,
    r2_np,
)


class LegacyCNNBiLSTM(nn.Module):
    """兼容旧版 checkpoint（marmousi_cnn_bilstm.pth）。

    结构由 state_dict 形状反推：
    - 3 层 Conv1d(k=3) + BN + ReLU: 1->32->64->64
    - 3 层 BiLSTM(hidden=64, bidirectional=True)
    - 1 层 LSTM(hidden=64, unidirectional)
    - Linear(64->1)
    """

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32)),
                nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64)),
                nn.Sequential(nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64)),
            ]
        )
        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=3,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_reg = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, 1, T]
        for block in self.convs:
            x = F.relu(block(x))
        x = x.transpose(1, 2)  # [B, T, 64]
        x, _ = self.bilstm(x)  # [B, T, 128]
        x, _ = self.lstm_reg(x)  # [B, T, 64]
        x = self.fc(x)  # [B, T, 1]
        return x.squeeze(-1)  # [B, T]


@dataclass
class SplitArgs:
    seed: int = 0
    split_mode: str = "uniform"  # uniform | random
    train_count: int = 20
    val_count: int = 5
    test_count: int = 5


def _try_parse_split_args_from_config(config_path: Path) -> Optional[SplitArgs]:
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    # 兼容 training 脚本中参数名
    try:
        seed = int(data.get("seed", 0))
        split_mode = str(data.get("split_mode", "uniform"))
        train_count = int(data.get("train_count", 20))
        val_count = int(data.get("val_count", 5))
        test_count = int(data.get("test_count", 5))
    except Exception:
        return None

    if split_mode not in ("uniform", "random"):
        split_mode = "uniform"
    return SplitArgs(
        seed=seed,
        split_mode=split_mode,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
    )


def _find_latest_run_config(data_root: Path) -> Optional[Path]:
    runs_dir = data_root / "runs"
    if not runs_dir.exists():
        return None

    candidates: List[Path] = []
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        cfg = p / "config.json"
        if cfg.exists():
            candidates.append(cfg)

    if not candidates:
        return None

    # 目录名通常是 YYYYMMDD_HHMMSS，按名字排序即可
    candidates.sort(key=lambda x: x.parent.name)
    return candidates[-1]


def build_split_indices(nx: int, split: SplitArgs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(split.seed)

    total = int(split.train_count + split.val_count + split.test_count)
    total = max(3, min(total, nx))

    if split.split_mode == "uniform":
        all_idx = np.unique(np.linspace(0, nx - 1, total, dtype=int))
        if len(all_idx) < total:
            remain = np.setdiff1d(np.arange(nx), all_idx)
            need = total - len(all_idx)
            all_idx = np.concatenate([all_idx, remain[:need]])
    else:
        all_idx = np.arange(nx)
        np.random.shuffle(all_idx)
        all_idx = all_idx[:total]

    train_end = min(int(split.train_count), len(all_idx))
    val_end = min(train_end + int(split.val_count), len(all_idx))

    train_idx = all_idx[:train_end]
    val_idx = all_idx[train_end:val_end]
    test_idx = all_idx[val_end:]

    if len(val_idx) == 0:
        val_idx = train_idx[:1]
    if len(test_idx) == 0:
        test_idx = val_idx[:1]

    return train_idx, val_idx, test_idx


def _select_indices(which: str, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, nx: int) -> np.ndarray:
    which = which.lower()
    if which == "train":
        return train_idx
    if which == "val":
        return val_idx
    if which == "test":
        return test_idx
    if which == "all":
        return np.arange(nx)
    raise ValueError(f"Unknown split: {which}. Use train|val|test|all")


@torch.no_grad()
def eval_one_model(
    model: CNNBiLSTM,
    seismic_norm: np.ndarray,
    impedance_norm: np.ndarray,
    norm_params: NormParams,
    trace_idx: np.ndarray,
    device: str,
    batch_size: int = 32,
) -> Dict[str, float]:
    """返回指标：loss(SmoothL1, in normalized space), PCC/R2 (in physical space)"""

    model.eval()

    # 组 batch：每条道一个样本，shape [B, 1, T]
    x = seismic_norm[:, trace_idx].T  # [N, T]
    y = impedance_norm[:, trace_idx].T  # [N, T]

    x_t = torch.from_numpy(x[:, None, :]).float()
    y_t = torch.from_numpy(y).float()

    criterion = nn.SmoothL1Loss(reduction="mean")

    all_pred_norm: List[np.ndarray] = []
    all_true_norm: List[np.ndarray] = []
    losses: List[float] = []

    for start in range(0, x_t.shape[0], batch_size):
        xb = x_t[start : start + batch_size].to(device)
        yb = y_t[start : start + batch_size].to(device)
        pred = model(xb)
        loss = criterion(pred, yb)
        losses.append(float(loss.item()))
        all_pred_norm.append(pred.detach().cpu().numpy())
        all_true_norm.append(yb.detach().cpu().numpy())

    pred_norm = np.concatenate(all_pred_norm, axis=0)
    true_norm = np.concatenate(all_true_norm, axis=0)

    pred_phys = denormalize_impedance(pred_norm, norm_params)
    true_phys = denormalize_impedance(true_norm, norm_params)

    return {
        "loss": float(np.mean(losses)),
        "pcc": float(pcc_np(pred_phys, true_phys)),
        "r2": float(r2_np(pred_phys, true_phys)),
    }


def _torch_load_weights(path: Path, device: str):
    """torch.load 包装：尽量使用 weights_only=True 以避免安全警告。"""
    try:
        return torch.load(str(path), map_location=device, weights_only=True)
    except TypeError:
        return torch.load(str(path), map_location=device)


def _extract_state_dict(obj):
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]
    return obj


def _make_model_for_state_dict(state_dict: Dict[str, Any]) -> nn.Module:
    keys = list(state_dict.keys())
    if any(k.startswith("conv1.") for k in keys):
        return CNNBiLSTM()
    if any(k.startswith("convs.") for k in keys):
        return LegacyCNNBiLSTM()
    # 默认回退到当前结构（若不匹配会在 load_state_dict 报错）
    return CNNBiLSTM()


def _maybe_load_state_dict(model: nn.Module, ckpt_path: Path, device: str) -> None:
    state = _torch_load_weights(ckpt_path, device=device)
    # 兼容未来若保存成 dict
    state = _extract_state_dict(state)
    model.load_state_dict(state)


def main():
    ap = argparse.ArgumentParser(description="Compute loss/PCC/R2 for Marmousi CNN-BiLSTM models")
    ap.add_argument("--data-root", type=str, default=r"D:\\SEISMIC_CODING\\comparison01")
    ap.add_argument("--eval-split", type=str, default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch-size", type=int, default=32)

    # 若你想手动指定划分参数，可用这些覆盖
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--split-mode", type=str, default=None, choices=["uniform", "random"])
    ap.add_argument("--train-count", type=int, default=None)
    ap.add_argument("--val-count", type=int, default=None)
    ap.add_argument("--test-count", type=int, default=None)

    ap.add_argument("--include-final", action="store_true", help="Also evaluate marmousi_cnn_bilstm.pth if present")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    seis_path = data_root / "seismic.npy"
    imp_path = data_root / "impedance.npy"
    norm_path = data_root / "norm_params.json"

    if not seis_path.exists():
        raise FileNotFoundError(f"Missing file: {seis_path}")
    if not imp_path.exists():
        raise FileNotFoundError(f"Missing file: {imp_path}")
    if not norm_path.exists():
        raise FileNotFoundError(f"Missing file: {norm_path}")

    # 设备
    if args.device == "cpu":
        device = "cpu"
    elif args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # split 参数：优先用最新 run 的 config.json，其次默认；CLI 覆盖
    split = SplitArgs()
    latest_cfg = _find_latest_run_config(data_root)
    if latest_cfg is not None:
        maybe = _try_parse_split_args_from_config(latest_cfg)
        if maybe is not None:
            split = maybe

    if args.seed is not None:
        split.seed = int(args.seed)
    if args.split_mode is not None:
        split.split_mode = str(args.split_mode)
    if args.train_count is not None:
        split.train_count = int(args.train_count)
    if args.val_count is not None:
        split.val_count = int(args.val_count)
    if args.test_count is not None:
        split.test_count = int(args.test_count)

    seismic = np.load(seis_path).astype(np.float32)  # [T, Nx]
    impedance = np.load(imp_path).astype(np.float32)  # [T, Nx]
    t, nx = seismic.shape

    norm_params = load_norm_params(str(norm_path))
    seismic_norm = normalize_seismic(seismic, norm_params)
    impedance_norm = normalize_impedance(impedance, norm_params)

    train_idx, val_idx, test_idx = build_split_indices(nx, split)
    use_idx = _select_indices(args.eval_split, train_idx, val_idx, test_idx, nx)

    ckpts: List[Tuple[str, Path]] = [
        ("supervised", data_root / "marmousi_cnn_bilstm_supervised.pth"),
        ("semi", data_root / "marmousi_cnn_bilstm_semi.pth"),
    ]
    if args.include_final:
        ckpts.append(("final", data_root / "marmousi_cnn_bilstm.pth"))

    rows: List[Tuple[str, Dict[str, float]]] = []
    for name, path in ckpts:
        if not path.exists():
            continue
        raw = _torch_load_weights(path, device=device)
        state_dict = _extract_state_dict(raw)
        model = _make_model_for_state_dict(state_dict).to(device)
        model.load_state_dict(state_dict)
        metrics = eval_one_model(
            model,
            seismic_norm=seismic_norm,
            impedance_norm=impedance_norm,
            norm_params=norm_params,
            trace_idx=use_idx,
            device=device,
            batch_size=int(args.batch_size),
        )
        rows.append((name, metrics))

    if not rows:
        raise FileNotFoundError("No checkpoints found to evaluate.")

    # 输出
    cfg_note = f"latest_config={latest_cfg}" if latest_cfg is not None else "latest_config=None"
    print(f"Eval split: {args.eval_split} (N={len(use_idx)} traces), device={device}")
    print(
        "SplitArgs: "
        f"seed={split.seed}, mode={split.split_mode}, train/val/test={split.train_count}/{split.val_count}/{split.test_count}"
    )
    print(cfg_note)
    print("\nModel\tloss(SmoothL1,norm)\tPCC(phys)\tR2(phys)")
    for name, m in rows:
        print(f"{name}\t{m['loss']:.6f}\t{m['pcc']:.6f}\t{m['r2']:.6f}")


if __name__ == "__main__":
    main()
