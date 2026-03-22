from __future__ import annotations

import importlib.util
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_DIR = REPO_ROOT / "new"
DATA_DIR = REPO_ROOT / "zmy_data" / "01" / "data"

SCRIPT_MAP = {
    "20Hz": NEW_DIR / "train_20Hz_thinlayer_v2.py",
    "30Hz": NEW_DIR / "train_30Hz_thinlayer_v3.py",
    "40Hz": NEW_DIR / "train_40Hz_thinlayer_v2.py",
}

DEFAULT_RESULT_DIRS = {
    "20Hz": NEW_DIR / "results" / "01_20Hz_thinlayer_v2",
    "30Hz": NEW_DIR / "results" / "01_30Hz_verified",
    "40Hz": NEW_DIR / "results" / "01_40Hz_thinlayer_v2",
}

DOMINANT_FREQS = {
    "20Hz": 20.0,
    "30Hz": 30.0,
    "40Hz": 40.0,
}

HIGHPASS_CUTOFFS = {
    "20Hz": 8.0,
    "30Hz": 12.0,
    "40Hz": 15.0,
}


@dataclass
class PreparedContext:
    freq: str
    module: ModuleType
    device: torch.device
    result_dir: Path
    seismic: np.ndarray
    impedance: np.ndarray
    norm_stats: dict[str, float]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    labeler: Any
    metrics_calc: Any


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_module(freq: str) -> ModuleType:
    script_path = SCRIPT_MAP[freq]
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load training module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_module_for_freq(module: ModuleType, freq: str, result_dir: Path) -> None:
    module.CFG.SEISMIC_PATH = str(DATA_DIR / f"01_{freq}_re.sgy")
    module.CFG.IMPEDANCE_PATH = str(DATA_DIR / f"01_{freq}_04.txt")
    module.CFG.OUTPUT_DIR = result_dir
    module.CFG.DOMINANT_FREQ = DOMINANT_FREQS[freq]
    module.CFG.DT = 0.001


def load_or_create_norm_stats(result_dir: Path, seismic: np.ndarray, impedance: np.ndarray, freq: str) -> dict[str, float]:
    norm_path = result_dir / "norm_stats.json"
    if norm_path.exists():
        with open(norm_path, "r", encoding="utf-8") as f:
            stats = json.load(f)
    else:
        stats = {
            "seis_mean": float(seismic.mean()),
            "seis_std": float(seismic.std()),
            "imp_mean": float(impedance.mean()),
            "imp_std": float(impedance.std()),
            "highpass_cutoff": HIGHPASS_CUTOFFS[freq],
        }
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    return stats


def make_splits(seed: int, n_traces: int, train_ratio: float, val_ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(n_traces)
    np.random.seed(seed)
    np.random.shuffle(idx)
    n_train = int(n_traces * train_ratio)
    n_val = int(n_traces * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def prepare_context(freq: str, result_dir: Path | None = None, augment_train: bool = True) -> PreparedContext:
    result_dir = result_dir or DEFAULT_RESULT_DIRS[freq]
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "checkpoints").mkdir(exist_ok=True)

    module = load_training_module(freq)
    configure_module_for_freq(module, freq, result_dir)
    _set_seed(int(module.CFG.SEED))

    seismic, _dt = module.load_seismic_data(module.CFG.SEISMIC_PATH)
    impedance = module.load_impedance_data(module.CFG.IMPEDANCE_PATH, seismic.shape[0])
    norm_stats = load_or_create_norm_stats(result_dir, seismic, impedance, freq)

    train_idx, val_idx, test_idx = make_splits(
        int(module.CFG.SEED),
        seismic.shape[0],
        float(module.CFG.TRAIN_RATIO),
        float(module.CFG.VAL_RATIO),
    )

    labeler = module.ThinLayerLabeler(dt=float(module.CFG.DT), dominant_freq=float(module.CFG.DOMINANT_FREQ))
    metrics_calc = module.ThinLayerMetrics(labeler)

    augmentor = None
    if augment_train:
        augmentor = module.ThinLayerAugmentor(
            prob=float(getattr(module.CFG, "AUGMENT_PROB", 0.5)),
            min_thick=int(getattr(module.CFG, "MIN_THIN_THICKNESS", 5)),
            max_thick=int(getattr(module.CFG, "MAX_THIN_THICKNESS", 30)),
        )

    train_ds = module.ThinLayerDatasetV2(seismic, impedance, train_idx, norm_stats, augmentor=augmentor, labeler=labeler)
    val_ds = module.ThinLayerDatasetV2(seismic, impedance, val_idx, norm_stats, augmentor=None, labeler=labeler)
    test_ds = module.ThinLayerDatasetV2(seismic, impedance, test_idx, norm_stats, augmentor=None, labeler=labeler)

    batch_size = int(getattr(module.CFG, "BATCH_SIZE", 4))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return PreparedContext(
        freq=freq,
        module=module,
        device=device,
        result_dir=result_dir,
        seismic=seismic,
        impedance=impedance,
        norm_stats=norm_stats,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        labeler=labeler,
        metrics_calc=metrics_calc,
    )


def build_model(context: PreparedContext) -> torch.nn.Module:
    return context.module.ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(context.device)


def load_checkpoint_into_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def evaluate_model(context: PreparedContext, model: torch.nn.Module) -> tuple[dict[str, Any], dict[str, Any]]:
    val_metrics = context.module.evaluate(
        model,
        context.val_loader,
        context.device,
        context.norm_stats,
        context.metrics_calc,
    )
    test_metrics = context.module.evaluate(
        model,
        context.test_loader,
        context.device,
        context.norm_stats,
        context.metrics_calc,
    )
    return val_metrics, test_metrics


def selection_score(metrics: dict[str, Any]) -> float:
    dpde_penalty = min(float(metrics["dpde_mean"]), 10.0)
    return (
        0.55 * float(metrics["pcc"])
        + 0.20 * float(metrics["thin_pcc"])
        + 0.15 * float(metrics["separability_mean"])
        + 0.10 * float(metrics["r2"])
        - 0.02 * dpde_penalty
    )


def infer_full_section(context: PreparedContext, model: torch.nn.Module, batch_size: int = 8) -> np.ndarray:
    model.eval()

    seis_mean = float(context.norm_stats["seis_mean"])
    seis_std = float(context.norm_stats["seis_std"]) + 1e-6
    imp_mean = float(context.norm_stats["imp_mean"])
    imp_std = float(context.norm_stats["imp_std"])
    cutoff = float(context.norm_stats.get("highpass_cutoff", HIGHPASS_CUTOFFS[context.freq]))

    seismic = context.seismic.astype(np.float32)
    seismic_norm = (seismic - seis_mean) / seis_std
    seismic_hf = context.module.highpass_filter(seismic, cutoff=cutoff, fs=1000)
    seismic_hf_norm = seismic_hf / (np.std(seismic_hf, axis=1, keepdims=True) + 1e-6)

    stacked = np.stack([seismic_norm, seismic_hf_norm], axis=1)
    outputs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, stacked.shape[0], batch_size):
            batch = torch.from_numpy(stacked[start:start + batch_size]).to(context.device)
            pred = model(batch).cpu().numpy().squeeze(1)
            outputs.append(pred.astype(np.float32))

    pred_norm = np.concatenate(outputs, axis=0)
    return pred_norm * imp_std + imp_mean


def ensure_prediction_cache(context: PreparedContext, ckpt_path: Path, cache_path: Path | None = None) -> np.ndarray:
    cache_path = cache_path or (ckpt_path.parent.parent / "pred_full.npy")
    if cache_path.exists():
        return np.load(cache_path)

    model = build_model(context)
    load_checkpoint_into_model(model, ckpt_path, context.device)
    pred = infer_full_section(context, model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pred.astype(np.float32))
    return pred


def to_plain_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain_dict(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
