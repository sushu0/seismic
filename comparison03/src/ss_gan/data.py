from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

@dataclass
class NPZDatasetConfig:
    path: str
    split: str
    normalize: bool = True

class SeisImpNPZ(Dataset):
    def __init__(self, cfg: NPZDatasetConfig, stats: dict | None = None):
        z = np.load(cfg.path, allow_pickle=True)
        self.split = cfg.split
        self.normalize = cfg.normalize
        self.stats = stats or (z["stats"].item() if "stats" in z else {})

        def get(name):
            if name not in z:
                raise KeyError(f"Missing '{name}' in {cfg.path}")
            return z[name].astype(np.float32)

        if cfg.split == "labeled":
            self.x = get("x_labeled")
            self.y = get("y_labeled")
        elif cfg.split == "unlabeled":
            self.x = get("x_unlabeled")
            self.y = None
        elif cfg.split == "val":
            self.x = get("x_val")
            self.y = get("y_val")
        elif cfg.split == "test":
            self.x = get("x_test")
            self.y = get("y_test")
        else:
            raise ValueError("split must be labeled/unlabeled/val/test")

        if self.normalize and self.stats:
            xm, xs = self.stats.get("x_mean", 0.0), self.stats.get("x_std", 1.0)
            self.x = (self.x - xm) / (xs + 1e-12)
            if self.y is not None:
                ym, ys = self.stats.get("y_mean", 0.0), self.stats.get("y_std", 1.0)
                self.y = (self.y - ym) / (ys + 1e-12)

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        x = torch.from_numpy(self.x[idx]).unsqueeze(0)
        if self.y is None:
            return {"x": x}
        y = torch.from_numpy(self.y[idx]).unsqueeze(0)
        return {"x": x, "y": y}

def make_loader(cfg: NPZDatasetConfig, batch_size: int, shuffle: bool, num_workers: int = 0, stats: dict | None = None):
    ds = SeisImpNPZ(cfg, stats=stats)
    # For evaluation we should not drop the last (possibly smaller) batch.
    drop_last = bool(shuffle)
    pin_memory = torch.cuda.is_available()
    persistent_workers = bool(num_workers > 0)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
