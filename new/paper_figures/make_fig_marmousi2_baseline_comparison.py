from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "new" / "paper_figures"


@dataclass(frozen=True)
class SourceInfo:
    label: str
    paths: tuple[Path, ...]
    note: str


def linear_resample(arr: np.ndarray, out_len: int) -> np.ndarray:
    """Linearly resample a [N, T] array along time."""
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D array [N, T], got {arr.shape}")
    if arr.shape[1] == out_len:
        return arr.astype(np.float32, copy=False)
    xp = np.linspace(0.0, 1.0, arr.shape[1], dtype=np.float64)
    xq = np.linspace(0.0, 1.0, out_len, dtype=np.float64)
    out = np.empty((arr.shape[0], out_len), dtype=np.float32)
    for i in range(arr.shape[0]):
        out[i] = np.interp(xq, xp, arr[i].astype(np.float64)).astype(np.float32)
    return out


def load_common_truth() -> tuple[np.ndarray, SourceInfo]:
    data_path = REPO_ROOT / "new" / "data.npy"
    data = np.load(data_path, allow_pickle=True).item()
    truth_full = np.asarray(data["acoustic_impedance"], dtype=np.float32)[:, 0, :]
    truth = linear_resample(truth_full, 470)
    source = SourceInfo(
        label="Reference",
        paths=(data_path,),
        note="Reference impedance taken from new/data.npy and linearly resampled from 1880 to 470 samples for a common comparison domain.",
    )
    return truth, source


def load_comparison01_prediction(device: torch.device) -> tuple[np.ndarray, SourceInfo]:
    import sys

    cmp_dir = REPO_ROOT / "comparison01"
    sys.path.insert(0, str(cmp_dir))
    from marmousi_cnn_bilstm import (  # type: ignore
        CNNBiLSTM,
        denormalize_impedance,
        load_norm_params,
        normalize_seismic,
    )

    seismic = np.load(cmp_dir / "seismic.npy").astype(np.float32)  # [T, Nx]
    norm_params = load_norm_params(cmp_dir / "norm_params.json")
    seismic_norm = normalize_seismic(seismic, norm_params).T  # [Nx, T]

    model = CNNBiLSTM().to(device)
    ckpt = torch.load(cmp_dir / "marmousi_cnn_bilstm_semi.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.eval()

    preds = []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, seismic_norm.shape[0], batch_size):
            batch = seismic_norm[start : start + batch_size]
            batch_tensor = torch.from_numpy(batch[:, None, :]).float().to(device)
            pred = model(batch_tensor).cpu().numpy()  # [B, T]
            preds.append(pred)
    pred_norm = np.concatenate(preds, axis=0).T  # [T, Nx]
    pred_phys = denormalize_impedance(pred_norm, norm_params).T.astype(np.float32)  # [Nx, T]

    source = SourceInfo(
        label="CNN-BiLSTM (semi-supervised)",
        paths=(
            cmp_dir / "seismic.npy",
            cmp_dir / "norm_params.json",
            cmp_dir / "marmousi_cnn_bilstm_semi.pth",
            cmp_dir / "marmousi_cnn_bilstm.py",
        ),
        note="Recomputed from the final semi-supervised CNN-BiLSTM checkpoint instead of using an existing screenshot figure.",
    )
    return pred_phys, source


def load_comparison02_prediction() -> tuple[np.ndarray, SourceInfo]:
    result_dir = REPO_ROOT / "comparison02" / "runs" / "marmousi2" / "results"
    pred = np.load(result_dir / "pred_impedance_all.npy").astype(np.float32)
    pred = linear_resample(pred, 470)
    source = SourceInfo(
        label="FCRSN-CW",
        paths=(
            result_dir / "pred_impedance_all.npy",
            result_dir / "true_impedance_all.npy",
        ),
        note="Full-resolution FCRSN-CW prediction resampled linearly from 1880 to 470 samples for fair side-by-side comparison.",
    )
    return pred, source


def load_comparison03_prediction() -> tuple[np.ndarray, SourceInfo]:
    run_dir = REPO_ROOT / "comparison03" / "runs" / "optimized_v3_advanced"
    pred_pack = np.load(run_dir / "pred_test.npz")
    with open(run_dir / "stats.json", "r", encoding="utf-8") as f:
        stats = json.load(f)
    y_mean = float(stats["y_mean"])
    y_std = float(stats["y_std"])
    pred_norm = np.asarray(pred_pack["pred"], dtype=np.float32)
    pred_phys = pred_norm[:, 0, :] * y_std + y_mean
    source = SourceInfo(
        label="Semi-supervised WGAN-GP",
        paths=(
            run_dir / "pred_test.npz",
            run_dir / "stats.json",
            REPO_ROOT / "comparison03" / "data" / "marmousi2_2721_like_l101.npz",
        ),
        note="Prediction denormalized from the optimized_v3_advanced semi-supervised GAN/WGAN-GP run. Its packaged Marmousi2 conversion is highly consistent with the repo-wide common truth.",
    )
    return pred_phys.astype(np.float32), source


def load_new_prediction(device: torch.device) -> tuple[np.ndarray, SourceInfo]:
    import sys
    import yaml

    new_dir = REPO_ROOT / "new"
    sys.path.insert(0, str(new_dir))
    from seisinv.models.baselines import UNet1D  # type: ignore

    data = np.load(new_dir / "data.npy", allow_pickle=True).item()
    seismic = np.asarray(data["seismic"], dtype=np.float32)[:, 0, :]  # [N, T]

    with open(new_dir / "configs" / "exp_real_data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(new_dir / "results" / "real_unet1d_optimized" / "norm_stats.json", "r", encoding="utf-8") as f:
        norm_stats = json.load(f)

    model = UNet1D(
        in_ch=1,
        out_ch=1,
        base=int(cfg["model"]["base"]),
        depth=int(cfg["model"]["depth"]),
    ).to(device)
    ckpt_path = new_dir / "results" / "real_unet1d_optimized" / "checkpoints" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    seis_norm = (seismic - float(norm_stats["seis_mean"])) / float(norm_stats["seis_std"])
    preds = []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, seis_norm.shape[0], batch_size):
            batch = seis_norm[start : start + batch_size]
            batch_tensor = torch.from_numpy(batch[:, None, :]).float().to(device)
            pred = model(batch_tensor)
            if isinstance(pred, tuple):
                pred = pred[0]
            preds.append(pred.cpu().numpy()[:, 0, :])
    pred_norm = np.concatenate(preds, axis=0)
    pred_phys = pred_norm * float(norm_stats["imp_std"]) + float(norm_stats["imp_mean"])

    source = SourceInfo(
        label="Proposed Method",
        paths=(
            new_dir / "data.npy",
            new_dir / "configs" / "exp_real_data.yaml",
            new_dir / "results" / "real_unet1d_optimized" / "norm_stats.json",
            ckpt_path,
            new_dir / "seisinv" / "models" / "baselines.py",
        ),
        note="Regenerated from the best checkpoint of the Marmousi2-derived real_unet1d_optimized line in new/.",
    )
    return pred_phys.astype(np.float32), source


def render_panel(
    ax: plt.Axes,
    img: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
    cmap: str,
    show_ylabel: bool,
) -> matplotlib.image.AxesImage:
    im = ax.imshow(
        img.T,
        cmap=cmap,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        extent=[0, img.shape[0] - 1, img.shape[1] - 1, 0],
    )
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=8)
    ax.set_xlabel("Trace number", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Time sample", fontsize=9)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="both", labelsize=8, width=0.6, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    return im


def make_figure(
    panels: list[tuple[str, np.ndarray]],
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> dict[str, float]:
    truth = panels[0][1]
    vmin = float(np.min(truth))
    vmax = float(np.max(truth))
    cmap = "viridis"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )

    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(16.5, 3.9),
        constrained_layout=False,
        facecolor="white",
    )
    images = []
    for idx, (ax, (title, img)) in enumerate(zip(axes, panels)):
        images.append(
            render_panel(
                ax=ax,
                img=img,
                title=title,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                show_ylabel=(idx == 0),
            )
        )

    fig.subplots_adjust(left=0.055, right=0.92, top=0.88, bottom=0.18, wspace=0.08)
    cax = fig.add_axes([0.935, 0.20, 0.012, 0.62])
    cbar = fig.colorbar(images[-1], cax=cax)
    cbar.set_label("Impedance", fontsize=9)
    cbar.ax.tick_params(labelsize=8, width=0.6, length=3)
    cbar.outline.set_linewidth(0.6)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"vmin": vmin, "vmax": vmax, "dpi": float(dpi)}


def write_metadata(path: Path, sources: list[SourceInfo], figure_meta: dict[str, float]) -> None:
    payload = {
        "figure": {
            "png": str(OUT_DIR / "fig_marmousi2_baseline_comparison.png"),
            "pdf": str(OUT_DIR / "fig_marmousi2_baseline_comparison.pdf"),
            "colormap": "viridis",
            "shared_vmin": figure_meta["vmin"],
            "shared_vmax": figure_meta["vmax"],
            "dpi": figure_meta["dpi"],
        },
        "sources": [
            {
                "label": src.label,
                "paths": [str(p) for p in src.paths],
                "note": src.note,
            }
            for src in sources
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Regenerate a publication-quality Marmousi2 comparison figure.")
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    truth, truth_src = load_common_truth()
    cmp01, cmp01_src = load_comparison01_prediction(device)
    cmp02, cmp02_src = load_comparison02_prediction()
    cmp03, cmp03_src = load_comparison03_prediction()
    proposed, proposed_src = load_new_prediction(device)

    panels = [
        ("(a) Reference", truth),
        ("(b) CNN-BiLSTM", cmp01),
        ("(c) FCRSN-CW", cmp02),
        ("(d) WGAN-GP", cmp03),
        ("(e) Proposed", proposed),
    ]

    out_png = OUT_DIR / "fig_marmousi2_baseline_comparison.png"
    out_pdf = OUT_DIR / "fig_marmousi2_baseline_comparison.pdf"
    figure_meta = make_figure(panels, out_png, out_pdf, args.dpi)
    write_metadata(
        OUT_DIR / "fig_marmousi2_baseline_comparison_sources.json",
        [truth_src, cmp01_src, cmp02_src, cmp03_src, proposed_src],
        figure_meta,
    )

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print("Shared display range:", figure_meta["vmin"], figure_meta["vmax"])


if __name__ == "__main__":
    main()
