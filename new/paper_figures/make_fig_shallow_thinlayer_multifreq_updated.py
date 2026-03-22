from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from thinlayer_multifreq_utils import ensure_prediction_cache, prepare_context


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "new" / "paper_figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make updated multi-frequency thin-layer paper figure.")
    parser.add_argument("--result-20", required=True)
    parser.add_argument("--result-30", required=True)
    parser.add_argument("--result-40", required=True)
    parser.add_argument("--out-png", default=str(OUT_DIR / "fig_shallow_thinlayer_multifreq_updated.png"))
    parser.add_argument("--out-pdf", default=str(OUT_DIR / "fig_shallow_thinlayer_multifreq_updated.pdf"))
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def load_truth_impedance() -> np.ndarray:
    path = REPO_ROOT / "zmy_data" / "01" / "data" / "01_40Hz_04.txt"
    raw = np.loadtxt(path, usecols=4, skiprows=1).astype(np.float32)
    return raw.reshape(100, 10001)


def load_prediction(freq: str, result_dir: Path) -> np.ndarray:
    context = prepare_context(freq, result_dir, augment_train=False)
    return ensure_prediction_cache(context, result_dir / "checkpoints" / "best.pt", result_dir / "pred_full.npy")


def panel_image(ax: plt.Axes, data: np.ndarray, title: str, vmin: float, vmax: float, panel: str) -> None:
    extent = [1, data.shape[0], data.shape[1] * 0.01, 0.0]
    im = ax.imshow(
        data.T,
        aspect="auto",
        cmap="viridis",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_title(f"{panel} {title}", fontsize=12, fontweight="bold", pad=6)
    ax.tick_params(labelsize=9, width=0.8, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    return im


def main() -> None:
    args = parse_args()
    out_png = Path(args.out_png)
    out_pdf = Path(args.out_pdf)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    result_20 = Path(args.result_20)
    result_30 = Path(args.result_30)
    result_40 = Path(args.result_40)

    true_imp = load_truth_impedance()
    pred_20 = load_prediction("20Hz", result_20)
    pred_30 = load_prediction("30Hz", result_30)
    pred_40 = load_prediction("40Hz", result_40)

    vmin = float(np.percentile(true_imp, 1.0))
    vmax = float(np.percentile(true_imp, 99.0))

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)

    im = panel_image(axes[0, 0], true_imp, "Reference Impedance", vmin, vmax, "(a)")
    panel_image(axes[0, 1], pred_20, "20 Hz", vmin, vmax, "(b)")
    panel_image(axes[1, 0], pred_30, "30 Hz", vmin, vmax, "(c)")
    panel_image(axes[1, 1], pred_40, "40 Hz", vmin, vmax, "(d)")

    axes[0, 0].set_ylabel("Time (ms)", fontsize=11)
    axes[1, 0].set_ylabel("Time (ms)", fontsize=11)
    axes[1, 0].set_xlabel("Trace", fontsize=11)
    axes[1, 1].set_xlabel("Trace", fontsize=11)

    cbar = fig.colorbar(im, ax=axes, shrink=0.95, pad=0.02)
    cbar.set_label("Impedance", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    sources = {
        "truth_path": str(REPO_ROOT / "zmy_data" / "01" / "data" / "01_40Hz_04.txt"),
        "result_20": str(result_20),
        "result_30": str(result_30),
        "result_40": str(result_40),
        "pred_20": str(result_20 / "pred_full.npy"),
        "pred_30": str(result_30 / "pred_full.npy"),
        "pred_40": str(result_40 / "pred_full.npy"),
        "out_png": str(out_png),
        "out_pdf": str(out_pdf),
        "color_scale_percentiles": [1.0, 99.0],
    }
    with open(out_png.with_suffix(".sources.json"), "w", encoding="utf-8") as f:
        json.dump(sources, f, indent=2)

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")


if __name__ == "__main__":
    main()
