import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import segyio


DEFAULT_SGY = Path(r"d:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy")
DEFAULT_OUTDIR = Path(r"d:\SEISMIC_CODING\new\output_images\real_sgy_spectrum_analysis")


def uniform_indices(total: int, count: int) -> np.ndarray:
    count = min(int(count), int(total))
    if count <= 0:
        raise ValueError("count must be positive")
    if count == total:
        return np.arange(total, dtype=np.int64)
    idx = np.linspace(0, total - 1, count)
    return np.round(idx).astype(np.int64)


def read_sgy_metadata(sgy_path: Path) -> Tuple[int, int, np.ndarray, float]:
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as f:
        tracecount = int(f.tracecount)
        samples_ms = np.asarray(f.samples, dtype=np.float32)
        interval_us = segyio.tools.dt(f) or 2000
    return tracecount, int(samples_ms.size), samples_ms, float(interval_us) / 1e6


def read_sgy_traces(sgy_path: Path, trace_indices: np.ndarray) -> np.ndarray:
    traces = []
    with segyio.open(str(sgy_path), "r", ignore_geometry=True) as f:
        for idx in trace_indices.tolist():
            traces.append(np.asarray(f.trace[int(idx)], dtype=np.float32))
    return np.stack(traces, axis=0)


def prepare_windowed_traces(traces: np.ndarray, sample_count: int) -> np.ndarray:
    clipped = traces[:, :sample_count].astype(np.float64)
    centered = clipped - clipped.mean(axis=1, keepdims=True)
    taper = np.hanning(sample_count)[None, :]
    return centered * taper


def spectrum_stats(traces: np.ndarray, dt_s: float, sample_count: int) -> Dict[str, float]:
    windowed = prepare_windowed_traces(traces, sample_count)
    spec = np.fft.rfft(windowed, axis=1)
    amp = np.abs(spec)
    power = amp ** 2
    mean_amp = amp.mean(axis=0)
    mean_power = power.mean(axis=0)
    freqs = np.fft.rfftfreq(sample_count, d=dt_s)

    peak_idx = int(np.argmax(mean_amp[1:]) + 1) if mean_amp.size > 1 else 0
    peak_freq = float(freqs[peak_idx])

    cumulative = np.cumsum(mean_power)
    cumulative = cumulative / np.maximum(cumulative[-1], 1e-12)
    f50 = float(freqs[int(np.searchsorted(cumulative, 0.50))])
    f95 = float(freqs[int(np.searchsorted(cumulative, 0.95))])

    return {
        "sample_count": int(sample_count),
        "duration_ms": float(sample_count * dt_s * 1000.0),
        "peak_hz": peak_freq,
        "f50_hz": f50,
        "f95_hz": f95,
        "freqs_hz": freqs.tolist(),
        "mean_amplitude": mean_amp.tolist(),
        "cumulative_power": cumulative.tolist(),
    }


def robust_section_limits(section: np.ndarray, pct: float = 99.0) -> float:
    return float(np.percentile(np.abs(section), pct))


def plot_spectrum_figure(
    section: np.ndarray,
    section_times_ms: np.ndarray,
    windows: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.20)

    ax0 = fig.add_subplot(gs[0, 0])
    clip = robust_section_limits(section)
    im = ax0.imshow(
        section.T,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-clip,
        vmax=clip,
        extent=[0, section.shape[0] - 1, float(section_times_ms[-1]), float(section_times_ms[0])],
    )
    ax0.axhline(float(section_times_ms[255]), color="#f39c12", lw=2.0, ls="--", label="Top 256 samples")
    ax0.axhline(float(section_times_ms[399]), color="#27ae60", lw=2.0, ls="--", label="Top 400 samples")
    ax0.set_title("(a) Real SGY Section Preview", fontsize=12, fontweight="bold")
    ax0.set_xlabel("Trace index (sampled)")
    ax0.set_ylabel("Time (ms)")
    ax0.legend(loc="lower right", fontsize=9, frameon=True)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    cbar.set_label("Amplitude")

    ax1 = fig.add_subplot(gs[0, 1])
    band_color = "#fdebd0"
    ax1.axvspan(20.0, 30.0, color=band_color, alpha=0.7, label="20-30 Hz thin-bed-sensitive band")
    colors = {
        "full_window": "#1f77b4",
        "top_256": "#ff7f0e",
        "top_400": "#2ca02c",
    }
    labels = {
        "full_window": "Full window",
        "top_256": "Upper window (top 256)",
        "top_400": "Upper window (top 400)",
    }
    for key in ("full_window", "top_256", "top_400"):
        freqs = np.asarray(windows[key]["freqs_hz"])
        amps = np.asarray(windows[key]["mean_amplitude"])
        amps = amps / np.maximum(amps.max(), 1e-12)
        ax1.plot(freqs, amps, lw=2.2, color=colors[key], label=labels[key])
        ax1.axvline(windows[key]["peak_hz"], color=colors[key], lw=1.2, ls=":")
    ax1.set_xlim(0, 60)
    ax1.set_ylim(0, 1.05)
    ax1.set_title("(b) Mean Normalized Amplitude Spectra", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Normalized amplitude")
    ax1.legend(loc="upper right", fontsize=9, frameon=True)
    ax1.grid(alpha=0.25, ls="--")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axvspan(20.0, 30.0, color=band_color, alpha=0.7)
    for key in ("full_window", "top_256", "top_400"):
        freqs = np.asarray(windows[key]["freqs_hz"])
        cum = np.asarray(windows[key]["cumulative_power"])
        ax2.plot(freqs, cum, lw=2.2, color=colors[key], label=labels[key])
        ax2.scatter([windows[key]["f50_hz"]], [0.50], color=colors[key], s=35, zorder=5)
        ax2.scatter([windows[key]["f95_hz"]], [0.95], color=colors[key], s=35, zorder=5, marker="s")
    ax2.axhline(0.50, color="gray", lw=1.0, ls="--")
    ax2.axhline(0.95, color="gray", lw=1.0, ls=":")
    ax2.set_xlim(0, 60)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("(c) Cumulative Power Curves", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Cumulative power")
    ax2.legend(loc="lower right", fontsize=9, frameon=True)
    ax2.grid(alpha=0.25, ls="--")

    ax3 = fig.add_subplot(gs[1, 1])
    metric_names = ["Peak", "F50", "F95"]
    metric_keys = ["peak_hz", "f50_hz", "f95_hz"]
    x = np.arange(len(metric_names))
    width = 0.22
    offsets = {"full_window": -width, "top_256": 0.0, "top_400": width}
    for key in ("full_window", "top_256", "top_400"):
        values = [windows[key][m] for m in metric_keys]
        ax3.bar(x + offsets[key], values, width=width, color=colors[key], alpha=0.88, label=labels[key])
        for xi, val in zip(x + offsets[key], values):
            ax3.text(xi, val + 0.5, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax3.axhspan(20.0, 30.0, color=band_color, alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metric_names)
    ax3.set_ylabel("Frequency (Hz)")
    ax3.set_ylim(0, max(windows["top_400"]["f95_hz"], 30.0) + 6.0)
    ax3.set_title("(d) Frequency Statistics Summary", fontsize=12, fontweight="bold")
    ax3.legend(loc="upper left", fontsize=9, frameon=True)
    ax3.grid(alpha=0.25, ls="--", axis="y")

    fig.suptitle(
        "Real SGY Spectral Analysis for Shallow Thin-bed Sensitive Frequency Interpretation",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.015,
        "The full-window dominant peak is around 12 Hz, while the upper target interval retains effective energy mainly within 10-30 Hz; "
        "therefore 20-30 Hz is treated as a thin-bed-sensitive reference band rather than the dominant peak frequency.",
        ha="center",
        va="bottom",
        fontsize=10,
    )
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real SGY spectrum and generate publication-ready figure")
    parser.add_argument("--sgy", type=Path, default=DEFAULT_SGY)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--sample-traces", type=int, default=4096)
    parser.add_argument("--section-traces", type=int, default=512)
    parser.add_argument("--top-window-a", type=int, default=256)
    parser.add_argument("--top-window-b", type=int, default=400)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    tracecount, sample_count, samples_ms, dt_s = read_sgy_metadata(args.sgy)
    fs_hz = 1.0 / dt_s
    sampled_idx = uniform_indices(tracecount, args.sample_traces)
    section_idx = uniform_indices(tracecount, args.section_traces)

    print(f"Reading {args.sample_traces} sampled traces from {args.sgy} ...")
    sampled_traces = read_sgy_traces(args.sgy, sampled_idx)
    section_traces = read_sgy_traces(args.sgy, section_idx)

    stats = {
        "sgy_path": str(args.sgy),
        "tracecount": tracecount,
        "sample_count": sample_count,
        "dt_seconds": dt_s,
        "fs_hz": fs_hz,
        "time_start_ms": float(samples_ms[0]),
        "time_end_ms": float(samples_ms[-1]),
        "sampled_trace_count": int(sampled_traces.shape[0]),
        "windows": {
            "full_window": spectrum_stats(sampled_traces, dt_s=dt_s, sample_count=sample_count),
            "top_256": spectrum_stats(sampled_traces, dt_s=dt_s, sample_count=args.top_window_a),
            "top_400": spectrum_stats(sampled_traces, dt_s=dt_s, sample_count=args.top_window_b),
        },
    }

    fig_path = args.outdir / "real_sgy_spectrum_analysis.png"
    json_path = args.outdir / "real_sgy_spectrum_summary.json"
    plot_spectrum_figure(section_traces, samples_ms, stats["windows"], fig_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Saved figure: {fig_path}")
    print(f"Saved summary: {json_path}")
    for key, label in (
        ("full_window", "full window"),
        ("top_256", "top 256"),
        ("top_400", "top 400"),
    ):
        item = stats["windows"][key]
        print(
            f"{label}: peak={item['peak_hz']:.4f} Hz, "
            f"f50={item['f50_hz']:.4f} Hz, "
            f"f95={item['f95_hz']:.4f} Hz"
        )


if __name__ == "__main__":
    main()
