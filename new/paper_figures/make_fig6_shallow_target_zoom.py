from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
NEW_DIR = REPO_ROOT / "new"
DEFAULT_RESULT_DIR = NEW_DIR / "sgy_inversion_v11_display_99bridge_full"
OUT_DIR = NEW_DIR / "paper_figures"

sys.path.insert(0, str(NEW_DIR))
from train_sgy_v8 import read_sgy_traces, read_tracecount_and_samples, uniform_indices  # type: ignore


@dataclass
class WindowInfo:
    window_id: str
    trace_start: int
    trace_end: int
    sample_start: int
    sample_end: int
    time_start_ms: float
    time_end_ms: float
    original_trace_start: int
    original_trace_end: int
    score: float
    reason: str


def parse_top_time_ms(sgy_path: str) -> float:
    matches = re.findall(r"(\d+)", Path(sgy_path).stem)
    if len(matches) >= 2:
        return float(matches[-2])
    return 2500.0


def shared_visual_map(section: np.ndarray, kind: str) -> np.ndarray:
    if kind == "seismic":
        centered = section.astype(np.float32)
        scale = float(np.percentile(np.abs(centered), 99.0)) + 1e-6
        return np.tanh(centered / scale).astype(np.float32)
    if kind == "impedance":
        log_section = np.log(np.clip(section, 1e5, None)).astype(np.float32)
        centered = log_section - np.median(log_section)
        scale = float(np.percentile(np.abs(centered), 99.0)) + 1e-6
        return np.tanh(centered / scale).astype(np.float32)
    raise ValueError(f"Unsupported kind: {kind}")


def select_time_window(
    observed: np.ndarray,
    upper_samples: int = 400,
    window_len: int = 260,
) -> tuple[int, int]:
    upper = observed[:, :upper_samples]
    mean_abs = np.mean(np.abs(upper), axis=0)
    mean_grad = np.r_[0.0, np.mean(np.abs(np.diff(upper, axis=1)), axis=0)]
    kernel = np.ones(21, dtype=np.float32) / 21.0
    mean_abs_s = np.convolve(mean_abs, kernel, mode="same")
    mean_grad_s = np.convolve(mean_grad, kernel, mode="same")
    score = 0.4 * (mean_abs_s / (mean_abs_s.max() + 1e-6)) + 0.6 * (mean_grad_s / (mean_grad_s.max() + 1e-6))

    best_score = -np.inf
    best = (0, window_len)
    for start in range(0, upper_samples - window_len + 1, 5):
        end = start + window_len
        current = float(score[start:end].mean())
        if current > best_score:
            best_score = current
            best = (start, end)
    return best


def score_trace_windows(
    observed: np.ndarray,
    impedance: np.ndarray,
    sample_start: int,
    sample_end: int,
    width: int = 700,
    step: int = 140,
) -> List[tuple[float, int, int, float, float, float, float]]:
    rows = []
    obs_top = observed[:, sample_start:sample_end]
    imp_top = impedance[:, sample_start:sample_end]
    for start in range(0, observed.shape[0] - width + 1, step):
        end = start + width
        sw = obs_top[start:end]
        iw = imp_top[start:end]
        event_energy = float(np.mean(np.abs(np.diff(sw, axis=1))))
        lateral_diff = float(np.mean(np.abs(np.diff(sw, axis=0))))
        continuity = event_energy / (lateral_diff + 1e-6)
        imp_edge = float(np.mean(np.abs(np.diff(iw, axis=1))))
        imp_lat = float(np.mean(np.abs(np.diff(iw, axis=0))))
        lat_penalty = 1.0 / (1.0 + imp_lat / (float(np.mean(np.abs(iw))) + 1e-6))
        rows.append((0.0, start, end, event_energy, continuity, imp_edge, lat_penalty))

    feats = np.asarray([r[3:] for r in rows], dtype=np.float64)
    for j in range(feats.shape[1]):
        mn = feats[:, j].min()
        mx = feats[:, j].max()
        if mx - mn > 1e-12:
            feats[:, j] = (feats[:, j] - mn) / (mx - mn)
        else:
            feats[:, j] = 0.0

    scores = 0.45 * feats[:, 0] + 0.25 * feats[:, 1] + 0.25 * feats[:, 2] + 0.05 * feats[:, 3]
    scored = []
    for idx, row in enumerate(rows):
        scored.append((float(scores[idx]), row[1], row[2], row[3], row[4], row[5], row[6]))
    scored.sort(reverse=True)
    return scored


def select_trace_windows(
    observed: np.ndarray,
    impedance: np.ndarray,
    sample_start: int,
    sample_end: int,
    original_trace_ids: np.ndarray,
    num_windows: int = 3,
    width: int = 700,
    step: int = 140,
    max_overlap_ratio: float = 0.15,
    top_time_ms: float = 2500.0,
    dt_ms: float = 2.0,
) -> List[WindowInfo]:
    scored = score_trace_windows(observed, impedance, sample_start, sample_end, width=width, step=step)
    selected: List[WindowInfo] = []
    for score, start, end, *_ in scored:
        ok = True
        for win in selected:
            overlap = max(0, min(end, win.trace_end) - max(start, win.trace_start))
            if overlap > width * max_overlap_ratio:
                ok = False
                break
        if not ok:
            continue

        center = 0.5 * (start + end)
        if center < observed.shape[0] / 3:
            region = "左侧目标段中反射事件较密、边界过渡较丰富，适合展示浅部层间变化。"
        elif center < 2 * observed.shape[0] / 3:
            region = "中部目标段结构最为复杂，能体现边界连续性与细节刻画能力。"
        else:
            region = "右侧目标段横向事件较可追踪，适合展示横向连续性与层间过渡。"

        selected.append(
            WindowInfo(
                window_id=f"W{len(selected) + 1}",
                trace_start=int(start),
                trace_end=int(end),
                sample_start=int(sample_start),
                sample_end=int(sample_end),
                time_start_ms=float(top_time_ms + sample_start * dt_ms),
                time_end_ms=float(top_time_ms + sample_end * dt_ms),
                original_trace_start=int(original_trace_ids[start]),
                original_trace_end=int(original_trace_ids[end - 1]),
                score=float(score),
                reason=region,
            )
        )
        if len(selected) >= num_windows:
            break
    return selected


def section_extent(n_traces: int, n_samples: int, top_time_ms: float, dt_ms: float) -> list[float]:
    return [0.0, float(n_traces - 1), float(top_time_ms + n_samples * dt_ms), float(top_time_ms)]


def show_section(
    ax: plt.Axes,
    section: np.ndarray,
    top_time_ms: float,
    dt_ms: float,
    title: str,
    add_xlabel: bool = True,
    add_ylabel: bool = False,
) -> None:
    ax.imshow(
        section.T,
        aspect="auto",
        cmap="RdBu_r",
        origin="upper",
        interpolation="nearest",
        extent=section_extent(section.shape[0], section.shape[1], top_time_ms, dt_ms),
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=6)
    if add_xlabel:
        ax.set_xlabel("Trace", fontsize=9)
    if add_ylabel:
        ax.set_ylabel("Time (ms)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8, width=0.6, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def build_figure(
    observed: np.ndarray,
    impedance: np.ndarray,
    windows: Sequence[WindowInfo],
    top_time_ms: float,
    dt_ms: float,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    obs_disp = shared_visual_map(observed, kind="seismic")
    imp_disp = shared_visual_map(impedance, kind="impedance")

    colors = ["tab:orange", "tab:green", "tab:purple"]

    fig = plt.figure(figsize=(16.5, 7.6), facecolor="white")
    gs = fig.add_gridspec(2, 4, width_ratios=[1.55, 1.0, 1.0, 1.0], wspace=0.18, hspace=0.18)

    ax_over_obs = fig.add_subplot(gs[0, 0])
    ax_over_imp = fig.add_subplot(gs[1, 0])
    show_section(ax_over_obs, obs_disp, top_time_ms, dt_ms, "(a) Overview: Observed seismic", add_xlabel=False, add_ylabel=True)
    show_section(ax_over_imp, imp_disp, top_time_ms, dt_ms, "Overview: Inverted impedance", add_xlabel=True, add_ylabel=True)

    for color, win in zip(colors, windows):
        rect = patches.Rectangle(
            (win.trace_start, win.time_start_ms),
            win.trace_end - win.trace_start,
            win.time_end_ms - win.time_start_ms,
            linewidth=1.4,
            edgecolor=color,
            facecolor="none",
        )
        rect2 = patches.Rectangle(
            (win.trace_start, win.time_start_ms),
            win.trace_end - win.trace_start,
            win.time_end_ms - win.time_start_ms,
            linewidth=1.4,
            edgecolor=color,
            facecolor="none",
        )
        ax_over_obs.add_patch(rect)
        ax_over_imp.add_patch(rect2)
        ax_over_obs.text(
            win.trace_start + 10,
            win.time_start_ms + 18,
            win.window_id,
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

    for idx, (color, win) in enumerate(zip(colors, windows), start=1):
        ax_obs = fig.add_subplot(gs[0, idx])
        ax_imp = fig.add_subplot(gs[1, idx])
        obs_win_raw = observed[win.trace_start : win.trace_end, win.sample_start : win.sample_end]
        imp_win_raw = impedance[win.trace_start : win.trace_end, win.sample_start : win.sample_end]
        obs_win = shared_visual_map(obs_win_raw, kind="seismic")
        imp_win = shared_visual_map(imp_win_raw, kind="impedance")
        show_section(
            ax_obs,
            obs_win,
            win.time_start_ms,
            dt_ms,
            f"({chr(ord('a') + idx)}) {win.window_id}: Observed",
            add_xlabel=False,
            add_ylabel=False,
        )
        show_section(
            ax_imp,
            imp_win,
            win.time_start_ms,
            dt_ms,
            f"{win.window_id}: Inverted impedance",
            add_xlabel=True,
            add_ylabel=False,
        )
        for ax in (ax_obs, ax_imp):
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readme(
    readme_path: Path,
    source_paths: Sequence[Path],
    windows: Sequence[WindowInfo],
    out_png: Path,
    out_pdf: Path,
    script_path: Path,
) -> None:
    lines = [
        "# Figure 6 浅部目标层段局部放大图说明",
        "",
        "## 输出文件",
        "",
        f"- `{out_png}`",
        f"- `{out_pdf}`",
        f"- `{script_path}`",
        "",
        "## 使用的源文件",
        "",
    ]
    for path in source_paths:
        lines.append(f"- `{path}`")
    lines += [
        "",
        "## 选窗原则",
        "",
        "- 使用 `v11_edgeplus` 主结果对应的同一批 5000 条均匀采样道",
        "- 在研究时间窗上部 400 个采样点内自动搜索浅部目标层段",
        "- 先按事件强度与时间梯度确定统一的浅部时间窗，再按事件丰富度、边界信息量与横向连续性自动挑选 3 个代表性横向窗口",
        "",
        "## 最终窗口范围",
        "",
    ]
    for win in windows:
        lines += [
            f"### {win.window_id}",
            f"- plotted trace range: `{win.trace_start} - {win.trace_end}`",
            f"- original SGY trace ids: `{win.original_trace_start} - {win.original_trace_end}`",
            f"- time range (ms): `{win.time_start_ms:.0f} - {win.time_end_ms:.0f}`",
            f"- score: `{win.score:.4f}`",
            f"- reason: {win.reason}",
            "",
        ]
    lines += [
        "## 绘图风格",
        "",
        "- 横轴为 trace，纵轴为 time (ms)",
        "- 左侧为全剖面概览，上方原始地震、下方主反演阻抗",
        "- 右侧为 3 个局部放大窗口，均采用与主文一致的上下对照排布",
        "- 原始地震与阻抗结果均采用与主文一致的共享显示映射和 `RdBu_r` 配色",
        "",
        "## 建议图题",
        "",
        "图 6 真实 SGY 浅部目标层段局部放大对比图",
        "",
        "## 建议图注",
        "",
        "左侧为真实 SGY 主结果对应的全剖面概览，矩形框标出浅部目标层段的 3 个局部放大位置；右侧为对应窗口的原始地震与主反演阻抗结果对照。局部窗口位于研究时间窗的上部目标层段，用于展示浅部目标层段的结构边界、层间过渡和横向连续性表征能力。",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 6 shallow target zoom figure for the paper.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--num-windows", type=int, default=3)
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    run_cfg_path = result_dir / "run_config.json"
    with open(run_cfg_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    sgy_path = run_cfg["sgy"]
    infer_traces = int(run_cfg["infer_traces"])
    tracecount, nsamples, _, dt_s = read_tracecount_and_samples(sgy_path)
    dt_ms = float(dt_s) * 1000.0
    top_time_ms = parse_top_time_ms(sgy_path)

    infer_trace_ids = uniform_indices(tracecount, infer_traces)
    observed = read_sgy_traces(sgy_path, infer_trace_ids).astype(np.float32)
    impedance = np.load(result_dir / "impedance_pred_final.npy").astype(np.float32)
    if observed.shape != impedance.shape:
        raise ValueError(f"Shape mismatch: observed {observed.shape}, impedance {impedance.shape}")

    sample_start, sample_end = select_time_window(observed, upper_samples=400, window_len=260)
    windows = select_trace_windows(
        observed=observed,
        impedance=impedance,
        sample_start=sample_start,
        sample_end=sample_end,
        original_trace_ids=np.asarray(infer_trace_ids, dtype=np.int64),
        num_windows=args.num_windows,
        width=700,
        step=140,
        max_overlap_ratio=0.15,
        top_time_ms=top_time_ms,
        dt_ms=dt_ms,
    )

    out_png = OUT_DIR / "fig6_shallow_target_zoom.png"
    out_pdf = OUT_DIR / "fig6_shallow_target_zoom.pdf"
    build_figure(
        observed=observed,
        impedance=impedance,
        windows=windows,
        top_time_ms=top_time_ms,
        dt_ms=dt_ms,
        out_png=out_png,
        out_pdf=out_pdf,
        dpi=args.dpi,
    )

    write_readme(
        readme_path=OUT_DIR / "README_fig6_shallow_target_zoom.md",
        source_paths=[
            run_cfg_path,
            Path(sgy_path),
            result_dir / "impedance_pred_final.npy",
            NEW_DIR / "train_sgy_v11.py",
        ],
        windows=windows,
        out_png=out_png,
        out_pdf=out_pdf,
        script_path=OUT_DIR / "make_fig6_shallow_target_zoom.py",
    )

    print(f"Saved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")
    print(f"Selected shallow time window: {top_time_ms + sample_start * dt_ms:.0f}-{top_time_ms + sample_end * dt_ms:.0f} ms")
    for win in windows:
        print(
            f"{win.window_id}: traces {win.trace_start}-{win.trace_end}, "
            f"time {win.time_start_ms:.0f}-{win.time_end_ms:.0f} ms, "
            f"original traces {win.original_trace_start}-{win.original_trace_end}"
        )


if __name__ == "__main__":
    main()
