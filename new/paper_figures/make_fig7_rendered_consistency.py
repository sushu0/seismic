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
DEFAULT_RESULT_DIR = NEW_DIR / "sgy_inversion_v12_rendered_99eval"
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


def map_with_scale(section: np.ndarray, scale: float) -> np.ndarray:
    return np.tanh(section.astype(np.float32) / (float(scale) + 1e-6)).astype(np.float32)


def observed_scaled_pair(observed: np.ndarray, rendered: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    scale = float(np.percentile(np.abs(observed), 99.0)) + 1e-6
    return map_with_scale(observed, scale), map_with_scale(rendered, scale), scale


def select_time_window(
    observed: np.ndarray,
    rendered: np.ndarray,
    upper_samples: int = 400,
    window_len: int = 260,
) -> tuple[int, int]:
    obs_upper = observed[:, :upper_samples]
    rend_upper = rendered[:, :upper_samples]
    mean_abs = np.mean(np.abs(obs_upper), axis=0)
    mean_grad = np.r_[0.0, np.mean(np.abs(np.diff(obs_upper, axis=1)), axis=0)]

    obs_mapped, rend_mapped, _ = observed_scaled_pair(obs_upper, rend_upper)
    corr_t = []
    for i in range(upper_samples):
        a = obs_mapped[:, i]
        b = rend_mapped[:, i]
        aa = a - a.mean()
        bb = b - b.mean()
        corr_t.append(float((aa * bb).sum() / (np.sqrt((aa * aa).sum() * (bb * bb).sum()) + 1e-6)))
    corr_t = np.asarray(corr_t, dtype=np.float64)

    kernel = np.ones(21, dtype=np.float64) / 21.0
    mean_abs_s = np.convolve(mean_abs, kernel, mode="same")
    mean_grad_s = np.convolve(mean_grad, kernel, mode="same")
    corr_t_s = np.convolve(corr_t, kernel, mode="same")
    score = (
        0.35 * (mean_abs_s / (mean_abs_s.max() + 1e-6))
        + 0.35 * (mean_grad_s / (mean_grad_s.max() + 1e-6))
        + 0.30 * ((corr_t_s + 1.0) / 2.0)
    )

    best_score = -np.inf
    best = (0, window_len)
    for start in range(0, upper_samples - window_len + 1, 5):
        end = start + window_len
        current = float(score[start:end].mean())
        if current > best_score:
            best_score = current
            best = (start, end)
    return best


def select_trace_windows(
    observed: np.ndarray,
    rendered: np.ndarray,
    sample_start: int,
    sample_end: int,
    original_trace_ids: np.ndarray,
    num_windows: int = 2,
    width: int = 800,
    step: int = 160,
    max_overlap_ratio: float = 0.18,
    top_time_ms: float = 2500.0,
    dt_ms: float = 2.0,
) -> List[WindowInfo]:
    obs_top = observed[:, sample_start:sample_end]
    rend_top = rendered[:, sample_start:sample_end]
    obs_map, rend_map, _ = observed_scaled_pair(obs_top, rend_top)

    rows = []
    for start in range(0, observed.shape[0] - width + 1, step):
        end = start + width
        ow = obs_top[start:end]
        rw = rend_top[start:end]
        om = obs_map[start:end]
        rm = rend_map[start:end]

        event_energy = float(np.mean(np.abs(np.diff(ow, axis=1))))
        continuity = event_energy / (float(np.mean(np.abs(np.diff(ow, axis=0)))) + 1e-6)

        a = om.ravel()
        b = rm.ravel()
        aa = a - a.mean()
        bb = b - b.mean()
        corr = float((aa * bb).sum() / (np.sqrt((aa * aa).sum() * (bb * bb).sum()) + 1e-6))
        agreement = float(1.0 - np.mean(np.abs(om - rm)))
        rows.append((0.0, start, end, event_energy, continuity, corr, agreement))

    feats = np.asarray([r[3:] for r in rows], dtype=np.float64)
    for j in range(feats.shape[1]):
        mn = feats[:, j].min()
        mx = feats[:, j].max()
        if mx - mn > 1e-12:
            feats[:, j] = (feats[:, j] - mn) / (mx - mn)
        else:
            feats[:, j] = 0.0

    scores = 0.28 * feats[:, 0] + 0.22 * feats[:, 1] + 0.25 * feats[:, 2] + 0.25 * feats[:, 3]
    scored = []
    for idx, row in enumerate(rows):
        scored.append((float(scores[idx]), row[1], row[2], row[3], row[4], row[5], row[6]))
    scored.sort(reverse=True)

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
            region = "左侧目标段反射事件较清晰，适合展示观测地震与 rendered view 的局部结构对应关系。"
        elif center < 2 * observed.shape[0] / 3:
            region = "中部目标段事件密集且层间过渡丰富，最能体现 rendered consistency 对解释直观性的增强。"
        else:
            region = "右侧目标段横向连续性较强，适合展示 rendered view 对可追踪结构的辅助表达。"

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


def extent_from_shape(n_traces: int, n_samples: int, top_time_ms: float, dt_ms: float) -> list[float]:
    return [0.0, float(n_traces - 1), float(top_time_ms + n_samples * dt_ms), float(top_time_ms)]


def show_section(
    ax: plt.Axes,
    mapped_section: np.ndarray,
    top_time_ms: float,
    dt_ms: float,
    title: str,
    add_xlabel: bool,
    add_ylabel: bool,
) -> None:
    ax.imshow(
        mapped_section.T,
        aspect="auto",
        cmap="RdBu_r",
        origin="upper",
        interpolation="nearest",
        extent=extent_from_shape(mapped_section.shape[0], mapped_section.shape[1], top_time_ms, dt_ms),
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
    rendered: np.ndarray,
    windows: Sequence[WindowInfo],
    top_time_ms: float,
    dt_ms: float,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
) -> None:
    obs_full_map, rend_full_map, _ = observed_scaled_pair(observed, rendered)
    colors = ["tab:orange", "tab:purple"]

    fig = plt.figure(figsize=(15.8, 8.0), facecolor="white")
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], wspace=0.18, hspace=0.22)

    ax_obs_full = fig.add_subplot(gs[0, 0:2])
    ax_rend_full = fig.add_subplot(gs[0, 2:4])
    show_section(ax_obs_full, obs_full_map, top_time_ms, dt_ms, "(a) Observed seismic", add_xlabel=True, add_ylabel=True)
    show_section(ax_rend_full, rend_full_map, top_time_ms, dt_ms, "(b) Seismic-like rendered view", add_xlabel=True, add_ylabel=True)

    for color, win in zip(colors, windows):
        for ax in (ax_obs_full, ax_rend_full):
            rect = patches.Rectangle(
                (win.trace_start, win.time_start_ms),
                win.trace_end - win.trace_start,
                win.time_end_ms - win.time_start_ms,
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
        ax_obs_full.text(
            win.trace_start + 12,
            win.time_start_ms + 20,
            win.window_id,
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )
        ax_rend_full.text(
            win.trace_start + 12,
            win.time_start_ms + 20,
            win.window_id,
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.2),
        )

    subplot_letters = ["(c)", "(d)", "(e)", "(f)"]
    for idx, (color, win) in enumerate(zip(colors, windows)):
        ax_obs = fig.add_subplot(gs[1, idx * 2])
        ax_rend = fig.add_subplot(gs[1, idx * 2 + 1])

        obs_win = observed[win.trace_start : win.trace_end, win.sample_start : win.sample_end]
        rend_win = rendered[win.trace_start : win.trace_end, win.sample_start : win.sample_end]
        obs_win_map, rend_win_map, _ = observed_scaled_pair(obs_win, rend_win)

        show_section(
            ax_obs,
            obs_win_map,
            win.time_start_ms,
            dt_ms,
            f"{subplot_letters[idx * 2]} {win.window_id}: Observed",
            add_xlabel=True,
            add_ylabel=(idx == 0),
        )
        show_section(
            ax_rend,
            rend_win_map,
            win.time_start_ms,
            dt_ms,
            f"{subplot_letters[idx * 2 + 1]} {win.window_id}: Rendered",
            add_xlabel=True,
            add_ylabel=False,
        )

        for ax in (ax_obs, ax_rend):
            for spine in ax.spines.values():
                spine.set_color(color)
                spine.set_linewidth(1.2)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.09)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_readme(
    readme_path: Path,
    source_paths: Sequence[Path],
    windows: Sequence[WindowInfo],
    result_dir: Path,
    out_png: Path,
    out_pdf: Path,
    script_path: Path,
) -> None:
    lines = [
        "# Figure 7 地震风格一致性展示图说明",
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
        "## 对应结果说明",
        "",
        f"- 本图使用 `{result_dir.name}` 中的 `rendered_seismic.npy` 作为 seismic-like rendered view",
        "- 该 rendered view 对应的是基于主反演结果构建的辅助展示层",
        "- 它用于增强观测地震与反演结果之间的展示一致性与解释直观性，不作为阻抗本体真实性的直接证明",
        "",
        "## 局部窗口选取原则",
        "",
        "- 先在研究时间窗上部 400 个采样点中自动搜索浅部目标层段",
        "- 再按事件丰富度、横向可追踪性与 observed-vs-rendered 局部一致性自动选取 2 个代表性窗口",
        "- 为保证主文版式紧凑，本图采用 2 个局部窗口",
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
        "## 图像用途说明",
        "",
        "- 该图属于辅助解释展示图，不属于主反演真实性证明图",
        "- 该图展示的是观测地震与 seismic-like rendered view 的显示一致性增强",
        "- 该图不表示阻抗本体图像与原始地震图像直接等价",
        "",
        "## 建议图题",
        "",
        "图 7 地震风格一致性展示图",
        "",
        "## 建议图注",
        "",
        "图 7 给出了观测地震剖面与基于主反演结果构建的 seismic-like rendered view 的对应展示。该结果主要用于增强反演结果的可解释一致性与可视化直观性。需要指出的是，该图反映的是展示层面的一致性增强，而非阻抗本体图像真实性的直接证明。",
    ]
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 7 rendered consistency figure for the paper.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--result-dir", type=str, default=str(DEFAULT_RESULT_DIR))
    parser.add_argument("--num-windows", type=int, default=2)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    run_cfg_path = result_dir / "run_config.json"
    with open(run_cfg_path, "r", encoding="utf-8") as f:
        run_cfg = json.load(f)

    sgy_path = str((REPO_ROOT / run_cfg["sgy"]).resolve()) if not Path(run_cfg["sgy"]).is_absolute() else run_cfg["sgy"]
    infer_traces = int(run_cfg["infer_traces"])
    top_time_ms = parse_top_time_ms(sgy_path)

    tracecount, _, _, dt_seconds = read_tracecount_and_samples(sgy_path)
    dt_ms = float(dt_seconds) * 1000.0
    infer_trace_ids = uniform_indices(tracecount, infer_traces)
    observed = read_sgy_traces(sgy_path, infer_trace_ids).astype(np.float32)
    rendered = np.load(result_dir / "rendered_seismic.npy").astype(np.float32)

    if observed.shape != rendered.shape:
        raise ValueError(f"Shape mismatch: observed {observed.shape}, rendered {rendered.shape}")

    sample_start, sample_end = select_time_window(observed, rendered, upper_samples=400, window_len=260)
    windows = select_trace_windows(
        observed=observed,
        rendered=rendered,
        sample_start=sample_start,
        sample_end=sample_end,
        original_trace_ids=np.asarray(infer_trace_ids, dtype=np.int64),
        num_windows=args.num_windows,
        width=800,
        step=160,
        max_overlap_ratio=0.18,
        top_time_ms=top_time_ms,
        dt_ms=dt_ms,
    )

    out_png = OUT_DIR / "fig7_rendered_consistency.png"
    out_pdf = OUT_DIR / "fig7_rendered_consistency.pdf"
    build_figure(
        observed=observed,
        rendered=rendered,
        windows=windows,
        top_time_ms=top_time_ms,
        dt_ms=dt_ms,
        out_png=out_png,
        out_pdf=out_pdf,
        dpi=args.dpi,
    )

    write_readme(
        readme_path=OUT_DIR / "README_fig7_rendered_consistency.md",
        source_paths=[
            run_cfg_path,
            result_dir / "rendered_seismic.npy",
            Path(sgy_path),
            NEW_DIR / "train_sgy_v11.py",
        ],
        windows=windows,
        result_dir=result_dir,
        out_png=out_png,
        out_pdf=out_pdf,
        script_path=OUT_DIR / "make_fig7_rendered_consistency.py",
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
