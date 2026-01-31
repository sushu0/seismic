# plot_impedance_marmousi2_raw.py
# 直接用 Vp.segy 和 Density.segy 计算 Z = Vp * rho，并按 gist_rainbow 画 Marmousi2 真值剖面
# 不对真值做任何平滑或单位变换处理

import os
import numpy as np
import segyio
import matplotlib.pyplot as plt

ROOT = r"D:\SEISMIC_CODING\comparison01"
VP_PATH = os.path.join(ROOT, "Vp.segy")
RHO_PATH = os.path.join(ROOT, "Density.segy")
DATA_NPY = os.path.join(ROOT, "data.npy")
IMP_NPY = os.path.join(ROOT, "impedance.npy")
OUT_FIG = os.path.join(ROOT, "true_impedance_marmousi2_raw.png")

def load_segy(path):
    print(f"[INFO] Loading {path}")
    with segyio.open(path, "r", ignore_geometry=True, strict=False) as f:
        n_traces = len(f.trace)
        n_samples = f.trace[0].size
        # 关键修复：使用 float64 避免溢出
        data = np.stack([np.array(f.trace[i], dtype="float64")
                         for i in range(n_traces)], axis=0)
        dt = f.bin[segyio.BinField.Interval] * 1e-6  # 秒
    print(f"       shape = {data.shape}, dt = {dt:.6f} s")
    print(f"       data range: [{data.min():.6e}, {data.max():.6e}]")
    return data, dt


def density_looks_invalid(rho: np.ndarray) -> bool:
    """粗判 Density.segy 是否被错误编码/损坏。

    典型密度（g/cc）约 1.5~3.5；如果中位数接近 0 且尾部分布到极大值，通常说明格式/尺度不对。
    """
    finite = np.isfinite(rho)
    if finite.mean() < 0.90:
        return True
    vals = rho[finite]
    if vals.size == 0:
        return True
    p1, p50, p99 = np.percentile(vals, [1, 50, 99])
    if np.abs(p50) < 0.1:
        return True
    if np.max(np.abs([p1, p99])) > 10.0:
        return True
    return False


def probe_segy_percentiles(path: str, percentiles=(1, 50, 99), stride: int = 200, max_traces: int = 20):
    """从 SEG-Y 抽样若干道，快速估计分布（避免整文件读入）。"""
    with segyio.open(path, "r", ignore_geometry=True, strict=False) as f:
        n_traces = len(f.trace)
        picks = list(range(0, n_traces, max(1, stride)))[:max_traces]
        if not picks:
            picks = [0]
        samples = np.concatenate([np.asarray(f.trace[i], dtype=np.float64).ravel() for i in picks], axis=0)
        finite = np.isfinite(samples)
        if not finite.any():
            return {
                "finite_ratio": 0.0,
                "p": {p: np.nan for p in percentiles},
                "min": np.nan,
                "max": np.nan,
                "picked_traces": len(picks),
            }
        vals = samples[finite]
        ps = np.percentile(vals, list(percentiles))
        return {
            "finite_ratio": float(finite.mean()),
            "p": {int(k): float(v) for k, v in zip(percentiles, ps)},
            "min": float(vals.min()),
            "max": float(vals.max()),
            "picked_traces": len(picks),
        }


def load_impedance_from_npy(dt_fallback: float) -> tuple[np.ndarray, float, str]:
    """加载一个更可信的真值阻抗来源。

    优先级：impedance.npy（如果已由 split 脚本生成）> data.npy(acoustic_impedance)
    返回：impedance (n_traces, n_samples), dt, source_desc
    """
    if os.path.exists(IMP_NPY):
        imp = np.load(IMP_NPY).astype(np.float64)  # [T, Nx]
        if imp.ndim != 2:
            raise ValueError(f"impedance.npy 期望 2D (T, Nx)，实际 {imp.shape}")
        return imp.T, dt_fallback, "impedance.npy"

    if os.path.exists(DATA_NPY):
        dic = np.load(DATA_NPY, allow_pickle=True).item()
        if "acoustic_impedance" not in dic:
            raise KeyError("data.npy 中未找到 key: acoustic_impedance")
        imp_raw = dic["acoustic_impedance"]
        if imp_raw.ndim == 3:
            imp_raw = np.squeeze(imp_raw)
        if imp_raw.ndim != 2:
            raise ValueError(f"data.npy acoustic_impedance 期望 2D (N_traces, T)，实际 {imp_raw.shape}")
        return imp_raw.astype(np.float64), dt_fallback, "data.npy:acoustic_impedance"

    raise FileNotFoundError("找不到 impedance.npy 或 data.npy，无法回退加载真值阻抗")

def main():
    # 1. 读入 Vp 和 Density（不做任何修改）
    vp, dt_vp = load_segy(VP_PATH)

    # 先抽样探测 Density 是否正常：如果明显异常，就不必整文件读入
    rho_probe = probe_segy_percentiles(RHO_PATH)
    print(
        "[INFO] Density.segy probe: "
        f"picked={rho_probe['picked_traces']} finite%={rho_probe['finite_ratio']*100:.3f}% "
        f"min/max={rho_probe['min']:.3e}/{rho_probe['max']:.3e} "
        f"p1/p50/p99={rho_probe['p'][1]:.3e}/{rho_probe['p'][50]:.3e}/{rho_probe['p'][99]:.3e}"
    )

    rho_invalid = (
        rho_probe["finite_ratio"] < 0.90
        or (abs(rho_probe["p"][50]) < 0.1)
        or (max(abs(rho_probe["p"][1]), abs(rho_probe["p"][99])) > 10.0)
    )

    if rho_invalid:
        print("[WARN] Density.segy 抽样即判为异常；将回退使用 NPY 真值阻抗。")
        rho = None
        dt_rho = dt_vp
    else:
        rho, dt_rho = load_segy(RHO_PATH)

    if dt_vp != dt_rho:
        print(f"[WARN] Vp dt={dt_vp}, rho dt={dt_rho}, 使用 Vp 的 dt")

    if (rho is not None) and (vp.shape != rho.shape):
        raise ValueError(f"Vp shape {vp.shape} != Density shape {rho.shape}")

    if rho is not None:
        rho_invalid = density_looks_invalid(rho)
        if rho_invalid:
            vals = rho[np.isfinite(rho)]
            if vals.size:
                p1, p50, p99 = np.percentile(vals, [1, 50, 99])
                print(
                    "[WARN] Density.segy 整体分布异常，疑似编码/尺度错误；将回退使用 NPY 真值阻抗。\n"
                    f"       rho finite%={np.isfinite(rho).mean()*100:.3f}% p1/p50/p99={p1:.3e}/{p50:.3e}/{p99:.3e}"
                )
            else:
                print("[WARN] Density.segy 全为非有限值；将回退使用 NPY 真值阻抗。")

    # 2. 计算/加载波阻抗
    if rho_invalid:
        impedance, dt_vp, imp_source = load_impedance_from_npy(dt_vp)
    else:
        impedance = vp * rho   # (n_traces, n_samples)
        imp_source = "Z = Vp × Density"
    
    print(f"\n[INFO] Impedance 就绪 (source: {imp_source})")
    print(f"       原始 min/max: [{np.nanmin(impedance):.6e}, {np.nanmax(impedance):.6e}]")

    # === 关键修复：清理异常值 ===
    # 先检查并移除 NaN/Inf
    finite_mask = np.isfinite(impedance)
    if not np.any(finite_mask):
        raise RuntimeError("Impedance 全是 NaN/inf，请检查原始 SEG-Y 数据。")

    finite_ratio = finite_mask.sum() / impedance.size * 100
    print(f"       有限值占比: {finite_ratio:.2f}%")
    
    # 计算有限值的统计信息
    vals_finite = impedance[finite_mask]
    print(f"       有限值 min/max: [{vals_finite.min():.6e}, {vals_finite.max():.6e}]")
    print(f"       有限值 mean: {vals_finite.mean():.6e}, std: {vals_finite.std():.6e}")
    
    # === 使用更稳健的百分位数方法清理离群值 ===
    # 直接用 5%-95% 分位数定义合理范围（比 IQR 更直接）
    p5, p95 = np.percentile(vals_finite, [5, 95])
    p_range = p95 - p5
    lower_bound = p5 - 2 * p_range  # 向下扩展2倍范围
    upper_bound = p95 + 2 * p_range # 向上扩展2倍范围
    
    print(f"\n[INFO] 百分位数清理边界:")
    print(f"       P5={p5:.6e}, P95={p95:.6e}")
    print(f"       清理边界: [{lower_bound:.6e}, {upper_bound:.6e}]")
    
    # 创建绘图副本，清理离群值
    imp_for_plot = impedance.copy()
    imp_for_plot[~finite_mask] = np.nan  # 非有限值
    outlier_mask = (impedance < lower_bound) | (impedance > upper_bound)
    imp_for_plot[outlier_mask] = np.nan  # 离群值
    
    # 检查清理后的数据
    clean_vals = imp_for_plot[np.isfinite(imp_for_plot)]
    if clean_vals.size == 0:
        raise RuntimeError("清理后无有效数据，请检查数据源")
    
    clean_ratio = clean_vals.size / impedance.size * 100
    print(f"       清理后保留: {clean_ratio:.2f}%")
    print(f"       清理后范围: [{clean_vals.min():.6e}, {clean_vals.max():.6e}]")

    # 使用更保守的分位数设置色标（保留更多细节）
    vmin = np.nanpercentile(clean_vals, 2)
    vmax = np.nanpercentile(clean_vals, 98)
    print(f"\n[INFO] 最终色标范围 (2%-98%): [{vmin:.6e}, {vmax:.6e}]")

    n_traces, n_samples = imp_for_plot.shape
    time_ms = np.arange(n_samples) * dt_vp * 1000.0  # 纵轴：毫秒

    # 4. 按你 FCN 那种风格画图：gist_rainbow + 道号 / 时间(ms)
    fig = plt.figure(figsize=(20, 6), dpi=150)
    ax = plt.gca()

    im = ax.imshow(
        imp_for_plot.T,
        aspect="auto",
        cmap="gist_rainbow",
        origin="upper",
        extent=[0, n_traces, time_ms[-1], time_ms[0]],
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear"  # 双线性插值
    )

    # 添加网格线
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, 
            color='white', alpha=0.3, axis='both')
    ax.set_xticks(np.linspace(0, n_traces, 10))
    ax.set_yticks(np.linspace(time_ms[0], time_ms[-1], 8))

    ax.set_xlabel("Trace number", fontsize=14, fontweight='bold')
    ax.set_ylabel("Time (ms)", fontsize=14, fontweight='bold')
    ax.set_title(f"True Impedance Section (Marmousi2, {imp_source})", 
                 fontsize=16, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, extend='both', pad=0.02, shrink=0.95)
    cbar.set_label(f"Impedance\n[{vmin:.2e}, {vmax:.2e}]", 
                   fontsize=12, fontweight='bold', rotation=270, labelpad=25)
    cbar.ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    print(f"[SAVE] Figure saved to: {OUT_FIG}")
    plt.show()

if __name__ == "__main__":
    main()