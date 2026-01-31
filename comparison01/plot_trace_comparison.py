import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse

from marmousi_cnn_bilstm import CNNBiLSTM, load_norm_params, normalize_seismic, denormalize_impedance


def _setup_matplotlib_style():
    """尽量匹配论文的简洁风格，并兼容中文图例。"""
    plt.rcParams["axes.unicode_minus"] = False
    # 优先使用常见中文字体（若系统无此字体会自动回退）
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"]


def _trace_no_to_index(trace_no: int) -> int:
    """论文中的 No.xxx 通常按 1-based 编号，这里统一转为 0-based 下标。"""
    return int(trace_no) - 1


def _build_time_axis_ms(T: int, t_end_ms: float) -> np.ndarray:
    return np.linspace(0.0, float(t_end_ms), T, dtype=np.float32)


def main(
    data_root: str = r"D:\SEISMIC_CODING\comparison01",
    trace_nos: list[int] | None = None,
    t_end_ms: float = 2200.0,
    y_min: float = 0.0,
    y_max: float = 12000.0,
    show: bool | None = None,
):
    """绘制论文指定的 4 条道，并按指定子图位置排版。

    默认排版：左上 299，右上 599，左下 1699，右下 2299。

    - trace_nos: 论文中的编号（1-based），默认 [299, 599, 1699, 2299]
    - t_end_ms: x 轴最大时间（毫秒）。默认 2200ms，与论文图更一致。
    - y_min/y_max: y 轴显示范围，默认 4000~12000，与论文标尺一致。
    - show: None 表示“被导入调用时不弹窗；脚本直接运行时弹窗”。
    """
    _setup_matplotlib_style()

    if trace_nos is None:
        trace_nos = [299, 599, 1699, 2299]

    if show is None:
        # 被 run_train_and_plot.py import 调用时避免阻塞；直接运行本脚本则默认展示。
        show = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seis_path = os.path.join(data_root, "seismic.npy")
    imp_path = os.path.join(data_root, "impedance.npy")
    norm_path = os.path.join(data_root, "norm_params.json")

    if not os.path.exists(seis_path):
        raise FileNotFoundError(f"Missing file: {seis_path}")
    if not os.path.exists(imp_path):
        raise FileNotFoundError(f"Missing file: {imp_path}")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalization file {norm_path} not found. Run training first to generate it."
        )

    seismic = np.load(seis_path).astype(np.float32)    # [T, Nx]
    impedance = np.load(imp_path).astype(np.float32)   # [T, Nx]
    T, Nx = seismic.shape

    norm_params = load_norm_params(norm_path)
    seismic_norm = normalize_seismic(seismic, norm_params)   # [T, Nx]

    # 加载监督 & 半监督模型
    sup_ckpt = os.path.join(data_root, "marmousi_cnn_bilstm_supervised.pth")
    semi_ckpt = os.path.join(data_root, "marmousi_cnn_bilstm_semi.pth")
    if not os.path.exists(sup_ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {sup_ckpt}")
    if not os.path.exists(semi_ckpt):
        raise FileNotFoundError(f"Missing checkpoint: {semi_ckpt}")

    sup_model = CNNBiLSTM().to(device)
    sup_model.load_state_dict(torch.load(sup_ckpt, map_location=device))
    sup_model.eval()

    semi_model = CNNBiLSTM().to(device)
    semi_model.load_state_dict(torch.load(semi_ckpt, map_location=device))
    semi_model.eval()

    t_ms = _build_time_axis_ms(T, t_end_ms=t_end_ms)

    # 子图排版顺序（axes.ravel()）：左上、右上、左下、右下
    letters = ["(a)", "(b)", "(c)", "(d)"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), dpi=300)
    axes = axes.ravel()

    for k, trace_no in enumerate(trace_nos[:4]):
        ax = axes[k]
        trace_idx = _trace_no_to_index(trace_no)
        if trace_idx < 0 or trace_idx >= Nx:
            raise ValueError(f"Trace No.{trace_no} out of range: valid 1..{Nx}")

        seis_trace_norm = seismic_norm[:, trace_idx]  # [T]
        true_imp_phys = impedance[:, trace_idx]       # [T]
        seis_tensor = torch.tensor(seis_trace_norm[None, None, :]).float().to(device)

        with torch.no_grad():
            sup_pred_norm = sup_model(seis_tensor).cpu().numpy().squeeze()
            semi_pred_norm = semi_model(seis_tensor).cpu().numpy().squeeze()

        sup_pred_phys = denormalize_impedance(sup_pred_norm, norm_params)
        semi_pred_phys = denormalize_impedance(semi_pred_norm, norm_params)

        # 颜色与图例顺序尽量匹配论文：红(增广) 绿(增广+半监督) 蓝(标签)
        ax.plot(t_ms, sup_pred_phys, color="red", linewidth=1.2, label="CNN-BiLSTM+增广")
        ax.plot(t_ms, semi_pred_phys, color="green", linewidth=1.2, label="CNN-BiLSTM+增广+半监督")
        ax.plot(t_ms, true_imp_phys, color="blue", linewidth=1.2, label="标签")

        ax.set_title(f"No.{trace_no}", fontsize=11)
        ax.set_xlabel("t/ms", fontsize=10)
        ax.set_ylabel(r"Impedance/(m/s·g/cm$^3$)", fontsize=10)
        ax.legend(loc="upper left", fontsize=9, frameon=True)
        ax.text(0.97, 0.06, letters[k], transform=ax.transAxes, ha="right", va="bottom", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.set_xlim(0.0, float(t_end_ms))
        ax.set_ylim(float(max(0.0, y_min)), float(y_max))

    fig.tight_layout()

    out_path = os.path.join(data_root, "impedance_paper_4traces_299_2299_599_1699.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {out_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot paper traces (No.299/2299/599/1699) with paper-like layout")
    parser.add_argument("--data-root", type=str, default=r"D:\SEISMIC_CODING\comparison01")
    parser.add_argument("--t-end-ms", type=float, default=2200.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=12000.0)
    parser.add_argument("--show", dest="show", action="store_true", default=True)
    parser.add_argument("--no-show", dest="show", action="store_false")
    args = parser.parse_args()
    main(data_root=args.data_root, t_end_ms=args.t_end_ms, y_min=args.y_min, y_max=args.y_max, show=args.show)
