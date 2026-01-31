# plot_impedance_section.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
import argparse

# 设置中文字体和西文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

from marmousi_cnn_bilstm import (
    CNNBiLSTM,
    load_norm_params,
    normalize_seismic,
    denormalize_impedance,
)


def load_data_and_norm(data_root):
    """读取 seismic / impedance，并做和训练阶段一致的全局归一化"""
    seis_path = os.path.join(data_root, "seismic.npy")
    imp_path = os.path.join(data_root, "impedance.npy")

    seismic = np.load(seis_path).astype(np.float32)    # [T, Nx]
    impedance = np.load(imp_path).astype(np.float32)   # [T, Nx]

    norm_path = os.path.join(data_root, "norm_params.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalization file {norm_path} not found. Run training first to generate it."
        )
    norm_params = load_norm_params(norm_path)
    seismic_norm = normalize_seismic(seismic, norm_params)

    return seismic, impedance, seismic_norm, norm_params


def run_model_on_section(model, seismic_norm, device, norm_params):
    """
    利用训练好的 CNN-BiLSTM 在整幅剖面上做预测：
    输入：seismic_norm [T, Nx]
    输出：pred_phys [T, Nx]（物理量空间中的阻抗）
    """
    T, Nx = seismic_norm.shape

    # [T, Nx] -> [Nx, T] -> [Nx, 1, T] 以适配模型输入 [B, 1, T]
    seis_traces = seismic_norm.T  # [Nx, T]
    seis_tensor = torch.from_numpy(seis_traces[:, None, :]).float().to(device)  # [Nx, 1, T]

    model.eval()
    with torch.no_grad():
        pred_norm = model(seis_tensor)        # [Nx, T]
        pred_norm = pred_norm.cpu().numpy()   # numpy [Nx, T]

    # [Nx, T] -> [T, Nx]，再反标准化回物理阻抗
    pred_norm = pred_norm.T
    pred_phys = denormalize_impedance(pred_norm, norm_params)   # [T, Nx]
    return pred_phys


def main():
    parser = argparse.ArgumentParser(description="Plot impedance sections (true / supervised / semi-supervised)")
    parser.add_argument("--data-root", type=str, default=r"D:\SEISMIC_CODING\comparison01")
    parser.add_argument("--t-end-ms", type=float, default=2200.0, help="Time axis end in ms (used for plotting only)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--no-show", dest="show", action="store_false", default=False)
    parser.add_argument("--show", dest="show", action="store_true")
    args = parser.parse_args()

    # 路径与你训练脚本保持一致
    data_root = args.data_root
    sup_ckpt = os.path.join(data_root, "marmousi_cnn_bilstm_supervised.pth")
    semi_ckpt = os.path.join(data_root, "marmousi_cnn_bilstm_semi.pth")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 1. 读数据 + 归一化
    seismic, impedance, seismic_norm, norm_params = load_data_and_norm(data_root)
    T, Nx = seismic.shape
    print(f"Data shape: T={T}, Nx={Nx}")

    # 2. 加载两个模型（监督 / 半监督）
    sup_model = CNNBiLSTM().to(device)
    semi_model = CNNBiLSTM().to(device)

    sup_model.load_state_dict(torch.load(sup_ckpt, map_location=device))
    semi_model.load_state_dict(torch.load(semi_ckpt, map_location=device))
    print("Loaded checkpoints:")
    print("  Supervised :", sup_ckpt)
    print("  Semi-super :", semi_ckpt)

    # 3. 整幅剖面预测
    print("Running supervised model on full section...")
    sup_pred = run_model_on_section(sup_model, seismic_norm, device, norm_params)

    print("Running semi-supervised model on full section...")
    semi_pred = run_model_on_section(semi_model, seismic_norm, device, norm_params)

    # 4. 画三幅剖面：真值 / 监督 / 半监督，统一色标范围
    all_imp = np.concatenate(
        [impedance.ravel(), sup_pred.ravel(), semi_pred.ravel()]
    )
    vmin = np.percentile(all_imp, 4)
    vmax = np.percentile(all_imp, 97)

    # 时间轴：为了与论文/四道图一致，这里按 0~t_end_ms 线性铺满 T 个采样点
    t_end_ms = float(args.t_end_ms)
    time_axis_ms = np.linspace(0.0, t_end_ms, T, dtype=np.float32)

    # 创建图像：x=Trace number (1..Nx)，y=Time(ms)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, dpi=int(args.dpi), constrained_layout=True)
    cmap = 'gist_rainbow'

    titles = ['True Impedance', 'Inverted (Supervised)', 'Inverted (Semi-supervised)']
    predictions = [impedance, sup_pred, semi_pred]
    images = []

    for i, (ax, title, pred) in enumerate(zip(axes, titles, predictions)):
        # pred 的形状为 [T, Nx]，行=时间，列=道号。
        im = ax.imshow(
            pred,
            aspect='auto',
            cmap=cmap,
            extent=[1, Nx, t_end_ms, 0.0],
            origin='upper',
            vmin=vmin,
            vmax=vmax,
            interpolation='nearest',
        )
        images.append(im)
        
        # 添加网格线
        ax.grid(True, which='major', linestyle='--', linewidth=0.5, 
                color='white', alpha=0.25, axis='both')
        
        # 设置刻度
        ax.set_xticks(np.linspace(1, Nx, 8))
        ax.set_yticks(np.linspace(0.0, t_end_ms, 6))
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Trace number', fontsize=11, fontweight='bold')
        if i == 0:
            ax.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')

    # 添加共享色条
    cbar = fig.colorbar(
        images[-1], ax=axes, orientation='vertical', pad=0.02, shrink=0.95, extend='both'
    )
    cbar.set_label("Impedance", fontsize=12, fontweight='bold', rotation=270, labelpad=18)
    
    # 添加总标题
    fig.suptitle('Marmousi2 Impedance Inversion Comparison (CNN-BiLSTM)', 
                 fontsize=15, fontweight='bold', y=0.98)

    out_path = os.path.join(data_root, "impedance_sections.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', format='png')
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("Section figure saved to:", out_path)


if __name__ == "__main__":
    main()

