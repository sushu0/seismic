import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from train_30Hz_thinlayer_v2 import (
    ThinLayerNetV2,
    load_seismic_data,
    load_impedance_data,
    ThinLayerDatasetV2,
    ThinLayerLabeler,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result_dir = Path(r'D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v2')
    figures_dir = result_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    seismic_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'
    impedance_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt'
    seismic, dt = load_seismic_data(seismic_path)
    n_traces = seismic.shape[0]
    impedance = load_impedance_data(impedance_path, n_traces)
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]

    with open(result_dir / 'norm_stats.json', 'r') as f:
        norm_stats = json.load(f)

    model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
    state = torch.load(result_dir / 'checkpoints' / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(state['model'])
    model.eval()

    np.random.seed(42)
    indices = np.random.permutation(seismic.shape[0])
    n_train = int(seismic.shape[0] * 0.6)
    n_val = int(seismic.shape[0] * 0.2)
    test_idx = indices[n_train + n_val:]

    labeler = ThinLayerLabeler(dt=0.001, dominant_freq=30.0)
    test_ds = ThinLayerDatasetV2(seismic, impedance, test_idx, norm_stats, augmentor=None, labeler=labeler)

    all_pred = []
    with torch.no_grad():
        for i in range(len(test_ds)):
            x, _, _, _ = test_ds[i]
            pred = model(x.unsqueeze(0).to(device)).cpu().numpy()[0, 0]
            all_pred.append(pred)

    pred = np.array(all_pred) * norm_stats['imp_std'] + norm_stats['imp_mean']
    true = impedance[test_idx] * norm_stats['imp_std'] + norm_stats['imp_mean']
    true = np.resize(np.asarray(true), pred.shape)

    mid_trace = len(pred) // 2
    samples = np.arange(pred.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=250)

    axes[0].plot(samples, pred[mid_trace], 'b-', linewidth=2.5, label='预测', alpha=0.8)
    axes[0].plot(samples, true[mid_trace], 'r--', linewidth=2.5, label='真实', alpha=0.8)
    axes[0].fill_between(samples, pred[mid_trace], true[mid_trace], alpha=0.2, color='gray', label='误差区间')
    axes[0].set_xlabel('采样点')
    axes[0].set_ylabel('阻抗')
    axes[0].set_title(f'单道对比 (地震道#{mid_trace})')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    error_trace = pred[mid_trace] - true[mid_trace]
    axes[1].fill_between(samples, error_trace, alpha=0.6, color='steelblue')
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=1)
    axes[1].set_xlabel('采样点')
    axes[1].set_ylabel('误差')
    axes[1].set_title('误差时间序列')
    mae_trace = np.mean(np.abs(error_trace))
    rmse_trace = np.sqrt(np.mean(error_trace ** 2))
    axes[1].text(0.02, 0.95, f'MAE={mae_trace:.2f}\nRMSE={rmse_trace:.2f}', transform=axes[1].transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.suptitle('30Hz 数据 - 单道详细对比', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    out_path = figures_dir / 'trace_comparison.png'
    plt.savefig(out_path, dpi=250, bbox_inches='tight')
    print(f"saved {out_path}")


if __name__ == '__main__':
    main()
