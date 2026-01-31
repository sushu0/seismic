import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from train_30Hz_thinlayer_v2 import (
    load_seismic_data,
    load_impedance_data,
    ThinLayerDatasetV2,
    ThinLayerLabeler,
    ThinLayerMetrics,
    evaluate,
    ThinLayerNetV2,
    CFG,
)


def main():
    result_dir = Path(r"D:\SEISMIC_CODING\new\results\01_30Hz_thinlayer_v2")
    ckpt = result_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint missing: {ckpt}")

    # 加载数据
    seismic, dt = load_seismic_data(CFG.SEISMIC_PATH)
    n_traces = seismic.shape[0]
    impedance = load_impedance_data(CFG.IMPEDANCE_PATH, n_traces)
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]

    # 归一化参数
    norm_stats = json.load(open(result_dir / "norm_stats.json", "r"))

    # 数据划分 (与训练一致)
    np.random.seed(CFG.SEED)
    indices = np.random.permutation(n_traces)
    n_train = int(n_traces * CFG.TRAIN_RATIO)
    n_val = int(n_traces * CFG.VAL_RATIO)
    test_idx = indices[n_train + n_val :]

    labeler = ThinLayerLabeler(dt=CFG.DT, dominant_freq=CFG.DOMINANT_FREQ)
    metrics_calc = ThinLayerMetrics(labeler)

    test_ds = ThinLayerDatasetV2(
        seismic, impedance, test_idx, norm_stats, augmentor=None, labeler=labeler
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
    # PyTorch 2.6 默认 weights_only=True 会阻止旧 checkpoint 反序列化，这里显式禁用
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])

    metrics_raw = evaluate(model, test_loader, device, norm_stats, metrics_calc)
    # 转成可JSON序列化的Python基础类型
    metrics = {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in metrics_raw.items()}

    out_path = result_dir / "test_metrics.json"
    json.dump(metrics, open(out_path, "w"), indent=2)
    print(f"saved {out_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
