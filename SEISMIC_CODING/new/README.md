# MS-PhysFormer: Multi-Scale Physics- & Frequency-Constrained Transformer U-Net (Seismic Impedance Inversion)

This is a **self-contained, runnable** PyTorch project that you can merge into your existing repo.
It provides:
- 2 baselines: **UNet1D**, **TCN1D**, and an optional **CNN-BiLSTM** baseline
- A new model: **MSPhysFormer** (U-Net + Transformer bottleneck + deep supervision)
- Physics-consistent forward modeling constraint (impedance -> reflectivity -> seismic via wavelet convolution)
- Frequency-domain loss (STFT magnitude) + Mean-Teacher consistency for unlabeled seismic
- Reproducibility (fixed seeds, config-driven training, logs & result artifacts)

## 1) Quick start (toy data)
```bash
pip install -r requirements.txt
python scripts/generate_toy_data.py --out_dir data/toy --n_train 64 --n_val 16 --n_test 16 --n_unlabeled 128
python train.py --config configs/exp_baseline_unet.yaml
python train.py --config configs/exp_newmodel.yaml
python train.py --config configs/abl_no_physics.yaml
python train.py --config configs/abl_no_freq.yaml
```

Outputs are saved to `results/<exp_name>/`:
- `metrics.csv` (per-epoch + best)
- `test_metrics.json`
- `pred_vs_true_traces.png`
- `seismic_recon.png`
- checkpoints: `checkpoints/best.pt`, `checkpoints/last.pt`

## 2) Plug in your real Marmousi2 / field dataset
Prepare `.npy` files under `data_root` (see configs):
- `train_labeled_seis.npy`, `train_labeled_imp.npy`
- `val_seis.npy`, `val_imp.npy`
- `test_seis.npy`, `test_imp.npy`
- optional: `train_unlabeled_seis.npy`

Each array should be shape `[N, T]` (single-channel traces). If you have sections, just flatten by trace.

Then run:
```bash
python train.py --config configs/exp_newmodel.yaml --override data.data_root=/path/to/data
```

## 3) Notes
- Wavelet: default is **30 Hz Ricker**, dt configurable.
- Forward model uses: r(t)=(I(t)-I(t-1))/(I(t)+I(t-1)), s(t)=r(t)*w(t) (same convolutional model used widely in synthetic experiments).
