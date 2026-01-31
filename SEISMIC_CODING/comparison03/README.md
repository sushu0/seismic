# ss_gan_impedance (paper-aligned core)

This repo implements the core method in:
《基于生成对抗网络的半监督地震波阻抗反演》

Included **ready-to-run datasets** in `data/`:
- `synth_toy.npz`
- `synth_marmousi2.npz` (smaller "marmousi2-like" for zip size)
- `synth_valve.npz`

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Train (toy)
```bash
python train.py --config configs/toy_fast.yaml
```

## Paper reproduction (commands)

**Strict paper reproduction policy**: this repo can run end-to-end on any compatible dataset, but the *strict reproduction mode* refuses to silently adapt dataset sizes/splits or introduce extra numerical tricks. If your data does not match the paper's Marmousi2 trace count/split rules, strict mode will error.

### 1) Generate paper-aligned synthetic datasets

Marmousi2 (paper split: 13601 total / 101 labeled / 1350 val / rest unlabeled; test uses all):
```bash
python scripts/make_synthetic.py --out data/marmousi2_full.npz --preset marmousi2 --seed 1234
```

Valve-like (paper split: 1300 total / 750 labeled / 275 val / 275 unlabeled; T=160 so k_large=80):
```bash
python scripts/make_synthetic.py --out data/valve_full.npz --preset valve --seed 1234
```

Noisy Marmousi2 (paper SNR experiments: 25/15/5 dB):
```bash
python scripts/make_synthetic.py --out data/marmousi2_snr25.npz --preset marmousi2 --seed 1234 --snr_db 25
python scripts/make_synthetic.py --out data/marmousi2_snr15.npz --preset marmousi2 --seed 1234 --snr_db 15
python scripts/make_synthetic.py --out data/marmousi2_snr05.npz --preset marmousi2 --seed 1234 --snr_db 5
```

### 2) Train

Paper-aligned hyperparameters are in `configs/marmousi2_paper.yaml` (epochs=1000, bs=10, lr=1e-3, n_critic=5, alpha=1100, beta=550, lambda_gp=10).

```bash
python train.py --config configs/marmousi2_paper.yaml
```

### Using a root-level Marmousi2 file (data.npy)

If you placed Marmousi2 data at project root as `data.npy` with keys `seismic` and `acoustic_impedance`, convert it to the repo's `.npz` format:

```bash
python scripts/convert_root_marmousi2.py --in data.npy --out data/marmousi2_paper.npz
```

Note: strict mode enforces the paper's Marmousi2 total trace count (13601) and split (101 labeled / 1350 val / rest unlabeled). If your file is smaller (e.g. 2721 traces), this is not the same dataset as the paper and strict mode will refuse to proceed.

## Inference
```bash
python infer.py --dataset data/synth_toy.npz --ckpt runs/toy_fast/checkpoints/best.pt --split test --out runs/toy_fast/pred_test.npz
```

## Repro alignment checklist
- Generator: 1D UNet, 4 down/4 up, conv kernels 299 then 3, channels up to 256.
- Discriminator: ResNet-like critic, 4 residual blocks, kernels 299 & 3, channels 8→128, FC256→FC1.
- Training: WGAN-GP, train D 5 steps then G 1 step per batch.
- Losses: L_G = L_adv + alpha * L_i + beta * L_s, with alpha=1100, beta=550; L_D uses gradient penalty with lambda=10.
- Forward constraint: Ricker wavelet (30Hz, dt=0.001s, dur=0.128s; Valve uses dur=0.064s), convolutional forward model.
- Metrics: PCC, r^2, MSE.

## Plot sections
```bash
python scripts/plot_section.py --pred runs/toy_fast/pred_test.npz --outdir runs/toy_fast/figures --n_traces 200
```

## Generate your own datasets
```bash
python scripts/make_synthetic.py --out data/new_marm.npz --preset marmousi2 --seed 1234
```
