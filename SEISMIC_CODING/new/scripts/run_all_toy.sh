#!/usr/bin/env bash
set -e
python scripts/generate_toy_data.py --out_dir data/toy --n_train 64 --n_val 16 --n_test 16 --n_unlabeled 128
python train.py --config configs/exp_baseline_unet.yaml
python train.py --config configs/exp_baseline_tcn.yaml
python train.py --config configs/exp_newmodel.yaml
python train.py --config configs/abl_no_physics.yaml
python train.py --config configs/abl_no_freq.yaml
