# One-click PowerShell script to run all experiments
$ErrorActionPreference = "Stop"

Write-Host "=== Generating toy data ===" -ForegroundColor Green
python scripts/generate_toy_data.py --out_dir data/toy --n_train 64 --n_val 16 --n_test 16 --n_unlabeled 128

Write-Host "=== Running baseline: UNet1D ===" -ForegroundColor Green
python train.py --config configs/exp_baseline_unet.yaml

Write-Host "=== Running baseline: TCN1D ===" -ForegroundColor Green
python train.py --config configs/exp_baseline_tcn.yaml

Write-Host "=== Running new model: MS-PhysFormer ===" -ForegroundColor Green
python train.py --config configs/exp_newmodel.yaml

Write-Host "=== Running ablation: no physics ===" -ForegroundColor Green
python train.py --config configs/abl_no_physics.yaml

Write-Host "=== Running ablation: no frequency ===" -ForegroundColor Green
python train.py --config configs/abl_no_freq.yaml

Write-Host "=== Collecting results ===" -ForegroundColor Green
python scripts/collect_results.py --results_root results --out_csv results/summary.csv --exp_names baseline_unet1d baseline_tcn1d new_ms_physformer abl_no_physics abl_no_freq

Write-Host "=== All experiments complete! Check results/summary.csv ===" -ForegroundColor Cyan
