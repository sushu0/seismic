"""
快速运行所有实验并汇总结果
"""
import subprocess
import sys

experiments = [
    ("Baseline UNet1D", "configs/exp_baseline_unet.yaml", "train.epochs=50"),
    ("Baseline TCN1D", "configs/exp_baseline_tcn.yaml", "train.epochs=50"),
    ("MS-PhysFormer (supervised)", "configs/exp_newmodel.yaml", "train.epochs=50 train.lambda_phys=0.0 train.lambda_freq=0.0 train.lambda_cons=0.0 train.use_teacher=false output.exp_name=ms_physformer_supervised"),
]

print("=" * 60)
print("开始运行完整实验...")
print("=" * 60)

for i, (name, config, overrides) in enumerate(experiments, 1):
    print(f"\n[{i}/{len(experiments)}] 运行: {name}")
    print(f"配置: {config}")
    print(f"覆盖: {overrides}")
    print("-" * 60)
    
    cmd = f"python train.py --config {config} --override {overrides}"
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"[FAILED] {name}")
        sys.exit(1)
    else:
        print(f"[OK] {name} completed")

print("\n" + "=" * 60)
print("All experiments completed! Collecting results...")
print("=" * 60)

# 汇总结果
subprocess.run("python scripts/collect_results.py --results_root results --out_csv results/summary.csv --exp_names baseline_unet1d baseline_tcn1d ms_physformer_supervised", shell=True)

print("\n[OK] Experiments completed! See results/summary.csv")
