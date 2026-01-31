#!/usr/bin/env python
"""Evaluation script for optimized_v3_advanced"""
import subprocess
import sys

run_dir = "runs/optimized_v3_advanced"
dataset = "data/marmousi2_2721_like_l101.npz"
ckpt = f"{run_dir}/checkpoints/best.pt"

print("=" * 60)
print("Plotting 4-trace comparison...")
print("=" * 60)
subprocess.run([
    sys.executable, "scripts/plot_traces_compare.py",
    "--pred", f"{run_dir}/pred_test.npz",
    "--out", f"{run_dir}/traces_compare.png",
    "--one_based",
    "--time_ms_max", "2350.0",
    "--trace_ids", "299", "599", "1699", "2299",
    "--ckpt", ckpt
], check=True)

print("\n" + "=" * 60)
print("Plotting section...")
print("=" * 60)
subprocess.run([
    sys.executable, "scripts/plot_section.py",
    "--pred", f"{run_dir}/pred_test.npz",
    "--outdir", f"{run_dir}/figures_section",
    "--time_ms_max", "9400.0",
    "--cmap", "jet"
], check=True)

print("\n" + "=" * 60)
print("Done! Check results in:")
print(f"  - {run_dir}/pred_test_metrics.json")
print(f"  - {run_dir}/traces_compare.png")
print(f"  - {run_dir}/figures_section/")
print("=" * 60)
