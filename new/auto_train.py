# -*- coding: utf-8 -*-
"""
简化版训练脚本 - 训练单个频率直到完成
"""
import subprocess
import sys
import time
import torch
from pathlib import Path

PYTHON = r"D:\SEISMIC_CODING\new\.venv\Scripts\python.exe"
TRAIN_SCRIPT = r"D:\SEISMIC_CODING\new\train_v6.py"

def train(freq):
    ckpt_path = Path(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\checkpoints\best.pt")
    test_metrics = Path(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\test_metrics.json")
    
    while not test_metrics.exists():
        # 检查进度
        if ckpt_path.exists():
            try:
                c = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                print(f"\nCurrent: Epoch {c['epoch']}, Best PCC: {c['best_pcc']:.4f}")
            except:
                pass
        
        cmd = [PYTHON, TRAIN_SCRIPT, "--freq", freq, "--epochs", "800"]
        if ckpt_path.exists():
            cmd.extend(["--resume", str(ckpt_path)])
        
        print(f"Starting training {freq}...")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            time.sleep(1)
            continue
    
    print(f"\n{freq} training completed!")

if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "40Hz"
    train(freq)
