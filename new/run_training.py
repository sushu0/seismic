# -*- coding: utf-8 -*-
"""
自动训练脚本 - 持续从检查点恢复训练
"""
import subprocess
import sys
import time
import os
from pathlib import Path

PYTHON = r"D:\SEISMIC_CODING\new\.venv\Scripts\python.exe"
TRAIN_SCRIPT = r"D:\SEISMIC_CODING\new\train_v6.py"

def train_freq(freq, max_retries=50):
    """训练指定频率的模型"""
    ckpt_path = Path(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\checkpoints\best.pt")
    
    for retry in range(max_retries):
        cmd = [PYTHON, TRAIN_SCRIPT, "--freq", freq, "--epochs", "800"]
        if ckpt_path.exists():
            cmd.extend(["--resume", str(ckpt_path)])
        
        print(f"\n{'='*60}")
        print(f"Training {freq} - Attempt {retry + 1}/{max_retries}")
        print(f"{'='*60}")
        
        try:
            result = subprocess.run(cmd, timeout=None)
            if result.returncode == 0:
                print(f"\n{freq} training completed successfully!")
                return True
        except KeyboardInterrupt:
            print(f"\n{freq} training interrupted. Restarting...")
            time.sleep(2)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(2)
    
    print(f"\n{freq} training failed after {max_retries} retries")
    return False

if __name__ == "__main__":
    freq = sys.argv[1] if len(sys.argv) > 1 else "40Hz"
    train_freq(freq)
