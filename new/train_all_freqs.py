# -*- coding: utf-8 -*-
"""
完整训练脚本 - 训练所有频率的V6模型
使用方法: python train_all_freqs.py [freq]
- 无参数: 依次训练 40Hz, 20Hz, 50Hz
- 有参数: 只训练指定频率
"""
import subprocess
import sys
import os
import torch
from pathlib import Path

PYTHON = r"D:\SEISMIC_CODING\new\.venv\Scripts\python.exe"
TRAIN_SCRIPT = r"D:\SEISMIC_CODING\new\train_v6.py"

def check_checkpoint(freq):
    """检查训练进度"""
    ckpt_path = Path(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\checkpoints\best.pt")
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            return ckpt.get('epoch', 0), ckpt.get('best_pcc', 0)
        except:
            return 0, 0
    return 0, 0

def check_completed(freq):
    """检查是否已完成训练"""
    metrics_path = Path(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\test_metrics.json")
    return metrics_path.exists()

def train_single_freq(freq, max_retries=100):
    """训练单个频率的模型"""
    
    # 检查是否已完成
    if check_completed(freq):
        import json
        metrics_path = Path(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\test_metrics.json")
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"\n{freq} already completed! PCC={m['test_pcc']:.4f}, R2={m['test_r2']:.4f}")
        return True
    
    ckpt_path = Path(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\checkpoints\best.pt")
    
    for retry in range(max_retries):
        ep, pcc = check_checkpoint(freq)
        
        print(f"\n{'='*60}")
        print(f"Training {freq} - Attempt {retry + 1}/{max_retries}")
        print(f"Current progress: Epoch {ep}, Best PCC: {pcc:.4f}")
        print(f"{'='*60}")
        
        cmd = [PYTHON, TRAIN_SCRIPT, "--freq", freq, "--epochs", "800"]
        if ckpt_path.exists():
            cmd.extend(["--resume", str(ckpt_path)])
        
        try:
            result = subprocess.run(cmd)
            if result.returncode == 0:
                print(f"\n{freq} training completed successfully!")
                return True
        except KeyboardInterrupt:
            print(f"\nTraining interrupted. Progress saved. Continuing...")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    return False

def main():
    # 要训练的频率列表
    if len(sys.argv) > 1:
        freqs = [sys.argv[1]]
    else:
        freqs = ["40Hz", "20Hz", "50Hz"]  # 30Hz已完成
    
    print("="*60)
    print("V6 Model Training Script")
    print("="*60)
    
    # 显示当前状态
    for freq in ["20Hz", "30Hz", "40Hz", "50Hz"]:
        if check_completed(freq):
            import json
            with open(rf"D:\SEISMIC_CODING\new\results\01_{freq}_v6\test_metrics.json") as f:
                m = json.load(f)
            print(f"  {freq}: COMPLETED (PCC={m['test_pcc']:.4f})")
        else:
            ep, pcc = check_checkpoint(freq)
            print(f"  {freq}: Epoch {ep}, Best PCC={pcc:.4f}")
    
    print("\nFrequencies to train:", freqs)
    print("="*60)
    
    # 依次训练
    for freq in freqs:
        success = train_single_freq(freq)
        if not success:
            print(f"Warning: {freq} may not have completed fully")
    
    print("\n" + "="*60)
    print("All training completed!")
    print("="*60)

if __name__ == "__main__":
    main()
