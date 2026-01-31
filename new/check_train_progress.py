#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
快速检查 30Hz 训练进度
"""
import sys
from pathlib import Path
import json

try:
    import torch
except ImportError:
    print("[错误] 请先安装 PyTorch: pip install torch")
    sys.exit(1)

LOG_PATH = Path("results/01_30Hz_thinlayer_v2/train_log.txt")
BEST_CKPT = Path("results/01_30Hz_thinlayer_v2/checkpoints/best.pt")
LAST_CKPT = Path("results/01_30Hz_thinlayer_v2/checkpoints/last.pt")
NORM_STATS = Path("results/01_30Hz_thinlayer_v2/norm_stats.json")

def extract_last_epoch_from_log():
    """从日志中提取最后一个 epoch 信息"""
    if not LOG_PATH.exists():
        return None
    
    with open(LOG_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    for line in reversed(lines):
        if line.strip().startswith("Epoch"):
            return line.strip()
    return None

def check_checkpoint(ckpt_path):
    """检查 checkpoint 信息"""
    if not ckpt_path.exists():
        return None
    
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        info = {
            'epoch': ckpt.get('epoch', '?'),
            'best_val_pcc': ckpt.get('best_val_pcc', -1)
        }
        if 'val_metrics' in ckpt:
            info['val_pcc'] = ckpt['val_metrics'].get('pcc', -1)
            info['val_r2'] = ckpt['val_metrics'].get('r2', -1)
        return info
    except Exception as e:
        return {'error': str(e)}

def main():
    print("=" * 70)
    print("  30Hz 薄层模型训练进度检查")
    print("=" * 70)
    print()
    
    # 1. 日志最后一行
    last_log = extract_last_epoch_from_log()
    if last_log:
        print("[训练日志最后一行]")
        print(f"  {last_log}")
    else:
        print("[训练日志] 未找到或为空")
    print()
    
    # 2. Best checkpoint
    best_info = check_checkpoint(BEST_CKPT)
    if best_info and 'error' not in best_info:
        print("[Best Checkpoint]")
        print(f"  文件: {BEST_CKPT}")
        print(f"  Epoch: {best_info['epoch']}")
        if 'val_pcc' in best_info:
            print(f"  Val PCC: {best_info['val_pcc']:.4f}")
            print(f"  Val R²:  {best_info['val_r2']:.4f}")
    else:
        print(f"[Best Checkpoint] 不存在或损坏")
    print()
    
    # 3. Last checkpoint (用于断点续训)
    last_info = check_checkpoint(LAST_CKPT)
    if last_info and 'error' not in last_info:
        print("[Last Checkpoint (断点续训用)]")
        print(f"  文件: {LAST_CKPT}")
        print(f"  Epoch: {last_info['epoch']}")
        print(f"  Best Val PCC (历史最佳): {last_info.get('best_val_pcc', -1):.4f}")
    else:
        print(f"[Last Checkpoint] 不存在")
    print()
    
    # 4. 归一化参数
    if NORM_STATS.exists():
        with open(NORM_STATS, 'r') as f:
            norm = json.load(f)
        print("[归一化参数]")
        print(f"  地震: mean={norm['seis_mean']:.2e}, std={norm['seis_std']:.2e}")
        print(f"  阻抗: mean={norm['imp_mean']:.2e}, std={norm['imp_std']:.2e}")
    else:
        print("[归一化参数] 不存在")
    print()
    
    # 5. 训练状态建议
    print("=" * 70)
    if last_info and 'epoch' in last_info:
        current_epoch = last_info['epoch']
        target_epoch = 500
        remaining = target_epoch - current_epoch
        print(f"[建议] 当前训练到 Epoch {current_epoch}/500，还需 {remaining} 个 epoch")
        print(f"       运行: python train_30Hz_thinlayer_v2.py  (自动从断点继续)")
        print(f"       或使用: resume_train_30Hz.bat")
    else:
        print("[建议] 未检测到断点，将从头开始训练")
        print("       运行: python train_30Hz_thinlayer_v2.py")
    print()
    
    # 6. 目标指标对比
    print("[目标指标 (参考 20Hz 模型)]")
    print("  Val PCC: ≈0.93")
    print("  Val R²:  ≈0.86")
    if best_info and 'val_pcc' in best_info:
        print()
        print(f"[当前差距]")
        print(f"  PCC 差距: {0.93 - best_info['val_pcc']:.4f}")
        print(f"  R² 差距:  {0.86 - best_info['val_r2']:.4f}")
    print("=" * 70)

if __name__ == '__main__':
    main()
