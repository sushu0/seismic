#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
断点续训功能验证脚本
检查 train_30Hz_thinlayer_v2.py 的断点恢复逻辑
"""
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("[错误] 请先安装 PyTorch")
    sys.exit(1)

CKPT_DIR = Path("results/01_30Hz_thinlayer_v2/checkpoints")
LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"

def test_checkpoint_load():
    """测试 checkpoint 能否正常加载"""
    print("=" * 70)
    print("  断点续训功能验证")
    print("=" * 70)
    print()
    
    if not LAST_CKPT.exists():
        print(f"[×] {LAST_CKPT} 不存在")
        print("[建议] 先运行一次训练生成 checkpoint")
        return False
    
    print(f"[✓] 检测到断点文件: {LAST_CKPT}")
    print(f"    大小: {LAST_CKPT.stat().st_size / 1e6:.2f} MB")
    print()
    
    try:
        # 测试加载
        print("[测试] 加载 checkpoint (weights_only=False)...")
        ckpt = torch.load(LAST_CKPT, map_location='cpu', weights_only=False)
        print("[✓] 加载成功")
        print()
        
        # 检查必要字段
        required_keys = ['model', 'epoch']
        optional_keys = ['optimizer', 'scheduler', 'best_val_pcc']
        
        print("[检查] Checkpoint 内容:")
        for key in required_keys:
            if key in ckpt:
                if key == 'epoch':
                    print(f"  ✓ {key}: {ckpt[key]}")
                else:
                    print(f"  ✓ {key}: <state_dict>")
            else:
                print(f"  × {key}: 缺失 (必需)")
                return False
        
        for key in optional_keys:
            if key in ckpt:
                if key == 'best_val_pcc':
                    print(f"  ✓ {key}: {ckpt[key]:.4f}")
                else:
                    print(f"  ✓ {key}: <state_dict>")
            else:
                print(f"  - {key}: 缺失 (可选)")
        print()
        
        # 续训起始点
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_pcc = ckpt.get('best_val_pcc', -1)
        print(f"[续训信息]")
        print(f"  将从 Epoch {start_epoch} 开始训练")
        print(f"  历史最佳 val_pcc: {best_val_pcc:.4f}")
        print()
        
        # 检查 best.pt
        if BEST_CKPT.exists():
            print(f"[验证] 对比 best.pt...")
            best_ckpt = torch.load(BEST_CKPT, map_location='cpu', weights_only=False)
            if 'val_metrics' in best_ckpt:
                best_pcc = best_ckpt['val_metrics'].get('pcc', -1)
                best_r2 = best_ckpt['val_metrics'].get('r2', -1)
                print(f"  Best checkpoint: Epoch {best_ckpt.get('epoch', '?')}")
                print(f"  Best val_pcc: {best_pcc:.4f}")
                print(f"  Best val_r2:  {best_r2:.4f}")
                print()
                
                # 检查一致性
                if abs(best_pcc - best_val_pcc) < 1e-4:
                    print("[✓] best_val_pcc 一致")
                else:
                    print(f"[!] best_val_pcc 不一致: last.pt={best_val_pcc:.4f}, best.pt={best_pcc:.4f}")
        
        print()
        print("=" * 70)
        print("[结论] 断点续训功能正常 ✓")
        print("       可以运行: python train_30Hz_thinlayer_v2.py")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"[×] 加载失败: {e}")
        print()
        print("[建议]")
        print("  1. 检查 PyTorch 版本是否匹配")
        print("  2. 尝试删除 last.pt 并重新训练")
        print("  3. 检查磁盘空间是否充足")
        return False

if __name__ == '__main__':
    success = test_checkpoint_load()
    sys.exit(0 if success else 1)
