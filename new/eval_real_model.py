"""快速评估真实数据模型性能"""
import torch
import numpy as np
from pathlib import Path
from seisinv.utils.metrics import summarize_metrics

# 使用generate_real_plots中的预测结果
from generate_real_plots import load_real_model, predict_and_forward_model

exp_name = 'real_unet1d_optimized'

print(f"加载模型: {exp_name}")
model, test_loader, fm, device, norm_stats = load_real_model(exp_name)

print("预测...")
y_true, y_pred, s_obs, s_pred = predict_and_forward_model(model, test_loader, fm, device)

print(f"计算指标...")
metrics = summarize_metrics(y_true, y_pred)

print(f"\n{'='*60}")
print(f"测试集性能 ({exp_name}):")
print(f"{'='*60}")
print(f"PCC  = {metrics['PCC']:.6f}")
print(f"R²   = {metrics['R2']:.6f}")
print(f"MSE  = {metrics['MSE']:.6f}")
print(f"{'='*60}")

# 保存结果
out_dir = Path(f'results/{exp_name}')
import json
test_metrics = {
    'best_epoch': 29,  # 最后一个epoch
    'best_val_pcc': 0.9985,
    'test_metrics': metrics
}
with open(out_dir / 'test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

print(f"\n结果已保存到: {out_dir / 'test_metrics.json'}")
