"""
检查实验是否完成并汇总结果
"""
import os
import json
import time

def check_experiment_complete(exp_name):
    """检查实验是否完成(test_metrics.json存在)"""
    metrics_file = f"results/{exp_name}/test_metrics.json"
    return os.path.exists(metrics_file)

def load_test_metrics(exp_name):
    """加载测试指标"""
    metrics_file = f"results/{exp_name}/test_metrics.json"
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    return data['test_metrics']

# 要检查的实验
experiments = {
    'baseline_unet1d': 'UNet1D Baseline',
    'baseline_tcn1d': 'TCN1D Baseline',
    'ms_physformer_supervised': 'MS-PhysFormer (supervised)'
}

print("=" * 80)
print("检查实验状态...")
print("=" * 80)

# 检查每个实验的状态
results = {}
for exp_name, display_name in experiments.items():
    if check_experiment_complete(exp_name):
        metrics = load_test_metrics(exp_name)
        results[exp_name] = metrics
        print(f"✓ {display_name:<35} | PCC: {metrics['PCC']:.4f} | R²: {metrics['R2']:.4f} | MSE: {metrics['MSE']:.4f}")
    else:
        print(f"⏳ {display_name:<35} | 运行中...")

print("\n" + "=" * 80)
print("实验结果摘要")
print("=" * 80)

if len(results) == len(experiments):
    print("\n所有实验已完成!\n")
    
    # 找出最佳模型
    best_pcc_name = max(results.items(), key=lambda x: x[1]['PCC'])[0]
    best_r2_name = max(results.items(), key=lambda x: x[1]['R2'])[0]
    
    print(f"最佳模型 (PCC): {experiments[best_pcc_name]} - PCC={results[best_pcc_name]['PCC']:.4f}")
    print(f"最佳模型 (R²):  {experiments[best_r2_name]} - R²={results[best_r2_name]['R2']:.4f}")
    
    # 保存汇总结果
    import pandas as pd
    df = pd.DataFrame([
        {
            'Model': experiments[exp_name],
            'PCC': metrics['PCC'],
            'R2': metrics['R2'],
            'MSE': metrics['MSE']
        }
        for exp_name, metrics in results.items()
    ])
    df = df.sort_values('PCC', ascending=False)
    df.to_csv('results/summary.csv', index=False)
    print(f"\n结果已保存到: results/summary.csv")
    print("\n详细结果:")
    print(df.to_string(index=False))
else:
    print(f"\n已完成: {len(results)}/{len(experiments)}")
    print("部分实验仍在运行中,请稍后再次运行此脚本")
