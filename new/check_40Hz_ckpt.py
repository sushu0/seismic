"""
40Hz模型评估脚本 - 简化版本
"""
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import segyio
from scipy.stats import pearsonr
from pathlib import Path

# 配置
SEISMIC_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_40Hz_re.sgy'
IMPEDANCE_PATH = 'D:/SEISMIC_CODING/zmy_data/01/data/01_40Hz_04.txt'
MODEL_PATH = 'D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer/checkpoints/best.pt'
NORM_PATH = 'D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer/norm_stats.json'
OUTPUT_DIR = Path('D:/SEISMIC_CODING/new/results/01_40Hz_thinlayer')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'设备: {device}')

# 加载数据
with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
    seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)

raw_imp = np.loadtxt(IMPEDANCE_PATH, usecols=4, skiprows=1).astype(np.float32)
n_traces = seismic.shape[0]
n_samples = len(raw_imp) // n_traces
impedance = raw_imp.reshape(n_traces, n_samples)

with open(NORM_PATH, 'r') as f:
    norm_stats = json.load(f)

seis_norm = (seismic - norm_stats['seis_mean']) / norm_stats['seis_std']

print(f'数据形状: seismic={seismic.shape}, impedance={impedance.shape}')

# 加载完整模型checkpoint
ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
print(f'Checkpoint keys: {list(ckpt.keys())}')
best_epoch = ckpt.get('epoch', 'N/A')
val_metrics = ckpt.get('val_metrics', {})
print(f'Best epoch: {best_epoch}')
print(f'Val metrics: {val_metrics}')

# 模型state_dict
model_state = ckpt['model']

# 检查模型结构 - 从state_dict推断
first_key = list(model_state.keys())[0]
print(f'\n模型第一个key: {first_key}')
print(f'模型权重数量: {len(model_state)}')

# 计算模型参数量
total_params = sum(p.numel() for p in model_state.values())
print(f'总参数量: {total_params:,}')

# 由于无法直接加载模型类，我们需要从训练脚本中手动复制模型定义
# 这里采用另一种方法：直接使用训练脚本进行推理

print('\n' + '='*50)
print('需要使用训练脚本进行推理')
print('='*50)
