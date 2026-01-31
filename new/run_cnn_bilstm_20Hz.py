# -*- coding: utf-8 -*-
"""
使用CNN-BiLSTM模型进行20Hz数据推理
"""
import os
import sys
import json
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

# ==================== CNN-BiLSTM模型 ====================
class CNNBiLSTM(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(128, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: [B, 1, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout(x)  # [B, 64, T]
        x = x.transpose(1, 2)  # [B, T, 64]
        out, _ = self.bilstm(x)  # [B, T, 128]
        out = self.fc(out)  # [B, T, 1]
        return out.squeeze(-1)  # [B, T]


def minmax_norm(arr, min_val, max_val):
    return (arr - min_val) / (max_val - min_val + 1e-8)

def minmax_denorm(arr, min_val, max_val):
    return arr * (max_val - min_val) + min_val


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*70)
    print("CNN-BiLSTM Model Inference on 20Hz Data")
    print("="*70)
    
    # ==================== 加载模型 ====================
    model_path = r'D:\SEISMIC_CODING\comparison01\marmousi_cnn_bilstm_supervised.pth'
    
    print(f"\nLoading model from: {model_path}")
    model = CNNBiLSTM(dropout=0.3)
    
    try:
        state = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(state, dict) and 'model' in state:
            model.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return
    
    model = model.to(device)
    model.eval()
    
    # ==================== 加载20Hz数据 ====================
    print("\nLoading 20Hz data...")
    SEISMIC_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy'
    IMPEDANCE_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt'
    
    try:
        with segyio.open(SEISMIC_PATH, 'r', ignore_geometry=True) as f:
            seismic = np.array([f.trace[i] for i in range(f.tracecount)], dtype=np.float32)
        print(f"✓ Seismic data shape: {seismic.shape}")
        
        df = pd.read_csv(IMPEDANCE_PATH, sep=r'\s+', skiprows=1, header=None)
        impedance_raw = df.iloc[:, -1].values.astype(np.float32)
        n_traces = seismic.shape[0]
        n_samples = seismic.shape[1]
        impedance = impedance_raw.reshape(n_traces, n_samples)
        print(f"✓ Impedance data shape: {impedance.shape}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # ==================== 数据归一化 ====================
    print("\nNormalizing data...")
    
    # 使用comparison01中的归一化参数（从marmousi模型训练的参数）
    seis_min, seis_max = float(seismic.min()), float(seismic.max())
    imp_min, imp_max = float(impedance.min()), float(impedance.max())
    
    print(f"  Seismic range: [{seis_min:.2f}, {seis_max:.2f}]")
    print(f"  Impedance range: [{imp_min:.2e}, {imp_max:.2e}]")
    
    # 按道进行归一化
    seis_norm = np.zeros_like(seismic)
    for i in range(seismic.shape[0]):
        trace_min = seismic[i].min()
        trace_max = seismic[i].max()
        if trace_max > trace_min:
            seis_norm[i] = (seismic[i] - trace_min) / (trace_max - trace_min)
        else:
            seis_norm[i] = seismic[i]
    
    imp_norm = minmax_norm(impedance, imp_min, imp_max)
    
    print("✓ Data normalized")
    
    # ==================== 模型推理 ====================
    print("\nRunning inference...")
    
    predictions = []
    with torch.no_grad():
        for i in range(n_traces):
            x = torch.from_numpy(seis_norm[i:i+1, np.newaxis, :]).to(device)
            pred = model(x).cpu().numpy()
            # 反归一化
            pred_denorm = minmax_denorm(pred, imp_min, imp_max)
            predictions.append(pred_denorm[0])
    
    predictions = np.array(predictions)
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"✓ Prediction range: [{predictions.min():.2e}, {predictions.max():.2e}]")
    
    # ==================== 评估 ====================
    print("\nEvaluating results...")
    
    pred_flat = predictions.flatten()
    true_flat = impedance.flatten()
    
    # 计算指标
    pcc, _ = pearsonr(pred_flat, true_flat)
    ss_res = np.sum((true_flat - pred_flat) ** 2)
    ss_tot = np.sum((true_flat - true_flat.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
    mae = np.mean(np.abs(pred_flat - true_flat))
    
    print(f"\n  PCC:  {pcc:.6f}")
    print(f"  R²:   {r2:.6f}")
    print(f"  RMSE: {rmse:.2e}")
    print(f"  MAE:  {mae:.2e}")
    
    # ==================== 保存结果 ====================
    output_dir = Path(r'D:\SEISMIC_CODING\new\results\cnn_bilstm_20hz')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存预测结果
    np.save(output_dir / 'predictions.npy', predictions)
    print(f"\n✓ Predictions saved to {output_dir / 'predictions.npy'}")
    
    # 保存指标
    metrics = {
        'test_pcc': float(pcc),
        'test_r2': float(r2),
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'model': 'CNN-BiLSTM (supervised)',
        'data': '20Hz'
    }
    
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {output_dir / 'metrics.json'}")
    
    # ==================== 可视化 ====================
    print("\nGenerating visualizations...")
    
    # 单道对比图
    for trace_idx in [25, 50, 75]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 8))
        
        time = np.arange(n_samples) * 0.001
        
        # 地震道
        ax1 = axes[0]
        ax1.plot(seismic[trace_idx], time, 'b-', linewidth=0.8)
        ax1.set_ylabel('Time (s)', fontsize=14)
        ax1.set_xlabel('Amplitude', fontsize=14)
        ax1.set_title(f'Seismic Trace #{trace_idx+1}', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        
        # 波阻抗对比
        ax2 = axes[1]
        ax2.plot(impedance[trace_idx], time, 'b-', linewidth=1.2, label='True', alpha=0.8)
        ax2.plot(predictions[trace_idx], time, 'r--', linewidth=1.2, label='Predicted', alpha=0.8)
        ax2.set_xlabel('Impedance', fontsize=14)
        ax2.set_title('Impedance Comparison (CNN-BiLSTM)', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 误差
        ax3 = axes[2]
        error = predictions[trace_idx] - impedance[trace_idx]
        ax3.fill_betweenx(time, 0, error, where=error>=0, color='red', alpha=0.5, label='Over')
        ax3.fill_betweenx(time, 0, error, where=error<0, color='blue', alpha=0.5, label='Under')
        ax3.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
        ax3.set_xlabel('Error', fontsize=14)
        ax3.set_title('Prediction Error', fontsize=14, fontweight='bold')
        ax3.invert_yaxis()
        ax3.legend(loc='upper right', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        pcc_trace, _ = pearsonr(predictions[trace_idx], impedance[trace_idx])
        ss_res_t = np.sum((impedance[trace_idx] - predictions[trace_idx]) ** 2)
        ss_tot_t = np.sum((impedance[trace_idx] - impedance[trace_idx].mean()) ** 2)
        r2_trace = 1 - ss_res_t / ss_tot_t
        
        fig.suptitle(f'CNN-BiLSTM (20Hz) - Trace #{trace_idx+1} | PCC={pcc_trace:.4f} | R²={r2_trace:.4f}', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'trace_{trace_idx+1}_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: trace_{trace_idx+1}_comparison.png")
    
    # 剖面对比图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    extent = [0, n_traces, n_samples*0.001, 0]
    
    ax1 = axes[0, 0]
    vmax = np.percentile(np.abs(seismic), 98)
    im1 = ax1.imshow(seismic.T, aspect='auto', cmap='seismic', extent=extent, vmin=-vmax, vmax=vmax)
    ax1.set_xlabel('Trace Number', fontsize=12)
    ax1.set_ylabel('Time (s)', fontsize=12)
    ax1.set_title('Seismic Section', fontsize=14, fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Amplitude')
    
    ax2 = axes[0, 1]
    vmin_imp = np.percentile(impedance, 2)
    vmax_imp = np.percentile(impedance, 98)
    im2 = ax2.imshow(impedance.T, aspect='auto', cmap='jet', extent=extent, vmin=vmin_imp, vmax=vmax_imp)
    ax2.set_xlabel('Trace Number', fontsize=12)
    ax2.set_ylabel('Time (s)', fontsize=12)
    ax2.set_title('True Impedance', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Impedance')
    
    ax3 = axes[1, 0]
    im3 = ax3.imshow(predictions.T, aspect='auto', cmap='jet', extent=extent, vmin=vmin_imp, vmax=vmax_imp)
    ax3.set_xlabel('Trace Number', fontsize=12)
    ax3.set_ylabel('Time (s)', fontsize=12)
    ax3.set_title('Predicted Impedance (CNN-BiLSTM)', fontsize=14, fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Impedance')
    
    ax4 = axes[1, 1]
    error_sect = predictions - impedance
    err_max = np.percentile(np.abs(error_sect), 98)
    im4 = ax4.imshow(error_sect.T, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-err_max, vmax=err_max)
    ax4.set_xlabel('Trace Number', fontsize=12)
    ax4.set_ylabel('Time (s)', fontsize=12)
    ax4.set_title('Prediction Error', fontsize=14, fontweight='bold')
    plt.colorbar(im4, ax=ax4, label='Error')
    
    fig.suptitle(f'CNN-BiLSTM (20Hz) - Section Comparison | PCC={pcc:.4f} | R²={r2:.4f}', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'section_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: section_comparison.png")
    
    print("\n" + "="*70)
    print("Inference completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
