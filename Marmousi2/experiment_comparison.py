#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验对比脚本 - 与传统方法的对比
Experiment Comparison Script - Comparison with Traditional Methods

作者: AI Assistant
日期: 2024
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import segyio
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

class ExperimentComparison:
    """实验对比类"""
    
    def __init__(self):
        self.results = {}
    
    def traditional_impedance_inversion(self, seismic_data, dt):
        """传统波阻抗反演方法 - 基于递归反演"""
        print("执行传统递归反演...")
        
        # 简化的递归反演算法
        impedance = np.zeros_like(seismic_data)
        impedance[:, 0] = 2000  # 初始波阻抗值
        
        for i in range(1, seismic_data.shape[1]):
            # 递归公式: I(t) = I(t-1) * (1 + R(t)) / (1 - R(t))
            # 其中 R(t) 是反射系数
            reflection_coeff = seismic_data[:, i]
            impedance[:, i] = impedance[:, i-1] * (1 + reflection_coeff) / (1 - reflection_coeff)
        
        return impedance
    
    def bandlimited_inversion(self, seismic_data, dt):
        """带限反演方法"""
        print("执行带限反演...")
        
        # 频域处理
        seismic_fft = np.fft.fft(seismic_data, axis=1)
        
        # 设计带通滤波器
        freqs = np.fft.fftfreq(seismic_data.shape[1], dt)
        low_freq = 5  # Hz
        high_freq = 50  # Hz
        
        # 创建滤波器
        filter_mask = (np.abs(freqs) >= low_freq) & (np.abs(freqs) <= high_freq)
        seismic_filtered_fft = seismic_fft * filter_mask
        
        # 反变换
        seismic_filtered = np.real(np.fft.ifft(seismic_filtered_fft, axis=1))
        
        # 递归反演
        impedance = self.traditional_impedance_inversion(seismic_filtered, dt)
        
        return impedance
    
    def evaluate_method(self, predicted, true, method_name):
        """评估方法性能"""
        # 展平数据
        pred_flat = predicted.flatten()
        true_flat = true.flatten()
        
        # 计算指标
        mse = mean_squared_error(true_flat, pred_flat)
        r2 = r2_score(true_flat, pred_flat)
        pcc, _ = pearsonr(true_flat, pred_flat)
        mae = np.mean(np.abs(pred_flat - true_flat))
        
        # 相对误差
        relative_error = np.mean(np.abs((pred_flat - true_flat) / true_flat)) * 100
        
        results = {
            'MSE': mse,
            'R²': r2,
            'PCC': pcc,
            'MAE': mae,
            'Relative_Error_%': relative_error
        }
        
        self.results[method_name] = results
        print(f"\n{method_name} 性能评估:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
        
        return results
    
    def compare_methods(self, seismic_data, true_impedance, dt, dl_prediction):
        """对比不同方法"""
        print("开始方法对比实验...")
        
        # 1. 传统递归反演
        traditional_pred = self.traditional_impedance_inversion(seismic_data, dt)
        self.evaluate_method(traditional_pred, true_impedance, "传统递归反演")
        
        # 2. 带限反演
        bandlimited_pred = self.bandlimited_inversion(seismic_data, dt)
        self.evaluate_method(bandlimited_pred, true_impedance, "带限反演")
        
        # 3. 深度学习方法
        self.evaluate_method(dl_prediction, true_impedance, "深度学习FCN")
        
        return self.results
    
    def visualize_comparison(self, seismic_data, true_impedance, predictions, dt, save_path=None):
        """可视化对比结果"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 选择几个代表性道进行对比
        trace_indices = [54, 800, 1500]
        
        for i, trace_idx in enumerate(trace_indices):
            # 地震道
            axes[0, i].plot(seismic_data[trace_idx], 'k-', linewidth=0.8)
            axes[0, i].set_title(f'地震道 {trace_idx}')
            axes[0, i].set_ylabel('振幅')
            axes[0, i].grid(True, alpha=0.3)
            
            # 波阻抗对比
            time_axis = np.arange(len(true_impedance[trace_idx])) * dt * 1000
            
            axes[1, i].plot(time_axis, true_impedance[trace_idx], 'k-', 
                          linewidth=2, label='真实波阻抗')
            axes[1, i].plot(time_axis, predictions['传统递归反演'][trace_idx], 'r--', 
                          linewidth=1.5, label='传统递归反演')
            axes[1, i].plot(time_axis, predictions['带限反演'][trace_idx], 'b--', 
                          linewidth=1.5, label='带限反演')
            axes[1, i].plot(time_axis, predictions['深度学习FCN'][trace_idx], 'g-', 
                          linewidth=1.5, label='深度学习FCN')
            
            axes[1, i].set_title(f'波阻抗对比 - 道 {trace_idx}')
            axes[1, i].set_xlabel('时间 (ms)')
            axes[1, i].set_ylabel('波阻抗')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_comparison_table(self, save_path=None):
        """生成对比表格"""
        if not self.results:
            print("没有结果数据，请先运行对比实验")
            return
        
        # 创建对比表格
        methods = list(self.results.keys())
        metrics = list(self.results[methods[0]].keys())
        
        print("\n" + "="*80)
        print("方法对比结果表")
        print("="*80)
        
        # 表头
        header = f"{'方法':<15}"
        for metric in metrics:
            header += f"{metric:>12}"
        print(header)
        print("-" * 80)
        
        # 数据行
        for method in methods:
            row = f"{method:<15}"
            for metric in metrics:
                row += f"{self.results[method][metric]:>12.6f}"
            print(row)
        
        print("="*80)
        
        # 保存到文件
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("方法对比结果表\n")
                f.write("="*80 + "\n")
                f.write(header + "\n")
                f.write("-" * 80 + "\n")
                for method in methods:
                    row = f"{method:<15}"
                    for metric in metrics:
                        row += f"{self.results[method][metric]:>12.6f}"
                    f.write(row + "\n")
                f.write("="*80 + "\n")
            print(f"对比表格已保存到: {save_path}")

def main():
    """主函数"""
    # 加载数据
    print("加载数据...")
    with segyio.open('data/SYNTHETIC.segy', "r", ignore_geometry=True) as f:
        seismic_data = np.stack([f.trace[i] for i in range(f.tracecount)])
        dt = f.bin[segyio.BinField.Interval] * 1e-6
    
    true_impedance = np.loadtxt('data/impedance.txt', 
                              usecols=4, skiprows=1).reshape(2721, 701)
    
    # 加载深度学习预测结果
    # 这里需要加载您训练好的模型预测结果
    # dl_prediction = load_dl_prediction()
    
    # 创建对比实验
    comparison = ExperimentComparison()
    
    # 执行对比
    # results = comparison.compare_methods(seismic_data, true_impedance, dt, dl_prediction)
    
    # 生成对比表格
    # comparison.generate_comparison_table('data/comparison_results.txt')
    
    print("对比实验完成!")

if __name__ == "__main__":
    main()
