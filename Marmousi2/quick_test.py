#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本 - 验证数据文件和运行简短训练
Quick Test Script - Verify data files and run short training

作者: AI Assistant
日期: 2024
"""

import os
import sys
import torch
import numpy as np
import segyio
from pathlib import Path

def check_environment():
    """检查环境和依赖"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    
    # 检查Python版本
    print(f"Python版本: {sys.version}")
    
    # 检查PyTorch
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name(0)}")
    
    # 检查其他依赖
    try:
        import segyio
        try:
            print(f"segyio版本: {segyio.__version__}")
        except AttributeError:
            print("segyio: 已安装 (版本信息不可用)")
    except ImportError:
        print("错误: 未安装segyio")
        return False
    
    try:
        import matplotlib
        print(f"matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("错误: 未安装matplotlib")
        return False
    
    try:
        import sklearn
        print(f"scikit-learn版本: {sklearn.__version__}")
    except ImportError:
        print("错误: 未安装scikit-learn")
        return False
    
    return True

def find_data_files():
    """查找数据文件"""
    print("\n" + "=" * 60)
    print("数据文件检查")
    print("=" * 60)
    
    possible_paths = [
        'data/SYNTHETIC.segy',
        './data/SYNTHETIC.segy',
        '../data/SYNTHETIC.segy',
        'Marmousi2/data/SYNTHETIC.segy'
    ]
    
    seismic_path = None
    impedance_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✓ 找到地震数据: {path}")
            seismic_path = path
            
            # 查找对应的impedance文件
            impedance_candidates = [
                path.replace('SYNTHETIC.segy', 'impedance.txt'),
                os.path.join(os.path.dirname(path), 'impedance.txt')
            ]
            for imp_path in impedance_candidates:
                if os.path.exists(imp_path):
                    print(f"✓ 找到波阻抗数据: {imp_path}")
                    impedance_path = imp_path
                    break
            break
    
    if not seismic_path:
        print("✗ 未找到地震数据文件 (SYNTHETIC.segy)")
        print("请检查以下位置:")
        for path in possible_paths:
            print(f"  - {path}")
        return None, None
    
    if not impedance_path:
        print("✗ 未找到波阻抗数据文件 (impedance.txt)")
        return None, None
    
    return seismic_path, impedance_path

def check_data_format(seismic_path, impedance_path):
    """检查数据格式"""
    print("\n" + "=" * 60)
    print("数据格式检查")
    print("=" * 60)
    
    try:
        # 检查地震数据
        with segyio.open(seismic_path, "r", ignore_geometry=True) as f:
            n_traces = f.tracecount
            n_samples = f.samples.size
            dt = f.bin[segyio.BinField.Interval] * 1e-6
            print(f"地震数据维度: {n_traces} 道 × {n_samples} 采样点")
            print(f"采样间隔: {dt:.6f} 秒")
            
            # 读取一小部分数据测试
            test_data = np.stack([f.trace[i] for i in range(min(10, n_traces))])
            print(f"地震数据范围: {np.min(test_data):.6f} ~ {np.max(test_data):.6f}")
            
    except Exception as e:
        print(f"✗ 地震数据读取失败: {e}")
        return False
    
    try:
        # 检查波阻抗数据
        impedance_data = np.loadtxt(impedance_path, usecols=4, skiprows=1)
        expected_shape = n_traces * n_samples
        if len(impedance_data) != expected_shape:
            print(f"✗ 波阻抗数据维度不匹配: 期望 {expected_shape}, 实际 {len(impedance_data)}")
            return False
        
        impedance_data = impedance_data.reshape(n_traces, n_samples)
        print(f"波阻抗数据维度: {impedance_data.shape}")
        print(f"波阻抗数据范围: {np.min(impedance_data):.2f} ~ {np.max(impedance_data):.2f}")
        
    except Exception as e:
        print(f"✗ 波阻抗数据读取失败: {e}")
        return False
    
    print("✓ 数据格式检查通过")
    return True

def quick_model_test():
    """快速模型测试"""
    print("\n" + "=" * 60)
    print("模型测试")
    print("=" * 60)
    
    try:
        import torch.nn as nn
        import torch.nn.functional as F
        
        # 简单的测试模型
        class SimpleFCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, 5, padding=2)
                self.conv2 = nn.Conv1d(32, 64, 5, padding=2)
                self.conv3 = nn.Conv1d(64, 1, 5, padding=2)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.conv3(x)
                return x
        
        # 创建测试数据
        batch_size, channels, length = 2, 1, 701
        test_input = torch.randn(batch_size, channels, length)
        
        # 测试模型
        model = SimpleFCN()
        output = model(test_input)
        
        print(f"✓ 模型测试通过")
        print(f"  输入形状: {test_input.shape}")
        print(f"  输出形状: {output.shape}")
        print(f"  模型参数量: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ 模型测试失败: {e}")
        return False

def main():
    """主函数"""
    print("基于深度学习的波阻抗反演 - 快速测试")
    print("=" * 60)
    
    # 1. 环境检查
    if not check_environment():
        print("\n环境检查失败，请安装必要的依赖包")
        return
    
    # 2. 数据文件检查
    seismic_path, impedance_path = find_data_files()
    if not seismic_path or not impedance_path:
        print("\n数据文件检查失败")
        return
    
    # 3. 数据格式检查
    if not check_data_format(seismic_path, impedance_path):
        print("\n数据格式检查失败")
        return
    
    # 4. 模型测试
    if not quick_model_test():
        print("\n模型测试失败")
        return
    
    print("\n" + "=" * 60)
    print("✓ 所有检查通过！")
    print("=" * 60)
    print("现在可以运行完整的训练脚本:")
    print("python complete_reproduction.py")
    print("\n或者运行对比实验:")
    print("python experiment_comparison.py")

if __name__ == "__main__":
    main()
