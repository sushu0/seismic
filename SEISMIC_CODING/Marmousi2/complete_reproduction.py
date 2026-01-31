#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于深度学习的波阻抗反演完整复现框架
Complete Reproduction Framework for Deep Learning-based Seismic Impedance Inversion

作者: AI Assistant
日期: 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import segyio
import json
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn.functional as F

# 设置标准输出编码为UTF-8，避免Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class SeismicImpedanceInversion:
    """基于深度学习的波阻抗反演主类"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.data_loader = None
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_pcc': [], 'val_pcc': [],
            'train_r2': [], 'val_r2': []
        }
        
        # 打印设备信息
        if torch.cuda.is_available():
            print(f"[OK] GPU检测成功: {torch.cuda.get_device_name(0)}")
            print(f"   GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("[WARNING] GPU不可用，使用CPU训练")
        
    def load_data(self):
        """加载和预处理数据"""
        print("正在加载数据...")
        
        # 加载地震数据
        with segyio.open(self.config['seismic_path'], "r", ignore_geometry=True) as f:
            seismic_data = np.stack([f.trace[i] for i in range(f.tracecount)])
            dt = f.bin[segyio.BinField.Interval] * 1e-6
            n_traces = f.tracecount
            n_samples_seis = len(f.samples)
        
        # 加载波阻抗数据
        imp_raw = np.loadtxt(self.config['impedance_path'], usecols=4, skiprows=1)
        
        # 自动检测波阻抗数据的维度
        if imp_raw.size % n_traces == 0:
            n_samples_imp = imp_raw.size // n_traces
            impedance_data = imp_raw.reshape(n_traces, n_samples_imp)
        else:
            raise ValueError(f"波阻抗数据大小({imp_raw.size})与道数({n_traces})不匹配")
        
        # 如果地震数据和波阻抗数据采样点数不一致，截取地震数据
        if n_samples_seis != n_samples_imp:
            print(f"检测到维度不匹配: 地震{n_samples_seis}点 vs 波阻抗{n_samples_imp}点")
            if n_samples_seis > n_samples_imp:
                seismic_data = seismic_data[:, :n_samples_imp]
                n_samples = n_samples_imp
                print(f"已截取地震数据到{n_samples_imp}个采样点")
            else:
                impedance_data = impedance_data[:, :n_samples_seis]
                n_samples = n_samples_seis
                print(f"已截取波阻抗数据到{n_samples_seis}个采样点")
        else:
            n_samples = n_samples_seis
        
        print(f"最终数据: {n_traces}道 x {n_samples}采样点")
        
        # 数据归一化
        def normalize(data):
            data_min, data_max = np.min(data), np.max(data)
            return (data - data_min) / (data_max - data_min), data_min, data_max
        
        seismic_norm, s_min, s_max = normalize(seismic_data)
        imp_norm, imp_min, imp_max = normalize(impedance_data)
        
        # 保存归一化参数
        norm_params = {
            's_min': float(s_min), 's_max': float(s_max),
            'imp_min': float(imp_min), 'imp_max': float(imp_max)
        }
        with open(self.config['norm_params_path'], 'w') as f:
            json.dump(norm_params, f, indent=4)
        
        # 数据划分
        test_idx = np.array([54, 800, 1500, 2700])
        train_val_indices = np.setdiff1d(np.arange(n_traces), test_idx)
        val_size = min(675, int(len(train_val_indices) * 0.25))  # 验证集大小
        train_idx, val_idx = train_test_split(train_val_indices, 
                                            test_size=val_size, random_state=42)
        
        # 转换为张量
        def to_tensor(data, indices):
            return torch.FloatTensor(data[indices]).unsqueeze(1)
        
        self.train_data = {
            'seismic': to_tensor(seismic_norm, train_idx),
            'impedance': to_tensor(imp_norm, train_idx)
        }
        self.val_data = {
            'seismic': to_tensor(seismic_norm, val_idx),
            'impedance': to_tensor(imp_norm, val_idx)
        }
        self.test_data = {
            'seismic': to_tensor(seismic_norm, test_idx),
            'impedance': to_tensor(imp_norm, test_idx)
        }
        
        print(f"数据加载完成: 训练集{len(train_idx)}道, 验证集{len(val_idx)}道, 测试集{len(test_idx)}道")
        return dt
    
    def build_model(self):
        """构建FCN模型"""
        print("正在构建模型...")
        
        class InceptionModule(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.branch1 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )
                self.branch2 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(out_channels//4, out_channels//4, 3, padding=1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )
                self.branch3 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(out_channels//4, out_channels//4, 5, padding=2),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )
                self.branch4 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 3, padding=2, dilation=2),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )
            
            def forward(self, x):
                return torch.cat([
                    self.branch1(x), self.branch2(x),
                    self.branch3(x), self.branch4(x)
                ], dim=1)
        
        class FCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.pool1_factor = 4
                self.pool2_factor = 5
                
                # 编码器
                self.enc1 = nn.Sequential(
                    InceptionModule(1, 64),
                    nn.MaxPool1d(self.pool1_factor)
                )
                self.enc2 = nn.Sequential(
                    InceptionModule(64, 128),
                    nn.MaxPool1d(self.pool2_factor)
                )
                self.enc3 = InceptionModule(128, 256)
                
                # 解码器
                self.dec1_conv = nn.Sequential(
                    nn.Conv1d(256+128, 128, 5, padding=2),
                    nn.LeakyReLU(0.1)
                )
                self.dec2_conv = nn.Sequential(
                    nn.Conv1d(128+64, 64, 9, padding=4),
                    nn.LeakyReLU(0.1)
                )
                self.final = nn.Conv1d(64, 1, 15, padding=7)
                
                # 残差连接
                total_pool_factor = self.pool1_factor * self.pool2_factor
                self.residual_conv = nn.Sequential(
                    nn.Conv1d(1, 256, 3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.MaxPool1d(total_pool_factor)
                )
            
            def forward(self, x):
                original_length = x.shape[2]
                
                # 编码
                e1 = self.enc1(x)
                e2 = self.enc2(e1)
                e3_features = self.enc3(e2)
                
                # 残差连接
                res = self.residual_conv(x)
                if res.shape[2] != e3_features.shape[2]:
                    res = F.interpolate(res, size=e3_features.shape[2], 
                                      mode='linear', align_corners=False)
                e3 = e3_features + res
                
                # 解码
                target_size_d1 = e1.shape[2]
                d1_up = F.interpolate(e3, size=target_size_d1, 
                                    mode='linear', align_corners=False)
                e2_up_for_dec1 = F.interpolate(e2, size=target_size_d1, 
                                             mode='linear', align_corners=False)
                d1_cat = torch.cat([d1_up, e2_up_for_dec1], dim=1)
                d1 = self.dec1_conv(d1_cat)
                
                target_size_d2 = original_length
                d2_up = F.interpolate(d1, size=target_size_d2, 
                                    mode='linear', align_corners=False)
                e1_up_for_dec2 = F.interpolate(e1, size=target_size_d2, 
                                             mode='linear', align_corners=False)
                d2_cat = torch.cat([d2_up, e1_up_for_dec2], dim=1)
                d2 = self.dec2_conv(d2_cat)
                
                out = self.final(d2)
                
                # 动态填充
                current_length = out.shape[2]
                if current_length < original_length:
                    padding_needed = original_length - current_length
                    out = F.pad(out, (0, padding_needed))
                elif current_length > original_length:
                    out = out[:, :, :original_length]
                
                return out
        
        self.model = FCN().to(self.device)
        print(f"模型构建完成，参数量: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train(self):
        """训练模型"""
        print("开始训练...")
        
        # 优化器和损失函数
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), 
                              lr=self.config['learning_rate'], 
                              weight_decay=self.config['weight_decay'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # 数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            self.train_data['seismic'], self.train_data['impedance'])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        def compute_metrics(pred, true):
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            pcc = np.corrcoef(pred.ravel(), true.ravel())[0,1]
            r2 = r2_score(true.ravel(), pred.ravel())
            return pcc, r2
        
        for epoch in range(self.config['max_epochs']):
            # 训练阶段
            self.model.train()
            epoch_loss = 0.0
            preds, labels = [], []
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                
                epoch_loss += loss.item()
                preds.append(outputs.detach().cpu())
                labels.append(targets.detach().cpu())
            
            # 计算训练指标
            preds = torch.cat(preds)
            labels = torch.cat(labels)
            train_pcc, train_r2 = compute_metrics(preds, labels)
            self.metrics['train_loss'].append(epoch_loss/len(train_loader))
            self.metrics['train_pcc'].append(train_pcc)
            self.metrics['train_r2'].append(train_r2)
            
            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                val_inputs = self.val_data['seismic'].to(self.device)
                val_pred = self.model(val_inputs)
                val_loss = criterion(val_pred, self.val_data['impedance'].to(self.device))
                
                val_pcc, val_r2 = compute_metrics(val_pred.cpu(), self.val_data['impedance'])
                self.metrics['val_loss'].append(val_loss.item())
                self.metrics['val_pcc'].append(val_pcc)
                self.metrics['val_r2'].append(val_r2)
            
            # 学习率调整
            scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                torch.save(self.model.state_dict(), self.config['model_save_path'])
                print(f"Epoch {epoch+1}: 发现新的最佳模型", flush=True)
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= 15:
                print("早停触发")
                break
            
            # 每轮都打印进度
            print(f'Epoch {epoch+1:03d}/{self.config["max_epochs"]} | '
                  f'Loss: {self.metrics["train_loss"][-1]:.4f}/{self.metrics["val_loss"][-1]:.4f} | '
                  f'PCC: {train_pcc:.3f}/{val_pcc:.3f} | '
                  f'R2: {train_r2:.3f}/{val_r2:.3f} | '
                  f'LR: {optimizer.param_groups[0]["lr"]:.6f}', flush=True)
        
        print("训练完成!")
    
    def predict(self, seismic_data):
        """预测波阻抗"""
        self.model.eval()
        with torch.no_grad():
            # 确保输入数据是正确的3D张量格式 [batch, channels, length]
            if len(seismic_data.shape) == 2:
                # 如果是2D数据 [traces, samples]，添加通道维度
                input_tensor = torch.FloatTensor(seismic_data).unsqueeze(1).to(self.device)
            elif len(seismic_data.shape) == 3:
                # 如果已经是3D数据，直接使用
                input_tensor = torch.FloatTensor(seismic_data).to(self.device)
            else:
                raise ValueError(f"不支持的输入数据维度: {seismic_data.shape}")
            
            prediction = self.model(input_tensor).cpu().numpy().squeeze()
        return prediction
    
    def visualize_results(self, prediction, dt, save_path=None):
        """可视化结果"""
        n_samples = prediction.shape[1]
        time_axis_ms = np.arange(n_samples) * dt * 1000
        
        vmin = np.percentile(prediction, 4)
        vmax = np.percentile(prediction, 97)
        
        plt.figure(figsize=(18, 6))
        im = plt.imshow(prediction.T, aspect='auto', cmap='gist_rainbow',
                       extent=[0, prediction.shape[0], time_axis_ms[-1], time_axis_ms[0]],
                       origin='upper', vmin=vmin, vmax=vmax)
        
        cbar = plt.colorbar(im, extend='both')
        cbar.set_label('波阻抗 (m/s·g/cm³)', rotation=270, labelpad=25)
        
        plt.xlabel('道号', fontsize=12)
        plt.ylabel('时间 (ms)', fontsize=12)
        plt.title('FCN预测波阻抗剖面', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()

def find_data_files():
    """自动查找数据文件"""
    possible_paths = [
        'data/extracted_synthetic/SYNTHETIC_time.segy',
        './data/extracted_synthetic/SYNTHETIC_time.segy',
        '../data/extracted_synthetic/SYNTHETIC_time.segy',
        'Marmousi2/data/extracted_synthetic/SYNTHETIC_time.segy',
        'data/SYNTHETIC.segy',
        './data/SYNTHETIC.segy',
    ]
    
    seismic_path = None
    impedance_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            # 验证文件可读性
            try:
                with segyio.open(path, "r", ignore_geometry=True) as f:
                    if f.tracecount > 0:
                        seismic_path = path
                        # 查找对应的impedance文件
                        base_dir = os.path.dirname(os.path.dirname(path)) if 'extracted_synthetic' in path else os.path.dirname(path)
                        impedance_candidates = [
                            os.path.join(base_dir, 'impedance.txt'),
                            path.replace('SYNTHETIC.segy', 'impedance.txt').replace('SYNTHETIC_time.segy', 'impedance.txt'),
                        ]
                        for imp_path in impedance_candidates:
                            if os.path.exists(imp_path):
                                impedance_path = imp_path
                                break
                        if seismic_path and impedance_path:
                            break
            except:
                continue
    
    return seismic_path, impedance_path

def main():
    """主函数"""
    print("="*60)
    print("开始训练 - Marmousi2波阻抗反演")
    print("="*60)
    
    # 自动查找数据文件
    seismic_path, impedance_path = find_data_files()
    
    if not seismic_path or not impedance_path:
        print("错误: 找不到数据文件")
        print("请确保以下文件存在:")
        print("- SYNTHETIC.segy (地震数据)")
        print("- impedance.txt (波阻抗数据)")
        print("\n可能的文件位置:")
        print("- data/SYNTHETIC.segy")
        print("- Marmousi2/data/SYNTHETIC.segy")
        return
    
    print(f"找到地震数据: {seismic_path}")
    print(f"找到波阻抗数据: {impedance_path}")
    
    # 配置参数
    config = {
        'seismic_path': seismic_path,
        'impedance_path': impedance_path,
        'norm_params_path': 'data/norm_params.json',
        'model_save_path': 'data/model.pth',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 8,
        'max_epochs': 120
    }
    
    # 创建输出目录
    os.makedirs('data', exist_ok=True)
    
    # 初始化系统
    system = SeismicImpedanceInversion(config)
    
    # 加载数据
    dt = system.load_data()
    
    # 构建模型
    system.build_model()
    
    # 训练模型
    system.train()
    
    # 预测和可视化
    print("正在进行预测...")
    prediction = system.predict(system.test_data['seismic'])
    system.visualize_results(prediction, dt, 'data/prediction_result.png')
    
    print("复现完成!")

if __name__ == "__main__":
    main()
