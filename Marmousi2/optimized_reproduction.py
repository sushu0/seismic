#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import segyio
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn.functional as F

# 设置字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

class OptimizedSeismicImpedanceInversion:
    """优化的波阻抗反演类 - 完全基于原始高性能实现"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.metrics = {
            'train_loss': [], 'val_loss': [],
            'train_pcc': [], 'val_pcc': [],
            'train_r2': [], 'val_r2': []
        }
        
    def safe_segy_load(self, path):
        """安全加载SEGY文件 - 完全复制原始实现"""
        path = Path(path)
        try:
            with segyio.open(str(path), "r", ignore_geometry=True) as f:
                if f.tracecount == 0:
                    raise ValueError("SEGY文件不包含任何地震道")
                return np.stack([f.trace[i] for i in range(f.tracecount)])
        except segyio.SegyError as e:
            print(f"SEGY文件格式错误：{str(e)}")
            exit(2)
        except Exception as e:
            print(f"加载{path.name}失败：{str(e)}")
            exit(3)

    def safe_txt_load(self, path):
        """安全加载TXT文件 - 完全复制原始实现"""
        path = Path(path)
        try:
            data = np.loadtxt(path, usecols=4, skiprows=1)
            if data.size == 0:
                raise ValueError("TXT文件无有效数据")
            return data.reshape(2721, 701)  # 根据实际维度调整
        except Exception as e:
            print(f"加载{path.name}失败：{str(e)}")
            exit(4)
    
    def load_data(self):
        """加载和预处理数据 - 完全复制原始实现"""
        print("正在加载数据...")
        
        # 加载数据
        seismic_data = self.safe_segy_load(self.config['seismic_path'])
        impedance_data = self.safe_txt_load(self.config['impedance_path'])

        # 数据归一化 - 完全复制原始实现
        def normalize(data):
            data_min, data_max = np.min(data), np.max(data)
            return (data - data_min) / (data_max - data_min), data_min, data_max

        seismic_norm, s_min, s_max = normalize(seismic_data)
        imp_norm, imp_min, imp_max = normalize(impedance_data)

        # 保存归一化参数
        norm_params = {
            's_min': float(s_min),
            's_max': float(s_max),
            'imp_min': float(imp_min),
            'imp_max': float(imp_max)
        }
        with open(self.config['norm_params_path'], 'w') as f:
            json.dump(norm_params, f, indent=4)
        print(f"√ 归一化参数已保存至：{self.config['norm_params_path']}")

        # 数据划分 - 完全复制原始实现
        test0_idx = np.array([54, 800, 1500, 2700])  
        train_val_indices = np.setdiff1d(np.arange(2701), test0_idx)
        train_idx, val_idx = train_test_split(train_val_indices, test_size=675, random_state=42)
        test_idx = test0_idx  # 直接使用预设道号作为测试集
        
        def to_tensor(data, indices):
            return torch.FloatTensor(data[indices]).unsqueeze(1)
        
        # 转换为PyTorch张量
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
        return seismic_data, impedance_data
    
    def build_model(self):
        """构建FCN模型 - 完全复制原始实现"""
        print("正在构建模型...")
        
        class InceptionModule(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                # 分支1：1x1卷积
                self.branch1 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )

                # 分支2：3x3卷积
                self.branch2 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(out_channels//4, out_channels//4, 3, padding=1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )

                # 分支3：5x5卷积
                self.branch3 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 1),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(out_channels//4, out_channels//4, 5, padding=2),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )

                # 分支4：3x3空洞卷积
                self.branch4 = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels//4, 3, padding=2, dilation=2),
                    nn.BatchNorm1d(out_channels//4),
                    nn.LeakyReLU(0.1)
                )

            def forward(self, x):
                return torch.cat([
                    self.branch1(x),
                    self.branch2(x),
                    self.branch3(x),
                    self.branch4(x)
                ], dim=1)

        class FCN(nn.Module):
            def __init__(self):
                super().__init__()

                # 编码器
                self.pool1_factor = 4
                self.pool2_factor = 5
                
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

                # 残差路径
                total_pool_factor = self.pool1_factor * self.pool2_factor
                self.residual_conv = nn.Sequential(
                    nn.Conv1d(1, 256, 3, padding=1),
                    nn.BatchNorm1d(256),
                    nn.MaxPool1d(total_pool_factor)
                )

            def forward(self, x):
                # 获取原始输入长度
                original_length = x.shape[2]

                # --- 编码 ---
                e1 = self.enc1(x)
                e2 = self.enc2(e1)
                e3_features = self.enc3(e2)

                # --- 残差连接 ---
                res = self.residual_conv(x)
                if res.shape[2] != e3_features.shape[2]:
                    res = F.interpolate(res, size=e3_features.shape[2], mode='linear', align_corners=False)
                e3 = e3_features + res

                # --- 解码 - 完全复制原始逻辑 ---
                target_size_dec1_in = e1.shape[2]
                d1_up = F.interpolate(e3, size=target_size_dec1_in, mode='linear', align_corners=False)
                e2_up_for_dec1 = F.interpolate(e2, size=target_size_dec1_in, mode='linear', align_corners=False)
                d1_cat = torch.cat([d1_up, e2_up_for_dec1], dim=1)
                d1 = self.dec1_conv(d1_cat)

                # 3. 上采样 d1 到 x 的尺寸 (original_length)
                target_size_d2 = original_length
                d2_up = F.interpolate(d1, size=target_size_d2, mode='linear', align_corners=False)

                # 4. 连接 d2_up 和 e1 (跳跃连接)
                e1_up_for_dec2 = F.interpolate(e1, size=target_size_d2, mode='linear', align_corners=False)
                d2_cat = torch.cat([d2_up, e1_up_for_dec2], dim=1)
                d2 = self.dec2_conv(d2_cat)

                # --- 最终输出 ---
                out = self.final(d2)

                # --- 动态填充 ---
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
        """训练模型 - 完全复制原始实现"""
        print("开始训练...")
        
        # 优化器和损失函数 - 完全复制原始实现
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

        # 数据加载器
        train_dataset = torch.utils.data.TensorDataset(
            self.train_data['seismic'], self.train_data['impedance'])
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=8, shuffle=True)
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        
        def compute_metrics(pred, true):
            pred = pred.detach().cpu().numpy()
            true = true.detach().cpu().numpy()
            pcc = np.corrcoef(pred.ravel(), true.ravel())[0,1]
            r2 = r2_score(true.ravel(), pred.ravel())
            return pcc, r2

        try:
            for epoch in range(120):
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
                    print(f"Epoch {epoch+1}: 发现新的最佳模型")
                else:
                    early_stop_counter += 1
                
                if early_stop_counter >= 15:
                    print("\nEarly stopping triggered")
                    break
                
                # 打印进度
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1:03d} | '
                        f'Train Loss: {self.metrics["train_loss"][-1]:.4f} | '
                        f'Val Loss: {self.metrics["val_loss"][-1]:.4f} | '
                        f'PCC: {train_pcc:.3f}/{val_pcc:.3f} | '
                        f'R²: {train_r2:.3f}/{val_r2:.3f}')

        except KeyboardInterrupt:
            print("\n检测到用户中断，正在进入可视化和保存...")
        
        print("训练完成!")
    
    def predict(self, seismic_data):
        """预测波阻抗"""
        self.model.eval()
        with torch.no_grad():
            if len(seismic_data.shape) == 2:
                input_tensor = torch.FloatTensor(seismic_data).unsqueeze(1).to(self.device)
            elif len(seismic_data.shape) == 3:
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
        'data/SYNTHETIC.segy',
        './data/SYNTHETIC.segy',
        '../data/SYNTHETIC.segy',
        'Marmousi2/data/SYNTHETIC.segy'
    ]
    
    seismic_path = None
    impedance_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            seismic_path = path
            impedance_candidates = [
                path.replace('SYNTHETIC.segy', 'impedance.txt'),
                os.path.join(os.path.dirname(path), 'impedance.txt')
            ]
            for imp_path in impedance_candidates:
                if os.path.exists(imp_path):
                    impedance_path = imp_path
                    break
            break
    
    return seismic_path, impedance_path

def main():
    """主函数"""
    print("优化的波阻抗反演复现 - 基于原始高性能实现")
    print("=" * 60)
    
    # 自动查找数据文件
    seismic_path, impedance_path = find_data_files()
    
    if not seismic_path or not impedance_path:
        print("错误: 找不到数据文件")
        print("请确保以下文件存在:")
        print("- SYNTHETIC.segy (地震数据)")
        print("- impedance.txt (波阻抗数据)")
        return
    
    print(f"找到地震数据: {seismic_path}")
    print(f"找到波阻抗数据: {impedance_path}")
    
    # 配置参数
    config = {
        'seismic_path': seismic_path,
        'impedance_path': impedance_path,
        'norm_params_path': 'data/norm_params.json',
        'model_save_path': 'data/optimized_model.pth',
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 8,
        'max_epochs': 120
    }
    
    # 创建输出目录
    os.makedirs('data', exist_ok=True)
    
    # 初始化系统
    system = OptimizedSeismicImpedanceInversion(config)
    
    # 加载数据
    seismic_data, impedance_data = system.load_data()
    
    # 构建模型
    system.build_model()
    
    # 训练模型
    system.train()
    
    # 预测和可视化
    print("正在进行预测...")
    prediction = system.predict(system.test_data['seismic'])
    
    # 获取采样间隔
    with segyio.open(seismic_path, "r", ignore_geometry=True) as f:
        dt = f.bin[segyio.BinField.Interval] * 1e-6
    
    system.visualize_results(prediction, dt, 'data/optimized_prediction_result.png')
    
    print("优化复现完成!")
    print(f"最终性能指标:")
    print(f"  训练损失: {system.metrics['train_loss'][-1]:.6f}")
    print(f"  验证损失: {system.metrics['val_loss'][-1]:.6f}")
    print(f"  训练PCC: {system.metrics['train_pcc'][-1]:.6f}")
    print(f"  验证PCC: {system.metrics['val_pcc'][-1]:.6f}")
    print(f"  训练R²: {system.metrics['train_r2'][-1]:.6f}")
    print(f"  验证R²: {system.metrics['val_r2'][-1]:.6f}")

if __name__ == "__main__":
    main()

