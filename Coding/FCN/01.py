# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import segyio
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn.functional as F

# 设置标准输出编码为UTF-8，避免Windows控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 输入文件路径配置
seismic_data_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'  # 地震数据文件路径 (SEGY格式)
impedance_data_path = r'D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt'  # 波阻抗数据文件路径 (TXT格式)

# 输出文件路径和文件名配置
output_image_dir = 'zmy_data/01/output_images'  # PNG格式图像输出目录
svg_image_dir = 'zmy_data/01/svg_images'  # SVG格式图像输出目录
model_save_path = 'zmy_data/01/FCN_01_model.pth'  # 训练好的模型保存路径
norm_params_path = 'zmy_data/01/norm_params.json'  # 数据归一化参数保存路径
loss_curve_filename = 'loss_curve'  # 损失曲线图像文件名
pcc_curve_filename = 'pcc_curve'  # 皮尔逊相关系数曲线图像文件名
r2_curve_filename = 'r2_curve'  # R平方值曲线图像文件名
test_visualization_filename = '01_30Hz'  # 测试集可视化对比图像文件名

# 设置字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# 创建输出目录
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(svg_image_dir, exist_ok=True)

#%% ****************** 数据加载与预处理 ******************
print("Loading data...")

def load_seismic_data(path):
    """加载SEGY格式地震数据"""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([f.trace[i] for i in range(100)])

def load_impedance_data(path):
    """加载波阻抗数据"""
    return np.loadtxt(path, usecols=4, skiprows=1).reshape(100, 10001)

# 加载数据
seismic_data = load_seismic_data(seismic_data_path)
impedance_data = load_impedance_data(impedance_data_path)
# 数据标准化
def normalize(data):
    data_min, data_max = np.min(data), np.max(data)
    return (data - data_min) / (data_max - data_min), data_min, data_max
#有区别


seismic_norm, s_min, s_max = normalize(seismic_data)
imp_norm, imp_min, imp_max = normalize(impedance_data)

# 保存归一化参数
norm_params = {
    's_min': float(s_min),
    's_max': float(s_max),
    'imp_min': float(imp_min),
    'imp_max': float(imp_max)
}
with open(norm_params_path, 'w') as f:
    json.dump(norm_params, f, indent=4)

#%% ****************** 数据集划分 ******************
indices = np.random.permutation(100)
train_idx, val_idx, test_idx = indices[:60], indices[60:80], indices[80:]

def to_tensor(data, indices):
    return torch.FloatTensor(data[indices]).unsqueeze(1)

# 转换为PyTorch张量
trainX = to_tensor(seismic_norm, train_idx)
valX = to_tensor(seismic_norm, val_idx)
testX = to_tensor(seismic_norm, test_idx)

trainImp = to_tensor(imp_norm, train_idx)
valImp = to_tensor(imp_norm, val_idx)
testImp = to_tensor(imp_norm, test_idx)

#%% ****************** 优化后的FCN模型定义 ******************
class AttentionModule(nn.Module):
    """注意力机制模块"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = (avg_out + max_out).unsqueeze(-1)
        return x * attention

class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，需要1x1卷积调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class OptimizedFCN(nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        
        # 编码器 - 使用残差块和注意力机制
        self.enc1 = nn.Sequential(
            nn.Conv1d(1, 64, 15, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            AttentionModule(64),
            nn.MaxPool1d(4),
            nn.Dropout(dropout_rate)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv1d(64, 128, 9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            AttentionModule(128),
            nn.MaxPool1d(5),
            nn.Dropout(dropout_rate)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv1d(128, 256, 5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            ResidualBlock(256, 256),
            AttentionModule(256),
            nn.Dropout(dropout_rate)
        )
        
        # 瓶颈层 - 增加深度
        self.bottleneck = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            ResidualBlock(512, 512),
            AttentionModule(512),
            nn.Conv1d(512, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 解码器 - 改进的上采样和跳跃连接
        self.up1 = nn.Upsample(scale_factor=5, mode='linear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv1d(256+128, 128, 5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            AttentionModule(128),
            nn.Dropout(dropout_rate)
        )
        
        self.up2 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv1d(128+64, 64, 9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            AttentionModule(64),
            nn.Dropout(dropout_rate)
        )
        
        # 最终输出层
        self.final = nn.Sequential(
            nn.Conv1d(64, 32, 15, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )
        
        # 全局残差连接
        self.global_residual = nn.Sequential(
            nn.Conv1d(1, 256, 1),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(4),
            nn.MaxPool1d(5)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 编码阶段保存特征
        e1 = self.enc1(x)       # [B,64,2500]
        e2 = self.enc2(e1)      # [B,128,500]
        e3 = self.enc3(e2)      # [B,256,500]
        
        # 瓶颈层处理
        bottleneck = self.bottleneck(e3)  # [B,256,500]
        
        # 全局残差连接
        res = self.global_residual(x)  # [B,256,500]
        bottleneck += res
        
        # 解码阶段（结合跳跃连接）
        d1 = self.up1(bottleneck)                    # [B,256,2500]
        d1 = torch.cat([d1, e2.repeat(1,1,5)], dim=1)  # 对齐维度
        d1 = self.dec1(d1)                           # [B,128,2500]
        
        d2 = self.up2(d1)                            # [B,128,10000]
        d2 = torch.cat([d2, e1.repeat(1,1,4)], dim=1)  # 对齐
        d2 = self.dec2(d2)                           # [B,64,10000]
        
        # 最终输出
        out = self.final(d2)                         # [B,1,10000]
        out = F.pad(out, (0,1))                      # 补齐到10001
        return out


#%% ****************** 训练配置 ******************
# 设备检测和配置
def check_device():
    """检测并配置训练设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"[OK] GPU检测成功: {gpu_name}")
        print(f"   GPU内存: {gpu_memory:.1f} GB")
        print(f"   CUDA版本: {torch.version.cuda}")
        print(f"   PyTorch版本: {torch.__version__}")
    else:
        device = torch.device('cpu')
        print("[WARNING] GPU不可用，使用CPU训练")
        print(f"   CPU核心数: {os.cpu_count()}")
        print(f"   PyTorch版本: {torch.__version__}")

    return device

# 设备配置
device = check_device()
print(f"\n[START] 训练设备: {device}")
print("="*50)

model = OptimizedFCN().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 数据加载器
train_dataset = torch.utils.data.TensorDataset(trainX, trainImp)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

#%% ****************** 训练循环 ******************
print(f"[INFO] 开始训练...")
print(f"   训练样本数: {len(trainX)}")
print(f"   验证样本数: {len(valX)}")
print(f"   测试样本数: {len(testX)}")
print(f"   批次大小: 8")
print(f"   最大训练轮数: 150")
print("="*50)

metrics = {
    'train_loss': [], 'val_loss': [],
    'train_pcc': [], 'val_pcc': [],
    'train_r2': [], 'val_r2': []
}

best_val_loss = float('inf')

def compute_metrics(pred, true):
    pred = pred.detach().cpu().numpy()
    true = true.detach().cpu().numpy()
    pcc = np.corrcoef(pred.ravel(), true.ravel())[0,1]
    r2 = r2_score(true.ravel(), pred.ravel())
    return pcc, r2

for epoch in range(150):
    # 训练阶段
    model.train()
    epoch_loss = 0.0
    preds, labels = [], []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        epoch_loss += loss.item()
        preds.append(outputs.detach().cpu())
        labels.append(targets.detach().cpu())
    
    # 计算训练指标
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    train_pcc, train_r2 = compute_metrics(preds, labels)
    metrics['train_loss'].append(epoch_loss/len(train_loader))
    metrics['train_pcc'].append(train_pcc)
    metrics['train_r2'].append(train_r2)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_inputs = valX.to(device)
        val_pred = model(val_inputs)
        val_loss = criterion(val_pred, valImp.to(device))
        
        val_pcc, val_r2 = compute_metrics(val_pred.cpu(), valImp)
        metrics['val_loss'].append(val_loss.item())
        metrics['val_pcc'].append(val_pcc)
        metrics['val_r2'].append(val_r2)
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}: 发现新的最佳模型")
    
    # 打印进度
    print(f'Epoch {epoch+1:03d} | '
          f'Train Loss: {metrics["train_loss"][-1]:.4f} | '
          f'Val Loss: {metrics["val_loss"][-1]:.4f} | '
          f'PCC: {train_pcc:.3f}/{val_pcc:.3f} | '
          f'R2: {train_r2:.3f}/{val_r2:.3f}')

print(f"\n[DONE] 训练完成!")
print(f"   最终验证损失: {best_val_loss:.4f}")
print(f"   最佳模型已保存至: {model_save_path}")
print("="*50)

#%% ****************** 可视化与保存 ******************
# 统一绘图参数配置
plot_config = {
    'figure.dpi': 300,
    'lines.linewidth': 1.5,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.3,
    'legend.frameon': False,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
}
plt.rcParams.update(plot_config)

#%% 训练曲线可视化
#%% ****************** 可视化模块 ******************
def plot_curves(train_vals, val_vals, title, ylabel, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(train_vals, label='Train')
    plt.plot(val_vals, label='Validation')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_image_dir}/{filename}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{svg_image_dir}/{filename}.svg', format='svg')
    plt.close()

# 绘制训练曲线
plot_curves(metrics['train_loss'], metrics['val_loss'],
        'Loss Curve', 'MSE Loss', loss_curve_filename)
plot_curves(metrics['train_pcc'], metrics['val_pcc'],
        'Pearson Correlation', 'PCC', pcc_curve_filename)
plot_curves(metrics['train_r2'], metrics['val_r2'],
        'R-squared', 'R2', r2_curve_filename)

# **********************测试集可视化对比**************************
with segyio.open('zmy_data/01/data/01_30Hz_re.sgy', "r", ignore_geometry=True) as f_seismic:
    sample_interval = f_seismic.bin[segyio.BinField.Interval]
dt = sample_interval * 1e-6
time = np.linspace(0, (testX.shape[2] - 1) * dt, testX.shape[2])
sample_numbers = np.array([10, 11, 14, 15])  # 选择几个样本

# 创建图像和子图
fig, axs = plt.subplots(1, 4, sharey=True)  # 调整图像比例
axs[0].invert_yaxis()  # 反转y轴，使时间从上到下

# 循环绘制每个样本
for i in range(4):
    sample_number = sample_numbers[i]
    TestingSetSeismicTrace = Variable(testX[sample_number:sample_number+1, :, :])
    CNN_ImpedancePrediction = model(TestingSetSeismicTrace.to(device))

    # 还原真实值与预测值
    TestingSetImpedanceTrace = testImp[sample_number, :].numpy().flatten() * (imp_max - imp_min) + imp_min
    CNN_ImpedancePrediction = CNN_ImpedancePrediction.data.cpu().numpy().flatten() * (imp_max - imp_min) + imp_min

    # 绘制真实值与预测值
    line1, = axs[i].plot(TestingSetImpedanceTrace, time, 'r-', label='真实值', linewidth=1.2)
    line2, = axs[i].plot(CNN_ImpedancePrediction, time, 'k--', label='FCN预测值', linewidth=0.6)

    axs[i].set_xlabel('波阻抗')
    if i == 0:
        axs[i].set_ylabel('时间/ms')

# 设置图例
axs[0].legend(loc='upper right', bbox_to_anchor=(1.05, 1))

# 保存测试集可视化对比图
plt.savefig(f'{output_image_dir}/{test_visualization_filename}.png', dpi=300, bbox_inches='tight')
plt.savefig(f'{svg_image_dir}/{test_visualization_filename}.svg', format='svg')

plt.show()