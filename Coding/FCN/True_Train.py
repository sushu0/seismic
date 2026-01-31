import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import segyio
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch.nn.functional as F
# 设置字体和绘图参数  
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# 文件路径配置
# output_image_dir = 'SEISMIC_CODING/output_images/HuCeng/FCN_30Hz1'
# svg_image_dir = 'SEISMIC_CODING/svg_images/HuCeng/FCN_30Hz1'
# model_save_path = 'SEISMIC_CODING/Model/HuCeng/FCN_30Hz/FCN_30Hz_model1.pth'
# norm_params_path = 'SEISMIC_CODING/Model/HuCeng/FCN_30Hz/norm_params1.json'

output_image_dir = 'SEISMIC_CODING/output_images/Hu6/FCN_20Hz'
svg_image_dir = 'SEISMIC_CODING/svg_images/Hu6/FCN_20Hz'
model_save_path = 'SEISMIC_CODING/Model/Hu6/FCN_20Hz/FCN_2s0Hz_model.pth'
norm_params_path = 'SEISMIC_CODING/Model/Hu6/FCN_20Hz/norm_params.json'

# output_image_dir = 'SEISMIC_CODING/output_images/FCN_40Hz'
# svg_image_dir = 'SEISMIC_CODING/svg_images/FCN_40Hz'
# model_save_path = 'SEISMIC_CODING/Model/FCN_40Hz/FCN_40Hz_model.pth'
# norm_params_path = 'SEISMIC_CODING/Model/FCN_40Hz/norm_params.json'
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(svg_image_dir, exist_ok=True)
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # 新增
os.makedirs(os.path.dirname(norm_params_path), exist_ok=True)  # 新增
#%% ****************** 数据加载与预处理 ******************
print("Loading data...")

def load_seismic_data(path):
    """加载SEGY格式地震数据"""
    with segyio.open(path, "r", ignore_geometry=True) as f:
        return np.stack([f.trace[i] for i in range(500)])

def load_impedance_data(path):
    """加载波阻抗数据"""
    return np.loadtxt(path, usecols=4, skiprows=1).reshape(500, 20001)

# 加载数据
# seismic_data = load_seismic_data('SEISMIC_CODING/Data/ThreeLevel/40Hz.segy')
# impedance_data = load_impedance_data('SEISMIC_CODING/Data/ThreeLevel/impedance.txt')
seismic_data = load_seismic_data('C:/Users/ZYH/Desktop/python_code/SEISMIC_CODING/Data/Hu6/20Hz.segy')
impedance_data = load_impedance_data('C:/Users/ZYH/Desktop/python_code/SEISMIC_CODING/Data/Hu6/impedance.txt')
# 数据标准化
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
with open(norm_params_path, 'w') as f:
    json.dump(norm_params, f, indent=4)

#%% ****************** 数据集划分 ******************
indices = np.random.permutation(500)
train_idx, val_idx, test_idx = indices[:300], indices[300:400], indices[400:]

def to_tensor(data, indices):
    return torch.FloatTensor(data[indices]).unsqueeze(1)

# 转换为PyTorch张量
trainX = to_tensor(seismic_norm, train_idx)
valX = to_tensor(seismic_norm, val_idx)
testX = to_tensor(seismic_norm, test_idx)

trainImp = to_tensor(imp_norm, train_idx)
valImp = to_tensor(imp_norm, val_idx)
testImp = to_tensor(imp_norm, test_idx)

#%% ****************** 改进的Inception模块 ******************
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

#%% ****************** 改进的Inception-FCN模型 ******************
class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器
        self.enc1 = nn.Sequential(
            InceptionModule(1, 64),
            nn.MaxPool1d(4)
        )
        self.enc2 = nn.Sequential(
            InceptionModule(64, 128),
            nn.MaxPool1d(5)
        )
        self.enc3 = nn.Sequential(
            InceptionModule(128, 256),
            nn.AdaptiveAvgPool1d(500)
        )
        
        # 解码器（调整上采样倍数）
        self.up1 = nn.Upsample(scale_factor=10, mode='linear', align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv1d(256+128, 128, 5, padding=2),
            nn.LeakyReLU(0.1)
        )
        self.up2 = nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.dec2 = nn.Sequential(
            nn.Conv1d(128+64, 64, 9, padding=4),
            nn.LeakyReLU(0.1)
        )
        self.final = nn.Conv1d(64, 1, 15, padding=7)
        
        # 残差路径
        self.residual = nn.Sequential(
            nn.Conv1d(1, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(4),
            nn.AdaptiveMaxPool1d(500)
        )

    def forward(self, x):
        # 编码
        e1 = self.enc1(x)    # [B,64,5000]
        e2 = self.enc2(e1)   # [B,128,1000]
        e3 = self.enc3(e2)   # [B,256,500]
        
        # 残差连接
        res = self.residual(x)  # [B,256,500]
        e3 += res
        
        # 解码
        d1 = self.up1(e3)
        e2_up = F.interpolate(e2, size=d1.shape[2], mode='linear', align_corners=False)
        d1 = torch.cat([d1, e2_up], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        e1_up = F.interpolate(e1, size=d2.shape[2], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e1_up], dim=1)
        d2 = self.dec2(d2)
        
        out = self.final(d2)
        out = F.pad(out, (0,1))  # 最终输出20001
        return out


#%% ****************** 训练配置 ******************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FCN().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
# 添加梯度裁剪
nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 数据加载器
train_dataset = torch.utils.data.TensorDataset(trainX, trainImp)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

#%% ****************** 训练循环 ******************
metrics = {
    'train_loss': [], 'val_loss': [],
    'train_pcc': [], 'val_pcc': [],
    'train_r2': [], 'val_r2': []
}

best_val_loss = float('inf')
early_stop_counter = 0

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
    
    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}: 发现新的最佳模型")
    else:
        early_stop_counter += 1
        if early_stop_counter >= 15:
            print("\nEarly stopping triggered")
            break
    
    # 打印进度
    print(f'Epoch {epoch+1:03d} | '
          f'Train Loss: {metrics["train_loss"][-1]:.4f} | '
          f'Val Loss: {metrics["val_loss"][-1]:.4f} | '
          f'PCC: {train_pcc:.3f}/{val_pcc:.3f} | '
          f'R²: {train_r2:.3f}/{val_r2:.3f}')

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
        'Loss Curve', 'MSE Loss', 'loss_curve')
plot_curves(metrics['train_pcc'], metrics['val_pcc'], 
        'Pearson Correlation', 'PCC', 'pcc_curve')
plot_curves(metrics['train_r2'], metrics['val_r2'], 
        'R-squared', 'R²', 'r2_curve')

# **********************测试集可视化对比**************************
with segyio.open('C:/Users/ZYH/Desktop/python_code/SEISMIC_CODING/Data/Hu6/20Hz.segy', "r", ignore_geometry=True) as f_seismic:
    sample_interval = f_seismic.bin[segyio.BinField.Interval]
# 自动计算时间轴（假设dt正确）
dt = sample_interval * 1e-6
time = np.linspace(0, (testX.shape[2] - 1) * dt, testX.shape[2])  # 自动适配20001
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
    line2, = axs[i].plot(CNN_ImpedancePrediction, time, 'k--', label='FCN预测值', linewidth=1.2)

    axs[i].set_xlabel('波阻抗')
    if i == 0:
        axs[i].set_ylabel('时间/ms')

# 设置图例
axs[0].legend(loc='upper right', bbox_to_anchor=(1.05, 1))

# 保存图像
output_image_path = os.path.join(output_image_dir, 'test_comparison.png')
svg_image_path = os.path.join(svg_image_dir, 'test_comparison.svg')
plt.tight_layout()
plt.savefig(output_image_path, dpi=300)
plt.savefig(svg_image_path, format='svg')

print(f"预测结果图像已保存至 {output_image_dir} 和 {svg_image_dir}.")
plt.show()