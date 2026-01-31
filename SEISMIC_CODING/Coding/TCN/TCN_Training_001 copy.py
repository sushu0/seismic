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
from torch.nn.utils import weight_norm
# 设置字体和绘图参数
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = ['Times New Roman', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# 文件路径配置
output_image_dir = 'SEISMIC_CODING/output_images/TCN_30Hz'
svg_image_dir = 'SEISMIC_CODING/svg_images/TCN_30Hzz'
model_save_path = 'SEISMIC_CODING/Model/TCN_30Hz/TCN_30Hz_model.pth'
norm_params_path = 'SEISMIC_CODING/Model/TCN_30Hz/norm_params.json'
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
seismic_data = load_seismic_data('SEISMIC_CODING/Data/ThreeLevel/30Hz.segy')
impedance_data = load_impedance_data('SEISMIC_CODING/Data/ThreeLevel/impedance.txt')

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
indices = np.random.permutation(100)
train_idx, val_idx, test_idx = indices[:60], indices[60:80], indices[80:]
def data_augmentation(data, noise_level=0.05, scale_range=(0.8, 1.2)):
    """添加随机噪声和幅度缩放"""
    # 随机幅度缩放
    scale = np.random.uniform(scale_range[0], scale_range[1])
    data = data * scale
    # 添加高斯噪声
    data += np.random.normal(0, noise_level, data.shape)
    return data

# 修改数据加载部分（在转换为张量之前）
train_seismic = seismic_norm[train_idx]
train_imp = imp_norm[train_idx]

# 对训练数据应用增强
for i in range(len(train_idx)):
    if np.random.rand() > 0.5:  # 50%概率增强
        train_seismic[i] = data_augmentation(train_seismic[i])
        

def to_tensor(data, indices):
    return torch.FloatTensor(data[indices]).unsqueeze(1)

# 转换为PyTorch张量
trainX = torch.FloatTensor(train_seismic).unsqueeze(1)
valX = to_tensor(seismic_norm, val_idx)
testX = to_tensor(seismic_norm, test_idx)

trainImp = to_tensor(imp_norm, train_idx)
valImp = to_tensor(imp_norm, val_idx)
testImp = to_tensor(imp_norm, test_idx)

#%% ****************** 优化后的FCN模型定义 ******************
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, bias=True))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, bias=True))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size - 1)/2 * dilation_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MustafaNet(nn.Module):
    def __init__(self):
        super(MustafaNet, self).__init__()
        self.tcn_local = TemporalConvNet(num_inputs=1, num_channels=[3, 6, 6, 6, 6, 6, 5], kernel_size=9, dropout=0.2)
        self.regression = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

    def forward(self, input):
        out = self.tcn_local(input)
        out = self.regression(out)
        return out            
#%% ****************** 训练配置 ******************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MustafaNet().to(device)
# 修改损失函数
class MultiScaleLoss(nn.Module):
    def __init__(self, weights=[0.5, 0.3, 0.2]):
        super().__init__()
        self.weights = weights
        self.mse = nn.MSELoss()
        
    def forward(self, outputs, targets):
        main_loss = self.mse(outputs[0], targets)
        aux1_loss = self.mse(outputs[1], targets)
        aux2_loss = self.mse(outputs[2], targets)
        return (self.weights[0] * main_loss + 
                self.weights[1] * aux1_loss +
                self.weights[2] * aux2_loss)

# 修改训练配置部分
criterion = MultiScaleLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)  # 替换原有调度器

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
        # 前向传播
        outputs, aux1, aux2 = model(inputs)
        loss = criterion((outputs, aux1, aux2), targets)
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
        val_outputs, val_aux1, val_aux2 = model(val_inputs)
        val_loss = criterion((val_outputs, val_aux1, val_aux2), valImp.to(device))
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
with segyio.open('SEISMIC_CODING/Data/ThreeLevel/30Hz.segy', "r", ignore_geometry=True) as f_seismic:
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
    line2, = axs[i].plot(CNN_ImpedancePrediction, time, 'k--', label='TCN_30Hz预测值', linewidth=0.6)

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