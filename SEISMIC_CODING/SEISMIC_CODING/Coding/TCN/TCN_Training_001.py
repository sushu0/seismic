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
from torch.utils.data import Dataset, DataLoader
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
#%% ****************** 修正后的EnhancedDataset类 ******************
#%% ****************** 修正后的EnhancedDataset类 ******************
class EnhancedDataset(Dataset):
    def __init__(self, data, targets, window_size=512, stride=256):
        """
        参数说明：
        data: 形状应为 (num_samples, 1, total_length)
        targets: 同data形状
        """
        assert data.dim() == 3, f"输入数据应为3维，实际维度: {data.dim()}"
        self.data = data          # (num_samples, 1, total_length)
        self.targets = targets    # (num_samples, 1, total_length)
        self.window_size = window_size
        self.stride = stride
        
        # 生成样本索引 (sample_index, start_position)
        self.indices = self._generate_indices()

    def _generate_indices(self):
        indices = []
        num_samples = self.data.size(0)  # 样本总数
        seq_length = self.data.size(2)   # 时间步总数
        
        for sample_idx in range(num_samples):
            num_windows = (seq_length - self.window_size) // self.stride + 1
            for win_idx in range(num_windows):
                start = win_idx * self.stride
                indices.append( (sample_idx, start) )
        return indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx, start = self.indices[idx]
        end = start + self.window_size
        
        # 数据增强
        gain = torch.FloatTensor(1).uniform_(0.8, 1.2)
        noise = torch.randn(1, self.window_size) * 0.05  # 修正噪声维度
        
        # 正确切片单个样本
        x = self.data[sample_idx, :, start:end]    # 形状 (1, window_size)
        y = self.targets[sample_idx, :, start:end]  # 形状 (1, window_size)
        
        # 应用增强
        x = x * gain + noise
        return x, y


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
trainX = trainX.permute(0, 1, 2)  # 保持原有维度
trainImp = trainImp.permute(0, 1, 2)

# 创建增强数据集
train_dataset = EnhancedDataset(trainX, trainImp, 
                            window_size=512, 
                            stride=256)
# 在创建数据集前添加维度检查
print(f"训练数据形状: {trainX.shape}")  # 应为 (60, 1, 10001)
print(f"目标数据形状: {trainImp.shape}")  # 应为 (60, 1, 10001)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.2):
        super().__init__()
        padding = (kernel_size-1) * dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        if self.downsample:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            ResidualBlock(1, 64, kernel_size=5, dilation=1),
            ResidualBlock(64, 128, kernel_size=5, dilation=2),
            ResidualBlock(128, 256, kernel_size=5, dilation=4),
            nn.Conv1d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        return self.network(x)  # 仅返回单一输出

    
#%% ****************** 训练配置 ******************
# 替换原有训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)  # 改为余弦退火

# 修改数据加载器
train_dataset = EnhancedDataset(trainX, trainImp)
train_loader = DataLoader(train_dataset, 
                        batch_size=16,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True)

# 在模型定义后添加参数初始化
def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

model.apply(init_weights)

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
    model.train()
    epoch_loss = 0.0
    preds, labels = [], []
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        
        # 前向传播（单输出）
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # 记录指标
        epoch_loss += loss.item()
        preds.append(outputs.detach().cpu())
        labels.append(targets.detach().cpu())
    
    # 后续代码保持不变...

    
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
        val_pred = model(val_inputs)  # 直接使用单个输出
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
    line2, = axs[i].plot(CNN_ImpedancePrediction, time, 'k--', label='FCN预测值', linewidth=0.6)

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