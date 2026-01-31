import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from torch.autograd import Variable
import segyio
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 设置字体和绘图参数
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.style.use('ggplot')

# 文件路径配置
output_image_dir = 'SEISMIC_CODING/output_images/CNN_10Hz'
svg_image_dir = 'SEISMIC_CODING/svg_images/CNN_10Hz' 
model_save_path = 'SEISMIC_CODING/Model/CNN_10Hz/Cnn_model_10Hz.pth'
norm_params_path = 'SEISMIC_CODING/Model/CNN_10Hz/norm_params.json'  # 新增参数保存路径
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(svg_image_dir, exist_ok=True)
os.makedirs(os.path.dirname(norm_params_path), exist_ok=True)

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
seismic_data = load_seismic_data('SEISMIC_CODING/Data/ThreeLevel/10Hz.segy')
impedance_data = load_impedance_data('SEISMIC_CODING/Data/ThreeLevel/impedance.txt')

# 数据标准化
def normalize(data):
    data_min, data_max = np.min(data), np.max(data)
    return (data - data_min) / (data_max - data_min), data_min, data_max

seismic_norm, s_min, s_max = normalize(seismic_data)
imp_norm, imp_min, imp_max = normalize(impedance_data)

# 保存归一化参数（新增部分）
norm_params = {
    's_min': float(s_min),
    's_max': float(s_max),
    'imp_min': float(imp_min),
    'imp_max': float(imp_max)
}
with open(norm_params_path, 'w') as f:
    json.dump(norm_params, f, indent=4)
print(f"归一化参数已保存至: {norm_params_path}")

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

#%% ****************** 模型定义 ******************
class SeismicCNN(nn.Module):
    def __init__(self):
        super(SeismicCNN, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=51, padding=25),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(4),
            
            nn.Conv1d(64, 128, kernel_size=25, padding=12),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(5),
            
            nn.Conv1d(128, 256, kernel_size=15, padding=7),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(5)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(256, 128, kernel_size=5, stride=5),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=5),
            nn.LeakyReLU(0.1),
            
            nn.ConvTranspose1d(64, 1, kernel_size=4, stride=4, output_padding=1),
            nn.Conv1d(1, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

#%% ****************** 训练配置 ******************
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SeismicCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# 数据加载器
train_dataset = torch.utils.data.TensorDataset(trainX, trainImp)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

#%% ****************** 训练循环 ******************
metrics = {
    'train_loss': [], 'val_loss': [],
    'train_pcc': [], 'val_pcc': [],
    'train_r2': [], 'val_r2': []
}

best_val_loss = float('inf')
early_stop_counter = 0

for epoch in range(400):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        epoch_loss += loss.item()
        preds.append(outputs.detach().cpu())
        labels.append(targets.detach().cpu())
    
    # 计算训练指标
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    metrics['train_loss'].append(epoch_loss/len(train_loader))
    
    # Pearson相关系数
    cov = torch.mean((preds - preds.mean()) * (labels - labels.mean()))
    std = preds.std() * labels.std()
    metrics['train_pcc'].append((cov / std).item())
    
    # R平方
    ss_res = torch.sum((labels - preds)**2)
    ss_tot = torch.sum((labels - labels.mean())**2)
    metrics['train_r2'].append((1 - ss_res/ss_tot).item())
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_inputs = valX.to(device)
        val_pred = model(val_inputs)
        val_loss = criterion(val_pred, valImp.to(device))
        
        # 验证指标计算
        val_pred = val_pred.cpu()
        metrics['val_loss'].append(val_loss.item())
        
        # PCC计算
        cov_val = torch.mean((val_pred - val_pred.mean()) * (valImp - valImp.mean()))
        std_val = val_pred.std() * valImp.std()
        metrics['val_pcc'].append((cov_val / std_val).item())
        
        # R²计算
        ss_res_val = torch.sum((valImp - val_pred)**2)
        ss_tot_val = torch.sum((valImp - valImp.mean())**2)
        metrics['val_r2'].append((1 - ss_res_val/ss_tot_val).item())
    
    # 学习率调整
    scheduler.step(val_loss)
    
    # 早停机制
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        # 保存最佳模型时同时保存归一化参数（新增部分）
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}: 发现新的最佳模型，已保存至 {model_save_path}")
    else:
        early_stop_counter += 1
        if early_stop_counter >= 50:
            print("\nEarly stopping triggered")
            break
    
    # 打印进度
    print(f'Epoch {epoch+1:03d} | '
        f'Train Loss: {metrics["train_loss"][-1]:.4f} | '
        f'Val Loss: {metrics["val_loss"][-1]:.4f} | '
        f'PCC: {metrics["train_pcc"][-1]:.3f}/{metrics["val_pcc"][-1]:.3f} | '
        f'R²: {metrics["train_r2"][-1]:.3f}/{metrics["val_r2"][-1]:.3f}')

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
with segyio.open('SEISMIC_CODING/Data/ThreeLevel/10Hz.segy', "r", ignore_geometry=True) as f_seismic:
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
    line2, = axs[i].plot(CNN_ImpedancePrediction, time, 'k--', label='CNN预测值', linewidth=0.6)

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
