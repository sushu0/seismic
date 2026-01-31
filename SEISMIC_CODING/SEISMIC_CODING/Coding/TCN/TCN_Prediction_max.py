import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
import json
from torch.nn.utils import weight_norm
# 设置中文字体为宋体，西文字体为 Times New Roman
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置 sans-serif 字体为黑体以支持中文
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置主要字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%% ****************** TCN模型定义 ******************
# 定义 TCN 模型结构
class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()

class TemporalBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(torch.nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, bias=True))
        self.relu1 = torch.nn.ReLU()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.conv2 = weight_norm(torch.nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        stride=stride, padding=padding, dilation=dilation, bias=True))
        self.relu2 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(dropout)

        self.net = torch.nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                self.conv2, self.relu2, self.dropout2)
        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = torch.nn.ReLU()
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

class TemporalConvNet(torch.nn.Module):
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

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class MustafaNet(torch.nn.Module):
    def __init__(self):
        super(MustafaNet, self).__init__()
        self.tcn_local = TemporalConvNet(num_inputs=1, num_channels=[3, 6, 6, 6, 6, 6, 5], kernel_size=9, dropout=0.2)
        self.regression = torch.nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

    def forward(self, input):
        out = self.tcn_local(input)
        out = self.regression(out)
        return out   

# ****************** 文件路径配置 ******************
SEISMIC_PATH = 'SEISMIC_CODING/Data/ThreeLevel/30Hz.segy'
MODEL_PATH = 'SEISMIC_CODING/Model/TCN_30Hz/TCN_30Hz_model.pth'
NORM_PARAMS_PATH = 'SEISMIC_CODING/Model/TCN_30Hz/norm_params.json'
OUTPUT_DIR = 'SEISMIC_CODING/output_images/TCN_30Hz'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ****************** 数据加载函数 ******************
def load_data_with_metadata(path):
    """加载地震数据并获取元数据"""
    with segyio.open(path, "r", ignore_geometry=True, strict=False) as f:
        seismic = np.stack([np.copy(f.trace[i]) for i in range(100)])
        dt = f.bin[segyio.BinField.Interval] * 1e-6
        return seismic, dt


def load_normalization_params():
    """加载归一化参数"""
    with open(NORM_PARAMS_PATH) as f:
        params = json.load(f)
    return (
        params['s_min'], params['s_max'],
        params['imp_min'], params['imp_max']
    )

# ****************** 预处理函数 ******************
def preprocess(seismic, s_min, s_max):
    """数据标准化"""
    return (seismic - s_min) / (s_max - s_min)

def postprocess(prediction, imp_min, imp_max):
    """反归一化"""
    return prediction * (imp_max - imp_min) + imp_min

# ****************** 主预测函数 ******************
def predict_full_section():
    # 加载参数
    s_min, s_max, imp_min, imp_max = load_normalization_params()
    
    # 加载地震数据
    seismic_raw, dt = load_data_with_metadata(SEISMIC_PATH)
    
    # 预处理
    seismic_norm = preprocess(seismic_raw, s_min, s_max)
    
    # 转换为张量
    input_tensor = torch.FloatTensor(seismic_norm).unsqueeze(1).to(device)
    
    # 初始化模型
    model = MustafaNet().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # 预测
    with torch.no_grad():
        prediction_norm = model(input_tensor).cpu().numpy().squeeze()
    
    return postprocess(prediction_norm, imp_min, imp_max), dt

# ****************** 可视化函数 ******************
def visualize_results(prediction, dt):
    """绘制波阻抗剖面"""
    n_samples = prediction.shape[1]
    time_axis = np.arange(n_samples) * dt * 1000  # 转换为毫秒
    
    plt.figure(figsize=(18, 6))  # 设置图像大小，与提供的代码一致
    plt.imshow(prediction.T, 
            aspect='auto',
            cmap='plasma',  # 使用相同的 colormap
            extent=[0, 100, time_axis[-1], time_axis[0]],  # 设置坐标范围
            origin='upper',  # 顶部为时间的开始
            vmin=6257730, vmax=10002548)  # 设置 vmin 和 vmax，依据实际数据调整
    
    plt.colorbar(label='波阻抗 (m/s * g/cm³)')
    plt.xlabel('道号', fontsize=12)
    plt.ylabel('时间 (ms)', fontsize=12)
    plt.title('CNN预测波阻抗剖面', fontsize=14)
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'impedance_section.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
    plt.close()
    print(f"结果已保存至：{output_path}")

# ****************** 执行预测 ******************
if __name__ == '__main__':
    prediction, dt = predict_full_section()
    visualize_results(prediction, dt)
    
    # 打印统计信息
    print("\n预测统计:") 
    print(f"形状: {prediction.shape} (道数 × 采样点数)") 
    print(f"最小值: {np.min(prediction):.2f}")
    print(f"最大值: {np.max(prediction):.2f}")
    print(f"平均值: {np.mean(prediction):.2f}")
