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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import segyio
import torchvision.ops  # 新增关键依赖

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch.nn.utils import weight_norm
from torchdiffeq import odeint_adjoint as odeint
# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EnhancedTCN(nn.Module):
    """优化后的TCN架构"""
    def __init__(self):
        super().__init__()
        self.tcn = nn.Sequential(
            self._build_block(1, 16, 7, 1),
            self._build_block(16, 32, 7, 2),
            self._build_block(32, 64, 7, 4),
            nn.AdaptiveAvgPool1d(512)
        )
        self.final = nn.Sequential(
            nn.Conv1d(64, 32, 1), nn.ReLU(),
            nn.Conv1d(32, 16, 1), nn.ReLU(),
            nn.Conv1d(16, 1, 1)
        )

    def _build_block(self, in_ch, out_ch, kernel, dilation):
        padding = (kernel-1) * dilation // 2
        return nn.Sequential(
            weight_norm(nn.Conv1d(in_ch, out_ch, kernel, 
                                padding=padding, dilation=dilation)),
            nn.ReLU(),
            nn.Dropout(0.2),
            weight_norm(nn.Conv1d(out_ch, out_ch, kernel,
                                padding=padding, dilation=dilation)),
            nn.ReLU()
        )

    def forward(self, x):
        return self.final(self.tcn(x)), []

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
    model = EnhancedTCN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # 预测（修改此处）
    with torch.no_grad():
        main_output, _ = model(input_tensor)  # 获取主输出
        prediction_norm = main_output.cpu().numpy().squeeze()
    
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
    plt.title('优化TCN预测波阻抗剖面', fontsize=14)
    
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
