import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
import json
import torch.nn.functional as F
# 设置中文字体为宋体，西文字体为 Times New Roman
import matplotlib
# 获取当前脚本的绝对路径，并设置工作目录为脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"当前工作目录已设置为：{os.getcwd()}")
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置 sans-serif 字体为黑体以支持中文
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置主要字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# ****************** 直接在代码中定义模型 ******************
#%% ****************** 改进的Inception模块 ******************
class InceptionModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 分支1：1x1卷积
        self.branch1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels//4, 1),
            torch.nn.BatchNorm1d(out_channels//4),
            torch.nn.LeakyReLU(0.1)
        )
        
        # 分支2：3x3卷积
        self.branch2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels//4, 1),
            torch.nn.BatchNorm1d(out_channels//4),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(out_channels//4, out_channels//4, 3, padding=1),
            torch.nn.BatchNorm1d(out_channels//4),
            torch.nn.LeakyReLU(0.1)
        )
        
        # 分支3：5x5卷积
        self.branch3 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels//4, 1),
            torch.nn.BatchNorm1d(out_channels//4),
            torch.nn.LeakyReLU(0.1),
            torch.nn.Conv1d(out_channels//4, out_channels//4, 5, padding=2),
            torch.nn.BatchNorm1d(out_channels//4),
            torch.nn.LeakyReLU(0.1)
        )
        
        # 分支4：3x3空洞卷积
        self.branch4 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, out_channels//4, 3, padding=2, dilation=2),
            torch.nn.BatchNorm1d(out_channels//4),
            torch.nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x)
        ], dim=1)

#%% ****************** 改进的Inception-FCN模型 ******************
class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # 编码器
        self.enc1 = torch.nn.Sequential(
            InceptionModule(1, 64),
            torch.nn.MaxPool1d(4)
        )
        self.enc2 = torch.nn.Sequential(
            InceptionModule(64, 128),
            torch.nn.MaxPool1d(5)
        )
        self.enc3 = torch.nn.Sequential(
            InceptionModule(128, 256),
            torch.nn.AdaptiveAvgPool1d(500)
        )
        
        # 解码器（调整上采样倍数）
        self.up1 = torch.nn.Upsample(scale_factor=10, mode='linear', align_corners=False)
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv1d(256+128, 128, 5, padding=2),
            torch.nn.LeakyReLU(0.1)
        )
        self.up2 = torch.nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.dec2 = torch.nn.Sequential(
            torch.nn.Conv1d(128+64, 64, 9, padding=4),
            torch.nn.LeakyReLU(0.1)
        )
        self.final = torch.nn.Conv1d(64, 1, 15, padding=7)
        
        # 残差路径
        self.residual = torch.nn.Sequential(
            torch.nn.Conv1d(1, 256, 3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.MaxPool1d(4),
            torch.nn.AdaptiveMaxPool1d(500)
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
# ****************** 文件路径配置 ******************
# SEISMIC_PATH = 'SEISMIC_CODING/Data/BoCeng/120Hz.segy'
# MODEL_PATH = 'SEISMIC_CODING/Model/BoCeng/FCN_120Hz/FCN_120Hz_model.pth'
# NORM_PARAMS_PATH = 'SEISMIC_CODING/Model/BoCeng/FCN_120Hz/norm_params.json'
# OUTPUT_DIR = 'SEISMIC_CODING/Data/BoCeng/output_images'
# ****************** 文件路径配置 ******************
SEISMIC_PATH = r'C:\Users\ZYH\Desktop\python_code\SEISMIC_CODING\Data\True\30Hz.segy'
MODEL_PATH = r'C:\Users\ZYH\Desktop\python_code\SEISMIC_CODING\Model\True\FCN_30Hz\FCN_30Hz_model.pth'
NORM_PARAMS_PATH = r'C:\Users\ZYH\Desktop\python_code\SEISMIC_CODING\Model\True\FCN_30Hz\norm_params.json'
OUTPUT_DIR = r'C:\Users\ZYH\Desktop\python_code\SEISMIC_CODING\Data\True\output_images'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# OUTPUT_DIR = 'SEISMIC_CODING/output_images/HuCeng/FCN_30Hz'

# SEISMIC_PATH = 'SEISMIC_CODING/Data/ThreeLevel/40Hz.segy'
# MODEL_PATH = 'SEISMIC_CODING/Model/FCN_40Hz/FCN_40Hz_model.pth'
# NORM_PARAMS_PATH = 'SEISMIC_CODING/Model/FCN_40Hz/norm_params.json'
# OUTPUT_DIR = 'SEISMIC_CODING/output_images/FCN_40Hz'

# SEISMIC_PATH = 'SEISMIC_CODING/Data/ThreeSha/30Hz.segy'
# MODEL_PATH = 'SEISMIC_CODING/Model/ThreeSha/FCN_30Hz/FCN_30Hz_model.pth'
# NORM_PARAMS_PATH = 'SEISMIC_CODING/Model/ThreeSha/FCN_30Hz/norm_params.json'
# OUTPUT_DIR = 'SEISMIC_CODING/output_images/ThreeSha/FCN_30Hz'
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
    model = FCN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    # 预测
    with torch.no_grad():
        prediction_norm = model(input_tensor).cpu().numpy().squeeze()
    
    return postprocess(prediction_norm, imp_min, imp_max), dt

##****************** 可视化函数 ******************
def visualize_results(prediction, dt):
    """绘制波阻抗剖面"""
    n_samples = prediction.shape[1]
    time_axis = np.arange(n_samples) * dt * 1000  # 转换为毫秒
    
    plt.figure(figsize=(18, 6))  # 设置图像大小，与提供的代码一致
    plt.imshow(prediction.T, 
            aspect='auto',
            cmap='plasma',  # 使用相同的 colormap
            extent=[0, 430, time_axis[-1], time_axis[0]],  # 设置坐标范围
            origin='upper',  # 顶部为时间的开始
            vmin=6257730, vmax=10002548)  # 设置 vmin 和 vmax，依据实际数据调整
    
    plt.colorbar(label='波阻抗 (m/s * g/cm³)')
    plt.xlabel('道号', fontsize=12)
    plt.ylabel('时间 (ms)', fontsize=12)
    plt.title('FCN预测波阻抗剖面', fontsize=14)
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, 'impedance30.png')
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
    plt.show()
    plt.close()

    print(f"结果已保存至：{output_path}")
def visualize_results(prediction, dt):
    """CFD风格专业可视化"""
    n_samples = prediction.shape[1]
    time_axis = np.arange(n_samples) * dt * 1000  # 时间轴（毫秒）

    plt.figure(figsize=(22, 8), dpi=100)
    
    # 使用jet色标模拟CFD风格（蓝-青-黄-红渐变）
    cfd_cmap = plt.cm.get_cmap('rainbow', 1000)  # 高分辨率色标
    img = plt.imshow(prediction.T,
               aspect='auto',
               cmap=cfd_cmap,
               extent=[0, prediction.shape[0], time_axis[-1], time_axis[0]],
               origin='upper',
               vmin=1e6,  # 固定色标最小值
               vmax=1e7) # 固定色标最大值
    
    # 专业色标设置（科学计数法格式）
    cbar = plt.colorbar(img, pad=0.01, aspect=30)
    cbar.set_label('Scalar Field Intensity (unit)',
                 fontsize=12,
                 fontname='Times New Roman',
                 labelpad=15)
    
    # 设置色标刻度格式
    cbar.formatter.set_powerlimits((6, 9))  # 自动科学计数法显示
    cbar.ax.yaxis.set_major_formatter(plt.ScalarFormatter(useMathText=True))
    cbar.update_ticks()
    
    # 坐标轴优化
    plt.xlabel('Position (m)', 
             fontsize=14,
             fontname='Times New Roman',
             labelpad=10)
    plt.ylabel('Time (ms)',
             fontsize=14,
             fontname='Times New Roman', 
             labelpad=10)
    
    # 保存高清图像
    output_path = os.path.join(OUTPUT_DIR, 'CFD_Style_Visualization.svg')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    plt.close()




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
