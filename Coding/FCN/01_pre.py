import torch
import numpy as np
import matplotlib.pyplot as plt
import segyio
import os
import json
import torch.nn.functional as F

# ****************** 文件路径配置 ******************
# 输入文件路径配置
SEISMIC_PATH = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'  # 地震数据文件路径 (SEGY格式)
MODEL_PATH = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\FCN_01_model.pth'  # 训练好的FCN模型文件路径
NORM_PARAMS_PATH = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\norm_params.json'  # 数据归一化参数文件路径

# 输出文件路径配置
OUTPUT_DIR = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\output_images'  # 预测结果图像输出目录
OUTPUT_FILENAME = '01_30Hz_pre.png'  # 输出图像文件名

# 设置中文字体为宋体，西文字体为 Times New Roman
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置 sans-serif 字体为黑体以支持中文
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置主要字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ****************** 直接在代码中定义模型 ******************
#%% ****************** 优化后的FCN模型定义 ******************
class AttentionModule(torch.nn.Module):
    """注意力机制模块"""
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.max_pool = torch.nn.AdaptiveMaxPool1d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channels, channels // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(channels // 4, channels),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        attention = (avg_out + max_out).unsqueeze(-1)
        return x * attention

class ResidualBlock(torch.nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size, 1, padding=kernel_size//2)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，需要1x1卷积调整
        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, 1, stride),
                torch.nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class OptimizedFCN(torch.nn.Module):
    def __init__(self, dropout_rate=0.1):
        super().__init__()
        
        # 编码器 - 使用残差块和注意力机制
        self.enc1 = torch.nn.Sequential(
            torch.nn.Conv1d(1, 64, 15, padding=7),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            ResidualBlock(64, 64),
            AttentionModule(64),
            torch.nn.MaxPool1d(4),
            torch.nn.Dropout(dropout_rate)
        )
        
        self.enc2 = torch.nn.Sequential(
            torch.nn.Conv1d(64, 128, 9, padding=4),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            ResidualBlock(128, 128),
            AttentionModule(128),
            torch.nn.MaxPool1d(5),
            torch.nn.Dropout(dropout_rate)
        )
        
        self.enc3 = torch.nn.Sequential(
            torch.nn.Conv1d(128, 256, 5, padding=2),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            ResidualBlock(256, 256),
            AttentionModule(256),
            torch.nn.Dropout(dropout_rate)
        )
        
        # 瓶颈层 - 增加深度
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv1d(256, 512, 3, padding=1),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            ResidualBlock(512, 512),
            AttentionModule(512),
            torch.nn.Conv1d(512, 256, 3, padding=1),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate)
        )
        
        # 解码器 - 改进的上采样和跳跃连接
        self.up1 = torch.nn.Upsample(scale_factor=5, mode='linear', align_corners=False)
        self.dec1 = torch.nn.Sequential(
            torch.nn.Conv1d(256+128, 128, 5, padding=2),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            ResidualBlock(128, 128),
            AttentionModule(128),
            torch.nn.Dropout(dropout_rate)
        )
        
        self.up2 = torch.nn.Upsample(scale_factor=4, mode='linear', align_corners=False)
        self.dec2 = torch.nn.Sequential(
            torch.nn.Conv1d(128+64, 64, 9, padding=4),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            ResidualBlock(64, 64),
            AttentionModule(64),
            torch.nn.Dropout(dropout_rate)
        )
        
        # 最终输出层
        self.final = torch.nn.Sequential(
            torch.nn.Conv1d(64, 32, 15, padding=7),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Conv1d(32, 1, 1)
        )
        
        # 全局残差连接
        self.global_residual = torch.nn.Sequential(
            torch.nn.Conv1d(1, 256, 1),
            torch.nn.BatchNorm1d(256),
            torch.nn.MaxPool1d(4),
            torch.nn.MaxPool1d(5)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, torch.nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, 0, 0.01)
                torch.nn.init.constant_(m.bias, 0)

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
    model = OptimizedFCN().to(device)
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
    plt.title('FCN预测波阻抗剖面', fontsize=14)
    
    # 保存图像
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')
    plt.show()
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
