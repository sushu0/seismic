from obspy.io.segy.segy import _read_segy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# =====================
# 设置matplotlib字体，支持中英文和负号显示
# =====================
# 设置中文字体为宋体，西文字体为 Times New Roman
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置 sans-serif 字体为黑体以支持中文
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置主要字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# =====================
# Step 1: 读取 SEGY 文件
# =====================
# 可切换不同的SEGY文件进行可视化
file_path = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_50Hz_re.sgy'  # 输入文件相对路径
segy_data = _read_segy(file_path)

# =====================
# Step 2: 提取地震道数目
# =====================
num_traces = len(segy_data.traces)
print(f"地震道数目: {num_traces}")

# =====================
# Step 3: 提取采样时间间隔 (微秒 -> 毫秒)
# =====================
dt = segy_data.binary_file_header.sample_interval_in_microseconds / 1000  # 时间采样间隔 (ms)
print(f"采样时间间隔: {dt:.2f} ms")

# =====================
# Step 4: 提取每道采样点数
# =====================
num_samples = len(segy_data.traces[0].data)
print(f"每道采样点数: {num_samples}")

# =====================
# Step 5: 提取地震道数据并绘制地震剖面
# =====================
# 提取所有道的数据，组成二维数组（道数 x 采样点数）
trace_data = [trace.data for trace in segy_data.traces]
trace_data = np.array(trace_data)  # shape: (num_traces, num_samples)

# 生成时间轴（单位：毫秒）
time = np.arange(0, dt * trace_data.shape[1], dt)  # 长度与采样点数一致

# 振幅标准化 (可选，增强对比度)
trace_data = trace_data / np.max(np.abs(trace_data))  # 归一化到 [-1, 1]

# =====================
# Step 6: 绘制地震剖面图
# =====================
plt.figure(figsize=(12, 6))  # 设置图像大小

# 使用 imshow 绘制剖面，横轴为道号，纵轴为时间，颜色表示振幅
plt.imshow(
    trace_data.T,  # 转置，确保时间为纵轴，地震道为横轴
    aspect='auto',  # 保持图像比例
    cmap='seismic',  # 红蓝配色方案
    extent=[0, trace_data.shape[0], time[-1] , time[0] ],  # X为道数，Y为时间ms
    vmax=1,
    vmin=-1
)

# 添加颜色条，显示振幅范围
plt.colorbar(label='振幅')

# 设置轴标签和标题
plt.xlabel('Trace No. (道号)')
plt.ylabel('Time (ms) 时间 (毫秒)')
plt.title('Seismic Section 地震剖面')

# =====================
# Step 7: 保存与显示图像
# =====================

# 指定输出目录和文件名，自动创建
output_dir = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\output_images'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '01_50Hz_cut.png')  # 输出文件名
plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')  # 高分辨率保存

plt.show()  # 显示图像

print(f"图像已保存至 {output_path}")
