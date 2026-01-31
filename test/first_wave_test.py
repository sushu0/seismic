
from obspy.io.segy.segy import _read_segy
import numpy as np
import matplotlib.pyplot as plt

# Step 1: 读取 SEG-Y 文件
file_path = 'Data/Two_ch_30HZ.sgy'
segy_data = _read_segy(file_path)

# Step 2: 提取第一道地震数据
# 提取第一道数据
first_trace_data = segy_data.traces[6 ].data

# 提取采样时间间隔 (微秒 -> 毫秒)
dt = segy_data.binary_file_header.sample_interval_in_microseconds / 1000  # 时间采样间隔 (ms)

# 生成时间轴
time = np.arange(0, len(first_trace_data) * dt, dt)  # 时间轴 (毫秒)

# Step 3: 绘制第一道地震数据
plt.figure(figsize=(10, 6))

# 绘制第一道振幅随时间变化的曲线
plt.plot(time, first_trace_data, color='blue', linewidth=1)

# 添加标题和标签
plt.title('First Trace Amplitude vs Time 第六道振幅-时间曲线', fontsize=16)
plt.xlabel('Time (ms) 时间 (毫秒)', fontsize=14)
plt.ylabel('Amplitude 振幅', fontsize=14)

# 显示网格
plt.grid(True, linestyle='--', alpha=0.7)

# 显示图像
plt.show()
