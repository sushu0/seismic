import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

# 设置中文字体为宋体，西文字体为 Times New Roman
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置 sans-serif 字体为黑体以支持中文
matplotlib.rcParams['font.family'] = ['Times New Roman', 'SimHei']  # 设置主要字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 显示负号

# 读取波阻抗数据
file_path = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\ThreeLevel\ThreeLevel_04.txt'  # 04_calculate_impedance_by_grid.py生成的文件
data = pd.read_csv(file_path, sep='\t')

# 提取必要信息
x_distances = data['X Distance'].unique()  # 获取所有的道位置
time = data['Time'].unique()  # 获取所有时间点
impedance_data = data.pivot(index='Time', columns='X Distance', values='Impedance')  # 创建二维矩阵

vmin = 6257730  # 紫色层大小
vmax = 10002548  # 土黄层大小
# 可视化波阻抗剖面
plt.figure(figsize=(18, 6))  # 设置图像大小
extent = [x_distances.min(), x_distances.max(), time.max(), time.min()]
plt.imshow(impedance_data.values, aspect='auto', cmap='plasma', extent=extent, origin='upper', vmin=vmin, vmax=vmax)
# 添加色条
cbar = plt.colorbar(label='Impedance')
cbar.ax.tick_params(labelsize=12)
cbar.set_label(r'impedance $(\mathrm{m/s \cdot g/cm^3})$', fontsize=12)

# 修改横轴标签为道号
plt.xlabel('Shot Number', fontsize=12)  # 横轴改为道号
plt.ylabel('Time (ms)', fontsize=12)

# 保存图像
output_dir = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\ThreeLevel\output_images'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'ThreeLevel.png')
plt.savefig(output_path, dpi=600, bbox_inches='tight', format='png')

plt.show()

print(f"图像已保存至 {output_path}")
