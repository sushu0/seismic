import pandas as pd
import numpy as np

# 生成 X Distance 和 Time 列
x_distance = np.arange(10, 1990.1, 20)  # 从0到1000，间隔10
time = np.arange(0, 100.01, 0.01)  # 从0到100，间隔0.01

# 为每个 X Distance 配置时间
data = []
for x in x_distance:
    for t in time:
        # 保证Time列数据不出现类似浮动的误差，保留两位小数
        data.append([x, 0.0, round(t, 3), 3000.0, 2285.910])

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['X Distance', 'Depth', 'Time', 'Velocity', 'Density'])

# 保存数据到文本文件，以空格分隔
df.to_csv(r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_40Hz_01.txt', sep='\t', index=False, header=True)

print("数据生成完成并保存为 01_40Hz_01.txt")
