# -*- coding: utf-8 -*-
"""
50Hz 数据生成脚本
整合4个步骤：
1. 生成初始网格数据 (01)
2. 解析时间区间数据 (02)
3. 更新网格数据 (03)
4. 生成最终阻抗文件 (04)
"""
import pandas as pd
import numpy as np
import os

# ==================== 配置参数 ====================
DATA_DIR = r'D:\SEISMIC_CODING\zmy_data\01\data'
TIMES_FILE = os.path.join(DATA_DIR, '01.txt')  # 原始时间文件

# 输出文件
OUTPUT_01 = os.path.join(DATA_DIR, '01_50Hz_01.txt')  # 初始网格
OUTPUT_02 = os.path.join(DATA_DIR, '01_50Hz_02.csv')  # 时间区间
OUTPUT_03 = os.path.join(DATA_DIR, '01_50Hz_03.txt')  # 更新后网格
OUTPUT_04 = os.path.join(DATA_DIR, '01_50Hz_04.txt')  # 最终阻抗

# 网格参数 - 与40Hz保持一致
X_START = 10
X_END = 1990
X_INTERVAL = 20  # X方向间隔
TIME_START = 0
TIME_END = 100
TIME_INTERVAL = 0.01  # 时间采样间隔 ms

# 初始物理参数（背景值）
VELOCITY_BG = 3000.0
DENSITY_BG = 2285.910

# 薄层物理参数（更新值）
VELOCITY_LAYER = 4000.0
DENSITY_LAYER = 2456.37

# 通道数
MAX_CHANNELS = 7

print("="*60)
print("50Hz 数据生成")
print("="*60)

# ==================== 步骤1: 生成初始网格数据 ====================
print("\n[步骤1] 生成初始网格数据...")

x_distance = np.arange(X_START, X_END + 0.1, X_INTERVAL)
time = np.arange(TIME_START, TIME_END + TIME_INTERVAL, TIME_INTERVAL)

print(f"  X Distance: {len(x_distance)} 个点 ({X_START} - {X_END})")
print(f"  Time: {len(time)} 个点 ({TIME_START} - {TIME_END} ms)")

data = []
for x in x_distance:
    for t in time:
        data.append([x, 0.0, round(t, 3), VELOCITY_BG, DENSITY_BG])

df_grid = pd.DataFrame(data, columns=['X Distance', 'Depth', 'Time', 'Velocity', 'Density'])
df_grid.to_csv(OUTPUT_01, sep='\t', index=False, header=True)
print(f"  保存: {OUTPUT_01}")
print(f"  数据行数: {len(df_grid)}")

# ==================== 步骤2: 解析时间区间数据 ====================
print("\n[步骤2] 解析时间区间数据...")

def load_times(file_path, max_channels):
    """加载 times 文件并解析数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                lines = f.readlines()

    parsed_data = []
    current_distance = None
    for line in lines[3:]:  # 跳过前三行标题
        line = line.strip()
        if not line or line.startswith("="):
            continue
        parts = line.split()
        if len(parts) >= max_channels + 1:
            try:
                current_distance = float(parts[0])
                values = parts[1:max_channels + 1]
            except ValueError:
                continue
        else:
            values = parts

        if current_distance is not None:
            for channel_id, value in enumerate(values, start=1):
                try:
                    time_value = float(value) if value != "NIL" else -1.0
                    parsed_data.append([current_distance, channel_id, time_value])
                except ValueError:
                    continue
    
    return pd.DataFrame(parsed_data, columns=['x_distance', 'channel', 'time'])

def split_times(data):
    """将数据拆分为每个 (x_distance, channel) 的时间对列表"""
    result = {}
    grouped = data.groupby(['x_distance', 'channel'])
    for (x, c), group in grouped:
        times = group['time'].tolist()
        if len(times) == 1:
            times.append(-1.0)
        times = sorted(times)
        result[(x, c)] = times
    return result

# 加载时间数据
times_data = load_times(TIMES_FILE, MAX_CHANNELS)
print(f"  加载时间数据: {len(times_data)} 条记录")

# 分割时间
split_result = split_times(times_data)
print(f"  分割后: {len(split_result)} 个 (x_distance, channel) 组合")

# 保存为CSV
rows = []
for (x_distance, channel), times in split_result.items():
    rows.append([x_distance, channel, times])
df_times = pd.DataFrame(rows, columns=['x_distance', 'channel', 'times'])
df_times.to_csv(OUTPUT_02, index=False)
print(f"  保存: {OUTPUT_02}")

# ==================== 步骤3: 更新网格数据 ====================
print("\n[步骤3] 更新网格数据...")

def update_grid(grid_df, times_df):
    """根据时间区间更新网格数据"""
    updated_count = 0
    for idx, row in times_df.iterrows():
        x_distance = row['x_distance']
        times = eval(row['times']) if isinstance(row['times'], str) else row['times']
        times = sorted(times)
        
        # 跳过无效时间区间
        if times[0] < 0 or times[1] < 0:
            continue
        
        # 在 grid 中找到对应的时间区间
        mask = (
            (grid_df['X Distance'] == x_distance) &
            (grid_df['Time'] >= times[0]) &
            (grid_df['Time'] <= times[1])
        )
        matches = grid_df[mask]
        
        if not matches.empty:
            grid_df.loc[mask, ['Velocity', 'Density']] = [VELOCITY_LAYER, DENSITY_LAYER]
            updated_count += len(matches)
    
    return grid_df, updated_count

# 重新加载网格数据
grid_df = pd.read_csv(OUTPUT_01, sep='\t')
times_df = pd.read_csv(OUTPUT_02)

# 更新
grid_df, updated_count = update_grid(grid_df, times_df)
print(f"  更新行数: {updated_count}")

# 保存
grid_df.to_csv(OUTPUT_03, sep='\t', index=False, header=True)
print(f"  保存: {OUTPUT_03}")

# ==================== 步骤4: 生成最终阻抗文件 ====================
print("\n[步骤4] 生成最终阻抗文件...")

# 加载更新后的网格
grid_df = pd.read_csv(OUTPUT_03, sep='\t')

# 标准化列名
grid_df.columns = grid_df.columns.str.strip().str.lower().str.replace(' ', '_')

# 获取可用的x_distance值
available_distances = sorted(grid_df['x_distance'].unique())
x_distances = [d for d in available_distances if X_START <= d <= X_END]
print(f"  选择的 x_distance 数量: {len(x_distances)}")

# 过滤数据
filtered_df = grid_df[grid_df['x_distance'].isin(x_distances)].copy()

# 时间对齐
filtered_df['time'] = (filtered_df['time'] / TIME_INTERVAL).round() * TIME_INTERVAL
filtered_df['time'] = filtered_df['time'].round(3)
filtered_df = filtered_df.drop_duplicates(subset=['x_distance', 'time'])

# 计算波阻抗
filtered_df['impedance'] = filtered_df['velocity'] * filtered_df['density']

# 添加道号
filtered_df['No.'] = filtered_df['x_distance'].apply(lambda x: x_distances.index(x) + 1)

# 选择和重命名列
filtered_df = filtered_df[['No.', 'x_distance', 'depth', 'time', 'impedance']]
filtered_df.rename(columns={
    'x_distance': 'X Distance',
    'depth': 'Depth',
    'time': 'Time',
    'impedance': 'Impedance'
}, inplace=True)

# 保存
filtered_df.to_csv(OUTPUT_04, sep='\t', index=False)
print(f"  保存: {OUTPUT_04}")
print(f"  数据行数: {len(filtered_df)}")

# ==================== 验证 ====================
print("\n" + "="*60)
print("验证生成的数据...")
print("="*60)

# 检查阻抗文件
imp_df = pd.read_csv(OUTPUT_04, sep='\t')
print(f"\n阻抗文件统计:")
print(f"  道数: {imp_df['No.'].nunique()}")
print(f"  每道采样点数: {len(imp_df) // imp_df['No.'].nunique()}")
print(f"  阻抗范围: {imp_df['Impedance'].min():.2f} - {imp_df['Impedance'].max():.2f}")
print(f"  唯一阻抗值数: {imp_df['Impedance'].nunique()}")

# 检查是否与SGY文件匹配
import segyio
sgy_path = os.path.join(DATA_DIR, '01_50Hz_re.sgy')
with segyio.open(sgy_path, 'r', ignore_geometry=True) as f:
    n_traces = f.tracecount
    n_samples = len(f.trace[0])
print(f"\nSGY文件 (01_50Hz_re.sgy):")
print(f"  道数: {n_traces}")
print(f"  每道采样数: {n_samples}")

expected_rows = imp_df['No.'].nunique() * (len(imp_df) // imp_df['No.'].nunique())
if imp_df['No.'].nunique() == n_traces:
    print(f"\n✓ 道数匹配!")
else:
    print(f"\n✗ 道数不匹配! 阻抗文件: {imp_df['No.'].nunique()}, SGY: {n_traces}")

print("\n" + "="*60)
print("50Hz 数据生成完成!")
print("="*60)
print(f"\n生成的文件:")
print(f"  1. {OUTPUT_01}")
print(f"  2. {OUTPUT_02}")
print(f"  3. {OUTPUT_03}")
print(f"  4. {OUTPUT_04}")
