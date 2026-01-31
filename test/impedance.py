import pandas as pd
import numpy as np
import os

# ============================================================================
# 数据处理参数配置
# ============================================================================

# 初始数据生成参数
X_DISTANCE_START = 25
X_DISTANCE_END = 4975.1
X_DISTANCE_INTERVAL = 50
TIME_START = 0
TIME_END = 100.01
TIME_INTERVAL = 0.01
INITIAL_VELOCITY = 3000.0
INITIAL_DENSITY = 2285.910

# 时间数据处理参数
MAX_CHANNELS = 3

# 网格更新参数
UPDATE_VELOCITY = 4000
UPDATE_DENSITY = 2456.37
GRID_CHANNELS = 3
GRID_X_INTERVAL = 25

# 波阻抗计算参数
IMPEDANCE_START_DISTANCE = 25
IMPEDANCE_END_DISTANCE = 4975
IMPEDANCE_CHANNELS = 100
TIME_SAMPLE_INTERVAL = 0.01

# ============================================================================
# 文件路径配置
# ============================================================================

# 输入文件路径
TIMES_INPUT_FILE = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\ThreeLevel.asc"

# 自动获取输入文件名（不含扩展名）
def get_input_filename():
    """从输入文件路径中提取文件名（不含扩展名）"""
    filename = os.path.basename(TIMES_INPUT_FILE)
    name_without_ext = os.path.splitext(filename)[0]
    return name_without_ext

# 动态生成输出路径
INPUT_NAME = get_input_filename()
BASE_OUTPUT_DIR = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data"
OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, INPUT_NAME)

# 转换后的txt文件路径
CONVERTED_TXT_FILE = os.path.join(OUTPUT_DIR, f"{INPUT_NAME}.txt")

# 输出文件路径
GRID_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{INPUT_NAME}_01.txt")
TIMES_CSV_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{INPUT_NAME}_02.csv")
GRID_UPDATED_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{INPUT_NAME}_03.txt")
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{INPUT_NAME}_04.txt")

def convert_asc_to_txt():
    """
    将.asc文件转换为.txt文件并移动到目标文件夹，同时复制原始.asc文件
    """
    print("=== 文件转换步骤 ===")
    print(f"转换文件: {TIMES_INPUT_FILE} -> {CONVERTED_TXT_FILE}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 复制原始.asc文件到目标文件夹
    original_asc_in_target = os.path.join(OUTPUT_DIR, os.path.basename(TIMES_INPUT_FILE))
    try:
        with open(TIMES_INPUT_FILE, 'rb') as f:
            content = f.read()
        with open(original_asc_in_target, 'wb') as f:
            f.write(content)
        print(f"原始.asc文件已复制到: {original_asc_in_target}")
    except Exception as e:
        print(f"复制原始.asc文件失败: {e}")
        raise
    
    # 复制文件（.asc到.txt）
    try:
        with open(TIMES_INPUT_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        with open(CONVERTED_TXT_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"文件转换完成: {CONVERTED_TXT_FILE}")
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        try:
            with open(TIMES_INPUT_FILE, 'r', encoding='gbk') as f:
                content = f.read()
            with open(CONVERTED_TXT_FILE, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"文件转换完成: {CONVERTED_TXT_FILE}")
        except UnicodeDecodeError:
            try:
                with open(TIMES_INPUT_FILE, 'r', encoding='latin-1') as f:
                    content = f.read()
                with open(CONVERTED_TXT_FILE, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"文件转换完成: {CONVERTED_TXT_FILE}")
            except Exception as e:
                print(f"文件转换失败: {e}")
                raise
    
    return CONVERTED_TXT_FILE

def print_config_info():
    """打印配置信息"""
    print("=" * 60)
    print("配置文件信息:")
    print("=" * 60)
    print(f"输入文件: {TIMES_INPUT_FILE}")
    print(f"提取的文件名: {INPUT_NAME}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"转换后的txt文件: {CONVERTED_TXT_FILE}")
    print(f"输出文件:")
    print(f"  - 初始数据: {GRID_OUTPUT_FILE}")
    print(f"  - 时间数据: {TIMES_CSV_OUTPUT_FILE}")
    print(f"  - 更新网格: {GRID_UPDATED_OUTPUT_FILE}")
    print(f"  - 最终结果: {FINAL_OUTPUT_FILE}")
    print("=" * 60)

# ============================================================================
# 函数定义
# ============================================================================

def generate_initial_data():
    """
    生成初始的 X Distance 和 Time 数据
    对应原文件 1data_deal_with.py 的功能
    """
    print("=== 步骤1: 生成初始数据 ===")
    
    # 生成 X Distance 和 Time 列
    x_distance = np.arange(X_DISTANCE_START, X_DISTANCE_END, X_DISTANCE_INTERVAL)
    time = np.arange(TIME_START, TIME_END, TIME_INTERVAL)

    # 为每个 X Distance 配置时间
    data = []
    for x in x_distance:
        for t in time:
            # 保证Time列数据不出现类似浮动的误差，保留两位小数
            data.append([x, 0.0, round(t, 3), INITIAL_VELOCITY, INITIAL_DENSITY])

    # 将数据转换为DataFrame
    df = pd.DataFrame(data, columns=['X Distance', 'Depth', 'Time', 'Velocity', 'Density'])

    # 保存数据到文本文件，以空格分隔
    os.makedirs(os.path.dirname(GRID_OUTPUT_FILE), exist_ok=True)
    df.to_csv(GRID_OUTPUT_FILE, sep='\t', index=False, header=True)

    print(f"数据生成完成并保存为 {GRID_OUTPUT_FILE}")
    return GRID_OUTPUT_FILE

def load_times(file_path, max_channels):
    """
    加载 times 文件并解析数据，返回 DataFrame 格式的结果。
    支持根据 `max_channels` 动态解析任意数量的通道数据。
    对应原文件 2update_times.py 的功能
    """
    print(f"=== 步骤2: 加载并解析时间数据 ===")
    print(f"加载文件: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                lines = f.readlines()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    lines = f.readlines()
            except UnicodeDecodeError:
                # 如果所有文本编码都失败，可能是二进制文件
                raise ValueError(f"文件 {file_path} 似乎是二进制文件，无法作为文本文件读取。请检查文件格式。")

    parsed_data = []
    current_distance = None
    for line in lines[3:]:  # 跳过前三行标题
        line = line.strip()
        if not line or line.startswith("="):  # 跳过空行或分隔符
            continue
        parts = line.split()
        if len(parts) >= max_channels + 1:  # 新的 x_distance 行
            try:
                current_distance = float(parts[0])
                values = parts[1:max_channels + 1]
            except ValueError as e:
                print(f"警告：无法解析距离值 '{parts[0]}'，跳过此行: {line}")
                continue
        else:
            values = parts  # 继续读取当前距离的通道值

        if current_distance is not None:
            for channel_id, value in enumerate(values, start=1):
                try:
                    time_value = float(value) if value != "NIL" else -1.0
                    parsed_data.append([current_distance, channel_id, time_value])
                except ValueError as e:
                    print(f"警告：无法解析时间值 '{value}'，跳过此值")
                    continue
    
    return pd.DataFrame(parsed_data, columns=['x_distance', 'channel', 'time'])

def split_times(data):
    """
    将加载的 DataFrame 数据拆分为每个 (x_distance, channel) 的时间对列表，并确保时间顺序正确。
    """
    result = {}
    grouped = data.groupby(['x_distance', 'channel'])
    for (x, c), group in grouped:
        times = group['time'].tolist()
        if len(times) == 1:
            times.append(-1.0)  # 补充 -1.0，保证两组时间对
        # 确保时间对顺序正确
        times = sorted(times)
        result[(x, c)] = times
    
    return result

def save_times_to_csv(parsed_times, output_file):
    """
    将解析后的数据保存为 CSV 文件。
    """
    rows = []
    for (x_distance, channel), times in parsed_times.items():
        rows.append([x_distance, channel, times])
    df = pd.DataFrame(rows, columns=['x_distance', 'channel', 'times'])
    df.to_csv(output_file, index=False)
    print(f"时间数据保存到 {output_file}")

def process_times_data(times_file_path, max_channels=MAX_CHANNELS):
    """
    处理时间数据的主要函数
    """
    os.makedirs(os.path.dirname(TIMES_CSV_OUTPUT_FILE), exist_ok=True)

    # 加载数据
    data = load_times(times_file_path, max_channels)
    print("加载的数据:")
    print(data.head())

    # 分割数据
    split_result = split_times(data)
    print("\n分割的时间数据:")
    for key, value in list(split_result.items())[:5]:  # 仅展示前5项
        print(f"{key}: {value}")

    # 保存数据到 CSV
    save_times_to_csv(split_result, TIMES_CSV_OUTPUT_FILE)
    return TIMES_CSV_OUTPUT_FILE

def load_grid(file_path):
    """
    加载并清洗 grid 文件。
    对应原文件 3update_grid.py 的功能
    """
    print(f"=== 步骤3: 加载网格数据 ===")
    print(f"加载 grid 文件: {file_path}")
    grid_df = pd.read_csv(
        file_path,
        sep="\t",  # 使用制表符分隔
        names=["x_distance", "depth", "time", "velocity", "density"],
        skiprows=1,  # 跳过标题行
        dtype={"x_distance": float, "depth": float, "time": float, "velocity": float, "density": float},
        na_values=["-99999"]  # 将 -99999 标记为 NaN
    )
    print(f"加载完成，原始 grid 数据行数: {len(grid_df)}")
    grid_df = grid_df.dropna()  # 删除包含 NaN 的行
    print(f"清洗完成，剩余 grid 数据行数: {len(grid_df)}")
    return grid_df

def load_times_csv(file_path):
    """
    加载并解析 times CSV 文件。
    """
    print(f"加载 times CSV 文件: {file_path}")
    times_df = pd.read_csv(file_path)
    print(f"times 文件加载完成，原始数据行数: {len(times_df)}")
    times_df['times'] = times_df['times'].apply(eval)  # 将字符串解析为列表
    print(f"解析 times 字段完成，示例: {times_df['times'].iloc[:5]}")
    return times_df

def update_grid(grid_df, times_df, channels, x_interval):
    """
    更新 grid 数据，根据 times 数据中定义的时间范围。
    """
    print("开始更新 grid 数据...")
    updated_count = 0
    for idx, row in times_df.iterrows():
        x_distance = row['x_distance']
        channel = row['channel']
        times = sorted(row['times'])  # 确保时间区间有序

        # 在 grid 中找到对应的时间区间
        mask = (
            (grid_df['x_distance'] == x_distance) &
            (grid_df['time'] >= times[0]) &
            (grid_df['time'] <= times[1])
        )
        matches = grid_df[mask]

        if not matches.empty:
            grid_df.loc[mask, ['velocity', 'density']] = [UPDATE_VELOCITY, UPDATE_DENSITY]
            updated_count += len(matches)

    print(f"更新完成，总共更新行数: {updated_count}")
    return grid_df

def save_grid(grid_df, output_path):
    """
    保存更新后的 grid 数据到文件，添加列标题。
    """
    print(f"保存更新后的 grid 文件到: {output_path}")
    grid_df.to_csv(
        output_path,
        sep="\t",
        index=False,
        header=["X Distance", "Depth", "Time", "Velocity", "Density"]
    )
    print(f"保存完成: {output_path}")

def process_grid_update(grid_file, times_file, output_file):
    """
    处理网格更新的主要函数
    """
    # 加载 grid 和 times 数据
    grid_df = load_grid(grid_file)
    times_df = load_times_csv(times_file)

    # 更新 grid 数据
    updated_grid_df = update_grid(grid_df, times_df, GRID_CHANNELS, GRID_X_INTERVAL)

    # 保存更新后的 grid 数据
    save_grid(updated_grid_df, output_file)
    return output_file

def filter_grid_by_time(grid_df, start_distance, end_distance, channels, time_interval):
    """
    根据时间采样间隔和其他参数过滤 grid 数据并生成道号。
    对应原文件 4Update_impedance.py 的功能
    """
    print(f"=== 步骤4: 计算波阻抗 ===")
    print(f"起始距离: {start_distance}, 终止距离: {end_distance}, 道数: {channels}, 时间采样间隔: {time_interval}ms")

    # 获取原始数据中实际存在的x_distance值
    available_distances = sorted(grid_df['x_distance'].unique())
    print(f"原始数据中可用的x_distance值数量: {len(available_distances)}")
    
    # 选择需要的x_distance值（从start_distance到end_distance，间隔为10）
    x_distances = []
    for dist in available_distances:
        if start_distance <= dist <= end_distance:
            x_distances.append(dist)
    
    print(f"选择的x_distance值数量: {len(x_distances)}")
    print(f"第一个值: {x_distances[0]}")
    print(f"最后一个值: {x_distances[-1]}")

    # 过滤符合 x_distance 范围的数据
    filtered_df = grid_df[grid_df['x_distance'].isin(x_distances)]
    print(f"过滤后的 x_distance 数据行数: {len(filtered_df)}")

    # 对时间进行修正和对齐，并处理浮点数精度问题
    filtered_df['time'] = (filtered_df['time'] / time_interval).round() * time_interval
    filtered_df['time'] = filtered_df['time'].round(3)  # 保留小数点后三位

    # 按时间采样间隔过滤 (去除重复的时间点)
    filtered_df = filtered_df.drop_duplicates(subset=['x_distance', 'time'])

    # 计算波阻抗
    filtered_df['impedance'] = filtered_df['velocity'] * filtered_df['density']

    # 添加道号 (No.)
    filtered_df['No.'] = filtered_df['x_distance'].apply(lambda x: x_distances.index(x) + 1)

    # 保留需要的列并重命名
    filtered_df = filtered_df[['No.', 'x_distance', 'depth', 'time', 'impedance']]
    filtered_df.rename(columns={
        'x_distance': 'X Distance',
        'depth': 'Depth',
        'time': 'Time',
        'impedance': 'Impedance'
    }, inplace=True)

    return filtered_df

def save_filtered_grid(filtered_df, output_path):
    """
    保存过滤后的数据到新文件。
    """
    print(f"保存数据到: {output_path}")
    filtered_df.to_csv(output_path, sep="\t", index=False)
    print(f"保存完成: {output_path}")

def process_impedance_calculation(input_file, output_file):
    """
    处理波阻抗计算的主要函数
    """
    # 加载数据后，检查并修正列名
    grid_df = pd.read_csv(
        input_file,
        sep="\t",
        header=0,
        dtype={"X Distance": float, "Depth": float, "Time": float, "Velocity": float, "Density": float},
        na_values=["-99999"],
        engine="python"
    )

    # 标准化列名
    grid_df.columns = grid_df.columns.str.strip().str.lower().str.replace(' ', '_')

    # 打印列名以确认
    print("修正后的列名:", grid_df.columns)

    # 输入参数
    start_distance = IMPEDANCE_START_DISTANCE
    end_distance = IMPEDANCE_END_DISTANCE
    channels = IMPEDANCE_CHANNELS
    time_interval = TIME_SAMPLE_INTERVAL

    # 过滤数据
    filtered_df = filter_grid_by_time(grid_df, start_distance, end_distance, channels, time_interval)

    # 保存结果
    save_filtered_grid(filtered_df, output_file)
    return output_file

def main():
    """
    主函数：执行完整的数据处理流程
    """
    print("开始执行完整的地震数据处理流程...")
    
    # 显示配置信息
    print_config_info()
    
    # 步骤0: 转换.asc文件为.txt文件
    converted_txt_file = convert_asc_to_txt()
    
    # 步骤1: 生成初始数据
    grid_file = generate_initial_data()
    
    # 步骤2: 处理时间数据（使用转换后的txt文件）
    times_csv_file = process_times_data(converted_txt_file)
    
    # 步骤3: 更新网格数据
    os.makedirs(os.path.dirname(GRID_UPDATED_OUTPUT_FILE), exist_ok=True)
    process_grid_update(grid_file, times_csv_file, GRID_UPDATED_OUTPUT_FILE)
    
    # 步骤4: 计算波阻抗
    os.makedirs(os.path.dirname(FINAL_OUTPUT_FILE), exist_ok=True)
    process_impedance_calculation(GRID_UPDATED_OUTPUT_FILE, FINAL_OUTPUT_FILE)
    
    print(f"\n=== 处理完成 ===")
    print(f"最终结果保存到: {FINAL_OUTPUT_FILE}")
    print("所有步骤已成功完成！")

if __name__ == "__main__":
    main() 