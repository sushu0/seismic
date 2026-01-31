import pandas as pd

def load_times(file_path, max_channels):
    """
    加载 times 文件并解析数据，返回 DataFrame 格式的结果。
    支持根据 `max_channels` 动态解析任意数量的通道数据。
    """
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


def save_to_csv(parsed_times, output_file):
    """
    将解析后的数据保存为 CSV 文件。
    """
    rows = []
    for (x_distance, channel), times in parsed_times.items():
        rows.append([x_distance, channel, times])
    df = pd.DataFrame(rows, columns=['x_distance', 'channel', 'times'])
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")


# 主函数执行代码
if __name__ == "__main__":
    # 文件路径 - 使用正确的文本文件
    file_path = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01.txt"

    output_file = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_40Hz_02.csv"
    
    # 确保输出目录存在
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 用户指定的 channel 数量
    max_channels = int(7)
    
    # 加载数据
    data = load_times(file_path, max_channels)
    print("Loaded Data:")
    print(data.head())

    # 分割数据
    split_result = split_times(data)
    print("\nSplit Times:")
    for key, value in list(split_result.items())[:1000]:  # 仅展示前10项
        print(f"{key}: {value}")

    # 保存数据到 CSV
    save_to_csv(split_result, output_file)
