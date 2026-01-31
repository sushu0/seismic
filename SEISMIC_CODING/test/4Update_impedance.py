import pandas as pd

def filter_grid_by_time(grid_df, start_distance, end_distance, channels, time_interval):
    """
    根据时间采样间隔和其他参数过滤 grid 数据并生成道号。
    """
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

# 主程序
if __name__ == "__main__":
    input_file = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_40Hz_03.txt'  # 输入文件
    output_file = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_40Hz_04.txt'  # 输出文件

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
    # start_distance = 25
    # end_distance = 4975
    # channels = 100
    # time_interval = float(0.01)
    start_distance = 10
    end_distance = 1990
    channels = 100
    time_interval = float(0.01)

    # 过滤数据
    filtered_df = filter_grid_by_time(grid_df, start_distance, end_distance, channels, time_interval)

    # 保存结果
    save_filtered_grid(filtered_df, output_file)
    print(f"程序执行完成，结果保存到 {output_file}")
