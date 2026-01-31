import pandas as pd

def load_grid(file_path):
    """
    加载并清洗 grid 文件。
    """
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

def load_times(file_path):
    """
    加载并解析 times 文件。
    """
    print(f"加载 times 文件: {file_path}")
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

        # 调试输出当前处理的 x_distance 和时间区间
        # print(f"处理 x_distance: {x_distance}, channel: {channel}, 时间区间: {times}")

        # 在 grid 中找到对应的时间区间
        mask = (
            (grid_df['x_distance'] == x_distance) &
            (grid_df['time'] >= times[0]) &
            (grid_df['time'] <= times[1])
        )
        matches = grid_df[mask]
        # print(f"找到匹配的行数: {len(matches)}")

        if not matches.empty:
            grid_df.loc[mask, ['velocity', 'density']] = [4000, 2456.37]
            updated_count += len(matches)

    # print(f"更新完成，总共更新行数: {updated_count}")
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

# 加载数据
grid_file = "zmy_data/01/data/01_40Hz_01.txt"  # 输入的 grid 文件路径
times_file = "zmy_data/01/data/01_40Hz_02.txt"  # 输入的 times 文件路径
output_file = "zmy_data/01/data/01_40Hz_03.txt"  # 输出的文件路径

# 加载 grid 和 times 数据
grid_df = load_grid(grid_file)
times_df = load_times(times_file)

# 更新 grid 数据
channels = 7
x_interval = 10
updated_grid_df = update_grid(grid_df, times_df, channels, x_interval)

# 保存更新后的 grid 数据
save_grid(updated_grid_df, output_file)

print(f"程序执行完成，结果保存到 {output_file}")
