import segyio
import numpy as np
from scipy import interpolate
import os

# ****************** 文件路径配置 ******************
# 输入文件路径配置
sgy_file_path = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_30Hz_xiufu.sgy'
# 输出文件路径配置
new_sgy_file_path = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'

# =====================
# Step 1: 使用segyio读取原始SGY文件
# =====================
print("正在读取SEGY文件...")
with segyio.open(sgy_file_path, "r", ignore_geometry=True) as f:
    # 读取所有地震道数据
    seismic_data = np.stack([f.trace[i] for i in range(f.tracecount)])
    # 获取采样间隔（微秒转换为秒）
    original_sampling_interval = f.bin[segyio.BinField.Interval] * 1e-6
    # 保存原始文件头信息以便复制
    original_bin = dict(f.bin)
    original_trace_headers = []
    for i in range(f.tracecount):
        original_trace_headers.append(dict(f.header[i]))
    
    print(f"原始数据形状: {seismic_data.shape}")
    print(f"原始采样间隔: {original_sampling_interval:.6f} 秒")

# =====================
# Step 2: 参数设置
# =====================
target_duration = 0.1                           # 目标总时长100ms（0.1秒）
target_sampling_interval = 0.00001              # 目标采样间隔0.01ms（0.00001秒）

# 计算扩展到100ms后的点数（原始采样间隔下）
target_npts_extended = int(target_duration / original_sampling_interval) + 1  # 101点

# =====================
# Step 3: 对每个地震道进行扩展和重采样
# =====================
print("正在进行重采样...")
resampled_traces = []
for i in range(seismic_data.shape[0]):
    data = seismic_data[i].copy()
    npts_original = len(data)

    # 扩展数据到100ms总时长
    if npts_original < target_npts_extended:
        # 补零
        extended_data = np.pad(data, (0, target_npts_extended - npts_original), mode='constant')
    elif npts_original > target_npts_extended:
        # 截断
        extended_data = data[:target_npts_extended]
    else:
        extended_data = data

    # 生成扩展后的时间轴（0到100ms，间隔1ms）
    time_extended = np.linspace(0, target_duration, target_npts_extended)

    # 生成目标时间轴（0到100ms，间隔0.01ms）
    n_target = int(target_duration / target_sampling_interval) + 1
    time_target = np.linspace(0, target_duration, n_target)

    # 线性插值重采样
    f_interp = interpolate.interp1d(time_extended, extended_data, kind='linear', fill_value='extrapolate')
    new_data = f_interp(time_target)

    resampled_traces.append(new_data.astype(np.float32))

# 转换为numpy数组
resampled_data = np.array(resampled_traces)
print(f"重采样后数据形状: {resampled_data.shape}")

# =====================
# Step 4: 使用segyio保存重采样后的数据
# =====================
print("正在保存重采样后的数据...")
# 创建输出目录（如果不存在）
output_dir = os.path.dirname(new_sgy_file_path)
os.makedirs(output_dir, exist_ok=True)

# 生成新的采样点数组
new_samples = np.arange(resampled_data.shape[1])

# 创建SEGY规范对象
spec = segyio.spec()
spec.format = 1  # IEEE 32位浮点数格式
spec.sorting = 2  # 按inline排序
spec.samples = new_samples
spec.ilines = range(1, resampled_data.shape[0] + 1)
spec.xlines = range(1, resampled_data.shape[0] + 1)  # 每个道使用不同的xline

# 使用正确的规范对象创建SEGY文件
with segyio.create(new_sgy_file_path, spec) as f:
    
    # 设置文件头信息
    f.bin.update({
        segyio.BinField.Interval: int(target_sampling_interval * 1e6),  # 采样间隔（微秒）
        segyio.BinField.Traces: resampled_data.shape[0],                # 道数
        segyio.BinField.Samples: resampled_data.shape[1],               # 采样点数
        segyio.BinField.Format: 1,                                      # 数据格式
    })
    
    # 复制其他重要的文件头信息
    for key, value in original_bin.items():
        if key not in [segyio.BinField.Interval, segyio.BinField.Traces, segyio.BinField.Samples, segyio.BinField.Format]:
            try:
                f.bin[key] = value
            except:
                pass  # 忽略无法设置的字段
    
    # 写入地震道数据和道头
    for i in range(resampled_data.shape[0]):
        f.trace[i] = resampled_data[i]
        
        # 设置道头信息
        trace_header = f.header[i]
        
        # 复制原始道头信息（如果存在）
        if i < len(original_trace_headers):
            for key, value in original_trace_headers[i].items():
                try:
                    trace_header[key] = value
                except:
                    pass  # 忽略无法设置的字段
        
        # 更新关键的道头信息
        trace_header.update({
            segyio.TraceField.TRACE_SEQUENCE_LINE: i + 1,
            segyio.TraceField.TRACE_SEQUENCE_FILE: i + 1,
            segyio.TraceField.FieldRecord: 1,
            segyio.TraceField.TraceNumber: i + 1,
            segyio.TraceField.INLINE_3D: i + 1,
            segyio.TraceField.CROSSLINE_3D: i + 1,
            segyio.TraceField.DelayRecordingTime: 0,
            segyio.TraceField.TRACE_SAMPLE_COUNT: resampled_data.shape[1],  # 采样点数
            segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(target_sampling_interval * 1e6),  # 采样间隔（微秒）
        })

# =====================
# Step 5: 输出信息
# =====================
print(f"重采样后的数据保存为：{new_sgy_file_path}")
print(f"新的采样间隔：{target_sampling_interval:.6f} 秒")
print(f"数据点数：{resampled_data.shape[1]} 点")
print(f"地震道数：{resampled_data.shape[0]} 道")
print(f"数据前10个点：{resampled_data[0][:10]}")
print("重采样完成！")