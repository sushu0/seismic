# 切割upper
# 处理完成，文件已保存至：C:/Users/ZYH/Desktop/python_code/SEISMIC_CODING/Data/ShiJi/upper_cut.segy
# 当前道数：430
# 单道采样点数：500
# 采样间隔保持：0.001秒
# 示例数据（前10点）： [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 切割lower
# 处理完成，文件已保存至：C:/Users/ZYH/Desktop/python_code/SEISMIC_CODING/Data/ShiJi/lower_cut.segy
# 当前道数：430
# 单道采样点数：500
# 采样间隔保持：0.001秒
# 示例数据（前10点）： [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# 开始合并
# 处理完成，输出文件：C:/Users/ZYH/Desktop/python_code/SEISMIC_CODING/Data/ShiJi/Real430.segy
# 合并后道数：430
# 最终采样点数：500
# PS C:\Users\ZYH\Desktop\python_code\SEISMIC_CODING>
import obspy
import numpy as np
from obspy.core import Trace
import sys
from scipy import interpolate
import os

# 切割upper
print("切割upper")
input_path_upper = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/upper.sgy'
output_path_upper = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/upper_cut.segy'
st = obspy.read(input_path_upper)
st_cut = st[0:430]
for trace in st_cut:
    # 保留前500采样点并保持float32类型
    trace.data = trace.data[:500].astype(np.float32)
st_cut.write(output_path_upper, format="SEGY")
# 验证结果
print(f"处理完成，文件已保存至：{output_path_upper}")
print(f"当前道数：{len(st_cut)}")
print(f"单道采样点数：{st_cut[0].stats.npts}")
print(f"采样间隔保持：{st_cut[0].stats.delta}秒")
print("示例数据（前10点）：", st_cut[0].data[:10])

# 切割lower
print("切割lower")
input_path_lower = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/lower.sgy'
output_path_lower = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/lower_cut.segy'
st = obspy.read(input_path_lower)
st_cut = st[0:430]
for trace in st_cut:
    # 保留前500采样点并保持float32类型
    trace.data = trace.data[:500].astype(np.float32)
st_cut.write(output_path_lower, format="SEGY")
# 验证结果
print(f"处理完成，文件已保存至：{output_path_lower}")
print(f"当前道数：{len(st_cut)}")
print(f"单道采样点数：{st_cut[0].stats.npts}")
print(f"采样间隔保持：{st_cut[0].stats.delta}秒")
print("示例数据（前10点）：", st_cut[0].data[:10])

# 合并upper_cut和lower_cut
def find_transition_points(data):
    """查找振幅突变位置，避免越界"""
    # 查找upper截止点（振幅>0 → 0）
    upper_cut = len(data)
    for i in range(len(data)-21):
        if (data[i] != 0 and data[i+1] == 0 and
            data[i+3] == 0 and data[i+5] == 0 and
            data[i+10] == 0 and data[i+20] == 0):
            upper_cut = i+1
            break
    # 查找lower起始点（0 → 振幅>0）
    lower_start = 0
    for i in range(21, len(data)-1):
        if (data[i-20] == 0 and data[i-10] == 0 and
            data[i-3] == 0 and data[i-1] == 0 and
            data[i] == 0 and data[i+1] != 0):
            lower_start = i+1
            break
    return upper_cut, lower_start

def process_traces(upper_trace, lower_trace):
    # 提取振幅数据
    upper_data = upper_trace.data.astype(np.float32)
    lower_data = lower_trace.data.astype(np.float32)
    # 查找关键位置
    upper_cut, lower_start = find_transition_points(upper_data)
    # 截取有效部分
    valid_upper = upper_data[:upper_cut]
    valid_lower = lower_data[lower_start:]
    # 合并并限制长度
    combined = np.concatenate([valid_upper, valid_lower])
    # 如果合并后不足500，补零
    if len(combined) < 500:
        combined = np.pad(combined, (0, 500-len(combined)), 'constant')
    return combined[:500]

print("开始合并")
upper_path = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/upper_cut.segy'
lower_path = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/lower_cut.segy'
output_path = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/Real430.segy'

try:
    # 读取数据并验证
    st_upper = obspy.read(upper_path)
    st_lower = obspy.read(lower_path)
    # 基础校验
    assert len(st_upper) == len(st_lower), "道数不匹配"
    assert all(u.stats.npts == 500 for u in st_upper), "Upper采样点数异常"
    assert all(l.stats.npts == 500 for l in st_lower), "Lower采样点数异常"
    # 创建新数据流
    merged_stream = obspy.Stream()
    # 逐道处理
    for u_tr, l_tr in zip(st_upper, st_lower):
        # 处理数据
        new_data = process_traces(u_tr, l_tr)
        # 创建新Trace（继承upper的头信息）
        new_tr = u_tr.copy()
        new_tr.data = new_data
        new_tr.stats.npts = len(new_data)
        merged_stream.append(new_tr)
    # 写入文件（强制500采样点）
    merged_stream.write(output_path, format="SEGY", data_encoding=5, byteorder=sys.byteorder)
    print(f"处理完成，输出文件：{output_path}")
    print(f"合并后道数：{len(merged_stream)}")
    print(f"最终采样点数：{merged_stream[0].stats.npts}")
except Exception as e:
    print(f"处理失败：{str(e)}")
