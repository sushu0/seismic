import obspy
import numpy as np
from obspy.core import Trace  # 修改导入方式
import os
import sys  # 添加系统模块

# =====================
# 函数：查找振幅突变点
# =====================
def find_transition_points(data):
    """
    查找振幅突变位置，避免无效零值区域重复拼接。
    返回：
        upper_cut   —— upper数据的有效截止点（振幅>0到0的突变点）
        lower_start —— lower数据的有效起始点（0到振幅>0的突变点）
    """
    # 查找upper截止点（振幅>0 → 0）
    upper_cut = len(data)
    for i in range(len(data)-1):
        if data[i] != 0 and data[i+1] == 0 and data[i+5]==0 and data[i+3]==0 and data[i+10] == 0 and data[i+20] == 0:
            upper_cut = i+1  # 包含突变点
            break

    # 查找lower起始点（0 → 振幅>0）
    lower_start = 0
    for i in range(len(data)-1):
        if  i>50 and data[i-10] == 0 and data[i-20] == 0 and data[i-3] == 0 and data[i-1] == 0 and data[i] == 0 and data[i+1] != 0:
            lower_start = i+1  # 从突变后开始
            break

    return upper_cut, lower_start

# =====================
# 函数：处理单道数据拼接
# =====================
def process_traces(upper_trace, lower_trace):
    """
    对每一对（同一索引）道，拼接upper和lower的有效部分，生成新的500采样点数据。
    """
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
    return combined[:500]  # 严格限制500采样点

# =====================
# 主流程：批量拼接并写出新SEGY文件
# =====================
# 输入输出配置
upper_path = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/upper_cut.segy'  # upper输入文件路径
lower_path = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/lower_cut.segy'  # lower输入文件路径
output_path = 'C:/Users/22639/Desktop/SEISMIC_CODING/Data/ShiJi/Real.segy'      # 合并输出文件路径

try:
    # 读取数据并验证
    st_upper = obspy.read(upper_path)
    st_lower = obspy.read(lower_path)

    # 基础校验：道数和采样点数一致性
    assert len(st_upper) == len(st_lower), "道数不匹配"
    assert all(u.stats.npts == 500 for u in st_upper), "Upper采样点数异常"
    assert all(l.stats.npts == 500 for l in st_lower), "Lower采样点数异常"
    # 创建新数据流
    merged_stream = obspy.Stream()

    # 逐道处理
    for u_tr, l_tr in zip(st_upper, st_lower):
        # 处理数据，拼接有效部分
        new_data = process_traces(u_tr, l_tr)

        # 创建新Trace（继承upper的头信息）
        new_tr = Trace()
        new_tr.data = new_data
        new_tr.stats = u_tr.stats.copy()

        # 更新关键头字段
        new_tr.stats.npts = len(new_data)
        new_tr.stats.delta = u_tr.stats.delta  # 保持采样率不变

        merged_stream.append(new_tr)

    # 写入文件（强制500采样点）
    merged_stream.write(
        output_path,
        format="SEGY",
        data_encoding=5,
        byteorder=sys.byteorder)  # 使用系统默认字节顺序

    print(f"处理完成，输出文件：{output_path}")
    print(f"合并后道数：{len(merged_stream)}")
    print(f"最终采样点数：{merged_stream[0].stats.npts}")

except Exception as e:
    print(f"处理失败：{str(e)}")
