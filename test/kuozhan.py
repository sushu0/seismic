# -*- coding: utf-8 -*-
"""
kuozhan.py

将指定的 SEGY 文件中每一道 trace 扩展（不足末尾补零，超出截断）到 100 个采样点，
输出到另一个 SEGY 文件。运行时无需任何参数，直接修改下面的路径即可。
依赖：obspy, numpy （pip install obspy numpy）
"""

import os
import numpy as np
from obspy import read
from obspy.core import Stream, Trace

# ————— 在这里填你的文件绝对路径 —————
INPUT_SGY  = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\test.sgy"  # 输入SEGY文件路径（相对路径）
OUTPUT_SGY = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\test_kz.sgy"   # 输出SEGY文件路径（相对路径）
TARGET_NSAMP = 100  # 目标采样点数，每道trace都扩展/截断为100点
# ————————————————————————————————

def process_traces_with_obspy(input_path, target_nsamp):
    """
    用 obspy 读取 SEGY 文件，并将每道 trace 扩展/截断为 target_nsamp 个采样点。
    返回处理后的 Stream 对象。
    """
    st = read(input_path, format="SEGY")  # 读取SEGY文件为Stream对象
    new_stream = Stream()
    for tr in st:
        data = tr.data.astype(np.float32)
        # 扩展或截断
        if len(data) < target_nsamp:
            new_data = np.pad(data, (0, target_nsamp - len(data)), 'constant')
        else:
            new_data = data[:target_nsamp]
        # 创建新Trace，继承原有头信息
        new_tr = Trace(data=new_data, header=tr.stats)
        new_stream.append(new_tr)
    return new_stream

if __name__ == "__main__":
    print("用 obspy 读取 traces 并扩展/截断到指定采样点数 …")
    # 处理所有trace
    processed_stream = process_traces_with_obspy(INPUT_SGY, TARGET_NSAMP)
    print(f"共处理 {len(processed_stream)} 条道，每道采样点数：{TARGET_NSAMP}")
    # 写新文件
    processed_stream.write(OUTPUT_SGY, format="SEGY")
    print(f"写入成功：{OUTPUT_SGY}")