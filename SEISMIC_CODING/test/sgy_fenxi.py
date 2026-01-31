import os
from obspy.io.segy.segy import _read_segy

sgy_file = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy'
segy = _read_segy(sgy_file)
print("binary_file_header属性：")
for attr in dir(segy.binary_file_header):
    if not attr.startswith('_'):
        try:
            print(f"{attr}: {getattr(segy.binary_file_header, attr)}")
        except Exception as e:
            print(f"{attr}: 读取出错({e})")

print("\n第一个trace header属性：")
for attr in dir(segy.traces[0].header):
    if not attr.startswith('_'):
        try:
            print(f"{attr}: {getattr(segy.traces[0].header, attr)}")
        except Exception as e:
            print(f"{attr}: 读取出错({e})")

output_path = r'C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re_fx.sgy'

# 获取采样间隔（单位：微秒），转换为毫秒
sample_interval_us = segy.binary_file_header.sample_interval_in_microseconds  # 单位: 微秒
sample_interval_ms = sample_interval_us / 1000.0

with open(output_path, 'w') as f:
    f.write('trace\ttime\tsmp\n')
    for trace_idx, trace in enumerate(segy.traces, 1):
        npts = len(trace.data)
        for time_idx in range(npts):
            time_ms = time_idx * sample_interval_ms
            smp = trace.data[time_idx]
            f.write(f'{trace_idx}\t\t{time_ms:.3f}\t{smp:.1f}\n')

print(f"提取完成，结果已保存到 {output_path}")