import struct
import os

# ====== 用户可修改的文件路径 ======
INPUT_FILE = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_30Hz.sgy"   # 修改为你的输入文件路径
OUTPUT_FILE = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_30Hz_xiufu.sgy"  # 输出文件路径

# ====== SEG-Y 转换 Little-endian -> Big-endian ======
def convert_sgy_le_to_be(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"❌ 文件不存在: {input_path}")
        return

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        # 1. 复制文本头 (3200 bytes)
        text_header = fin.read(3200)
        fout.write(text_header)

        # 2. 读取并转换二进制头 (400 bytes)
        bin_header_le = fin.read(400)
        bin_header_list = list(struct.unpack("<200H", bin_header_le))  # 小端解析为 200 个 unsigned short
        bin_header_be = struct.pack(">" + "H"*200, *bin_header_list)   # 转换为大端
        fout.write(bin_header_be)

        # 获取重要字段（用于确定每个样点字节数）
        sample_interval = bin_header_list[8]   # bytes 17-18 (小端解析)
        samples_per_trace = bin_header_list[10] # bytes 21-22
        data_format = bin_header_list[12]      # bytes 25-26

        if data_format != 5:
            print(f"⚠ 警告: 检测到数据格式码 {data_format}，非 IEEE 4字节浮点 (code=5)，可能需要修改代码解析方式。")

        # 3. 循环读取 trace
        trace_count = 0
        trace_header_size = 240
        sample_size = 4  # IEEE float
        trace_data_size = samples_per_trace * sample_size

        while True:
            trace_header = fin.read(trace_header_size)
            if not trace_header or len(trace_header) < trace_header_size:
                break  # EOF

            # 转换 trace header 小端 -> 大端
            trace_header_list = list(struct.unpack("<" + "H"*(trace_header_size//2), trace_header))
            trace_header_be = struct.pack(">" + "H"*(trace_header_size//2), *trace_header_list)
            fout.write(trace_header_be)

            # 读取并转换 trace data
            trace_data = fin.read(trace_data_size)
            if len(trace_data) < trace_data_size:
                break  # 数据不完整

            trace_samples = struct.unpack("<" + "f"*samples_per_trace, trace_data)
            trace_data_be = struct.pack(">" + "f"*samples_per_trace, *trace_samples)
            fout.write(trace_data_be)

            trace_count += 1

        print(f"✅ 转换完成: {trace_count} 道数据已写入 {output_path}")
        print(f"原始文件: {input_path}")
        print(f"输出文件: {output_path}")


if __name__ == "__main__":
    convert_sgy_le_to_be(INPUT_FILE, OUTPUT_FILE)
