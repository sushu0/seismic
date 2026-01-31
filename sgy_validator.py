import os
import struct
import chardet
import segyio

# ========= 配置检测文件路径 =========
SGY_FILE = r"C:\Users\22639\Desktop\SEISMIC_CODING\zmy_data\01\data\01_20Hz_xiufu.sgy"
# ===================================

def detect_sgy(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"文件: {file_path}  大小: {file_size} bytes ({file_size/1024:.2f}KB)")

    with open(file_path, "rb") as f:
        text_header = f.read(3200)
        bin_header = f.read(400)

    # ==== 文本头检查 ====
    encoding_guess = chardet.detect(text_header)["encoding"] or "ascii"
    print("\n== 文本头 (3200 bytes) 检查 ==")
    print(f"推断编码: {encoding_guess.lower()}")
    try:
        preview = text_header.decode(encoding_guess, errors="replace")[:200]
        print(f"文本头前 200 字符预览（如果可读）:\n{preview}")
    except:
        print("⚠ 无法解码文本头")

    # ==== 二进制头检查（标准字节位置） ====
    print("\n== 二进制头 (400 bytes) 检查 ==")

    # big-endian
    be_sample_interval = struct.unpack(">H", bin_header[16:18])[0]
    be_samples_per_trace = struct.unpack(">H", bin_header[20:22])[0]
    be_format_code = struct.unpack(">H", bin_header[24:26])[0]

    # little-endian
    le_sample_interval = struct.unpack("<H", bin_header[16:18])[0]
    le_samples_per_trace = struct.unpack("<H", bin_header[20:22])[0]
    le_format_code = struct.unpack("<H", bin_header[24:26])[0]

    print(f"big-endian 解析 -> sample interval: {be_sample_interval} μs, samples/trace: {be_samples_per_trace}, data format code: {be_format_code}")
    print(f"little-endian 解析 -> sample interval: {le_sample_interval} μs, samples/trace: {le_samples_per_trace}, data format code: {le_format_code}")

    # 判断字节序
    endian = "big-endian"
    si, ns, fmt_code = be_sample_interval, be_samples_per_trace, be_format_code
    if be_sample_interval > 100000 or be_samples_per_trace > 20000:
        endian = "little-endian"
        si, ns, fmt_code = le_sample_interval, le_samples_per_trace, le_format_code

    print(f"最终选定字节序: {endian}")
    print(f"二进制头读出: sample interval = {si} μs, samples/trace = {ns}, data format code = {fmt_code}")

    # 格式码检查
    fmt_map = {
        1: "4-byte IBM floating point",
        2: "4-byte int",
        3: "2-byte int",
        4: "4-byte fixed-point",
        5: "4-byte IEEE floating point",
        8: "1-byte int"
    }
    if fmt_code in fmt_map:
        print(f"数据格式码 {fmt_code} -> {fmt_map[fmt_code]}")
    else:
        print(f"⚠ 数据格式码 ({fmt_code}) 不是常见标准码 {list(fmt_map.keys())}")

    # 估算 trace 数量
    if fmt_code in fmt_map and fmt_code in [1,2,4,5]:
        bytes_per_sample = 4
    elif fmt_code == 3:
        bytes_per_sample = 2
    elif fmt_code == 8:
        bytes_per_sample = 1
    else:
        bytes_per_sample = None

    if bytes_per_sample:
        est_trace_size = 240 + ns * bytes_per_sample
        trace_count = (file_size - 3600) // est_trace_size
        print(f"估算每道大小: {est_trace_size} bytes, 估计道数 = {trace_count}")

    # ==== 尝试用 segyio 读取 ====
    print("\n== segyio 读取测试 ==")
    try:
        with segyio.open(file_path, "r", ignore_geometry=True) as segyfile:
            traces = len(segyfile.trace)
            si_segyio = segyfile.bin[segyio.BinField.Interval]
            print(f"✅ segyio 成功读取: 道数={traces}, 采样间隔={si_segyio} μs")
    except Exception as e:
        print(f"⚠ segyio 无法正常读取: {e}")

if __name__ == "__main__":
    detect_sgy(SGY_FILE)