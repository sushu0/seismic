# -*- coding: utf-8 -*-
"""
SGY文件可视化脚本
- 色彩鲜艳
- 局部特点突出
- 优化处理速度（向量化+下采样）
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter, uniform_filter1d
from scipy.signal import hilbert
from pathlib import Path
import segyio

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_rainbow_colormap():
    """创建鲜艳的彩虹色图"""
    colors = [
        (0.0, '#000080'),    # 深蓝
        (0.1, '#0000FF'),    # 蓝
        (0.2, '#00BFFF'),    # 深天空蓝
        (0.3, '#00FFFF'),    # 青
        (0.4, '#00FF7F'),    # 春绿
        (0.5, '#FFFF00'),    # 黄
        (0.6, '#FFA500'),    # 橙
        (0.7, '#FF4500'),    # 橙红
        (0.8, '#FF0000'),    # 红
        (0.9, '#DC143C'),    # 深红
        (1.0, '#8B0000'),    # 暗红
    ]
    return LinearSegmentedColormap.from_list('vibrant_rainbow', 
                                              [(c[0], c[1]) for c in colors])


def create_seismic_wiggle_colormap():
    """创建地震波形红蓝色图（波形可视化用）"""
    colors = [
        (0.0, '#0000AA'),    # 深蓝（负值）
        (0.25, '#4444FF'),   # 蓝
        (0.5, '#FFFFFF'),    # 白（零值）
        (0.75, '#FF4444'),   # 红
        (1.0, '#AA0000'),    # 深红（正值）
    ]
    return LinearSegmentedColormap.from_list('seismic_rb', 
                                              [(c[0], c[1]) for c in colors])


def create_hot_cold_colormap():
    """创建冷暖对比色图"""
    colors = [
        (0.0, '#000033'),    # 深蓝黑
        (0.15, '#0066CC'),   # 蓝
        (0.3, '#33CCFF'),    # 天蓝
        (0.45, '#AAFFFF'),   # 浅青
        (0.5, '#FFFFFF'),    # 白
        (0.55, '#FFFFAA'),   # 浅黄
        (0.7, '#FFAA00'),    # 金橙
        (0.85, '#FF3300'),   # 红橙
        (1.0, '#660000'),    # 深红
    ]
    return LinearSegmentedColormap.from_list('hot_cold', 
                                              [(c[0], c[1]) for c in colors])


def enhance_local_contrast(data, sigma=3):
    """增强局部对比度（CLAHE风格）"""
    print("  应用局部对比度增强...")
    smooth = gaussian_filter(data.astype(np.float32), sigma=sigma)
    local_std = np.sqrt(gaussian_filter((data.astype(np.float32) - smooth)**2, sigma=sigma) + 1e-8)
    enhanced = (data - smooth) / local_std
    return enhanced


def apply_agc_fast(data, window_size=50):
    """快速向量化AGC处理"""
    print("  应用AGC增益控制...")
    data = data.astype(np.float32)
    # 沿时间轴（axis=0）计算滑动窗口RMS
    squared = data ** 2
    # 使用uniform_filter1d进行快速滑动窗口平均
    rms = np.sqrt(uniform_filter1d(squared, size=window_size, axis=0, mode='constant') + 1e-10)
    result = data / rms
    return result


def downsample_data(data, target_traces=2000, target_samples=None):
    """下采样数据以加速可视化"""
    n_samples, n_traces = data.shape
    
    # 计算下采样步长
    trace_step = max(1, n_traces // target_traces)
    sample_step = 1 if target_samples is None else max(1, n_samples // target_samples)
    
    # 下采样
    downsampled = data[::sample_step, ::trace_step]
    
    return downsampled, trace_step, sample_step


def read_sgy_file(filepath):
    """读取SGY文件"""
    print(f"正在读取: {filepath}")
    
    with segyio.open(str(filepath), "r", ignore_geometry=True) as f:
        # 获取基本信息
        n_traces = f.tracecount
        n_samples = len(f.samples)
        sample_interval = f.bin[segyio.BinField.Interval]
        
        print(f"  道数: {n_traces}")
        print(f"  采样点数: {n_samples}")
        print(f"  采样间隔: {sample_interval} μs ({sample_interval/1000} ms)")
        
        # 从文件名提取时间范围
        filename = Path(filepath).stem
        if '2500-6000' in filename:
            time_start = 2500
            time_end = 6000
        else:
            time_start = 0
            time_end = n_samples * sample_interval / 1000  # ms
        
        print(f"  时间范围: {time_start} - {time_end} ms")
        
        # 读取所有道
        data = segyio.tools.collect(f.trace[:]).T  # (traces, samples) -> (samples, traces)
        
    return data, time_start, time_end, sample_interval, n_traces


def main():
    # ==================== 配置 ====================
    SGY_FILE = Path(r'D:\SEISMIC_CODING\new\0908_Q1JB_PSTMR_ChengGuo-2500-6000.Sgy')
    OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\output_images')
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # ==================== 读取数据 ====================
    data, time_start, time_end, sample_interval, original_n_traces = read_sgy_file(SGY_FILE)
    n_samples, n_traces = data.shape
    print(f"\n原始数据形状: {data.shape} (采样点×道数)")
    print(f"数据范围: [{data.min():.2e}, {data.max():.2e}]")
    
    # ==================== 下采样用于可视化 ====================
    TARGET_TRACES = 2500  # 目标道数（用于可视化）
    print(f"\n下采样到 ~{TARGET_TRACES} 道...")
    data_ds, trace_step, sample_step = downsample_data(data, target_traces=TARGET_TRACES)
    n_samples_ds, n_traces_ds = data_ds.shape
    print(f"下采样后形状: {data_ds.shape}")
    
    # 创建时间轴
    time_axis = np.linspace(time_start, time_end, n_samples_ds)
    
    # ==================== 数据预处理 ====================
    print("\n数据预处理...")
    
    # 1. AGC处理 - 增强弱信号（使用快速向量化方法）
    data_agc = apply_agc_fast(data_ds, window_size=100)
    
    # 2. 局部对比度增强
    data_enhanced = enhance_local_contrast(data_agc, sigma=5)
    
    # ==================== 可视化 ====================
    print("\n生成可视化...")
    
    # 色图
    rainbow_cmap = create_rainbow_colormap()
    seismic_cmap = create_seismic_wiggle_colormap()
    hot_cold_cmap = create_hot_cold_colormap()
    
    # ---------- 图1: 原始地震剖面 (鲜艳红蓝) ----------
    fig, ax = plt.subplots(figsize=(18, 10))
    
    vmax = np.percentile(np.abs(data_ds), 99)
    
    im = ax.imshow(data_ds, 
                   aspect='auto', 
                   cmap=seismic_cmap,
                   vmin=-vmax, vmax=vmax,
                   extent=[0, original_n_traces, time_end, time_start],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('地震剖面 - 原始数据', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('振幅', fontsize=12)
    
    plt.tight_layout()
    output_path1 = OUTPUT_DIR / 'seismic_original.png'
    plt.savefig(output_path1, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path1}")
    
    # ---------- 图2: AGC增强后的地震剖面 ----------
    fig, ax = plt.subplots(figsize=(18, 10))
    
    vmax = np.percentile(np.abs(data_agc), 99)
    
    im = ax.imshow(data_agc, 
                   aspect='auto', 
                   cmap=seismic_cmap,
                   vmin=-vmax, vmax=vmax,
                   extent=[0, original_n_traces, time_end, time_start],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('地震剖面 - AGC增强', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('归一化振幅', fontsize=12)
    
    plt.tight_layout()
    output_path2 = OUTPUT_DIR / 'seismic_agc_enhanced.png'
    plt.savefig(output_path2, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path2}")
    
    # ---------- 图3: 局部对比增强 (彩虹色) ----------
    fig, ax = plt.subplots(figsize=(18, 10))
    
    vmin = np.percentile(data_enhanced, 1)
    vmax = np.percentile(data_enhanced, 99)
    
    im = ax.imshow(data_enhanced, 
                   aspect='auto', 
                   cmap=rainbow_cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[0, original_n_traces, time_end, time_start],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('地震剖面 - 局部特征增强 (彩虹色)', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('增强振幅', fontsize=12)
    
    # 添加网格以便于分析
    ax.grid(True, alpha=0.3, linestyle='--', color='white', linewidth=0.5)
    
    plt.tight_layout()
    output_path3 = OUTPUT_DIR / 'seismic_local_enhanced_rainbow.png'
    plt.savefig(output_path3, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path3}")
    
    # ---------- 图4: 瞬时振幅 (热图) ----------
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # 计算瞬时振幅（包络）
    print("  计算瞬时振幅...")
    analytic = hilbert(data_agc, axis=0)
    envelope = np.abs(analytic)
    
    vmax = np.percentile(envelope, 98)
    
    im = ax.imshow(envelope, 
                   aspect='auto', 
                   cmap='magma',  # 鲜艳的热图
                   vmin=0, vmax=vmax,
                   extent=[0, original_n_traces, time_end, time_start],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('瞬时振幅 (能量分布)', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('振幅', fontsize=12)
    
    plt.tight_layout()
    output_path4 = OUTPUT_DIR / 'seismic_instantaneous_amplitude.png'
    plt.savefig(output_path4, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path4}")
    
    # ---------- 图5: 综合对比图 (2x2) ----------
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    
    # 原始
    vmax_orig = np.percentile(np.abs(data_ds), 99)
    im1 = axes[0, 0].imshow(data_ds, aspect='auto', cmap=seismic_cmap,
                            vmin=-vmax_orig, vmax=vmax_orig,
                            extent=[0, original_n_traces, time_end, time_start],
                            interpolation='bilinear')
    axes[0, 0].set_title('原始数据', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('时间 (ms)', fontsize=12)
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # AGC
    vmax_agc = np.percentile(np.abs(data_agc), 99)
    im2 = axes[0, 1].imshow(data_agc, aspect='auto', cmap=seismic_cmap,
                            vmin=-vmax_agc, vmax=vmax_agc,
                            extent=[0, original_n_traces, time_end, time_start],
                            interpolation='bilinear')
    axes[0, 1].set_title('AGC增强', fontsize=14, fontweight='bold')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # 彩虹色局部增强
    im3 = axes[1, 0].imshow(data_enhanced, aspect='auto', cmap=rainbow_cmap,
                            vmin=np.percentile(data_enhanced, 1),
                            vmax=np.percentile(data_enhanced, 99),
                            extent=[0, original_n_traces, time_end, time_start],
                            interpolation='bilinear')
    axes[1, 0].set_title('局部特征增强 (彩虹色)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('道号', fontsize=12)
    axes[1, 0].set_ylabel('时间 (ms)', fontsize=12)
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.8)
    
    # 瞬时振幅
    im4 = axes[1, 1].imshow(envelope, aspect='auto', cmap='inferno',
                            vmin=0, vmax=np.percentile(envelope, 98),
                            extent=[0, original_n_traces, time_end, time_start],
                            interpolation='bilinear')
    axes[1, 1].set_title('瞬时振幅', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('道号', fontsize=12)
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.8)
    
    plt.suptitle('地震剖面多视角分析\n0908_Q1JB_PSTMR_ChengGuo (2500-6000ms)', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    output_path5 = OUTPUT_DIR / 'seismic_comprehensive_comparison.png'
    plt.savefig(output_path5, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path5}")
    
    # ---------- 图6: 高对比度冷暖色图 ----------
    fig, ax = plt.subplots(figsize=(18, 10))
    
    vmax = np.percentile(np.abs(data_agc), 98)
    
    im = ax.imshow(data_agc, 
                   aspect='auto', 
                   cmap=hot_cold_cmap,
                   vmin=-vmax, vmax=vmax,
                   extent=[0, original_n_traces, time_end, time_start],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14, fontweight='bold')
    ax.set_ylabel('时间 (ms)', fontsize=14, fontweight='bold')
    ax.set_title('地震剖面 - 高对比度冷暖色图', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('振幅', fontsize=12)
    
    plt.tight_layout()
    output_path6 = OUTPUT_DIR / 'seismic_hot_cold.png'
    plt.savefig(output_path6, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  已保存: {output_path6}")
    
    print(f"\n所有图像已保存到: {OUTPUT_DIR}")
    print("完成!")


if __name__ == '__main__':
    main()
