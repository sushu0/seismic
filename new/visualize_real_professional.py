# -*- coding: utf-8 -*-
"""
生成专业风格的阻抗剖面可视化
模仿地震解释软件的显示效果
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def create_seismic_colormap():
    """创建类似地震解释软件的彩虹色图"""
    colors = [
        (0.0, 'darkblue'),
        (0.15, 'blue'),
        (0.3, 'cyan'),
        (0.45, 'green'),
        (0.55, 'yellow'),
        (0.7, 'orange'),
        (0.85, 'red'),
        (1.0, 'darkred')
    ]
    return LinearSegmentedColormap.from_list('seismic_rainbow', 
                                              [(c[0], c[1]) for c in colors])


def main():
    # ==================== 配置 ====================
    DATA_DIR = Path(r'D:\SEISMIC_CODING\new\real_data_inference')
    OUTPUT_DIR = DATA_DIR
    
    # 时间参数（根据SGY文件：时间范围2500-6000ms）
    TIME_START = 2500  # ms
    TIME_END = 6000    # ms
    
    # ==================== 加载数据 ====================
    print("Loading data...")
    seismic = np.load(DATA_DIR / 'seismic.npy')
    impedance = np.load(DATA_DIR / 'impedance_pred.npy')
    
    n_traces, n_samples = seismic.shape
    print(f"Seismic: {seismic.shape}")
    print(f"Impedance: {impedance.shape}")
    
    # 创建时间轴
    time_axis = np.linspace(TIME_START, TIME_END, n_samples)
    
    # ==================== 创建专业风格可视化 ====================
    print("Generating professional visualizations...")
    
    # 自定义colormap
    rainbow_cmap = create_seismic_colormap()
    
    # ----- 1. 阻抗剖面（彩虹色，类似专业软件） -----
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # 使用百分位数确定色标范围
    vmin = np.percentile(impedance, 2)
    vmax = np.percentile(impedance, 98)
    
    im = ax.imshow(impedance.T, 
                   aspect='auto', 
                   cmap=rainbow_cmap,
                   vmin=vmin, vmax=vmax,
                   extent=[0, n_traces, TIME_END, TIME_START],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14)
    ax.set_ylabel('时间 (ms)', fontsize=14)
    ax.set_title('波阻抗反演剖面 - V6模型', fontsize=16, fontweight='bold')
    
    # 添加colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('波阻抗 (kg/m²·s)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加网格
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'impedance_professional.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'impedance_professional.png'}")
    
    # ----- 2. 地震剖面 + 阻抗剖面对比 -----
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    
    # 地震剖面
    vmax_seis = np.percentile(np.abs(seismic), 98)
    im1 = axes[0].imshow(seismic.T, 
                         aspect='auto', 
                         cmap='gray',
                         vmin=-vmax_seis, vmax=vmax_seis,
                         extent=[0, n_traces, TIME_END, TIME_START],
                         interpolation='bilinear')
    axes[0].set_xlabel('道号 (Trace)', fontsize=12)
    axes[0].set_ylabel('时间 (ms)', fontsize=12)
    axes[0].set_title('地震剖面', fontsize=14, fontweight='bold')
    cbar1 = plt.colorbar(im1, ax=axes[0], pad=0.02, shrink=0.85)
    cbar1.set_label('振幅', fontsize=10)
    
    # 阻抗剖面
    im2 = axes[1].imshow(impedance.T, 
                         aspect='auto', 
                         cmap=rainbow_cmap,
                         vmin=vmin, vmax=vmax,
                         extent=[0, n_traces, TIME_END, TIME_START],
                         interpolation='bilinear')
    axes[1].set_xlabel('道号 (Trace)', fontsize=12)
    axes[1].set_ylabel('时间 (ms)', fontsize=12)
    axes[1].set_title('波阻抗反演剖面', fontsize=14, fontweight='bold')
    cbar2 = plt.colorbar(im2, ax=axes[1], pad=0.02, shrink=0.85)
    cbar2.set_label('波阻抗 (kg/m²·s)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'seismic_impedance_comparison.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'seismic_impedance_comparison.png'}")
    
    # ----- 3. 局部放大（中间1000道） -----
    start_trace = n_traces // 2 - 500
    end_trace = start_trace + 1000
    
    fig, ax = plt.subplots(figsize=(16, 10))
    
    imp_subset = impedance[start_trace:end_trace]
    vmin_sub = np.percentile(imp_subset, 2)
    vmax_sub = np.percentile(imp_subset, 98)
    
    im = ax.imshow(imp_subset.T, 
                   aspect='auto', 
                   cmap=rainbow_cmap,
                   vmin=vmin_sub, vmax=vmax_sub,
                   extent=[start_trace, end_trace, TIME_END, TIME_START],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14)
    ax.set_ylabel('时间 (ms)', fontsize=14)
    ax.set_title(f'波阻抗反演剖面 - 局部放大 (道 {start_trace}-{end_trace})', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('波阻抗 (kg/m²·s)', fontsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'impedance_zoomed.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'impedance_zoomed.png'}")
    
    # ----- 4. Jet colormap版本 -----
    fig, ax = plt.subplots(figsize=(16, 10))
    
    im = ax.imshow(impedance.T, 
                   aspect='auto', 
                   cmap='jet',
                   vmin=vmin, vmax=vmax,
                   extent=[0, n_traces, TIME_END, TIME_START],
                   interpolation='bilinear')
    
    ax.set_xlabel('道号 (Trace)', fontsize=14)
    ax.set_ylabel('时间 (ms)', fontsize=14)
    ax.set_title('波阻抗反演剖面 - V6模型 (Jet配色)', fontsize=16, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label('波阻抗 (kg/m²·s)', fontsize=12)
    
    ax.grid(True, alpha=0.3, linestyle='--', color='white')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'impedance_jet_style.png', dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'impedance_jet_style.png'}")
    
    # ----- 5. 水平切片（时间切片） -----
    # 选择几个关键时间点
    time_slices = [3000, 4000, 5000]  # ms
    
    fig, axes = plt.subplots(len(time_slices), 1, figsize=(16, 4*len(time_slices)))
    
    for i, t in enumerate(time_slices):
        # 找到对应的采样点
        idx = int((t - TIME_START) / (TIME_END - TIME_START) * n_samples)
        idx = max(0, min(idx, n_samples - 1))
        
        # 取该时间点附近的平均值
        slice_data = impedance[:, max(0, idx-5):min(n_samples, idx+5)].mean(axis=1)
        
        axes[i].plot(slice_data, 'b-', linewidth=0.8)
        axes[i].fill_between(range(len(slice_data)), slice_data.min(), slice_data, alpha=0.3)
        axes[i].set_xlabel('道号 (Trace)', fontsize=12)
        axes[i].set_ylabel('波阻抗', fontsize=12)
        axes[i].set_title(f'时间切片 @ {t}ms', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlim(0, n_traces)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'time_slices.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'time_slices.png'}")
    
    print("\n" + "=" * 50)
    print("所有专业可视化已生成完毕！")
    print(f"输出目录: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
