"""
生成类似参考图风格的波阻抗预测剖面可视化
风格特点：
- 简洁的单图布局
- viridis颜色映射
- 中文标题和标签
- 时间轴(ms)和道号(道号)
- 1e7量级的colorbar
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import json
import sys
import os
import segyio

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.size'] = 12

# 添加项目路径
sys.path.insert(0, r'D:\SEISMIC_CODING\new')

def load_data_and_model(freq):
    """加载数据和模型进行推理"""
    from train_v6 import InversionNet
    from scipy.signal import butter, filtfilt
    
    # 数据路径
    base_data_path = r'D:\SEISMIC_CODING\zmy_data\01\data'
    results_path = rf'D:\SEISMIC_CODING\new\results\01_{freq}Hz_v6'
    
    # 读取地震数据 (使用segyio正确读取)
    seismic_file = os.path.join(base_data_path, f'01_{freq}Hz_re.sgy')
    with segyio.open(seismic_file, 'r', ignore_geometry=True) as f:
        seismic = np.stack([f.trace[i] for i in range(f.tracecount)], axis=0).astype(np.float32)
    
    n_traces = seismic.shape[0]
    
    # 读取阻抗数据
    impedance_file = os.path.join(base_data_path, f'01_{freq}Hz_04.txt')
    impedance_raw = np.loadtxt(impedance_file, usecols=4, skiprows=1).astype(np.float32)
    n_samples = len(impedance_raw) // n_traces
    impedance = impedance_raw.reshape(n_traces, n_samples)
    
    # 确保长度一致
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]
    
    print(f'{freq}Hz - Seismic shape: {seismic.shape}, Impedance shape: {impedance.shape}')
    print(f'Impedance range: {impedance.min():.2e} - {impedance.max():.2e}')
    
    # 加载归一化参数
    norm_file = os.path.join(results_path, 'norm_stats.json')
    with open(norm_file, 'r') as f:
        norm_stats = json.load(f)
    
    # 加载模型
    checkpoint_file = os.path.join(results_path, 'checkpoints', 'best.pt')
    checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    
    # 创建模型 (V6版本参数: in_ch=2, base=48)
    model = InversionNet(in_ch=2, base=48)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # 归一化地震数据
    seis_mean = norm_stats['seis_mean']
    seis_std = norm_stats['seis_std']
    seismic_norm = (seismic - seis_mean) / (seis_std + 1e-8)
    
    # 高通滤波
    cutoff = norm_stats.get('highpass_cutoff', 12)
    nyq = 500  # 采样率/2
    
    try:
        b, a = butter(4, cutoff / nyq, btype='high')
        seismic_hp = filtfilt(b, a, seismic, axis=-1).astype(np.float32)
        seismic_hp_norm = seismic_hp / (np.std(seismic_hp, axis=1, keepdims=True) + 1e-6)
    except:
        seismic_hp_norm = seismic_norm
    
    # 转换为tensor
    X = np.stack([seismic_norm, seismic_hp_norm], axis=1)  # (100, 2, n_samples)
    X_tensor = torch.FloatTensor(X)
    
    # 推理
    with torch.no_grad():
        pred_norm = model(X_tensor).squeeze(1).numpy()  # (100, n_samples)
    
    # 反归一化预测
    imp_mean = norm_stats['imp_mean']
    imp_std = norm_stats['imp_std']
    pred = pred_norm * imp_std + imp_mean
    
    print(f'Prediction range: {pred.min():.2e} - {pred.max():.2e}')
    
    return seismic, impedance, pred


def plot_impedance_section(data, title, save_path, vmin=None, vmax=None, time_max=100):
    """
    绘制类似参考图风格的波阻抗剖面图
    
    参数:
        data: 2D数组 (n_traces, n_samples)
        title: 图标题
        save_path: 保存路径
        vmin, vmax: colorbar范围
        time_max: 最大时间(ms)
    """
    n_traces, n_samples = data.shape
    
    # 对数据进行下采样以便更好地可视化（类似参考图的稀疏感）
    # 目标：约100个时间点
    target_samples = 100
    if n_samples > target_samples:
        step = n_samples // target_samples
        data_display = data[:, ::step][:, :target_samples]
    else:
        data_display = data
    
    n_traces_disp, n_samples_disp = data_display.shape
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 6), dpi=150)
    
    # 计算显示范围
    if vmin is None:
        vmin = np.percentile(data_display, 1)
    if vmax is None:
        vmax = np.percentile(data_display, 99)
    
    # 绘制热图
    extent = [0, n_traces_disp, time_max, 0]  # [left, right, bottom, top]
    im = ax.imshow(data_display.T, aspect='auto', cmap='viridis', 
                   extent=extent, vmin=vmin, vmax=vmax)
    
    # 设置标签
    ax.set_xlabel('道号', fontsize=14)
    ax.set_ylabel('时间（ms）', fontsize=14)
    ax.set_title(title, fontsize=16, fontweight='bold')
    
    # 设置刻度
    ax.set_xlim(0, n_traces_disp)
    ax.set_ylim(time_max, 0)
    
    # 添加colorbar (增加间距)
    cbar = plt.colorbar(im, ax=ax, pad=0.08, shrink=0.9)
    cbar.set_label('波阻抗（m/s*g/cm3）', fontsize=12)
    
    # 设置colorbar为科学计数法
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.88)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f'已保存: {save_path}')


def plot_comparison_horizontal(true_data, pred_data, freq, save_dir):
    """
    绘制真实和预测的水平对比图（并排）
    """
    # 下采样
    n_traces, n_samples = true_data.shape
    target_samples = 100
    if n_samples > target_samples:
        step = n_samples // target_samples
        true_disp = true_data[:, ::step][:, :target_samples]
        pred_disp = pred_data[:, ::step][:, :target_samples]
    else:
        true_disp = true_data
        pred_disp = pred_data
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=150)
    
    n_traces_disp, n_samples_disp = true_disp.shape
    time_max = 100  # ms
    
    # 统一颜色范围
    vmin = min(np.percentile(true_disp, 1), np.percentile(pred_disp, 1))
    vmax = max(np.percentile(true_disp, 99), np.percentile(pred_disp, 99))
    
    extent = [0, n_traces_disp, time_max, 0]
    
    # 真实波阻抗
    im1 = axes[0].imshow(true_disp.T, aspect='auto', cmap='viridis',
                          extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('道号', fontsize=12)
    axes[0].set_ylabel('时间（ms）', fontsize=12)
    axes[0].set_title(f'{freq}Hz 真实波阻抗剖面', fontsize=14, fontweight='bold')
    
    # 预测波阻抗
    im2 = axes[1].imshow(pred_disp.T, aspect='auto', cmap='viridis',
                          extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('道号', fontsize=12)
    axes[1].set_ylabel('时间（ms）', fontsize=12)
    axes[1].set_title(f'{freq}Hz 预测波阻抗剖面', fontsize=14, fontweight='bold')
    
    # 添加共享colorbar (放在最右边)
    cbar = fig.colorbar(im2, ax=axes, orientation='vertical', 
                        fraction=0.03, pad=0.08, shrink=0.85)
    cbar.set_label('波阻抗（m/s*g/cm3）', fontsize=11)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    save_path = os.path.join(save_dir, f'{freq}Hz_true_vs_pred_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f'已保存: {save_path}')


def main():
    """主函数"""
    save_dir = r'D:\SEISMIC_CODING\comparison01\figures_v6'
    os.makedirs(save_dir, exist_ok=True)
    
    # 处理所有可用频率
    frequencies = [20, 30, 40]  # 检查可用的频率
    
    for freq in frequencies:
        try:
            print(f'\n处理 {freq}Hz 数据...')
            
            # 检查模型是否存在
            results_path = rf'D:\SEISMIC_CODING\new\results\01_{freq}Hz_v6'
            if not os.path.exists(results_path):
                print(f'  跳过 {freq}Hz: 未找到模型')
                continue
            
            # 加载数据和进行预测
            seismic, true_imp, pred_imp = load_data_and_model(freq)
            
            # 绘制单独的预测剖面图（参考图风格）
            plot_impedance_section(
                pred_imp,
                f'{freq}Hz  预测波阻抗剖面',
                os.path.join(save_dir, f'{freq}Hz_predicted_impedance_section.png'),
                time_max=100
            )
            
            # 绘制单独的真实剖面图
            plot_impedance_section(
                true_imp,
                f'{freq}Hz  真实波阻抗剖面',
                os.path.join(save_dir, f'{freq}Hz_true_impedance_section.png'),
                time_max=100
            )
            
            # 绘制真实vs预测对比图
            plot_comparison_horizontal(true_imp, pred_imp, freq, save_dir)
            
        except Exception as e:
            print(f'处理 {freq}Hz 时出错: {e}')
            import traceback
            traceback.print_exc()
    
    print(f'\n✓ 所有图片已保存到: {save_dir}')


if __name__ == '__main__':
    main()
