"""
生成反演后的地震剖面图和四道对比图
"""
import numpy as np
import torch
from pathlib import Path
import yaml
from torch.utils.data import DataLoader

from seisinv.data.dataset import TraceDataset
from seisinv.models.baselines import UNet1D
from seisinv.losses.physics import ForwardModel
from seisinv.utils.plotting import save_section_with_contour, save_four_trace_comparison

def load_model_and_data(exp_name='baseline_unet1d'):
    """加载训练好的模型和测试数据"""
    
    # 加载配置
    config_path = 'configs/exp_baseline_unet.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载归一化统计量
    import json
    norm_path = Path(f'results/{exp_name}/norm_stats.json')
    with open(norm_path, 'r') as f:
        norm_stats = json.load(f)
    
    # 加载原始测试数据
    data_root = Path(cfg['data']['data_root'])
    test_seis = np.load(data_root / "test_seis.npy")
    test_imp = np.load(data_root / "test_imp.npy")
    
    # 手动归一化
    test_seis_norm = (test_seis - norm_stats['seis_mean']) / norm_stats['seis_std']
    test_imp_norm = (test_imp - norm_stats['imp_mean']) / norm_stats['imp_std']
    
    # 创建tensor dataset
    from torch.utils.data import TensorDataset
    test_ds = TensorDataset(
        torch.from_numpy(test_seis_norm[:, None, :].astype(np.float32)),
        torch.from_numpy(test_imp_norm[:, None, :].astype(np.float32))
    )
    
    def collate_fn(batch):
        seis_list, imp_list = zip(*batch)
        return {
            'seis': torch.stack(seis_list),
            'imp': torch.stack(imp_list)
        }
    
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    
    # 构建模型
    model = UNet1D(
        in_ch=1,
        out_ch=1,
        base=cfg['model']['base'],
        depth=cfg['model']['depth']
    ).to(device)
    
    # 加载最佳checkpoint
    ckpt_path = Path(f'results/{exp_name}/checkpoints/best.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    # 创建前向模型
    from seisinv.utils.wavelet import ricker
    phys_cfg = cfg['physics']
    wavelet = ricker(
        f0=phys_cfg['wavelet_f0_hz'],
        length=phys_cfg['wavelet_length_s'],
        dt=phys_cfg['dt_s']
    )
    wavelet_tensor = torch.from_numpy(wavelet).float().to(device)
    fm = ForwardModel(wavelet=wavelet_tensor, eps=float(phys_cfg['eps']))
    
    return model, test_loader, fm, device, norm_stats

def denormalize(data, stats):
    """反归一化"""
    mean = stats['mean']
    std = stats['std']
    return data * std + mean

@torch.no_grad()
def predict_and_forward_model(model, loader, fm, device):
    """预测阻抗并通过物理模型生成地震记录"""
    model.eval()
    
    ys = []
    yhats = []
    seis = []
    seis_hat = []
    
    for batch in loader:
        x = batch["seis"].to(device)
        y = batch["imp"].to(device)
        
        yhat = model(x)
        if isinstance(yhat, tuple):
            yhat, _ = yhat
        
        # 通过物理模型生成合成地震
        s_hat = fm(yhat)
        
        ys.append(y.cpu().numpy())
        yhats.append(yhat.cpu().numpy())
        seis.append(x.cpu().numpy())
        seis_hat.append(s_hat.cpu().numpy())
    
    y_true = np.concatenate(ys, axis=0)[:, 0, :]  # (N, T)
    y_pred = np.concatenate(yhats, axis=0)[:, 0, :]
    s_obs = np.concatenate(seis, axis=0)[:, 0, :]
    s_pred = np.concatenate(seis_hat, axis=0)[:, 0, :]
    
    return y_true, y_pred, s_obs, s_pred

def main():
    exp_name = 'baseline_unet1d'
    out_dir = Path(f'results/{exp_name}')
    
    print(f"加载模型和数据: {exp_name}")
    model, test_loader, fm, device, norm_stats = load_model_and_data(exp_name)
    
    print("进行预测和前向建模...")
    y_true, y_pred, s_obs, s_pred = predict_and_forward_model(model, test_loader, fm, device)
    
    print(f"数据形状: s_obs={s_obs.shape}, s_pred={s_pred.shape}")
    
    # 构建stats字典用于反归一化
    seismic_stats = {'mean': norm_stats['seis_mean'], 'std': norm_stats['seis_std']}
    impedance_stats = {'mean': norm_stats['imp_mean'], 'std': norm_stats['imp_std']}
    
    # 反归一化地震数据到原始域
    s_obs_denorm = denormalize(s_obs, seismic_stats)
    s_pred_denorm = denormalize(s_pred, seismic_stats)
    y_pred_denorm = denormalize(y_pred, impedance_stats)
    y_true_denorm = denormalize(y_true, impedance_stats)
    
    # 1. 生成反演后的地震剖面图 (合成地震记录)
    print("生成合成地震剖面图...")
    save_section_with_contour(
        s_pred_denorm,
        out_dir / 'seis_synthetic_section.png',
        title='Synthetic Seismic Section (from inverted impedance)',
        xlabel='Trace number',
        ylabel='Sample',
        cmap='seismic'
    )
    
    # 2. 生成预测阻抗剖面图 (带等值线)
    print("生成预测阻抗剖面图...")
    save_section_with_contour(
        y_pred_denorm,
        out_dir / 'pred_imp_section_contour.png',
        title='Predicted Impedance Section (npy)',
        xlabel='Trace number',
        ylabel='Sample',
        cmap='jet'
    )
    
    # 3. 生成四道地震记录对比图
    trace_ids = [2, 5, 12, 15]  # 对应原图中的 299, 599, 1699, 2299 (如果有足够的traces)
    if s_obs.shape[0] >= 16:
        trace_ids = [2, 5, 12, 15]
    else:
        trace_ids = list(range(min(4, s_obs.shape[0])))
    
    # 转换为实际的trace编号用于显示
    display_ids = [299, 599, 1699, 2299] if s_obs.shape[0] >= 16 else trace_ids
    
    print(f"生成四道地震对比图 (traces: {trace_ids})...")
    
    # 创建自定义的四道对比图,显示观测vs合成地震
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (trace_id, display_id) in enumerate(zip(trace_ids, display_ids)):
        ax = axes[i]
        t = np.arange(len(s_obs_denorm[trace_id]))
        
        # 地震记录对比: 观测(红) vs 合成(蓝)
        ax.plot(t, s_obs_denorm[trace_id], 'r-', linewidth=1.2, label='观测')
        ax.plot(t, s_pred_denorm[trace_id], 'b-', linewidth=1.2, label='合成')
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'No. {display_id}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'four_trace_seismic_comparison.png', dpi=200)
    plt.close()
    print(f"✓ 保存到: {out_dir / 'four_trace_seismic_comparison.png'}")
    
    # 4. 额外生成阻抗对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (trace_id, display_id) in enumerate(zip(trace_ids, display_ids)):
        ax = axes[i]
        t = np.arange(len(y_pred_denorm[trace_id]))
        
        # 阻抗对比: 真实(红) vs 预测(蓝)
        ax.plot(t, y_true_denorm[trace_id], 'r-', linewidth=1.2, label='真实')
        ax.plot(t, y_pred_denorm[trace_id], 'b-', linewidth=1.2, label='预测')
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('Impedance(m/s*g/cm^3)', fontsize=10)
        ax.set_title(f'No. {display_id}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'four_trace_impedance_comparison.png', dpi=200)
    plt.close()
    print(f"✓ 保存到: {out_dir / 'four_trace_impedance_comparison.png'}")
    
    print("\n" + "="*60)
    print("所有图像生成完成!")
    print("="*60)
    print(f"输出目录: {out_dir}")
    print(f"1. 合成地震剖面: seis_synthetic_section.png")
    print(f"2. 预测阻抗剖面: pred_imp_section_contour.png")
    print(f"3. 四道地震对比: four_trace_seismic_comparison.png")
    print(f"4. 四道阻抗对比: four_trace_impedance_comparison.png")

if __name__ == '__main__':
    main()
