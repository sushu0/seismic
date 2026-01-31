"""
为真实数据生成反演可视化图
"""
import numpy as np
import torch
from pathlib import Path
import yaml
from torch.utils.data import DataLoader, TensorDataset

from seisinv.models.baselines import UNet1D
from seisinv.losses.physics import ForwardModel
from seisinv.utils.wavelet import ricker

def load_real_model(exp_name='real_unet1d_optimized'):
    """加载训练好的真实数据模型"""
    
    config_path = 'configs/exp_real_data.yaml'
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
    
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
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
    phys_cfg = cfg['physics']
    wavelet = ricker(
        f0=phys_cfg['wavelet_f0_hz'],
        length=phys_cfg['wavelet_length_s'],
        dt=phys_cfg['dt_s']
    )
    wavelet_tensor = torch.from_numpy(wavelet).float().to(device)
    fm = ForwardModel(wavelet=wavelet_tensor, eps=float(phys_cfg['eps']))
    
    return model, test_loader, fm, device, norm_stats

def denormalize(data, mean, std):
    """反归一化"""
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False
    
    exp_name = 'real_unet1d_optimized'
    out_dir = Path(f'results/{exp_name}')
    
    print(f"加载模型和数据: {exp_name}")
    model, test_loader, fm, device, norm_stats = load_real_model(exp_name)
    
    print("进行预测和前向建模...")
    y_true, y_pred, s_obs, s_pred = predict_and_forward_model(model, test_loader, fm, device)
    
    print(f"数据形状: y_pred={y_pred.shape}, s_pred={s_pred.shape}")
    
    # 反归一化到原始域
    s_obs_denorm = denormalize(s_obs, norm_stats['seis_mean'], norm_stats['seis_std'])
    s_pred_denorm = denormalize(s_pred, norm_stats['seis_mean'], norm_stats['seis_std'])
    y_pred_denorm = denormalize(y_pred, norm_stats['imp_mean'], norm_stats['imp_std'])
    y_true_denorm = denormalize(y_true, norm_stats['imp_mean'], norm_stats['imp_std'])
    
    # 选择特定的道进行可视化 (299, 599, 1699, 2299映射到实际索引)
    # 测试集有273个样本，选择代表性的4个
    trace_ids = [29, 59, 169, 229] if y_pred.shape[0] > 229 else [10, 50, 100, 200]
    display_ids = [299, 599, 1699, 2299]
    
    print(f"生成四道阻抗对比图 (traces: {trace_ids})...")
    
    # 1. 四道阻抗对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (trace_id, display_id) in enumerate(zip(trace_ids, display_ids)):
        ax = axes[i]
        t = np.arange(len(y_pred_denorm[trace_id]))
        
        ax.plot(t, y_true_denorm[trace_id], 'r-', linewidth=1.2, label='观')
        ax.plot(t, y_pred_denorm[trace_id], 'b-', linewidth=1.2, label='预')
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('Impedance(m/s*g/cm^3)', fontsize=10)
        ax.set_title(f'No. {display_id}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'four_trace_impedance_comparison.png', dpi=200)
    plt.close()
    print(f"✓ 保存: {out_dir / 'four_trace_impedance_comparison.png'}")
    
    # 2. 四道地震对比图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, (trace_id, display_id) in enumerate(zip(trace_ids, display_ids)):
        ax = axes[i]
        t = np.arange(len(s_obs_denorm[trace_id]))
        
        ax.plot(t, s_obs_denorm[trace_id], 'r-', linewidth=1.2, label='观')
        ax.plot(t, s_pred_denorm[trace_id], 'b-', linewidth=1.2, label='预')
        
        ax.set_xlabel('t', fontsize=10)
        ax.set_ylabel('Amplitude', fontsize=10)
        ax.set_title(f'No. {display_id}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'four_trace_seismic_comparison.png', dpi=200)
    plt.close()
    print(f"✓ 保存: {out_dir / 'four_trace_seismic_comparison.png'}")
    
    # 3. 预测阻抗剖面图
    print("生成预测阻抗剖面图...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    im_T = y_pred_denorm.T
    n_samples, n_traces = im_T.shape
    X, Y = np.meshgrid(np.arange(n_traces), np.arange(n_samples))
    
    levels = 30
    cf = ax.contourf(X, Y, im_T, levels=levels, cmap='jet')
    
    cbar = plt.colorbar(cf, ax=ax)
    cbar.set_label('[0,30,1,228]', rotation=0, labelpad=15)
    
    ax.set_xlabel('Trace number')
    ax.set_ylabel('Sample')
    ax.set_title('Predicted Impedance Section (npy)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(out_dir / 'pred_imp_section_contour.png', dpi=200)
    plt.close()
    print(f"✓ 保存: {out_dir / 'pred_imp_section_contour.png'}")
    
    print("\n" + "="*60)
    print("所有图像生成完成!")
    print("="*60)

if __name__ == '__main__':
    main()
