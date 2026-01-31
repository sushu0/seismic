# -*- coding: utf-8 -*-
"""
优化版 01_20Hz 数据训练脚本
加入边缘感知损失、混合精度训练、学习率预热等优化

使用方式:
  cd D:\SEISMIC_CODING\new
  .\.venv\Scripts\python.exe train_01_20Hz_optimized.py
"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 设置标准输出编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ==================== 配置 ====================
class Config:
    # 数据路径
    DATA_DIR = Path(r'D:\SEISMIC_CODING\new\data\01_20Hz')
    RESULT_DIR = Path(r'D:\SEISMIC_CODING\new\results\01_20Hz_unet1d_optimized')
    
    # 模型参数
    MODEL_BASE = 48      # 基础通道数 (适中)
    MODEL_DEPTH = 5      # 网络深度 (同原版)
    
    # 训练参数
    EPOCHS = 500         # 更多轮次
    BATCH_SIZE = 4       # 更小批次，更多更新
    LR = 1e-3            # 稍高初始学习率
    LR_MIN = 1e-6
    WEIGHT_DECAY = 1e-4
    GRAD_CLIP = 1.0
    WARMUP_EPOCHS = 20   # 学习率预热
    
    # 损失权重 - 调整以更注重边缘
    LAMBDA_MSE = 1.0
    LAMBDA_EDGE = 0.5    # 增强边缘感知
    LAMBDA_SSIM = 0.3    # 增强结构相似性
    
    # 设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 随机种子
    SEED = 42

# ==================== 改进的 UNet1D 模型 ====================
class ConvBlock(nn.Module):
    """改进的卷积块：加入残差连接和Dropout"""
    def __init__(self, c_in: int, c_out: int, k: int = 3, p: int = 1, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(c_out, c_out, k, padding=p),
            nn.BatchNorm1d(c_out),
            nn.GELU(),
        )
        # 残差连接
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        return self.net(x) + self.skip(x)


class AttentionGate(nn.Module):
    """注意力门控机制，用于跳跃连接"""
    def __init__(self, ch: int):
        super().__init__()
        self.W_g = nn.Conv1d(ch, ch, 1)
        self.W_x = nn.Conv1d(ch, ch, 1)
        self.psi = nn.Conv1d(ch, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        """g: 来自解码器的特征, x: 来自编码器的跳跃连接"""
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # 对齐尺寸
        if g1.shape[-1] != x1.shape[-1]:
            diff = x1.shape[-1] - g1.shape[-1]
            if diff > 0:
                g1 = F.pad(g1, (0, diff))
            else:
                g1 = g1[..., :x1.shape[-1]]
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi


class ImprovedUNet1D(nn.Module):
    """改进的UNet1D：加入注意力门控、残差连接、更深的网络"""
    def __init__(self, in_ch: int = 1, base: int = 64, depth: int = 6, out_ch: int = 1, dropout: float = 0.1):
        super().__init__()
        self.depth = depth
        chs = [base * (2 ** i) for i in range(depth)]
        
        # 编码器
        self.enc = nn.ModuleList()
        self.pool = nn.ModuleList()
        prev = in_ch
        for c in chs:
            self.enc.append(ConvBlock(prev, c, dropout=dropout))
            self.pool.append(nn.MaxPool1d(2))
            prev = c
        
        # 瓶颈层
        self.bottleneck = ConvBlock(prev, prev * 2, dropout=dropout)
        prev = prev * 2
        
        # 解码器
        self.up = nn.ModuleList()
        self.attn = nn.ModuleList()  # 注意力门控
        self.dec = nn.ModuleList()
        for c in reversed(chs):
            self.up.append(nn.ConvTranspose1d(prev, c, kernel_size=2, stride=2))
            self.attn.append(AttentionGate(c))
            self.dec.append(ConvBlock(prev, c, dropout=dropout))
            prev = c
        
        # 输出层
        self.out = nn.Sequential(
            nn.Conv1d(prev, prev // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(prev // 2, out_ch, kernel_size=1)
        )

    def forward(self, x):
        skips = []
        for i in range(self.depth):
            x = self.enc[i](x)
            skips.append(x)
            x = self.pool[i](x)
        
        x = self.bottleneck(x)
        
        for i in range(self.depth):
            x = self.up[i](x)
            skip = skips[-(i + 1)]
            
            # 对齐尺寸
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    x = x[..., :skip.shape[-1]]
            
            # 应用注意力门控
            skip = self.attn[i](x, skip)
            x = torch.cat([x, skip], dim=1)
            x = self.dec[i](x)
        
        return self.out(x)


# ==================== 自定义损失函数 ====================
class EdgeAwareLoss(nn.Module):
    """边缘感知损失：对边界区域给予更高权重"""
    def __init__(self):
        super().__init__()
        # Sobel 算子用于检测边缘
        self.sobel = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        sobel_kernel = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3)
        self.sobel.weight = nn.Parameter(sobel_kernel, requires_grad=False)
        
    def forward(self, pred, target):
        # 计算梯度（边缘）
        pred_grad = torch.abs(self.sobel(pred))
        target_grad = torch.abs(self.sobel(target))
        
        # 边缘权重图
        edge_weight = 1.0 + torch.abs(target_grad)
        
        # 加权MSE损失
        diff = (pred - target) ** 2
        weighted_loss = (diff * edge_weight).mean()
        
        # 梯度差异损失
        grad_loss = F.mse_loss(pred_grad, target_grad)
        
        return weighted_loss + 0.5 * grad_loss


class SSIM1DLoss(nn.Module):
    """1D结构相似性损失"""
    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2
        
    def forward(self, pred, target):
        # 使用滑动窗口计算局部均值和方差
        kernel = torch.ones(1, 1, self.window_size, device=pred.device) / self.window_size
        
        mu_x = F.conv1d(pred, kernel, padding=self.window_size // 2)
        mu_y = F.conv1d(target, kernel, padding=self.window_size // 2)
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv1d(pred ** 2, kernel, padding=self.window_size // 2) - mu_x_sq
        sigma_y_sq = F.conv1d(target ** 2, kernel, padding=self.window_size // 2) - mu_y_sq
        sigma_xy = F.conv1d(pred * target, kernel, padding=self.window_size // 2) - mu_xy
        
        ssim = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
               ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))
        
        return 1 - ssim.mean()


class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, lambda_mse=1.0, lambda_edge=0.3, lambda_ssim=0.2):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.edge_loss = EdgeAwareLoss()
        self.ssim_loss = SSIM1DLoss()
        self.lambda_mse = lambda_mse
        self.lambda_edge = lambda_edge
        self.lambda_ssim = lambda_ssim
        
    def forward(self, pred, target):
        l_mse = self.mse_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)
        
        total = self.lambda_mse * l_mse + self.lambda_edge * l_edge + self.lambda_ssim * l_ssim
        return total, {'mse': l_mse.item(), 'edge': l_edge.item(), 'ssim': l_ssim.item()}


# ==================== 数据集 ====================
class SeismicDataset(Dataset):
    def __init__(self, seis_path, imp_path, norm_stats=None, fit_norm=False):
        self.seis = np.load(seis_path).astype(np.float32)
        self.imp = np.load(imp_path).astype(np.float32)
        
        if fit_norm:
            self.norm_stats = {
                'seis_mean': float(self.seis.mean()),
                'seis_std': float(self.seis.std()),
                'imp_mean': float(self.imp.mean()),
                'imp_std': float(self.imp.std()),
            }
        else:
            self.norm_stats = norm_stats
        
        # Z-score 归一化
        self.seis = (self.seis - self.norm_stats['seis_mean']) / (self.norm_stats['seis_std'] + 1e-6)
        self.imp = (self.imp - self.norm_stats['imp_mean']) / (self.norm_stats['imp_std'] + 1e-6)
    
    def __len__(self):
        return len(self.seis)
    
    def __getitem__(self, idx):
        seis = torch.from_numpy(self.seis[idx:idx+1])  # [1, T]
        imp = torch.from_numpy(self.imp[idx:idx+1])    # [1, T]
        return {'seis': seis, 'imp': imp}


# ==================== 评估指标 ====================
def compute_metrics(pred, target):
    """计算评估指标"""
    pred = pred.flatten()
    target = target.flatten()
    
    mse = float(((pred - target) ** 2).mean())
    
    # PCC
    pred_mean = pred.mean()
    target_mean = target.mean()
    cov = ((pred - pred_mean) * (target - target_mean)).mean()
    std_pred = pred.std()
    std_target = target.std()
    pcc = float(cov / (std_pred * std_target + 1e-8))
    
    # R²
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target_mean) ** 2).sum()
    r2 = float(1 - ss_res / (ss_tot + 1e-8))
    
    return {'MSE': mse, 'PCC': pcc, 'R2': r2}


# ==================== 训练函数 ====================
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip):
    model.train()
    total_loss = 0
    loss_components = {'mse': 0, 'edge': 0, 'ssim': 0}
    
    pbar = tqdm(loader, desc='Training')
    for batch in pbar:
        seis = batch['seis'].to(device)
        imp = batch['imp'].to(device)
        
        optimizer.zero_grad()
        pred = model(seis)
        loss, components = criterion(pred, imp)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        for k in loss_components:
            loss_components[k] += components[k]
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_target = []
    
    for batch in loader:
        seis = batch['seis'].to(device)
        imp = batch['imp'].to(device)
        
        pred = model(seis)
        all_pred.append(pred.cpu().numpy())
        all_target.append(imp.cpu().numpy())
    
    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)
    
    metrics = compute_metrics(all_pred, all_target)
    return metrics


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, lr_min):
    """带预热的余弦退火调度器"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return max(lr_min / Config.LR, 0.5 * (1 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== 主函数 ====================
def main():
    print("=" * 60)
    print("优化版 UNet1D 训练 - 01_20Hz 数据")
    print("=" * 60)
    
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(Config.SEED)
    
    # 创建输出目录
    Config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Config.RESULT_DIR / 'checkpoints'
    ckpt_dir.mkdir(exist_ok=True)
    
    device = torch.device(Config.DEVICE)
    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    print("\n加载数据...")
    train_ds = SeismicDataset(
        Config.DATA_DIR / 'train_labeled_seis.npy',
        Config.DATA_DIR / 'train_labeled_imp.npy',
        fit_norm=True
    )
    norm_stats = train_ds.norm_stats
    
    val_ds = SeismicDataset(
        Config.DATA_DIR / 'val_seis.npy',
        Config.DATA_DIR / 'val_imp.npy',
        norm_stats=norm_stats
    )
    test_ds = SeismicDataset(
        Config.DATA_DIR / 'test_seis.npy',
        Config.DATA_DIR / 'test_imp.npy',
        norm_stats=norm_stats
    )
    
    print(f"  训练集: {len(train_ds)} 道")
    print(f"  验证集: {len(val_ds)} 道")
    print(f"  测试集: {len(test_ds)} 道")
    
    # 保存归一化参数
    with open(Config.RESULT_DIR / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 创建模型
    print(f"\n创建模型: ImprovedUNet1D (base={Config.MODEL_BASE}, depth={Config.MODEL_DEPTH})")
    model = ImprovedUNet1D(
        in_ch=1,
        base=Config.MODEL_BASE,
        depth=Config.MODEL_DEPTH,
        out_ch=1,
        dropout=0.1
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数: {n_params:,}")
    
    # 损失函数和优化器
    criterion = CombinedLoss(
        lambda_mse=Config.LAMBDA_MSE,
        lambda_edge=Config.LAMBDA_EDGE,
        lambda_ssim=Config.LAMBDA_SSIM
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    scheduler = get_lr_scheduler(optimizer, Config.WARMUP_EPOCHS, Config.EPOCHS, Config.LR_MIN)
    
    # 训练
    print(f"\n开始训练 ({Config.EPOCHS} epochs)...")
    print("-" * 60)
    
    best_val_r2 = -float('inf')
    best_epoch = 0
    history = []
    
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss, loss_comp = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.GRAD_CLIP
        )
        
        val_metrics = evaluate(model, val_loader, device)
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'loss_mse': loss_comp['mse'],
            'loss_edge': loss_comp['edge'],
            'loss_ssim': loss_comp['ssim'],
            'val_mse': val_metrics['MSE'],
            'val_pcc': val_metrics['PCC'],
            'val_r2': val_metrics['R2'],
            'lr': current_lr
        })
        
        # 打印进度
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] Epoch {epoch:3d}/{Config.EPOCHS}: "
              f"loss={train_loss:.4f} (mse={loss_comp['mse']:.4f}, edge={loss_comp['edge']:.4f}, ssim={loss_comp['ssim']:.4f}) "
              f"| val_mse={val_metrics['MSE']:.4f} val_pcc={val_metrics['PCC']:.4f} val_r2={val_metrics['R2']:.4f} "
              f"| lr={current_lr:.2e}")
        
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'norm_stats': norm_stats
        }, ckpt_dir / 'last.pt')
        
        if val_metrics['R2'] > best_val_r2:
            best_val_r2 = val_metrics['R2']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'norm_stats': norm_stats
            }, ckpt_dir / 'best.pt')
            print(f"  ★ 新的最佳模型! (val_r2={best_val_r2:.4f})")
    
    # 测试
    print("\n" + "=" * 60)
    print("测试最佳模型...")
    best_ckpt = torch.load(ckpt_dir / 'best.pt', map_location=device)
    model.load_state_dict(best_ckpt['model'])
    
    test_metrics = evaluate(model, test_loader, device)
    print(f"  最佳 Epoch: {best_epoch}")
    print(f"  测试集 MSE: {test_metrics['MSE']:.6f}")
    print(f"  测试集 PCC: {test_metrics['PCC']:.6f}")
    print(f"  测试集 R²:  {test_metrics['R2']:.6f}")
    
    # 保存结果
    with open(Config.RESULT_DIR / 'test_metrics.json', 'w') as f:
        json.dump({
            'best_epoch': best_epoch,
            'test_metrics': test_metrics,
            'history': history
        }, f, indent=2)
    
    print("\n" + "=" * 60)
    print(f"训练完成! 结果保存到: {Config.RESULT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
