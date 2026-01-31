# -*- coding: utf-8 -*-
"""
ThinLayerNet V2 - 20Hz 薄层反演训练 (优化版)
改进内容：
1. 梯度匹配损失 (Gradient Matching Loss)
2. 边界区域加权损失
3. 在线薄层数据增强
4. 薄层三分类标签自动生成
5. 新评估指标：双峰距误差、分离度、薄层F1
6. 高频辅助输入通道
"""
import numpy as np
import segyio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import sys
import os
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

if sys.platform == 'win32':
    # Avoid wrapping sys.stdout.buffer which can break when stdout is redirected.
    # Prefer reconfigure when available (py3.7+).
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _enforce_workspace_venv_on_windows():
    """Prevent accidental runs under system Python on Windows.

    This repo frequently ends up launching the same script with multiple Python installs.
    To keep checkpoints/logs consistent, we hard-require the local `.venv` interpreter.

    Set env var `ALLOW_NON_VENV=1` to bypass (for debugging).
    """
    if sys.platform != 'win32':
        return
    if os.environ.get('ALLOW_NON_VENV', '').strip() == '1':
        return

    try:
        script_dir = Path(__file__).resolve().parent
        expected = (script_dir / '.venv' / 'Scripts' / 'python.exe').resolve()
        actual = Path(sys.executable).resolve()
        if expected.exists() and actual != expected:
            msg = (
                f"[FATAL] Please run with workspace venv python:\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"Tip: run `new/run_train_v2.bat` or `new/start_train_v2_guard.ps1`."
            )
            print(msg, file=sys.stderr, flush=True)
            raise SystemExit(2)
    except SystemExit:
        raise
    except Exception:
        # If anything goes wrong with the check, do not block training.
        return

# ==================== 配置 ====================
class Config:
    # 路径 - 20Hz数据
    SEISMIC_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_re.sgy'
    IMPEDANCE_PATH = r'D:\SEISMIC_CODING\zmy_data\01\data\01_20Hz_04.txt'
    OUTPUT_DIR = Path(r'D:\SEISMIC_CODING\new\results\01_20Hz_thinlayer_v2')
    
    # 训练参数
    EPOCHS = 500
    BATCH_SIZE = 4
    LR = 3e-4
    WEIGHT_DECAY = 1e-5
    
    # 损失权重
    LAMBDA_GRAD = 0.3        # 梯度匹配损失权重
    LAMBDA_EDGE = 0.5        # 边界加权系数
    LAMBDA_SPARSE = 0.05     # 稀疏正则权重
    LAMBDA_FWD = 0.1         # 正演一致性损失权重
    
    # 薄层参数
    DOMINANT_FREQ = 20.0     # 主频 Hz - 20Hz
    DT = 0.001               # 采样间隔 s
    TUNING_THICKNESS = 1.0 / (4 * DOMINANT_FREQ)  # 调谐厚度 ~12.5ms
    
    # 数据增强
    AUGMENT_PROB = 0.5       # 薄层注入概率
    MIN_THIN_THICKNESS = 8   # 最小薄层厚度(采样点)
    MAX_THIN_THICKNESS = 50  # 最大薄层厚度(采样点) - 20Hz分辨率较低，薄层较厚
    
    # 划分
    TRAIN_RATIO = 0.6
    VAL_RATIO = 0.2
    SEED = 42

CFG = Config()

# ==================== 数据加载 ====================
def load_seismic_data(path):
    with segyio.open(path, "r", ignore_geometry=True, strict=False) as f:
        n_traces = f.tracecount
        data = np.stack([np.copy(f.trace[i]) for i in range(n_traces)])
        dt = f.bin[segyio.BinField.Interval] * 1e-6
    return data.astype(np.float32), dt

def load_impedance_data(path, n_traces):
    raw = np.loadtxt(path, usecols=4, skiprows=1)
    n_samples = len(raw) // n_traces
    return raw.reshape(n_traces, n_samples).astype(np.float32)

def highpass_filter(data, cutoff=8, fs=1000, order=4):
    """高通滤波提取高频成分 - 20Hz数据用8Hz截止频率"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data, axis=-1).astype(np.float32)

# ==================== 薄层三分类标签生成 ====================
class ThinLayerLabeler:
    """根据阻抗自动生成薄层三分类标签"""
    
    def __init__(self, dt=0.001, dominant_freq=20.0):
        self.dt = dt
        self.dominant_freq = dominant_freq
        # 调谐厚度(采样点数)
        self.tuning_samples = int(1.0 / (4 * dominant_freq) / dt)
        self.half_wavelength = int(1.0 / (2 * dominant_freq) / dt)
        
    def compute_reflection_coef(self, impedance):
        """计算反射系数"""
        rc = np.zeros_like(impedance)
        rc[1:] = (impedance[1:] - impedance[:-1]) / (impedance[1:] + impedance[:-1] + 1e-10)
        return rc
    
    def find_layer_boundaries(self, impedance, threshold_percentile=70):
        """检测层界面位置"""
        rc = self.compute_reflection_coef(impedance)
        threshold = np.percentile(np.abs(rc), threshold_percentile)
        
        # 找正负峰
        pos_peaks, _ = find_peaks(rc, height=threshold, distance=3)
        neg_peaks, _ = find_peaks(-rc, height=threshold, distance=3)
        
        all_peaks = np.sort(np.concatenate([pos_peaks, neg_peaks]))
        return all_peaks, rc
    
    def classify_thin_layers(self, impedance):
        """
        分类薄层类型:
        0 = 重叠 (thickness < tuning_thickness)
        1 = 半重叠 (tuning_thickness <= thickness < half_wavelength)  
        2 = 不重叠 (thickness >= half_wavelength)
        -1 = 非薄层区域
        """
        boundaries, rc = self.find_layer_boundaries(impedance)
        n_samples = len(impedance)
        
        # 初始化标签为-1(非薄层)
        labels = np.full(n_samples, -1, dtype=np.int32)
        layer_info = []
        
        # 遍历相邻界面对
        for i in range(len(boundaries) - 1):
            top = boundaries[i]
            bottom = boundaries[i + 1]
            thickness = bottom - top
            
            # 确定类别
            if thickness < self.tuning_samples * 0.8:
                category = 0  # 重叠
            elif thickness < self.tuning_samples * 1.5:
                category = 1  # 半重叠
            else:
                category = 2  # 不重叠
            
            # 标记该区间
            labels[top:bottom+1] = category
            layer_info.append({
                'top': top,
                'bottom': bottom,
                'thickness': thickness,
                'category': category
            })
        
        return labels, layer_info, boundaries
    
    def get_boundary_weight(self, impedance, alpha=2.0, sigma=3):
        """生成边界加权图"""
        rc = self.compute_reflection_coef(impedance)
        grad_abs = np.abs(rc)
        # 平滑
        grad_smooth = gaussian_filter1d(grad_abs, sigma=sigma)
        # 归一化并加权
        threshold = np.percentile(grad_smooth, 70)
        weight = 1.0 + alpha * (grad_smooth > threshold).astype(np.float32)
        return weight

# ==================== 数据增强：薄层注入 ====================
class ThinLayerAugmentor:
    """在线薄层数据增强"""
    
    def __init__(self, prob=0.5, min_thick=5, max_thick=30):
        self.prob = prob
        self.min_thick = min_thick
        self.max_thick = max_thick
        
    def inject_thin_layer(self, seismic, impedance):
        """向数据中注入人工薄层"""
        if np.random.rand() > self.prob:
            return seismic, impedance
        
        n_samples = len(impedance)
        imp_aug = impedance.copy()
        
        # 随机选择注入位置(避开边缘)
        margin = self.max_thick * 2
        if n_samples < margin * 2:
            return seismic, impedance
            
        pos = np.random.randint(margin, n_samples - margin)
        thickness = np.random.randint(self.min_thick, self.max_thick)
        
        # 随机阻抗变化幅度
        imp_mean = np.mean(impedance)
        imp_std = np.std(impedance)
        delta = np.random.uniform(0.5, 2.0) * imp_std
        if np.random.rand() > 0.5:
            delta = -delta
        
        # 注入薄层
        imp_aug[pos:pos+thickness] += delta
        
        # 重新生成对应的地震道(简化：用卷积近似)
        rc_aug = np.zeros_like(imp_aug)
        rc_aug[1:] = (imp_aug[1:] - imp_aug[:-1]) / (imp_aug[1:] + imp_aug[:-1] + 1e-10)
        
        # Ricker子波
        wavelet = self._ricker_wavelet(CFG.DOMINANT_FREQ, CFG.DT, 0.1)
        seis_aug = np.convolve(rc_aug, wavelet, mode='same').astype(np.float32)
        
        # 混合原始和增强(保留原始的低频趋势)
        alpha = 0.7
        seis_mixed = alpha * seis_aug + (1 - alpha) * seismic
        
        return seis_mixed, imp_aug
    
    def _ricker_wavelet(self, freq, dt, length):
        """生成Ricker子波"""
        t = np.arange(-length/2, length/2, dt)
        a = (np.pi * freq) ** 2
        wavelet = (1 - 2 * a * t**2) * np.exp(-a * t**2)
        return wavelet.astype(np.float32)

# ==================== 数据集 ====================
class ThinLayerDatasetV2(Dataset):
    def __init__(self, seismic, impedance, indices, norm_stats, augmentor=None, labeler=None):
        self.seismic = seismic
        self.impedance = impedance
        self.indices = indices
        self.norm_stats = norm_stats
        self.augmentor = augmentor
        self.labeler = labeler
        
        # 预计算高频成分 (20Hz数据用8Hz截止频率)
        self.seismic_hf = highpass_filter(seismic, cutoff=8, fs=1000)
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        i = self.indices[idx]
        seis = self.seismic[i].copy()
        imp = self.impedance[i].copy()
        seis_hf = self.seismic_hf[i].copy()
        
        # 数据增强
        if self.augmentor is not None:
            seis, imp = self.augmentor.inject_thin_layer(seis, imp)
        
        # 归一化
        seis_norm = (seis - self.norm_stats['seis_mean']) / (self.norm_stats['seis_std'] + 1e-6)
        seis_hf_norm = seis_hf / (np.std(seis_hf) + 1e-6)  # 零均值单位方差
        imp_norm = (imp - self.norm_stats['imp_mean']) / (self.norm_stats['imp_std'] + 1e-6)
        
        # 边界权重
        if self.labeler is not None:
            boundary_weight = self.labeler.get_boundary_weight(imp, alpha=CFG.LAMBDA_EDGE)
            thin_labels, _, _ = self.labeler.classify_thin_layers(imp)
        else:
            boundary_weight = np.ones_like(imp)
            thin_labels = np.full_like(imp, -1, dtype=np.int32)
        
        # 双通道输入: [原始地震, 高频成分]
        x = np.stack([seis_norm, seis_hf_norm], axis=0)
        
        return (torch.from_numpy(x),
                torch.from_numpy(imp_norm).unsqueeze(0),
                torch.from_numpy(boundary_weight),
                torch.from_numpy(thin_labels))

# ==================== 模型定义 ====================
class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilations=[1, 2, 4, 8]):
        super().__init__()
        self.branches = nn.ModuleList()
        branch_ch = out_ch // len(dilations)
        for d in dilations:
            self.branches.append(nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=3, padding=d, dilation=d),
                nn.BatchNorm1d(branch_ch),
                nn.GELU()
            ))
        self.fusion = nn.Sequential(
            nn.Conv1d(branch_ch * len(dilations), out_ch, kernel_size=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        branches = [branch(x) for branch in self.branches]
        out = torch.cat(branches, dim=1)
        out = self.fusion(out)
        return out + self.skip(x)


class BoundaryEnhanceModule(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.edge_conv = nn.Conv1d(ch, ch, kernel_size=3, padding=1, groups=ch, bias=False)
        with torch.no_grad():
            edge_kernel = torch.tensor([-1.0, 2.0, -1.0]).view(1, 1, 3)
            self.edge_conv.weight.data = edge_kernel.repeat(ch, 1, 1)
        self.refine = nn.Sequential(
            nn.Conv1d(ch * 2, ch, kernel_size=1),
            nn.BatchNorm1d(ch),
            nn.GELU(),
            nn.Conv1d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(ch),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        edge = torch.abs(self.edge_conv(x))
        combined = torch.cat([x, edge], dim=1)
        attention = self.refine(combined)
        return x * attention + x


class ThinLayerBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_boundary=True):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.GELU()
        )
        self.boundary = BoundaryEnhanceModule(out_ch) if use_boundary else nn.Identity()
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.boundary(out)
        return out + self.skip(x)


class ThinLayerNetV2(nn.Module):
    """改进版ThinLayerNet - 双通道输入"""
    def __init__(self, in_ch=2, base_ch=64, out_ch=1):
        super().__init__()
        # 输入卷积 - 处理双通道(原始+高频)
        self.input_conv = nn.Sequential(
            nn.Conv1d(in_ch, base_ch, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_ch),
            nn.GELU()
        )
        
        self.multi_scale = DilatedConvBlock(base_ch, base_ch, dilations=[1, 2, 4, 8])
        
        # Encoder - 减少下采样次数
        self.enc1 = ThinLayerBlock(base_ch, base_ch * 2)
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = ThinLayerBlock(base_ch * 2, base_ch * 4)
        self.pool2 = nn.MaxPool1d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            DilatedConvBlock(base_ch * 4, base_ch * 8, dilations=[1, 2, 4, 8, 16]),
            ThinLayerBlock(base_ch * 8, base_ch * 8)
        )
        
        # Decoder
        self.up2 = nn.ConvTranspose1d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec2 = ThinLayerBlock(base_ch * 8, base_ch * 4)
        
        self.up1 = nn.ConvTranspose1d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec1 = ThinLayerBlock(base_ch * 4, base_ch * 2)
        
        # 细化模块
        self.refine = nn.Sequential(
            ThinLayerBlock(base_ch * 2 + base_ch, base_ch * 2),
            nn.Conv1d(base_ch * 2, base_ch, kernel_size=3, padding=1),
            nn.GELU(),
            BoundaryEnhanceModule(base_ch),
        )
        
        # 输出
        self.output = nn.Sequential(
            nn.Conv1d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(base_ch // 2, out_ch, kernel_size=1)
        )
    
    def forward(self, x):
        x0 = self.input_conv(x)
        x0 = self.multi_scale(x0)
        
        e1 = self.enc1(x0)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        b = self.bottleneck(p2)
        
        d2 = self.up2(b)
        if d2.shape[-1] != e2.shape[-1]:
            d2 = F.interpolate(d2, size=e2.shape[-1], mode='linear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        if d1.shape[-1] != e1.shape[-1]:
            d1 = F.interpolate(d1, size=e1.shape[-1], mode='linear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        if d1.shape[-1] != x0.shape[-1]:
            d1 = F.interpolate(d1, size=x0.shape[-1], mode='linear', align_corners=False)
        
        out = torch.cat([d1, x0], dim=1)
        out = self.refine(out)
        out = self.output(out)
        
        return out

# ==================== 损失函数 ====================
class GradientMatchingLoss(nn.Module):
    """梯度匹配损失 - 确保预测和真实阻抗的梯度一致"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # 一阶差分
        pred_grad = pred[:, :, 1:] - pred[:, :, :-1]
        target_grad = target[:, :, 1:] - target[:, :, :-1]
        return F.mse_loss(pred_grad, target_grad)


class WeightedMSELoss(nn.Module):
    """边界加权MSE损失"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, weight):
        # weight: (B, L)
        mse = (pred.squeeze(1) - target.squeeze(1)) ** 2
        weighted_mse = mse * weight
        return weighted_mse.mean()


class SparseGradientLoss(nn.Module):
    """稀疏梯度正则 - 促进反射系数稀疏"""
    def __init__(self):
        super().__init__()
    
    def forward(self, pred):
        grad = pred[:, :, 1:] - pred[:, :, :-1]
        return torch.mean(torch.abs(grad))


class ForwardConsistencyLoss(nn.Module):
    """正演一致性损失 - 确保预测阻抗正演后与输入地震一致"""
    def __init__(self, dominant_freq=40.0, dt=0.001):
        super().__init__()
        # 创建Ricker子波
        length = 0.1
        t = torch.arange(-length/2, length/2, dt)
        a = (np.pi * dominant_freq) ** 2
        wavelet = (1 - 2 * a * t**2) * torch.exp(-a * t**2)
        self.register_buffer('wavelet', wavelet.float().view(1, 1, -1))
    
    def forward(self, pred_imp, input_seis):
        # 计算反射系数
        rc = pred_imp[:, :, 1:] - pred_imp[:, :, :-1]
        rc = F.pad(rc, (1, 0))
        
        # 卷积生成合成地震
        synthetic = F.conv1d(rc, self.wavelet, padding=self.wavelet.shape[-1]//2)
        
        # 只取第一通道(原始地震)比较
        if input_seis.shape[1] > 1:
            input_seis = input_seis[:, 0:1, :]
        
        # 尺寸对齐
        min_len = min(synthetic.shape[-1], input_seis.shape[-1])
        synthetic = synthetic[:, :, :min_len]
        input_seis = input_seis[:, :, :min_len]
        
        return F.mse_loss(synthetic, input_seis)


class CombinedLossV2(nn.Module):
    """组合损失函数"""
    def __init__(self, lambda_grad=0.3, lambda_sparse=0.05, lambda_fwd=0.1):
        super().__init__()
        self.weighted_mse = WeightedMSELoss()
        self.grad_loss = GradientMatchingLoss()
        self.sparse_loss = SparseGradientLoss()
        self.fwd_loss = ForwardConsistencyLoss()
        
        self.lambda_grad = lambda_grad
        self.lambda_sparse = lambda_sparse
        self.lambda_fwd = lambda_fwd
    
    def forward(self, pred, target, weight, input_seis):
        loss_mse = self.weighted_mse(pred, target, weight)
        loss_grad = self.grad_loss(pred, target)
        loss_sparse = self.sparse_loss(pred)
        loss_fwd = self.fwd_loss(pred, input_seis)
        
        total = (loss_mse + 
                 self.lambda_grad * loss_grad + 
                 self.lambda_sparse * loss_sparse +
                 self.lambda_fwd * loss_fwd)
        
        return total, {
            'mse': loss_mse.item(),
            'grad': loss_grad.item(),
            'sparse': loss_sparse.item(),
            'fwd': loss_fwd.item()
        }

# ==================== 评估指标 ====================
class ThinLayerMetrics:
    """薄层专用评估指标"""
    
    def __init__(self, labeler):
        self.labeler = labeler
    
    def compute_all_metrics(self, pred, true, pred_denorm, true_denorm):
        """计算所有指标"""
        metrics = {}
        
        # 全局指标
        metrics['mse'] = float(np.mean((pred - true) ** 2))
        
        # 处理常数情况
        pred_flat = pred.flatten()
        true_flat = true.flatten()
        if np.std(pred_flat) > 1e-6 and np.std(true_flat) > 1e-6:
            metrics['pcc'] = float(np.corrcoef(pred_flat, true_flat)[0, 1])
        else:
            metrics['pcc'] = 0.0
        
        ss_res = np.sum((true_flat - pred_flat) ** 2)
        ss_tot = np.sum((true_flat - true_flat.mean()) ** 2) + 1e-10
        metrics['r2'] = float(1 - ss_res / ss_tot)
        
        # 薄层指标(使用反归一化的数据)
        thin_metrics = self._compute_thin_layer_metrics(pred_denorm, true_denorm)
        metrics.update(thin_metrics)
        
        return metrics
    
    def _compute_thin_layer_metrics(self, pred, true):
        """计算薄层专用指标"""
        results = {
            'thin_mse': [],
            'thin_pcc': [],
            'dpde': [],          # 双峰距误差
            'separability': [],  # 分离度
            'layer_f1': {'tp': 0, 'fp': 0, 'fn': 0}
        }
        
        n_traces = pred.shape[0]
        
        for i in range(n_traces):
            pred_trace = pred[i]
            true_trace = true[i]
            
            # 获取薄层信息
            labels, layer_info, true_boundaries = self.labeler.classify_thin_layers(true_trace)
            
            # 检测预测中的边界
            pred_boundaries, pred_rc = self.labeler.find_layer_boundaries(pred_trace, threshold_percentile=60)
            
            # 薄层区域指标
            thin_mask = labels >= 0
            if np.sum(thin_mask) > 10:
                thin_pred = pred_trace[thin_mask]
                thin_true = true_trace[thin_mask]
                results['thin_mse'].append(np.mean((thin_pred - thin_true) ** 2))
                
                if np.std(thin_pred) > 1e-6 and np.std(thin_true) > 1e-6:
                    pcc = np.corrcoef(thin_pred, thin_true)[0, 1]
                    if not np.isnan(pcc):
                        results['thin_pcc'].append(pcc)
            
            # 逐薄层分析
            for layer in layer_info:
                top, bottom = layer['top'], layer['bottom']
                thickness_true = layer['thickness']
                category = layer['category']
                
                # 在预测边界中寻找对应的边界对
                matched_top, matched_bottom = self._match_boundaries(
                    top, bottom, pred_boundaries, tolerance=5
                )
                
                if matched_top is not None and matched_bottom is not None:
                    # 计算双峰距误差
                    thickness_pred = matched_bottom - matched_top
                    dpde = abs(thickness_pred - thickness_true)
                    results['dpde'].append(dpde)
                    
                    # 计算分离度
                    sep = self._compute_separability(pred_rc, matched_top, matched_bottom)
                    results['separability'].append(sep)
                    
                    # F1统计
                    results['layer_f1']['tp'] += 1
                else:
                    # 未能匹配 - False Negative
                    results['dpde'].append(thickness_true)  # 最大误差
                    results['separability'].append(0.0)     # 未分离
                    results['layer_f1']['fn'] += 1
            
            # 检查False Positive(预测了不存在的薄层)
            fp_count = self._count_false_positives(pred_boundaries, true_boundaries, tolerance=5)
            results['layer_f1']['fp'] += fp_count
        
        # 汇总
        final = {}
        final['thin_mse'] = np.mean(results['thin_mse']) if results['thin_mse'] else 0.0
        final['thin_pcc'] = np.mean(results['thin_pcc']) if results['thin_pcc'] else 0.0
        final['dpde_mean'] = np.mean(results['dpde']) if results['dpde'] else 0.0
        final['separability_mean'] = np.mean(results['separability']) if results['separability'] else 0.0
        
        # F1计算
        tp = results['layer_f1']['tp']
        fp = results['layer_f1']['fp']
        fn = results['layer_f1']['fn']
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        final['thin_f1'] = 2 * precision * recall / (precision + recall + 1e-10)
        final['thin_precision'] = precision
        final['thin_recall'] = recall
        
        return final
    
    def _match_boundaries(self, true_top, true_bottom, pred_boundaries, tolerance=5):
        """匹配预测边界与真实边界"""
        matched_top = None
        matched_bottom = None
        
        for pb in pred_boundaries:
            if abs(pb - true_top) <= tolerance and matched_top is None:
                matched_top = pb
            elif abs(pb - true_bottom) <= tolerance and matched_bottom is None:
                matched_bottom = pb
        
        return matched_top, matched_bottom
    
    def _compute_separability(self, rc, top, bottom):
        """计算分离度(0-1)"""
        if bottom <= top + 2:
            return 0.0
        
        segment = rc[top:bottom+1]
        if len(segment) < 3:
            return 0.0
        
        peak_val = np.max(np.abs(segment))
        if peak_val < 1e-10:
            return 0.0
        
        # 找谷值
        mid = len(segment) // 2
        valley_region = segment[max(0, mid-2):min(len(segment), mid+3)]
        valley_val = np.min(np.abs(valley_region))
        
        # 分离度 = 1 - valley/peak
        separability = 1.0 - valley_val / (peak_val + 1e-10)
        return max(0.0, min(1.0, separability))
    
    def _count_false_positives(self, pred_boundaries, true_boundaries, tolerance=5):
        """统计误报的边界数"""
        fp = 0
        for pb in pred_boundaries:
            matched = False
            for tb in true_boundaries:
                if abs(pb - tb) <= tolerance:
                    matched = True
                    break
            if not matched:
                fp += 1
        return fp // 2  # 边界对计数

# ==================== 训练函数 ====================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    loss_components = {'mse': 0, 'grad': 0, 'sparse': 0, 'fwd': 0}
    
    for x, y, weight, _ in loader:
        x, y, weight = x.to(device), y.to(device), weight.to(device)
        
        optimizer.zero_grad()
        pred = model(x)
        
        loss, components = criterion(pred, y, weight, x)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for k, v in components.items():
            loss_components[k] += v
    
    n = len(loader)
    return total_loss / n, {k: v / n for k, v in loss_components.items()}


def evaluate(model, loader, device, norm_stats, metrics_calculator):
    model.eval()
    all_pred = []
    all_true = []
    all_pred_denorm = []
    all_true_denorm = []
    
    with torch.no_grad():
        for x, y, _, _ in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            
            all_pred.append(pred.cpu().numpy())
            all_true.append(y.cpu().numpy())
            
            # 反归一化
            pred_denorm = pred.cpu().numpy() * norm_stats['imp_std'] + norm_stats['imp_mean']
            true_denorm = y.cpu().numpy() * norm_stats['imp_std'] + norm_stats['imp_mean']
            all_pred_denorm.append(pred_denorm)
            all_true_denorm.append(true_denorm)
    
    pred = np.concatenate(all_pred, axis=0).squeeze()
    true = np.concatenate(all_true, axis=0).squeeze()
    pred_denorm = np.concatenate(all_pred_denorm, axis=0).squeeze()
    true_denorm = np.concatenate(all_true_denorm, axis=0).squeeze()
    
    metrics = metrics_calculator.compute_all_metrics(pred, true, pred_denorm, true_denorm)
    return metrics


def main():
    _enforce_workspace_venv_on_windows()

    # Some libraries may spawn child processes on Windows; ensure they use this interpreter.
    if sys.platform == 'win32':
        try:
            import multiprocessing as mp
            mp.set_executable(str(Path(sys.executable).resolve()))
        except Exception:
            pass

    print("=" * 70)
    print("ThinLayerNet V2 训练 - 薄层优化版")
    print("=" * 70)
    
    # 创建输出目录
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (CFG.OUTPUT_DIR / 'checkpoints').mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    seismic, dt = load_seismic_data(CFG.SEISMIC_PATH)
    n_traces = seismic.shape[0]
    impedance = load_impedance_data(CFG.IMPEDANCE_PATH, n_traces)
    
    min_len = min(seismic.shape[1], impedance.shape[1])
    seismic = seismic[:, :min_len]
    impedance = impedance[:, :min_len]
    print(f"数据形状: {seismic.shape}")
    
    # 归一化统计
    norm_stats = {
        'seis_mean': float(np.mean(seismic)),
        'seis_std': float(np.std(seismic)),
        'imp_mean': float(np.mean(impedance)),
        'imp_std': float(np.std(impedance))
    }
    
    with open(CFG.OUTPUT_DIR / 'norm_stats.json', 'w') as f:
        json.dump(norm_stats, f, indent=2)
    
    # 划分数据
    np.random.seed(CFG.SEED)
    indices = np.random.permutation(n_traces)
    n_train = int(n_traces * CFG.TRAIN_RATIO)
    n_val = int(n_traces * CFG.VAL_RATIO)
    
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    print(f"划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # 初始化工具
    labeler = ThinLayerLabeler(dt=CFG.DT, dominant_freq=CFG.DOMINANT_FREQ)
    augmentor = ThinLayerAugmentor(prob=CFG.AUGMENT_PROB, 
                                    min_thick=CFG.MIN_THIN_THICKNESS,
                                    max_thick=CFG.MAX_THIN_THICKNESS)
    metrics_calc = ThinLayerMetrics(labeler)
    
    # 数据集
    train_ds = ThinLayerDatasetV2(seismic, impedance, train_idx, norm_stats, 
                                   augmentor=augmentor, labeler=labeler)
    val_ds = ThinLayerDatasetV2(seismic, impedance, val_idx, norm_stats,
                                 augmentor=None, labeler=labeler)
    test_ds = ThinLayerDatasetV2(seismic, impedance, test_idx, norm_stats,
                                  augmentor=None, labeler=labeler)
    
    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 模型
    model = ThinLayerNetV2(in_ch=2, base_ch=64, out_ch=1).to(device)
    print(f"模型参数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失和优化器
    criterion = CombinedLossV2(
        lambda_grad=CFG.LAMBDA_GRAD,
        lambda_sparse=CFG.LAMBDA_SPARSE,
        lambda_fwd=CFG.LAMBDA_FWD
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    
    # 训练
    best_val_pcc = -1
    history = []
    
    print(f"\n开始训练 (Epochs={CFG.EPOCHS})...")
    print("-" * 100)
    
    for epoch in range(1, CFG.EPOCHS + 1):
        train_loss, loss_comp = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, norm_stats, metrics_calc)
        scheduler.step()
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            **loss_comp,
            **{f'val_{k}': v for k, v in val_metrics.items()}
        })
        
        # 保存最佳模型
        if val_metrics['pcc'] > best_val_pcc:
            best_val_pcc = val_metrics['pcc']
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_metrics': val_metrics
            }, CFG.OUTPUT_DIR / 'checkpoints' / 'best.pt')
        
        # 定期保存
        if epoch % 100 == 0:
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
            }, CFG.OUTPUT_DIR / 'checkpoints' / 'last.pt')
        
        # 日志（每个 epoch 打印，便于在重定向日志中实时观察训练是否在跑）
        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} "
              f"[mse={loss_comp['mse']:.4f} grad={loss_comp['grad']:.4f}] | "
              f"val_pcc={val_metrics['pcc']:.4f} val_r2={val_metrics['r2']:.4f} | "
              f"thin_pcc={val_metrics['thin_pcc']:.4f} thin_f1={val_metrics['thin_f1']:.4f} "
              f"sep={val_metrics['separability_mean']:.4f}", flush=True)
    
    # 最终保存
    torch.save({'epoch': epoch, 'model': model.state_dict()}, 
               CFG.OUTPUT_DIR / 'checkpoints' / 'last.pt')
    
    # 测试评估
    print("\n" + "=" * 70)
    print("测试集评估")
    print("=" * 70)
    
    checkpoint = torch.load(CFG.OUTPUT_DIR / 'checkpoints' / 'best.pt', map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    test_metrics = evaluate(model, test_loader, device, norm_stats, metrics_calc)
    
    print(f"\n全局指标:")
    print(f"  MSE:  {test_metrics['mse']:.4f}")
    print(f"  PCC:  {test_metrics['pcc']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    
    print(f"\n薄层指标:")
    print(f"  薄层 MSE:      {test_metrics['thin_mse']:.4f}")
    print(f"  薄层 PCC:      {test_metrics['thin_pcc']:.4f}")
    print(f"  双峰距误差:    {test_metrics['dpde_mean']:.2f} 采样点")
    print(f"  分离度:        {test_metrics['separability_mean']:.4f}")
    print(f"  薄层 F1:       {test_metrics['thin_f1']:.4f}")
    print(f"  薄层 Precision:{test_metrics['thin_precision']:.4f}")
    print(f"  薄层 Recall:   {test_metrics['thin_recall']:.4f}")
    
    # 保存指标
    with open(CFG.OUTPUT_DIR / 'test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print(f"\n训练完成! 结果保存到: {CFG.OUTPUT_DIR}")
    
    # ========== 可视化 ==========
    print("\n" + "=" * 70)
    print("生成可视化...")
    print("=" * 70)
    
    generate_visualizations(model, seismic, impedance, norm_stats, device, CFG.OUTPUT_DIR)


def generate_visualizations(model, seismic, impedance, norm_stats, device, output_dir):
    """生成评估可视化图"""
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr
    
    model.eval()
    n_traces = seismic.shape[0]
    n_samples = seismic.shape[1]
    
    # 归一化地震数据
    seis_norm = (seismic - norm_stats['seis_mean']) / norm_stats['seis_std']
    
    # 推理所有道
    print("推理全部数据...")
    all_pred = []
    with torch.no_grad():
        for i in range(n_traces):
            x = torch.from_numpy(seis_norm[i:i+1]).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(x)
            pred_np = pred.cpu().numpy().squeeze()
            all_pred.append(pred_np)
    
    pred_full = np.array(all_pred)
    
    # 反归一化
    pred_full_denorm = pred_full * norm_stats['imp_std'] + norm_stats['imp_mean']
    
    # 计算全局指标
    pcc_full, _ = pearsonr(pred_full_denorm.flatten(), impedance.flatten())
    ss_res = np.sum((impedance - pred_full_denorm) ** 2)
    ss_tot = np.sum((impedance - np.mean(impedance)) ** 2)
    r2_full = 1 - ss_res / ss_tot
    
    print(f"\n全数据集 ({n_traces} traces):")
    print(f"  PCC: {pcc_full:.4f}")
    print(f"  R²:  {r2_full:.4f}")
    
    # 坐标转换参数
    dt_ms = 0.01  # 采样间隔 ms
    total_time = n_samples * dt_ms  # 约100.01 ms
    shot_per_trace = 20  # 每个trace代表20个shot
    
    # ===== 剖面对比图 =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    extent = [0, n_traces * shot_per_trace, total_time, 0]
    
    vmin, vmax = impedance.min(), impedance.max()
    im0 = axes[0].imshow(impedance.T, aspect='auto', cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title('True Impedance', fontsize=14)
    axes[0].set_xlabel('Shot Number', fontsize=12)
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    plt.colorbar(im0, ax=axes[0], label='Impedance')
    
    im1 = axes[1].imshow(pred_full_denorm.T, aspect='auto', cmap='seismic', extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title('Predicted Impedance', fontsize=14)
    axes[1].set_xlabel('Shot Number', fontsize=12)
    axes[1].set_ylabel('Time (ms)', fontsize=12)
    plt.colorbar(im1, ax=axes[1], label='Impedance')
    
    diff = pred_full_denorm - impedance
    diff_max = np.percentile(np.abs(diff), 99)
    im2 = axes[2].imshow(diff.T, aspect='auto', cmap='RdBu_r', extent=extent, vmin=-diff_max, vmax=diff_max)
    axes[2].set_title('Difference (Pred - True)', fontsize=14)
    axes[2].set_xlabel('Shot Number', fontsize=12)
    axes[2].set_ylabel('Time (ms)', fontsize=12)
    plt.colorbar(im2, ax=axes[2], label='Difference')
    
    plt.suptitle(f'40Hz Model: PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'section_comparison_40Hz.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: section_comparison_40Hz.png")
    
    # ===== 薄层区域放大图 =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    t_start, t_end = 30, 60
    sample_start = int(t_start / dt_ms)
    sample_end = int(t_end / dt_ms)
    extent_zoom = [0, n_traces * shot_per_trace, t_end, t_start]
    
    imp_zoom = impedance[:, sample_start:sample_end]
    pred_zoom = pred_full_denorm[:, sample_start:sample_end]
    vmin_z, vmax_z = imp_zoom.min(), imp_zoom.max()
    
    im0 = axes[0].imshow(imp_zoom.T, aspect='auto', cmap='seismic', extent=extent_zoom, vmin=vmin_z, vmax=vmax_z)
    axes[0].set_title('True Impedance (30-60 ms)', fontsize=14)
    axes[0].set_xlabel('Shot Number', fontsize=12)
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(pred_zoom.T, aspect='auto', cmap='seismic', extent=extent_zoom, vmin=vmin_z, vmax=vmax_z)
    axes[1].set_title('Predicted Impedance (30-60 ms)', fontsize=14)
    axes[1].set_xlabel('Shot Number', fontsize=12)
    axes[1].set_ylabel('Time (ms)', fontsize=12)
    plt.colorbar(im1, ax=axes[1])
    
    diff_zoom = pred_zoom - imp_zoom
    diff_max_z = np.percentile(np.abs(diff_zoom), 99)
    im2 = axes[2].imshow(diff_zoom.T, aspect='auto', cmap='RdBu_r', extent=extent_zoom, vmin=-diff_max_z, vmax=diff_max_z)
    axes[2].set_title('Difference (30-60 ms)', fontsize=14)
    axes[2].set_xlabel('Shot Number', fontsize=12)
    axes[2].set_ylabel('Time (ms)', fontsize=12)
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle('40Hz Model - Thin Layer Zone', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'thin_layer_zone_40Hz.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: thin_layer_zone_40Hz.png")
    
    # ===== 道对比图 =====
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    test_traces = [10, 50, 90]
    time_axis = np.arange(n_samples) * dt_ms
    
    for idx, trace_idx in enumerate(test_traces):
        ax = axes[0, idx]
        ax.plot(impedance[trace_idx], time_axis, 'b-', linewidth=1.5, label='True')
        ax.plot(pred_full_denorm[trace_idx], time_axis, 'r--', linewidth=1.5, label='Predicted')
        ax.set_xlabel('Impedance', fontsize=11)
        ax.set_ylabel('Time (ms)', fontsize=11)
        ax.set_title(f'Trace {trace_idx} (Shot {trace_idx*shot_per_trace})', fontsize=12)
        ax.invert_yaxis()
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        trace_pcc, _ = pearsonr(impedance[trace_idx], pred_full_denorm[trace_idx])
        ax.text(0.05, 0.95, f'PCC={trace_pcc:.4f}', transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax2 = axes[1, idx]
        ax2.plot(imp_zoom[trace_idx], time_axis[sample_start:sample_end], 'b-', linewidth=1.5, label='True')
        ax2.plot(pred_zoom[trace_idx], time_axis[sample_start:sample_end], 'r--', linewidth=1.5, label='Predicted')
        ax2.set_xlabel('Impedance', fontsize=11)
        ax2.set_ylabel('Time (ms)', fontsize=11)
        ax2.set_title(f'Trace {trace_idx} (30-60 ms)', fontsize=12)
        ax2.invert_yaxis()
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'40Hz Model - Trace Comparison\nOverall PCC={pcc_full:.4f}, R²={r2_full:.4f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'trace_comparison_40Hz.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"保存: trace_comparison_40Hz.png")
    
    # 保存全数据集指标
    full_metrics = {
        'full_pcc': float(pcc_full),
        'full_r2': float(r2_full)
    }
    with open(output_dir / 'full_metrics.json', 'w') as f:
        json.dump(full_metrics, f, indent=2)
    
    print("\n可视化完成!")


if __name__ == '__main__':
    main()

