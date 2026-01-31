# 工程目录结构

## 1) 项目目录树（核心部分）

```
d:\SEISMIC_CODING\new\
├── train.py                        # 训练入口（主脚本）
├── requirements.txt                # 依赖清单
├── README.md                       # 项目说明
├── QUICK_START.md                  # 快速开始指南
├── PROJECT_COMPLETION_REPORT.md    # 完成报告
│
├── configs/                        # 配置文件目录
│   ├── exp_baseline_unet.yaml      # UNet1D配置
│   ├── exp_baseline_tcn.yaml       # TCN1D配置
│   ├── exp_newmodel.yaml           # MS-PhysFormer配置
│   ├── abl_no_physics.yaml         # 消融：无物理约束
│   └── abl_no_freq.yaml            # 消融：无频域约束
│
├── data/                           # 数据目录
│   └── toy/                        # 玩具数据
│       ├── train_labeled_seis.npy
│       ├── train_labeled_imp.npy
│       ├── train_unlabeled_seis.npy
│       ├── val_seis.npy
│       ├── val_imp.npy
│       ├── test_seis.npy
│       └── test_imp.npy
│
├── results/                        # 结果目录
│   ├── baseline_unet1d/            # UNet1D实验结果
│   │   ├── metrics.csv
│   │   ├── test_metrics.json
│   │   ├── *.png                   # 可视化图像
│   │   └── checkpoints/
│   │       ├── best.pt
│   │       └── last.pt
│   ├── baseline_tcn1d/             # TCN1D实验结果
│   └── new_ms_physformer/          # MS-PhysFormer实验结果
│
├── scripts/                        # 实用脚本
│   ├── generate_toy_data.py        # 生成合成数据
│   ├── collect_results.py          # 汇总实验结果
│   ├── run_all_toy.ps1             # Windows一键运行
│   └── run_all_toy.sh              # Linux/Mac一键运行
│
└── seisinv/                        # 核心模块包
    ├── __init__.py
    │
    ├── models/                     # 模型定义
    │   ├── baselines.py            # Baseline模型
    │   └── ms_physformer.py        # MS-PhysFormer新模型
    │
    ├── losses/                     # 损失函数
    │   ├── physics.py              # 物理正演损失
    │   └── frequency.py            # STFT频域损失
    │
    ├── data/                       # 数据加载
    │   └── dataset.py              # Dataset类
    │
    ├── trainer/                    # 训练器
    │   └── train.py                # 训练循环
    │
    └── utils/                      # 工具函数
        ├── logger.py               # 日志记录
        ├── metrics.py              # 评估指标
        ├── plotting.py             # 可视化
        ├── seed.py                 # 随机种子
        └── wavelet.py              # 子波生成
```

## 2) Baseline模型类名与训练入口

### 2.1 模型类（位于 `seisinv/models/baselines.py`）

```python
# Baseline模型类名：
class UNet1D(nn.Module):
    """
    1D U-Net for seismic inversion
    Args:
        in_ch: int = 1          # 输入通道数
        base: int = 32          # 基础通道数
        depth: int = 4          # U-Net深度
        out_ch: int = 1         # 输出通道数
    
    Forward:
        Input:  [B, 1, T]       # 地震数据
        Output: [B, 1, T]       # 阻抗预测
    """

class TCN1D(nn.Module):
    """
    Temporal Convolutional Network
    Args:
        in_ch: int = 1          # 输入通道数
        ch: int = 64            # 通道数
        layers: int = 6         # 层数
        kernel_size: int = 3
        dropout: float = 0.1
        out_ch: int = 1
    
    Forward:
        Input:  [B, 1, T]
        Output: [B, 1, T]
    """

class CNNBiLSTM(nn.Module):
    """
    CNN + BiLSTM hybrid model
    Args:
        in_ch: int = 1
        cnn_ch: int = 32
        lstm_hidden: int = 64
        lstm_layers: int = 2
        out_ch: int = 1
    
    Forward:
        Input:  [B, 1, T]
        Output: [B, 1, T]
    """
```

### 2.2 新模型类（位于 `seisinv/models/ms_physformer.py`）

```python
class MSPhysFormer(nn.Module):
    """
    Multi-Scale U-Net with Transformer bottleneck and deep supervision
    
    Args:
        in_ch: int = 1              # 输入通道数
        base: int = 48              # 基础通道数
        depth: int = 4              # U-Net深度
        nhead: int = 4              # Transformer注意力头数
        tf_dim_mult: int = 2        # Transformer维度倍数
        tf_layers: int = 2          # Transformer层数
        out_ch: int = 1             # 输出通道数
    
    Forward:
        Input:  [B, 1, T]           # 地震数据
        Output: (final_pred, multi_scale_preds)
            - final_pred: [B, 1, T]             # 最终阻抗预测
            - multi_scale_preds: list of 3 tensors  # Deep supervision辅助输出
                [0]: [B, 1, T//16]  (1/16尺度)
                [1]: [B, 1, T//8]   (1/8尺度)
                [2]: [B, 1, T//4]   (1/4尺度)
    """
```

### 2.3 训练入口（`train.py`）

```python
# 主函数结构：
def main():
    # 1. 解析配置
    cfg = yaml.safe_load(Path(args.config).read_text())
    apply_overrides(cfg, parse_override_kv(args.override))
    
    # 2. 设置随机种子
    seed_everything(cfg["seed"], deterministic=cfg["deterministic"])
    
    # 3. 构建数据集
    train_ds, val_ds, test_ds, unl_ds, stats = build_datasets(
        cfg["data"]["data_root"], 
        norm=NormConfig(**cfg["data"]["norm"])
    )
    
    # 4. 创建DataLoader
    train_loader_l = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], ...)
    train_loader_u = DataLoader(unl_ds, batch_size=..., collate_fn=collate_fn_with_none)
    
    # 5. 构建模型
    model = build_model(cfg).to(device)
    teacher = build_model(cfg).to(device) if cfg["train"]["use_teacher"] else None
    
    # 6. 创建物理正演模型
    wavelet = wavelet_tensor(f0=..., dt=..., length=...)
    fm = ForwardModel(wavelet=wavelet, eps=...).to(device)
    
    # 7. 训练
    test_metrics = train_one_experiment(
        model=model,
        teacher=teacher,
        train_loader_l=train_loader_l,
        train_loader_u=train_loader_u,
        val_loader=val_loader,
        test_loader=test_loader,
        out_dir=out_dir,
        device=device,
        fm=fm,
        cfg=cfg,
        logger=logger
    )

# 模型构建函数：
def build_model(cfg: dict):
    name = cfg["model"]["name"]
    if name == "unet1d":
        return UNet1D(in_ch=1, base=cfg["model"]["base"], depth=cfg["model"]["depth"])
    if name == "tcn1d":
        return TCN1D(in_ch=1, ch=cfg["model"]["ch"], layers=cfg["model"]["layers"], dropout=cfg["model"]["dropout"])
    if name == "cnn_bilstm":
        return CNNBiLSTM(...)
    if name == "ms_physformer":
        return MSPhysFormer(
            in_ch=1, 
            base=cfg["model"]["base"], 
            depth=cfg["model"]["depth"],
            nhead=cfg["model"]["nhead"], 
            tf_dim_mult=cfg["model"]["tf_dim_mult"], 
            tf_layers=cfg["model"]["tf_layers"]
        )
    raise ValueError(f"Unknown model name: {name}")
```

### 2.4 运行命令

```bash
# 训练UNet1D baseline
python train.py --config configs/exp_baseline_unet.yaml

# 训练TCN1D baseline
python train.py --config configs/exp_baseline_tcn.yaml

# 训练MS-PhysFormer
python train.py --config configs/exp_newmodel.yaml

# 参数覆盖示例
python train.py --config configs/exp_newmodel.yaml \
  --override train.epochs=50 train.lr=0.0005 train.batch_size=16
```

## 3) 数据加载接口（`seisinv/data/dataset.py`）

### 3.1 Dataset类

```python
class TraceDataset(Dataset):
    """
    地震-阻抗trace数据集
    
    初始化参数:
        seis: np.ndarray [N, T]         # 地震数据（N条trace，每条T个采样点）
        imp: np.ndarray [N, T] | None   # 阻抗数据（有标签）或None（无标签）
        norm: NormConfig                # 归一化配置
    
    输出格式:
        {
            "seis": Tensor [1, T],      # 归一化后的地震trace
            "imp": Tensor [1, T] | None # 归一化后的阻抗trace（无标签时为None）
        }
    """
    
    def __init__(self, seis: np.ndarray, imp: np.ndarray | None, norm: NormConfig):
        # seis: [N, T], imp: [N, T] or None
        ...
    
    def __getitem__(self, idx: int):
        x = self._norm_seis(self.seis[idx])  # [T]
        x = torch.from_numpy(x).unsqueeze(0)  # [1, T]
        
        if self.imp is None:
            return {"seis": x, "imp": None}
        
        y = self._norm_imp(self.imp[idx])  # [T]
        y = torch.from_numpy(y).unsqueeze(0)  # [1, T]
        
        return {"seis": x, "imp": y}
```

### 3.2 DataLoader输出张量shape

```python
# 有标签DataLoader (用于训练集、验证集、测试集)
for batch in train_loader:
    batch["seis"]  # Tensor [B, 1, T]  地震数据
    batch["imp"]   # Tensor [B, 1, T]  阻抗标签

# 无标签DataLoader (用于半监督训练)
for batch in unlabeled_loader:
    batch["seis"]  # Tensor [B, 1, T]  地震数据
    batch["imp"]   # None               无标签

# 默认参数示例：
# B (batch_size) = 8
# T (trace_length) = 512
# 因此典型shape为：[8, 1, 512]
```

### 3.3 归一化配置

```python
@dataclass
class NormConfig:
    seismic: str = "zscore"      # "none" | "zscore"
    impedance: str = "zscore"    # "none" | "zscore" | "log_zscore"
    eps: float = 1e-6

# 归一化公式：
# zscore: (x - mean) / (std + eps)
# log_zscore: log(x + eps) -> zscore
```

### 3.4 数据构建函数

```python
def build_datasets(data_root: str | Path, norm: NormConfig):
    """
    从目录加载所有.npy文件并构建数据集
    
    输入：
        data_root: 数据根目录
        norm: 归一化配置
    
    输出：
        (train_ds, val_ds, test_ds, unl_ds, stats)
        
        train_ds: TraceDataset  # 训练集（有标签）
        val_ds: TraceDataset    # 验证集（有标签）
        test_ds: TraceDataset   # 测试集（有标签）
        unl_ds: TraceDataset | None  # 无标签集（可选）
        stats: dict  # 归一化统计信息
            {
                "seis_mean": float,
                "seis_std": float,
                "imp_mean": float,
                "imp_std": float,
                "norm": {...}
            }
    
    期望的文件结构：
        data_root/
            train_labeled_seis.npy      # [N_train, T]
            train_labeled_imp.npy       # [N_train, T]
            val_seis.npy                # [N_val, T]
            val_imp.npy                 # [N_val, T]
            test_seis.npy               # [N_test, T]
            test_imp.npy                # [N_test, T]
            train_unlabeled_seis.npy    # [N_unlabeled, T] (可选)
    """
    ...
    return train_ds, val_ds, test_ds, unl_ds, stats
```

### 3.5 实际数据示例（玩具数据）

```python
# 当前data/toy/下的数据shape：
train_labeled_seis.npy:     (64, 512)   # 64条trace，每条512个采样点
train_labeled_imp.npy:      (64, 512)
val_seis.npy:               (16, 512)
val_imp.npy:                (16, 512)
test_seis.npy:              (16, 512)
test_imp.npy:               (16, 512)
train_unlabeled_seis.npy:   (128, 512)  # 无标签数据

# DataLoader batch shape示例（batch_size=8）：
batch["seis"]: torch.Size([8, 1, 512])
batch["imp"]:  torch.Size([8, 1, 512])
```

## 4) 核心接口总结

### 模型接口
- **输入**: `[B, 1, T]` 地震数据
- **输出**: 
  - Baseline: `[B, 1, T]` 阻抗预测
  - MS-PhysFormer: `([B, 1, T], [多尺度列表])` 主输出+辅助输出

### 训练循环接口
```python
train_one_experiment(
    model: nn.Module,           # 学生模型
    teacher: nn.Module | None,  # 教师模型（Mean Teacher）
    train_loader_l: DataLoader, # 有标签训练集
    train_loader_u: DataLoader | None,  # 无标签训练集
    val_loader: DataLoader,     # 验证集
    test_loader: DataLoader,    # 测试集
    out_dir: Path,              # 输出目录
    device: torch.device,       # 设备
    fm: ForwardModel,           # 物理正演模型
    cfg: dict,                  # 配置字典
    logger: SimpleLogger        # 日志记录器
) -> dict:  # 返回测试指标 {"MSE": ..., "PCC": ..., "R2": ...}
```

### 损失函数接口
```python
# 物理正演损失
phys_loss = PhysicsLoss(forward_model=fm, mode="mse")
loss = phys_loss(pred_imp, obs_seis)  # 输入阻抗预测，输出损失

# 频域损失
freq_loss = STFTMagLoss(n_fft=256, hop_length=64, win_length=256)
loss = freq_loss(pred_seis, obs_seis)  # 输入地震数据，输出损失
```

---

**完整代码位置**: `d:\SEISMIC_CODING\new\`  
**文档**: [README.md](README.md), [QUICK_START.md](QUICK_START.md), [PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md)
