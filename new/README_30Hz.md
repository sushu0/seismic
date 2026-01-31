# 30Hz 地震反演训练项目结构说明

## 📁 项目文件夹结构

```
D:\SEISMIC_CODING\new\
├── train_30Hz_thinlayer_v2.py           ⭐ 30Hz 数据训练脚本 (主要)
├── visualize_30Hz.py                    ⭐ 30Hz 数据可视化脚本
├── train_thinlayer_v2.py                📄 20Hz 数据训练脚本 (参考)
├── visualize_thinlayer_v2.py            📄 20Hz 数据可视化脚本 (参考)
│
└── results/
    │
    ├── 01_20Hz_thinlayer_v2/            📊 20Hz 训练结果目录
    │   ├── checkpoints/
    │   │   ├── best.pt                  (最佳模型 - 验证集PCC最高)
    │   │   └── last.pt                  (最后模型)
    │   ├── figures/
    │   │   ├── figures_improved/        (改进版高质量图像)
    │   │   ├── beautiful_comparison_test.png
    │   │   ├── beautiful_comparison_all.png
    │   │   ├── error_analysis.png
    │   │   └── trace_comparison.png
    │   ├── logs/
    │   ├── norm_stats.json              (归一化参数)
    │   └── test_metrics.json            (测试集指标)
    │
    └── 01_30Hz_thinlayer_v2/            📊 30Hz 训练结果目录 (NEW)
        ├── checkpoints/
        │   ├── best.pt                  (最佳模型)
        │   └── last.pt                  (最后模型)
        ├── figures/
        │   ├── beautiful_comparison_test.png
        │   ├── beautiful_comparison_all.png
        │   ├── error_analysis.png
        │   ├── trace_comparison.png
        │   └── metrics_summary.txt
        ├── logs/
        ├── norm_stats.json              (归一化参数)
        └── test_metrics.json            (测试集指标)

```

## 📊 30Hz 数据信息

### 输入训练数据 (2个)

| 名称 | 路径 | 用途 | 说明 |
|------|------|------|------|
| **地震数据** | `D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_re.sgy` | 模型输入 (X) | SEG-Y 格式, 30Hz 主频地震记录 |
| **阻抗数据** | `D:\SEISMIC_CODING\zmy_data\01\data\01_30Hz_04.txt` | 标签/目标 (Y) | 文本格式, 对应的声阻抗剖面 |

### 任务描述
- **地震反演**: 从地震波形预测声阻抗分布
- **输入**: 地震记录 (2通道: 原始地震 + 高频成分)
- **输出**: 阻抗预测
- **应用**: 薄层识别、地质界面检测

## 🚀 快速开始

### 1. 训练模型 (30Hz)

```bash
# 进入项目目录
cd D:\SEISMIC_CODING\new

# 激活虚拟环境
.\.venv\Scripts\activate

# 运行训练脚本
python train_30Hz_thinlayer_v2.py
```

**训练参数:**
- Epochs: 500
- Batch Size: 4
- Learning Rate: 3e-4
- Loss: 组合损失 (MSE + 梯度匹配 + 稀疏 + 正演一致性)

**预期输出:**
- 控制台日志: 实时训练进度 (PCC, R², 薄层F1等)
- 模型文件: `results/01_30Hz_thinlayer_v2/checkpoints/best.pt`
- 指标文件: `results/01_30Hz_thinlayer_v2/test_metrics.json`

### 2. 生成可视化 (30Hz)

```bash
python visualize_30Hz.py
```

**生成图像:**
- `beautiful_comparison_test.png`: 测试集三通道对比 (预测/真实/误差)
- `beautiful_comparison_all.png`: 截断数据集对比
- `error_analysis.png`: 误差分布、散点密度、道均误差
- `trace_comparison.png`: 单道波形和误差时间序列

**图像特点:**
- DPI: 250 (高分辨率)
- Colormap: Jet (色彩丰富)
- 百分位数裁剪: [3%, 97%] (突出细节)

## 📈 模型架构

### ThinLayerNetV2 特点

```
输入: 双通道 [原始地震, 高频成分]
     ↓
编码器: Multi-scale (膨胀卷积)
     ↓
瓶颈: 深层特征提取 (dilations=[1,2,4,8,16])
     ↓
解码器: U-Net 结构 + 边界增强
     ↓
输出: 单通道阻抗预测
```

**关键模块:**
- **膨胀卷积块** (Dilated Conv Block): 多尺度上下文
- **边界增强模块** (Boundary Enhancement): 强化界面特征
- **薄层块** (ThinLayer Block): 薄层优化

## 📉 损失函数

```
总损失 = MSE + λ_grad*梯度损失 + λ_sparse*稀疏损失 + λ_fwd*正演一致性

其中:
  λ_grad = 0.3      (梯度匹配权重)
  λ_sparse = 0.05   (稀疏正则权重)
  λ_fwd = 0.1       (正演一致性权重)
```

## 📊 评估指标

### 全局指标
- **PCC**: 皮尔逊相关系数 (0-1, 越高越好)
- **R²**: 决定系数 (0-1, 越高越好)
- **MSE**: 均方误差 (越低越好)

### 薄层专用指标
- **薄层 F1**: 薄层检测准确率
- **薄层 PCC**: 薄层区域相关系数
- **DPDE**: 双峰距误差 (采样点, 越小越好)
- **分离度**: 薄层分离程度 (0-1, 越高越好)

## 📁 文件说明

### 核心脚本

| 文件 | 功能 | 输入 | 输出 |
|------|------|------|------|
| `train_30Hz_thinlayer_v2.py` | 30Hz 数据训练 | SEG-Y + TXT | 模型 + 指标 |
| `visualize_30Hz.py` | 结果可视化 | 模型 + 数据 | PNG 图像 |
| `train_20Hz_thinlayer_v2.py` | 20Hz 数据训练 | SEG-Y + TXT | 模型 + 指标 |
| `visualize_thinlayer_v2.py` | 20Hz 可视化 | 模型 + 数据 | PNG 图像 |

### 输出目录结构

```
checkpoints/
  ├── best.pt         (最佳模型权重)
  └── last.pt         (最后保存的模型)

figures/
  ├── beautiful_comparison_test.png
  ├── beautiful_comparison_all.png
  ├── error_analysis.png
  ├── trace_comparison.png
  └── metrics_summary.txt

logs/
  (训练日志保存目录，留作扩展)

norm_stats.json     (数据归一化参数)
test_metrics.json   (测试集评估指标)
```

## 🔧 数据增强

### 薄层注入增强
- **概率**: 50%
- **方式**: 随机注入人工薄层
- **厚度范围**: 5-30 采样点
- **强度**: 0.5-2.0× 标准差

### 高频滤波
- **高通滤波器**: Butterworth, order=4
- **截止频率**: 18Hz (原始地震), 12Hz (高频通道)
- **目的**: 提取高频成分作为辅助输入通道

## 📊 数据划分

```
总道数: ~100 traces

训练集: 60% (60条)
验证集: 20% (20条)
测试集: 20% (20条)
```

## 🎯 预期性能

基于 20Hz 数据的参考指标 (500 epochs):

| 指标 | 20Hz | 预期30Hz |
|------|------|----------|
| PCC | 0.93 | 0.92-0.94 |
| R² | 0.86 | 0.85-0.88 |
| 薄层F1 | 0.78 | 0.75-0.80 |

## 🛠️ 常见问题

### Q1: 如何修改训练参数?
编辑 `train_30Hz_thinlayer_v2.py` 中的 `Config` 类:
```python
class Config:
    EPOCHS = 500      # 修改训练轮数
    BATCH_SIZE = 4    # 修改批大小
    LR = 3e-4         # 修改学习率
    DOMINANT_FREQ = 30.0  # 修改主频
```

### Q2: 如何使用预训练模型?
```python
checkpoint = torch.load('results/01_30Hz_thinlayer_v2/checkpoints/best.pt')
model.load_state_dict(checkpoint['model'])
```

### Q3: 如何预测新数据?
参考 `visualize_30Hz.py` 中的预测部分,调整数据路径即可。

## 📝 许可与参考

- 数据来源: zmy_data 目录
- 模型: ThinLayerNet V2 (薄层反演优化版)
- 框架: PyTorch 2.6.0+cu124

## 📧 联系方式

如有问题,请检查:
1. 数据路径是否正确
2. 虚拟环境是否激活
3. CUDA 是否可用 (可选)

---

**最后更新**: 2026-01-03  
**项目状态**: ✅ 生产就绪  
**30Hz 数据**: ✅ 准备训练  
