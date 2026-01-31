# 30Hz 薄层模型训练 - 快速开始

## 📊 当前状态

- ✅ **训练进度**: 18/500 epochs (3.6%)
- ✅ **最佳指标**: val_pcc=0.8937, val_r2=0.7560 (Epoch 16)
- 🎯 **目标指标**: val_pcc≥0.93, val_r2≥0.86
- ✅ **断点续训**: 已集成并验证
- 📁 **断点文件**: `results/01_30Hz_thinlayer_v2/checkpoints/last.pt` (29.84 MB)

## 🚀 立即开始训练

### 最简单的方式 (推荐)

双击运行:
```
START_TRAIN_30Hz.bat
```

该脚本会自动:
1. ✅ 检测虚拟环境
2. ✅ 验证 PyTorch 安装
3. ✅ 检测并加载断点
4. ✅ 启动训练 (从 Epoch 19 继续)

### 命令行方式

```batch
cd D:\SEISMIC_CODING\new

# 激活虚拟环境 (如果需要)
.venv\Scripts\activate

# 运行训练 (自动从断点继续)
python train_30Hz_thinlayer_v2.py
```

## 📈 实时监控训练

在**另一个终端**运行:

```powershell
cd D:\SEISMIC_CODING\new
.\monitor_train_30Hz.ps1
```

显示内容:
- 当前最佳模型的 epoch 和指标
- 实时训练日志 (最后 20 行)
- 日志文件状态
- 每 5 秒自动刷新

## 🔍 检查训练进度

```batch
# 快速查看日志最后 10 行
cd D:\SEISMIC_CODING\new
Get-Content results\01_30Hz_thinlayer_v2\train_log.txt -Tail 10

# 或运行进度检查脚本 (需要虚拟环境)
python check_train_progress.py
```

## 🎨 训练完成后的可视化

训练达到 500 epochs 或提前停止后:

```batch
cd D:\SEISMIC_CODING\new
python visualize_complete.py
```

生成内容:
- 阻抗反演剖面对比图
- 误差分布和统计
- 薄层检测效果
- 所有图像保存到 `results/01_30Hz_thinlayer_v2/visualizations/`

## ⏱️ 预计时间

- **剩余**: 482 epochs
- **预估**: 
  - GPU (RTX 3090): ~16-20 小时
  - GPU (GTX 1080): ~30-40 小时
  - CPU: 不推荐 (可能需要数天)

## 🛠️ 故障排查

### 训练中断了怎么办?

直接重新运行 `START_TRAIN_30Hz.bat` 或 `python train_30Hz_thinlayer_v2.py`，会自动从断点继续。

### 显存不足 (CUDA out of memory)

编辑 `train_30Hz_thinlayer_v2.py`:
```python
class Config:
    BATCH_SIZE = 2  # 从 4 改为 2 或 1
```

### 如何从头开始训练?

```python
# 方法 1: 修改配置
class Config:
    RESUME = False  # train_30Hz_thinlayer_v2.py 第 95 行

# 方法 2: 删除断点
Remove-Item results\01_30Hz_thinlayer_v2\checkpoints\last.pt
```

## 📁 项目文件

```
new/
├── train_30Hz_thinlayer_v2.py       ← 训练脚本 (已集成断点续训)
├── visualize_complete.py            ← 可视化脚本
├── START_TRAIN_30Hz.bat             ← 一键启动 (新增)
├── resume_train_30Hz.bat            ← 简化启动 (新增)
├── monitor_train_30Hz.ps1           ← 实时监控 (新增)
├── check_train_progress.py          ← 进度检查 (新增)
├── test_resume.py                   ← 断点验证 (新增)
└── results/01_30Hz_thinlayer_v2/
    ├── checkpoints/
    │   ├── best.pt                  ← 最佳模型
    │   └── last.pt                  ← 断点续训用
    ├── train_log.txt                ← 训练日志
    ├── norm_stats.json              ← 归一化参数
    └── test_metrics.json            ← 测试集指标 (训练完成后)
```

## 📚 详细文档

查看完整指南: [TRAINING_GUIDE_30Hz.md](TRAINING_GUIDE_30Hz.md)

## ✅ 操作清单

- [ ] 1. 双击运行 `START_TRAIN_30Hz.bat` 启动训练
- [ ] 2. 另开终端运行 `.\monitor_train_30Hz.ps1` 监控进度
- [ ] 3. 等待训练完成 (可能需要 16-40 小时)
- [ ] 4. 运行 `python visualize_complete.py` 生成可视化
- [ ] 5. 检查指标是否达到目标 (val_pcc≥0.93, val_r2≥0.86)

---

**最后更新**: 2026-01-05  
**状态**: ✅ 断点续训已集成，可以安全继续训练  
**下一步**: 运行 `START_TRAIN_30Hz.bat`
