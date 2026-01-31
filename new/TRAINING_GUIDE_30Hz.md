# 30Hz 薄层模型训练完整指南

## 当前状态

**训练进度**: Epoch 18/500 (已完成 3.6%)  
**最佳指标**: Epoch 16 → val_pcc=**0.8937**, val_r2=**0.7560**  
**目标指标**: val_pcc≈**0.93**, val_r2≈**0.86** (参考 20Hz 模型)  
**断点文件**: `results/01_30Hz_thinlayer_v2/checkpoints/last.pt` (29.84 MB, 包含 Epoch 18 状态)

## 训练操作

### 方法 1: 使用快捷脚本 (推荐)

```batch
# Windows 命令行
cd D:\SEISMIC_CODING\new
resume_train_30Hz.bat
```

该脚本会：
- ✅ 自动检测虚拟环境 (.venv 或 venv)
- ✅ 检测断点文件 last.pt
- ✅ 从 Epoch 19 继续训练到 Epoch 500

### 方法 2: 直接运行 Python (手动指定环境)

```bash
# 如果有虚拟环境
cd D:\SEISMIC_CODING\new
.venv\Scripts\python.exe train_30Hz_thinlayer_v2.py

# 或激活虚拟环境后运行
.venv\Scripts\activate
python train_30Hz_thinlayer_v2.py
```

### 方法 3: 后台运行 (长时间训练推荐)

**PowerShell 后台运行**:
```powershell
cd D:\SEISMIC_CODING\new
Start-Process -NoNewWindow -FilePath ".venv\Scripts\python.exe" `
  -ArgumentList "train_30Hz_thinlayer_v2.py" `
  -RedirectStandardOutput "train_output.log" `
  -RedirectStandardError "train_error.log"
```

**或使用 nohup (如果安装了 Git Bash)**:
```bash
cd /d/SEISMIC_CODING/new
nohup .venv/Scripts/python.exe train_30Hz_thinlayer_v2.py > train_output.log 2>&1 &
```

## 训练监控

### 实时监控 (PowerShell)

```powershell
cd D:\SEISMIC_CODING\new
.\monitor_train_30Hz.ps1
```

每 5 秒刷新一次，显示：
- 最佳模型 epoch 和指标
- 最后 20 行训练日志
- 日志文件状态

### 快速查看进度

```powershell
# 查看最后 10 个 epoch
Get-Content results\01_30Hz_thinlayer_v2\train_log.txt -Tail 20

# 监控文件变化
Get-Item results\01_30Hz_thinlayer_v2\train_log.txt | Select-Object Name, Length, LastWriteTime
```

## 断点续训机制

训练脚本已集成断点续训功能 (修改点在 `train_30Hz_thinlayer_v2.py` Line ~803):

```python
# Config 类中设置
RESUME = True  # 若为 True 且存在 last.pt，则从断点继续

# 训练开始前自动检测
if CFG.RESUME and ckpt_last.exists():
    # 加载: model, optimizer, scheduler, 起始 epoch, best_val_pcc
    start_epoch = ckpt.get('epoch', 0) + 1
    # 从 Epoch 19 继续训练
```

**保存策略**:
- `best.pt`: 验证集 PCC 创新高时保存 (包含完整状态)
- `last.pt`: 每 **10** 个 epoch 保存一次 + 训练结束时保存

## 预计训练时间

- **已训练**: 18 epochs
- **剩余**: 482 epochs
- **预计时间**: 取决于硬件 (GPU/CPU)
  - 如果每 epoch ≈ 2 分钟 → 剩余约 **16 小时**
  - 如果每 epoch ≈ 5 分钟 → 剩余约 **40 小时**

建议在**稳定的机器上后台运行**，或使用云 GPU 加速。

## 训练完成后的可视化

训练达到 500 epochs (或提前停止) 后，运行可视化：

```bash
cd D:\SEISMIC_CODING\new
python visualize_complete.py
```

该脚本会：
1. 加载 `results/01_30Hz_thinlayer_v2/checkpoints/best.pt`
2. 在测试集上评估
3. 生成完整的可视化图像:
   - 阻抗反演剖面对比 (真值 vs 预测)
   - 误差分布图
   - 薄层检测效果
4. 保存到 `results/01_30Hz_thinlayer_v2/visualizations/`

## 关键指标说明

| 指标 | 当前 (Epoch 16) | 目标 (20Hz 参考) | 说明 |
|------|----------------|------------------|------|
| **val_pcc** | 0.8937 | ~0.93 | Pearson 相关系数，衡量整体波形相似性 |
| **val_r2** | 0.7560 | ~0.86 | R² 决定系数，衡量拟合质量 |
| **thin_pcc** | 0.2522 | - | 薄层区域的 PCC (更难预测) |
| **thin_f1** | 0.0019 | - | 薄层三分类 F1 分数 |
| **separability** | 0.8942 | - | 双峰分离度 |

## 疑难排查

### 1. 训练中断了怎么办?

直接重新运行训练命令，脚本会自动从 `last.pt` 恢复。

### 2. 如何从头开始训练?

```python
# 方法 1: 修改 train_30Hz_thinlayer_v2.py
class Config:
    RESUME = False  # 改为 False

# 方法 2: 删除 last.pt
Remove-Item results\01_30Hz_thinlayer_v2\checkpoints\last.pt
```

### 3. 训练卡住/显存不足

```python
# 减小 batch size (修改 train_30Hz_thinlayer_v2.py)
class Config:
    BATCH_SIZE = 2  # 从 4 改为 2
```

### 4. 可视化脚本报错

确保使用 `weights_only=False`:
```python
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
```

## 文件结构

```
new/
├── train_30Hz_thinlayer_v2.py       # 训练脚本 (已集成断点续训)
├── visualize_complete.py            # 可视化脚本
├── resume_train_30Hz.bat            # 快捷启动脚本 (新增)
├── monitor_train_30Hz.ps1           # 实时监控脚本 (新增)
├── check_train_progress.py          # 进度检查工具 (新增)
└── results/
    └── 01_30Hz_thinlayer_v2/
        ├── checkpoints/
        │   ├── best.pt              # 最佳模型 (验证集最高 PCC)
        │   └── last.pt              # 断点续训用 (每 10 epochs 更新)
        ├── train_log.txt            # 训练日志 (实时追加)
        ├── norm_stats.json          # 归一化参数
        └── test_metrics.json        # 测试集指标 (训练完成后生成)
```

## 下一步行动

1. **立即操作**: 运行 `resume_train_30Hz.bat` 继续训练
2. **监控**: 另开一个 PowerShell 窗口运行 `.\monitor_train_30Hz.ps1`
3. **耐心等待**: 训练到 500 epochs (或观察到 val_pcc 不再上升)
4. **验证结果**: 运行 `visualize_complete.py` 生成最终图像和指标

---

**最后更新**: 2026-01-05  
**当前状态**: ✅ 断点续训功能已集成，可以安全继续训练
