# 快速开始指南

## 一、环境准备

### 1. 安装Python依赖
```powershell
pip install -r requirements.txt
```

需要的包：numpy, pyyaml, matplotlib, torch, tqdm

### 2. 检查GPU（可选）
```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

如果没有GPU，会自动使用CPU（速度较慢）。

## 二、数据准备

### 选项A：使用玩具数据（推荐用于快速测试）

```powershell
python scripts/generate_toy_data.py --out_dir data/toy --n_train 64 --n_val 16 --n_test 16 --n_unlabeled 128
```

生成的文件：
- `data/toy/train_labeled_seis.npy` - 训练地震数据 (64, 512)
- `data/toy/train_labeled_imp.npy` - 训练阻抗数据 (64, 512)
- `data/toy/val_seis.npy` / `val_imp.npy` - 验证集 (16, 512)
- `data/toy/test_seis.npy` / `test_imp.npy` - 测试集 (16, 512)
- `data/toy/train_unlabeled_seis.npy` - 无标签数据 (128, 512)

### 选项B：使用你自己的数据

准备以下.npy文件（格式 `[N, T]`，N是trace数量，T是采样点数）：
```
your_data/
  ├── train_labeled_seis.npy
  ├── train_labeled_imp.npy
  ├── val_seis.npy
  ├── val_imp.npy
  ├── test_seis.npy
  ├── test_imp.npy
  └── train_unlabeled_seis.npy  (可选)
```

然后修改配置：
```powershell
python train.py --config configs/exp_newmodel.yaml --override data.data_root=your_data
```

## 三、运行实验

### 1. Baseline模型

**UNet1D**:
```powershell
python train.py --config configs/exp_baseline_unet.yaml
```

**TCN1D**:
```powershell
python train.py --config configs/exp_baseline_tcn.yaml
```

### 2. 新模型 MS-PhysFormer

**推荐配置（纯监督）**:
```powershell
python train.py --config configs/exp_newmodel.yaml --override train.lambda_phys=0.0 train.lambda_freq=0.0 train.lambda_cons=0.0 train.use_teacher=false
```

**完整配置（含物理约束，可能不稳定）**:
```powershell
python train.py --config configs/exp_newmodel.yaml
```

### 3. 快速测试（减少训练轮数）

```powershell
# 5 epochs快速测试
python train.py --config configs/exp_baseline_unet.yaml --override train.epochs=5
python train.py --config configs/exp_newmodel.yaml --override train.epochs=5 train.lambda_phys=0.0 train.lambda_freq=0.0
```

### 4. 消融实验

**去掉物理约束**:
```powershell
python train.py --config configs/abl_no_physics.yaml
```

**去掉频域约束**:
```powershell
python train.py --config configs/abl_no_freq.yaml
```

## 四、查看结果

### 每个实验的输出位置

```
results/<exp_name>/
  ├── metrics.csv                      # 训练曲线（每epoch的loss和指标）
  ├── test_metrics.json                # 最终测试指标
  ├── norm_stats.json                  # 归一化统计
  ├── pred_vs_true_traces.png          # 预测vs真实阻抗对比
  ├── pred_imp_section.png             # 预测阻抗剖面
  ├── true_imp_section.png             # 真实阻抗剖面
  ├── seis_obs_section.png             # 观测地震剖面
  ├── seis_recon_section.png           # 重构地震剖面
  └── checkpoints/
      ├── best.pt                      # 最佳模型
      └── last.pt                      # 最后一个epoch
```

### 汇总多个实验结果

```powershell
python scripts/collect_results.py --results_root results --out_csv results/summary.csv --exp_names baseline_unet1d baseline_tcn1d new_ms_physformer
```

会生成 `results/summary.csv`：
```csv
exp_name,best_epoch,best_val_mse,test_MSE,test_PCC,test_R2
baseline_unet1d,5,0.867,0.865,0.544,0.227
baseline_tcn1d,3,0.974,1.014,0.394,0.093
new_ms_physformer,5,0.951,0.961,0.464,0.141
```

## 五、参数调整

### 常用覆盖参数

**训练轮数**:
```powershell
--override train.epochs=100
```

**学习率**:
```powershell
--override train.lr=0.0005
```

**批大小**:
```powershell
--override train.batch_size=16
```

**损失权重**:
```powershell
--override train.lambda_phys=0.01 train.lambda_freq=0.01 train.lambda_cons=0.2
```

**关闭半监督**:
```powershell
--override train.use_unlabeled=false train.use_teacher=false
```

**组合多个参数**:
```powershell
python train.py --config configs/exp_newmodel.yaml --override train.epochs=50 train.lr=0.0005 train.batch_size=16
```

### 模型架构调整

**UNet通道数**:
```powershell
--override model.base=64
```

**UNet深度**:
```powershell
--override model.depth=5
```

**Transformer配置**:
```powershell
--override model.nhead=8 model.tf_layers=4
```

## 六、常见问题

### Q1: 训练损失爆炸或出现NaN？
**A**: 降低物理损失权重或关闭：
```powershell
python train.py --config configs/exp_newmodel.yaml --override train.lambda_phys=0.0 train.lambda_freq=0.0
```

### Q2: GPU内存不足？
**A**: 减小批大小或模型通道数：
```powershell
--override train.batch_size=4 model.base=32
```

### Q3: 训练太慢？
**A**: 减少训练轮数或使用更小的模型：
```powershell
--override train.epochs=50 model.depth=3
```

### Q4: 如何使用CPU训练？
**A**: 修改配置：
```powershell
--override device=cpu
```

### Q5: 如何加载训练好的模型？
**A**: 使用PyTorch加载：
```python
import torch
checkpoint = torch.load('results/new_ms_physformer/checkpoints/best.pt')
model.load_state_dict(checkpoint['model'])
```

## 七、完整实验流程示例

### 示例1：对比baseline和新模型

```powershell
# 1. 生成数据
python scripts/generate_toy_data.py --out_dir data/toy --n_train 64 --n_val 16 --n_test 16 --n_unlabeled 128

# 2. 训练baseline
python train.py --config configs/exp_baseline_unet.yaml
python train.py --config configs/exp_baseline_tcn.yaml

# 3. 训练新模型
python train.py --config configs/exp_newmodel.yaml --override train.lambda_phys=0.0 train.lambda_freq=0.0 train.lambda_cons=0.0 train.use_teacher=false output.exp_name=new_ms_physformer_supervised

# 4. 汇总结果
python scripts/collect_results.py --results_root results --out_csv results/comparison.csv --exp_names baseline_unet1d baseline_tcn1d new_ms_physformer_supervised
```

### 示例2：超参数调优

```powershell
# 测试不同学习率
python train.py --config configs/exp_newmodel.yaml --override train.lr=0.0005 output.exp_name=ms_lr_0005
python train.py --config configs/exp_newmodel.yaml --override train.lr=0.001 output.exp_name=ms_lr_001
python train.py --config configs/exp_newmodel.yaml --override train.lr=0.002 output.exp_name=ms_lr_002

# 汇总
python scripts/collect_results.py --results_root results --out_csv results/lr_tuning.csv --exp_names ms_lr_0005 ms_lr_001 ms_lr_002
```

### 示例3：消融研究

```powershell
# 完整模型
python train.py --config configs/exp_newmodel.yaml --override train.lambda_phys=0.0 output.exp_name=full_model

# 去掉Transformer
python train.py --config configs/exp_baseline_unet.yaml --override model.base=48 model.depth=4 output.exp_name=no_transformer

# 去掉Deep Supervision（修改配置或代码）
# ...

# 汇总对比
python scripts/collect_results.py --results_root results --out_csv results/ablation.csv --exp_names full_model no_transformer
```

## 八、下一步

1. **修改真实数据**: 参考"二、数据准备 - 选项B"
2. **调整超参数**: 参考"五、参数调整"
3. **修复物理损失**: 查看 [PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md) 中的"已知问题与建议"
4. **论文写作**: 查看报告中的"论文写作建议"

## 九、获取帮助

- **详细文档**: [README.md](README.md)
- **完成报告**: [PROJECT_COMPLETION_REPORT.md](PROJECT_COMPLETION_REPORT.md)
- **代码注释**: 查看各模块源代码中的docstring

祝实验顺利！🚀
