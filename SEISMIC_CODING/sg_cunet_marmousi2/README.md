# SG-CUnet（论文复现）— Marmousi2 全流程

本项目基于你提供的论文思路，给出**从头训练到推理与可视化**的完整实现：
- **数据**：Marmousi2 阻抗 `marmousi_Ip_model.npy` 与合成地震 `marmousi_synthetic_seismic.npy`（二维，同尺寸）。
- **模型**：SG-CUnet（U-Net 主干 + 双分支头：阻抗 / 反射）。
- **损失**：四项闭环物理约束 + Kendall 同方差不确定性加权。
- **扩增**：弹性形变（论文 α=1500, δ=100 的思路）。
- **复现图**：
  - 图3：原始阻抗 vs 弹性形变后的“扩展阻抗”（`fig3_aug_impedance.png`）
  - 图5：道 1343 的**实际/预测/平均**三曲线（`fig5_trace1343.png`）

> 若你手头是 SEGY，可先自行转换为 `.npy`（形状 `(T, X)`，时间在第0维、道在第1维）。

## 1. 安装
```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. 放置数据
将两张 `.npy` 放到项目根目录：
```
marmousi_Ip_model.npy
marmousi_synthetic_seismic.npy
```

## 3. 训练
```bash
python scripts/train.py --z_path marmousi_Ip_model.npy                         --s_path marmousi_synthetic_seismic.npy                         --workdir ./exp_sg_cunet                         --epochs 50 --batch_size 8
```
输出：`exp_sg_cunet/train_log.txt`, `loss_curve.png`, `model_last.pth`, 以及图3/图5结果图。

> **复现实验**：论文使用更长训练（~500 epochs）和更大数据块；你可以把 `--epochs` 调大获取更稳定的结果。

## 4. 推理（可单独运行）
```bash
python scripts/infer.py --z_path marmousi_Ip_model.npy                         --s_path marmousi_synthetic_seismic.npy                         --workdir ./exp_sg_cunet                         --model ./exp_sg_cunet/model_last.pth
```

## 5. 生成图 3 / 图 5（可重复）
```bash
# 图3：扩展阻抗可视化
python scripts/plot_fig3.py --z_path marmousi_Ip_model.npy --workdir ./exp_sg_cunet

# 图5：道 1343 实际/预测/平均
python scripts/plot_fig5.py --z_path marmousi_Ip_model.npy                             --s_path marmousi_synthetic_seismic.npy                             --workdir ./exp_sg_cunet                             --model ./exp_sg_cunet/model_last.pth
```

## 目录结构
```
sg_cunet_marmousi2/
  models/sg_cunet.py
  utils/{data.py,geo.py,phys.py,vis.py}
  scripts/{train.py,infer.py,plot_fig3.py,plot_fig5.py}
  requirements.txt
  README.md
```

---

**说明**：本实现严格遵循论文的核心逻辑（2D 小块、双头输出、四项损失 + Kendall 加权、35Hz Ricker 正演），并补齐工程细节（弹性形变、滑窗推理、可视化脚本）。
