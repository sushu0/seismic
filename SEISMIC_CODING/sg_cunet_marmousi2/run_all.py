# -*- coding: utf-8 -*-
# 一键运行脚本：自动安装依赖 -> 数据准备 -> 两阶段训练(50+150) -> 画图

import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
WORK = os.path.join(ROOT, "exp_sg_cunet")
REQS = os.path.join(ROOT, "requirements.txt")

def run(cmd, cwd=None):
    print("\n[RUN]", " ".join(map(str, cmd)))
    subprocess.check_call(cmd, cwd=cwd or ROOT)

def ensure_dirs():
    os.makedirs(DATA, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)

def ensure_requirements():
    if os.path.isfile(REQS):
        try:
            run([sys.executable, "-m", "pip", "install", "-r", REQS])
        except subprocess.CalledProcessError as e:
            print("[WARN] 依赖安装出错，继续尝试运行。", e)

def main():
    ensure_dirs()
    ensure_requirements()

    # 1) 数据准备
    prep_py = os.path.join(DATA, "prepare_data.py")
    if not os.path.isfile(prep_py):
        raise FileNotFoundError(f"未找到: {prep_py}")
    run([sys.executable, prep_py], cwd=DATA)

    z_path = os.path.join(DATA, "marmousi_Ip_model.npy")
    s_path = os.path.join(DATA, "marmousi_synthetic_seismic.npy")
    if not (os.path.isfile(z_path) and os.path.isfile(s_path)):
        raise FileNotFoundError("数据准备失败：缺 marmousi_Ip_model.npy 或 marmousi_synthetic_seismic.npy")

    # 2) 两阶段训练（适配 4050：小模型+小窗+累积梯度+AMP）
    train_py = os.path.join(ROOT, "scripts", "train.py")
    if not os.path.isfile(train_py):
        raise FileNotFoundError(f"未找到训练脚本: {train_py}")

    # —— Stage 1: 50 轮（预训练，较强正则/增强）——
    run([sys.executable, train_py,
         "--z_path", z_path, "--s_path", s_path, "--workdir", WORK,
         "--epochs", "50",
         "--batch_size", "4",
         "--accum_steps", "2",
         "--lr", "2e-4",
         "--base", "24",
         "--t_win", "128", "--x_win", "16",
         "--stride_t", "64",  "--stride_x", "8",
         "--alpha", "600", "--sigma", "50",
         "--dt", "0.005", "--nt", "128"])

    # —— Stage 2: 150 轮（微调，弱增强/低学习率，从上阶段继续）——
    model_pth = os.path.join(WORK, "model_last.pth")
    run([sys.executable, train_py,
         "--z_path", z_path, "--s_path", s_path, "--workdir", WORK,
         "--epochs", "150",
         "--batch_size", "4",
         "--accum_steps", "2",
         "--lr", "1e-4",
         "--base", "24",
         "--t_win", "128", "--x_win", "16",
         "--stride_t", "64",  "--stride_x", "8",
         "--alpha", "400", "--sigma", "40",
         "--dt", "0.005", "--nt", "128",
         "--resume_from", model_pth])

    # 3) 画图（图3 + 图5（两条线：真实/预测））
    plot3 = os.path.join(ROOT, "scripts", "plot_fig3.py")
    plot5 = os.path.join(ROOT, "scripts", "plot_fig5.py")
    if os.path.isfile(plot3):
        run([sys.executable, plot3, "--z_path", z_path, "--workdir", WORK])
    if os.path.isfile(plot5):
        run([sys.executable, plot5, "--z_path", z_path, "--s_path", s_path,
             "--workdir", WORK, "--model", model_pth, "--trace_idx", "1343"])

    print("\n[OK] 全流程完成。结果输出目录：", WORK)
    print(" - 训练日志与曲线：train_log.txt, loss_curve.png")
    print(" - 图3：fig3_aug_impedance.png")
    print(" - 图5：fig5_trace1343_2lines.png")

if __name__ == "__main__":
    main()
