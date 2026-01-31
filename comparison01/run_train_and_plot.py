# run_train_and_plot.py
from marmousi_cnn_bilstm import main as train_main
import plot_trace_comparison


def main():
    # 1) 先完整跑一遍训练（监督 + 半监督）
    train_main()

    # 2) 训练结束后，调用绘图脚本的 main()，画 4 条代表性测试道
    plot_trace_comparison.main()


if __name__ == "__main__":
    main()

