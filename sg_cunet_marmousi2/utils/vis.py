import os
import numpy as np
import matplotlib.pyplot as plt

def imshow2(A, B, titles=("A","B"), out=None, cmap='jet'):
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    im0 = ax[0].imshow(A, cmap=cmap, aspect='auto'); ax[0].set_title(titles[0]); ax[0].invert_yaxis()
    im1 = ax[1].imshow(B, cmap=cmap, aspect='auto'); ax[1].set_title(titles[1]); ax[1].invert_yaxis()
    plt.colorbar(im0, ax=ax[0], fraction=0.046); plt.colorbar(im1, ax=ax[1], fraction=0.046)
    plt.tight_layout()
    if out: plt.savefig(out, dpi=200)
    plt.close()

def plot_trace3(true_imp, pred_imp, mean_imp, dt=0.001, trace_id=1343, out=None):
    depth_axis = np.arange(len(true_imp)) * dt * 1000.0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,6))
    plt.plot(true_imp, depth_axis, label="实际阻抗", linewidth=2)
    plt.plot(pred_imp, depth_axis, label="预测阻抗(一次)", linewidth=1)
    plt.plot(mean_imp, depth_axis, label="平均阻抗", linewidth=2)
    plt.gca().invert_yaxis()
    plt.xlabel("Impedance"); plt.ylabel("Time (ms)")
    plt.legend(); plt.title(f"道 {trace_id}：实际/预测/平均")
    plt.tight_layout()
    if out: plt.savefig(out, dpi=200)
    plt.close()
