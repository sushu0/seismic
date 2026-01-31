from __future__ import annotations
import os, argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="pred .npz from infer.py")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_traces", type=int, default=0, help="0 means all")
    ap.add_argument("--time_ms_max", type=float, default=None, help="If set, y-axis is Time (ms) from 0..time_ms_max")
    ap.add_argument("--cmap", default="jet")
    ap.add_argument("--title_suffix", default="", help="e.g. ' (Marmousi2, marmousi2_2721_like)'")
    ap.add_argument("--vmin", type=float, default=None)
    ap.add_argument("--vmax", type=float, default=None)
    args = ap.parse_args()

    z = np.load(args.pred, allow_pickle=True)
    pred = z["pred"][:,0,:]
    true = z["true"][:,0,:]
    n_total = int(pred.shape[0])
    n = n_total if int(args.n_traces) <= 0 else min(int(args.n_traces), n_total)

    pred2 = pred[:n].T  # [T, N]
    true2 = true[:n].T  # [T, N]
    err2 = pred2 - true2

    os.makedirs(args.outdir, exist_ok=True)

    extent = None
    y_label = "sample"
    if args.time_ms_max is not None:
        extent = [0, n - 1, float(args.time_ms_max), 0.0]
        y_label = "Time (ms)"

    def _plot_one(fname: str, arr: np.ndarray, title: str, cbar_label: str, vmin: float | None, vmax: float | None):
        plt.figure(figsize=(12, 4.5))
        im = plt.imshow(
            arr,
            aspect="auto",
            origin="upper",
            cmap=args.cmap,
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        cb = plt.colorbar(im)
        cb.set_label(cbar_label)
        plt.title(title)
        plt.xlabel("Trace number")
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, fname), dpi=300)
        plt.close()

    # For impedance plots, default color limits: use provided vmin/vmax, else min/max of TRUE.
    vmin_imp = args.vmin
    vmax_imp = args.vmax
    if (vmin_imp is None) or (vmax_imp is None):
        vmin_auto = float(np.nanmin(true2))
        vmax_auto = float(np.nanmax(true2))
        if vmin_imp is None:
            vmin_imp = vmin_auto
        if vmax_imp is None:
            vmax_imp = vmax_auto

    vmin_err = float(np.nanmin(err2))
    vmax_err = float(np.nanmax(err2))

    _plot_one(
        "true.png",
        true2,
        f"True Impedance Section{args.title_suffix}",
        f"Impedance [{vmin_imp:.3g}, {vmax_imp:.3g}]",
        vmin_imp,
        vmax_imp,
    )
    _plot_one(
        "pred.png",
        pred2,
        f"Predicted Impedance Section{args.title_suffix}",
        f"Impedance [{vmin_imp:.3g}, {vmax_imp:.3g}]",
        vmin_imp,
        vmax_imp,
    )
    _plot_one(
        "error.png",
        err2,
        f"Error Section (Pred - True){args.title_suffix}",
        f"Error [{vmin_err:.3g}, {vmax_err:.3g}]",
        None,
        None,
    )

if __name__ == "__main__":
    main()
