#!/usr/bin/env python
"""Run inference on user-provided seismic traces.

Input: .npy array of shape [n_traces, n_samples] (float32/float64)
Output: .npy predicted impedance (same shape), plus optional .png section preview.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1]))  # add project root
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from fcrsn_cw.data.scaler import load_scalers
from fcrsn_cw.models.fcrsn_cw import FCRSN_CW
from fcrsn_cw.utils.plotting import plot_section

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seismic_npy", type=str, required=True)
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default="checkpoints/best.pt")
    ap.add_argument("--out", type=str, default="pred_impedance.npy")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=32)

    ap.add_argument("--k_first", type=int, default=299)
    ap.add_argument("--k_last", type=int, default=3)
    ap.add_argument("--k_res1", type=int, default=299)
    ap.add_argument("--k_res2", type=int, default=3)
    ap.add_argument(
        "--last_relu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ReLU at output (paper uses ReLU). Use --no-last-relu to disable.",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    seismic = np.load(args.seismic_npy).astype(np.float32)
    assert seismic.ndim == 2
    seismic_scaler, imp_scaler = load_scalers(run_dir/"scalers.json")

    x = seismic_scaler.transform(seismic)
    x_t = torch.from_numpy(x).unsqueeze(1)  # (N,1,T)
    dl = DataLoader(TensorDataset(x_t), batch_size=args.batch_size, shuffle=False)

    model = FCRSN_CW(k_first=args.k_first, k_last=args.k_last, k_res1=args.k_res1, k_res2=args.k_res2, last_relu=args.last_relu).to(args.device)
    ckpt = torch.load(str(run_dir/args.ckpt), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds = []
    for (xb,) in dl:
        xb = xb.to(args.device)
        yb = model(xb).cpu().numpy()[:, 0, :]
        preds.append(yb)
    pred_scaled = np.concatenate(preds, axis=0)
    pred = imp_scaler.inverse_transform(pred_scaled)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, pred.astype(np.float32))
    # quick preview
    plot_section(out.with_suffix(".png"), pred[: min(512, pred.shape[0])], title="Predicted impedance (preview)")
    print(f"[OK] Saved {out} and {out.with_suffix('.png')}")

if __name__ == "__main__":
    main()
