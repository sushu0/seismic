from __future__ import annotations
import argparse
from pathlib import Path
import json
import csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--out_csv", type=str, default="results/summary.csv")
    ap.add_argument("--exp_names", nargs="+", required=True, help="List of exp_name folders")
    args = ap.parse_args()

    rows = []
    for exp in args.exp_names:
        p = Path(args.results_root) / exp / "test_metrics.json"
        if not p.exists():
            print(f"[WARN] Missing {p}")
            continue
        obj = json.loads(p.read_text(encoding="utf-8"))
        m = obj.get("test_metrics", {})
        rows.append({
            "exp_name": exp,
            "best_epoch": obj.get("best_epoch"),
            "best_val_mse": obj.get("best_val_mse"),
            "test_MSE": m.get("MSE"),
            "test_PCC": m.get("PCC"),
            "test_R2": m.get("R2"),
        })

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OK] Wrote {out} with {len(rows)} rows")

if __name__ == "__main__":
    main()
