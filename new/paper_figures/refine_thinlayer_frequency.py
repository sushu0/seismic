from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from thinlayer_multifreq_utils import (
    DEFAULT_RESULT_DIRS,
    build_model,
    ensure_prediction_cache,
    evaluate_model,
    load_checkpoint_into_model,
    prepare_context,
    selection_score,
    to_plain_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine thin-layer inversion checkpoint selection with light fine-tuning.")
    parser.add_argument("--freq", required=True, choices=["20Hz", "30Hz"])
    parser.add_argument("--init-ckpt", required=True, help="Initial checkpoint to start from.")
    parser.add_argument("--output-dir", required=True, help="New output directory for the refined run.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--disable-augment", action="store_true", help="Disable training-set augmentation during fine-tuning.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    context = prepare_context(args.freq, output_dir, augment_train=not args.disable_augment)
    init_ckpt = Path(args.init_ckpt)
    if not init_ckpt.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {init_ckpt}")

    model = build_model(context)
    initial_payload = load_checkpoint_into_model(model, init_ckpt, context.device)
    baseline_val, baseline_test = evaluate_model(context, model)
    best_score = selection_score(baseline_val)
    best_payload = {
        "source": str(init_ckpt),
        "epoch": int(initial_payload.get("epoch", 0)),
        "val_metrics": baseline_val,
        "test_metrics": baseline_test,
        "selection_score": best_score,
    }

    shutil.copy2(init_ckpt, ckpt_dir / "init.pt")
    shutil.copy2(init_ckpt, ckpt_dir / "best.pt")
    torch.save({"epoch": int(initial_payload.get("epoch", 0)), "model": model.state_dict()}, ckpt_dir / "last.pt")

    criterion = context.module.CombinedLossV2(
        lambda_grad=float(context.module.CFG.LAMBDA_GRAD),
        lambda_sparse=float(context.module.CFG.LAMBDA_SPARSE),
        lambda_fwd=float(context.module.CFG.LAMBDA_FWD),
    ).to(context.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=max(args.lr * 0.1, 1e-6),
    )

    history: list[dict[str, object]] = []
    no_improve = 0

    print(f"Refining {args.freq}")
    print(f"  init ckpt : {init_ckpt}")
    print(f"  output dir: {output_dir}")
    print(f"  augment   : {'off' if args.disable_augment else 'on'}")
    print(f"  baseline  : score={best_score:.4f}, val_pcc={baseline_val['pcc']:.4f}, thin_pcc={baseline_val['thin_pcc']:.4f}, sep={baseline_val['separability_mean']:.4f}")

    for epoch in range(1, args.epochs + 1):
        train_loss, loss_comp = context.module.train_epoch(model, context.train_loader, criterion, optimizer, context.device)
        val_metrics, test_metrics = evaluate_model(context, model)
        scheduler.step()

        current_score = selection_score(val_metrics)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "loss_components": loss_comp,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "val_selection_score": current_score,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        improved = current_score > best_score + 1e-4
        if improved:
            best_score = current_score
            best_payload = {
                "source": "refined",
                "epoch": epoch,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "selection_score": current_score,
            }
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "selection_score": current_score,
                    "init_ckpt": str(init_ckpt),
                },
                ckpt_dir / "best.pt",
            )
            no_improve = 0
        else:
            no_improve += 1

        if epoch % args.save_every == 0 or epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "selection_score": current_score,
                    "init_ckpt": str(init_ckpt),
                },
                ckpt_dir / "last.pt",
            )

        print(
            f"Epoch {epoch:03d} | train={train_loss:.4f} | "
            f"val_pcc={val_metrics['pcc']:.4f} val_r2={val_metrics['r2']:.4f} | "
            f"thin_pcc={val_metrics['thin_pcc']:.4f} sep={val_metrics['separability_mean']:.4f} "
            f"dpde={val_metrics['dpde_mean']:.2f} | score={current_score:.4f}"
        )

        if no_improve >= args.patience:
            print(f"Early stop triggered after {epoch} epochs without score improvement.")
            break

    best_model = build_model(context)
    best_ckpt = load_checkpoint_into_model(best_model, ckpt_dir / "best.pt", context.device)
    final_val, final_test = evaluate_model(context, best_model)
    pred_full = ensure_prediction_cache(context, ckpt_dir / "best.pt", output_dir / "pred_full.npy")

    summary = {
        "freq": args.freq,
        "init_ckpt": str(init_ckpt),
        "default_reference_dir": str(DEFAULT_RESULT_DIRS[args.freq]),
        "output_dir": str(output_dir),
        "train_epochs_requested": args.epochs,
        "train_epochs_completed": len(history),
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "augment_train": not args.disable_augment,
        "baseline": {
            "val_metrics": baseline_val,
            "test_metrics": baseline_test,
            "selection_score": selection_score(baseline_val),
        },
        "selected_checkpoint": {
            "path": str(ckpt_dir / "best.pt"),
            "stored_epoch": int(best_ckpt.get("epoch", 0)),
            "selection_score": selection_score(final_val),
            "val_metrics": final_val,
            "test_metrics": final_test,
        },
        "best_payload": best_payload,
        "pred_full_path": str(output_dir / "pred_full.npy"),
    }

    with open(output_dir / "refine_history.json", "w", encoding="utf-8") as f:
        json.dump(to_plain_dict(history), f, indent=2)
    with open(output_dir / "refine_summary.json", "w", encoding="utf-8") as f:
        json.dump(to_plain_dict(summary), f, indent=2)
    with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(to_plain_dict(final_test), f, indent=2)

    print("Selected checkpoint summary:")
    print(
        f"  val score={selection_score(final_val):.4f}, "
        f"val_pcc={final_val['pcc']:.4f}, thin_pcc={final_val['thin_pcc']:.4f}, "
        f"sep={final_val['separability_mean']:.4f}, dpde={final_val['dpde_mean']:.2f}"
    )
    print(
        f"  test score={selection_score(final_test):.4f}, "
        f"test_pcc={final_test['pcc']:.4f}, thin_pcc={final_test['thin_pcc']:.4f}, "
        f"sep={final_test['separability_mean']:.4f}, dpde={final_test['dpde_mean']:.2f}"
    )
    print(f"  pred_full={pred_full.shape}")


if __name__ == "__main__":
    main()
