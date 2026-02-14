from __future__ import annotations

import argparse
import pathlib
import platform
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.engine import (  # noqa: E402
    RuntimeContext,
    build_criterion,
    build_dataloaders,
    build_model,
    resolve_device,
    run_eval_epoch,
    run_train_epoch,
)
from benchmark.utils import load_yaml, save_json, save_yaml, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline segmentation model on FracAtlas.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--output-dir", default="runs/baseline")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = (REPO_ROOT / args.config).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)

    device = resolve_device(str(config.get("runtime", {}).get("device", "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(config.get("runtime", {}).get("amp", False)))

    train_loader, val_loader, _ = build_dataloaders(config, REPO_ROOT)
    model = build_model(config).to(device)

    training_cfg = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 3e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1e-4)),
    )
    criterion = build_criterion(config)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(runtime.amp and device.type == "cuda"))

    epochs = int(training_cfg.get("epochs", 20))
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(training_cfg.get("min_lr", 1e-6)),
        )
    max_train_batches = training_cfg.get("max_train_batches")
    max_eval_batches = training_cfg.get("max_eval_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None

    history = []
    selection_metric = str(training_cfg.get("selection_metric", "dice_pos"))
    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_loss = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            runtime=runtime,
            scaler=scaler,
            max_batches=max_train_batches,
        )
        val_metrics = run_eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            runtime=runtime,
            max_batches=max_eval_batches,
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(row)

        current_score = float(val_metrics.get(selection_metric, val_metrics.get("dice", 0.0)))
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        if scheduler is not None:
            scheduler.step()

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    save_yaml(output_dir / "resolved_config.yaml", config)
    save_json(
        output_dir / "environment.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    )
    save_json(
        output_dir / "metrics.json",
        {
            "selection_metric": selection_metric,
            "best_epoch": best_epoch,
            "best_val_score": best_score,
            "history": history,
        },
    )
    print(f"saved outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
