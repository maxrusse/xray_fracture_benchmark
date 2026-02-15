from __future__ import annotations

import argparse
import pathlib
import platform
import sys
from typing import Any

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
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def _build_optimizer(model: torch.nn.Module, training_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(training_cfg.get("optimizer", "adamw")).lower()
    lr = float(training_cfg.get("learning_rate", 3e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    if name == "adamw":
        betas_raw = training_cfg.get("betas", [0.9, 0.999])
        betas = (float(betas_raw[0]), float(betas_raw[1])) if isinstance(betas_raw, (list, tuple)) else (0.9, 0.999)
        eps = float(training_cfg.get("eps", 1e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(training_cfg.get("momentum", 0.9))
        nesterov = bool(training_cfg.get("nesterov", True))
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_cfg: dict[str, Any],
    epochs: int,
    steps_per_epoch: int,
) -> tuple[torch.optim.lr_scheduler.LRScheduler | None, bool]:
    scheduler_name = str(training_cfg.get("scheduler", "none")).lower()
    if scheduler_name == "none":
        return None, False
    if scheduler_name == "cosine":
        return (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(epochs, 1),
                eta_min=float(training_cfg.get("min_lr", 1e-6)),
            ),
            False,
        )
    if scheduler_name == "onecycle":
        max_lr = float(training_cfg.get("max_lr", training_cfg.get("learning_rate", 3e-4)))
        pct_start = float(training_cfg.get("pct_start", 0.15))
        div_factor = float(training_cfg.get("div_factor", 10.0))
        final_div_factor = float(training_cfg.get("final_div_factor", 1000.0))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=max(epochs, 1),
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=pct_start,
            anneal_strategy="cos",
            div_factor=div_factor,
            final_div_factor=final_div_factor,
        )
        return scheduler, True
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def _clone_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _clone_state_to_cpu(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _update_ema_state(model: torch.nn.Module, ema_state: dict[str, torch.Tensor], decay: float) -> None:
    with torch.no_grad():
        for name, value in model.state_dict().items():
            src = value.detach()
            if name not in ema_state:
                ema_state[name] = src.clone()
                continue
            if not torch.is_floating_point(src):
                ema_state[name].copy_(src)
                continue
            ema_state[name].mul_(decay).add_(src, alpha=1.0 - decay)


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
    if args.init_checkpoint:
        init_path = (REPO_ROOT / args.init_checkpoint).resolve()
        state = torch.load(init_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded init checkpoint: {init_path}")
        if missing:
            print(f"missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    training_cfg = config["training"]
    optimizer = _build_optimizer(model, training_cfg)
    criterion = build_criterion(config)
    loss_cfg = config.get("loss", {})
    presence_bce_weight = float(loss_cfg.get("presence_bce_weight", 0.0))
    presence_bce_warmup_epochs = int(loss_cfg.get("presence_bce_warmup_epochs", 0))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
    scaler = torch.amp.GradScaler(device.type, enabled=bool(runtime.amp and device.type == "cuda"))

    epochs = int(training_cfg.get("epochs", 20))
    max_train_batches = training_cfg.get("max_train_batches")
    max_eval_batches = training_cfg.get("max_eval_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    steps_per_epoch = len(train_loader)
    if max_train_batches is not None:
        steps_per_epoch = min(steps_per_epoch, max_train_batches)
    scheduler, scheduler_step_per_batch = _build_scheduler(optimizer, training_cfg, epochs, steps_per_epoch)

    ema_cfg = training_cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_state = _clone_model_state(model) if ema_enabled else None
    ema_model = build_model(config).to(device) if ema_enabled else None
    update_ema_fn = (
        (lambda src_model: _update_ema_state(src_model, ema_state, ema_decay)) if ema_enabled and ema_state is not None else None
    )

    eval_cfg = config.get("evaluation", {})
    eval_threshold = float(eval_cfg.get("threshold", 0.5))
    eval_tta_mode = str(eval_cfg.get("tta", "none"))
    eval_presence_score_mode = str(eval_cfg.get("presence_score_mode", "max"))
    eval_presence_topk_frac = float(eval_cfg.get("presence_topk_frac", 0.01))
    eval_presence_threshold = float(eval_cfg.get("presence_threshold", 0.5))

    history = []
    selection_metric = str(training_cfg.get("selection_metric", "dice_pos"))
    best_score = -1.0
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        effective_presence_bce_weight = float(presence_bce_weight)
        if presence_bce_warmup_epochs > 0 and presence_bce_weight > 0.0:
            ramp = min(float(epoch) / float(presence_bce_warmup_epochs), 1.0)
            effective_presence_bce_weight = float(presence_bce_weight * ramp)

        train_loss = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            runtime=runtime,
            scaler=scaler,
            scheduler=scheduler,
            scheduler_step_per_batch=scheduler_step_per_batch,
            max_batches=max_train_batches,
            presence_bce_weight=effective_presence_bce_weight,
            grad_clip_norm=grad_clip_norm if grad_clip_norm > 0 else None,
            update_ema_fn=update_ema_fn,
        )
        eval_model = model
        if ema_enabled and ema_state is not None and ema_model is not None:
            ema_model.load_state_dict(ema_state, strict=False)
            eval_model = ema_model

        val_metrics = run_eval_epoch(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            runtime=runtime,
            max_batches=max_eval_batches,
            threshold=eval_threshold,
            tta_mode=eval_tta_mode,
            presence_score_mode=eval_presence_score_mode,
            presence_topk_frac=eval_presence_topk_frac,
            presence_threshold=eval_presence_threshold,
        )
        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "presence_bce_weight": float(effective_presence_bce_weight),
            "eval_model": "ema" if ema_enabled else "online",
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(row)

        current_score = float(val_metrics.get(selection_metric, val_metrics.get("dice", 0.0)))
        if current_score > best_score:
            best_score = current_score
            best_epoch = epoch
            if ema_enabled and ema_state is not None:
                torch.save(_clone_state_to_cpu(ema_state), output_dir / "best_model.pt")
            else:
                torch.save(model.state_dict(), output_dir / "best_model.pt")

        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

    torch.save(model.state_dict(), output_dir / "last_model.pt")
    if ema_enabled and ema_state is not None:
        torch.save(_clone_state_to_cpu(ema_state), output_dir / "ema_model.pt")
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
