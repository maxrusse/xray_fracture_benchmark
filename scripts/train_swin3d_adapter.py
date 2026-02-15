from __future__ import annotations

import argparse
import copy
import json
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SWIN3D_SRC_DIR = REPO_ROOT.parent / "swin3d_dnp" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SWIN3D_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SWIN3D_SRC_DIR))

from benchmark.engine import (  # noqa: E402
    RuntimeContext,
    build_dataloaders,
    resolve_device,
    run_eval_epoch,
    run_train_epoch,
)
from benchmark.losses import build_loss  # noqa: E402
from benchmark.utils import load_yaml, save_json, set_seed  # noqa: E402
from swin3d_dnp.models import build_simple_swin3d_dnp  # noqa: E402


class Swin3DFracAtlasAdapter(nn.Module):
    """Adapt Swin3D-DNP simple model to 2D fracture segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        context_channels: int = 24,
        pseudo_depth: int = 4,
        depth_reduce: str = "max",
        pseudo_mode: str = "repeat",
        use_depth_bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.pseudo_depth = int(pseudo_depth)
        if self.pseudo_depth < 1:
            raise ValueError("pseudo_depth must be >= 1")
        self.depth_reduce = str(depth_reduce).lower()
        if self.depth_reduce not in {"max", "mean"}:
            raise ValueError("depth_reduce must be one of: max, mean")
        self.pseudo_mode = str(pseudo_mode).lower()
        if self.pseudo_mode not in {"repeat", "pyramid", "dnp_plus"}:
            raise ValueError("pseudo_mode must be one of: repeat, pyramid, dnp_plus")

        if bool(use_depth_bias):
            self.depth_bias = nn.Parameter(torch.zeros(1, self.in_channels, self.pseudo_depth, 1, 1))
        else:
            self.register_parameter("depth_bias", None)

        self.core = build_simple_swin3d_dnp(
            in_channels=self.in_channels,
            out_channels=out_channels,
            context_channels=context_channels,
        )

    def _reduce_depth(self, logits_3d: torch.Tensor) -> torch.Tensor:
        if self.depth_reduce == "max":
            return logits_3d.max(dim=2).values
        return logits_3d.mean(dim=2)

    def _smooth2d(self, x: torch.Tensor, kernel_size: int) -> torch.Tensor:
        if kernel_size <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def _build_pseudo_volume(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.pseudo_mode == "repeat":
            vol = x.unsqueeze(2).repeat(1, 1, self.pseudo_depth, 1, 1)
        elif self.pseudo_mode == "pyramid":
            bank = torch.stack(
                [
                    self._smooth2d(x, 9),
                    self._smooth2d(x, 7),
                    self._smooth2d(x, 5),
                    self._smooth2d(x, 3),
                    x,
                ],
                dim=2,
            )
            vol = F.interpolate(bank, size=(self.pseudo_depth, h, w), mode="trilinear", align_corners=False)
        else:
            smooth3 = self._smooth2d(x, 3)
            smooth7 = self._smooth2d(x, 7)
            detail = x - smooth3
            sharpen = x + detail
            bank = torch.stack([smooth7, smooth3, x, sharpen, detail], dim=2)
            vol = F.interpolate(bank, size=(self.pseudo_depth, h, w), mode="trilinear", align_corners=False)

        if self.depth_bias is not None:
            vol = vol + self.depth_bias
        return vol

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        if x.shape[1] != self.in_channels:
            if self.in_channels == 1:
                # Convert RGB-style inputs to a single grayscale channel.
                x = x.mean(dim=1, keepdim=True)
            elif x.shape[1] == 1 and self.in_channels > 1:
                x = x.repeat(1, self.in_channels, 1, 1)
            else:
                raise ValueError(
                    f"Input channels mismatch: got {x.shape[1]}, expected {self.in_channels}"
                )
        b, _, h, w = x.shape
        fine = self._build_pseudo_volume(x)

        coarse_h = max(16, h // 2)
        coarse_w = max(16, w // 2)
        coarse = F.interpolate(
            fine,
            size=(self.pseudo_depth, coarse_h, coarse_w),
            mode="trilinear",
            align_corners=False,
        )

        centers = torch.zeros((b, 3), dtype=fine.dtype, device=fine.device)
        spacing_fine = torch.tensor([[1.0, 1.0, 1.0]], dtype=fine.dtype, device=fine.device).repeat(b, 1)
        spacing_coarse = torch.tensor([[1.0, 2.0, 2.0]], dtype=fine.dtype, device=fine.device).repeat(b, 1)

        _, fine_logits_3d = self.core(
            image_coarse=coarse,
            image_fine=fine,
            centers_coarse_norm_dhw=centers,
            fine_shape=(self.pseudo_depth, h, w),
            spacing_fine_dhw_mm=spacing_fine,
            spacing_coarse_dhw_mm=spacing_coarse,
        )
        logits_2d = self._reduce_depth(fine_logits_3d)
        if logits_2d.shape[-2:] != (h, w):
            logits_2d = F.interpolate(logits_2d, size=(h, w), mode="bilinear", align_corners=False)
        return logits_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Swin3D adapter on FracAtlas.")
    parser.add_argument("--config", default="configs/swin3d_adapter_fast.yaml")
    parser.add_argument("--output-dir", default="runs/swin3d_adapter_fast")
    parser.add_argument("--init-checkpoint", default=None)
    return parser.parse_args()


def _threshold_grid(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    cur = float(start)
    while cur <= stop + 1e-12:
        values.append(round(cur, 6))
        cur += step
    return values


def _eval_with_params(
    model: nn.Module,
    loader: Any,
    criterion: nn.Module,
    runtime: RuntimeContext,
    threshold: float,
    tta_mode: str,
    max_batches: int | None = None,
) -> dict[str, float]:
    metrics = run_eval_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        runtime=runtime,
        max_batches=max_batches,
        threshold=threshold,
        tta_mode=tta_mode,
    )
    metrics["threshold"] = float(threshold)
    metrics["tta_mode"] = str(tta_mode)
    return metrics


def main() -> int:
    args = parse_args()
    config = load_yaml((REPO_ROOT / args.config).resolve())
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 42)))

    device = resolve_device(str(config.get("runtime", {}).get("device", "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(config.get("runtime", {}).get("amp", False)))
    use_amp = runtime.amp and runtime.device.type == "cuda"

    train_loader, val_loader, test_loader = build_dataloaders(config, REPO_ROOT)
    criterion = build_loss(config)

    model_cfg = config.get("model", {})
    model = Swin3DFracAtlasAdapter(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 1)),
        context_channels=int(model_cfg.get("context_channels", 24)),
        pseudo_depth=int(model_cfg.get("pseudo_depth", 4)),
        depth_reduce=str(model_cfg.get("depth_reduce", "max")),
        pseudo_mode=str(model_cfg.get("pseudo_mode", "repeat")),
        use_depth_bias=bool(model_cfg.get("use_depth_bias", True)),
    ).to(device)
    if args.init_checkpoint:
        init_path = (REPO_ROOT / args.init_checkpoint).resolve()
        state = torch.load(init_path, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"loaded init checkpoint: {init_path}")
        if missing:
            print(f"missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"unexpected keys ({len(unexpected)}): {unexpected[:8]}")

    tr_cfg = config.get("training", {})
    optimizer_name = str(tr_cfg.get("optimizer", "adamw")).lower()
    if optimizer_name not in {"adamw", "adam"}:
        raise ValueError("training.optimizer must be one of: adamw, adam")
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(tr_cfg.get("learning_rate", 1e-4)),
        weight_decay=float(tr_cfg.get("weight_decay", 1e-4)),
        betas=(
            float(tr_cfg.get("beta1", 0.9)),
            float(tr_cfg.get("beta2", 0.999)),
        ),
        eps=float(tr_cfg.get("eps", 1e-8)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    epochs = int(tr_cfg.get("epochs", 5))
    max_train_batches = tr_cfg.get("max_train_batches")
    max_train_batches = int(max_train_batches) if max_train_batches is not None else None
    max_eval_batches = tr_cfg.get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    selection_metric = str(tr_cfg.get("selection_metric", "dice_pos"))
    grad_clip_norm = tr_cfg.get("grad_clip_norm")
    grad_clip_norm = float(grad_clip_norm) if grad_clip_norm is not None else None

    scheduler_name = str(tr_cfg.get("scheduler", "none")).lower()
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    scheduler_step_per_batch = False
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(tr_cfg.get("min_lr", 1e-6)),
        )
    elif scheduler_name == "onecycle":
        steps_per_epoch = min(len(train_loader), max_train_batches) if max_train_batches is not None else len(train_loader)
        steps_per_epoch = max(1, int(steps_per_epoch))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(tr_cfg.get("learning_rate", 1e-4)),
            epochs=max(epochs, 1),
            steps_per_epoch=steps_per_epoch,
            pct_start=float(tr_cfg.get("onecycle_pct_start", 0.2)),
            anneal_strategy=str(tr_cfg.get("onecycle_anneal_strategy", "cos")),
            div_factor=float(tr_cfg.get("onecycle_div_factor", 25.0)),
            final_div_factor=float(tr_cfg.get("onecycle_final_div_factor", 1e4)),
        )
        scheduler_step_per_batch = True
    elif scheduler_name not in {"none", ""}:
        raise ValueError("training.scheduler must be one of: none, cosine, onecycle")

    loss_cfg = config.get("loss", {})
    presence_bce_weight = float(loss_cfg.get("presence_bce_weight", 0.0))
    presence_bce_warmup_epochs = int(loss_cfg.get("presence_bce_warmup_epochs", 0))

    ema_cfg = tr_cfg.get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema_warmup_steps = int(ema_cfg.get("warmup_steps", 0))
    ema_model: nn.Module | None = None
    if ema_enabled:
        ema_model = copy.deepcopy(model).to(device)
        ema_model.eval()
        for p in ema_model.parameters():
            p.requires_grad_(False)
    step_counter = 0

    def update_ema_fn(source_model: nn.Module) -> None:
        nonlocal step_counter
        if ema_model is None:
            return
        step_counter += 1
        decay = ema_decay if step_counter > ema_warmup_steps else 0.0
        with torch.no_grad():
            for ema_p, p in zip(ema_model.parameters(), source_model.parameters()):
                ema_p.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
            for ema_b, b in zip(ema_model.buffers(), source_model.buffers()):
                ema_b.data.copy_(b.data)

    eval_cfg = config.get("evaluation", {})
    eval_threshold = float(eval_cfg.get("threshold", 0.5))
    eval_tta = str(eval_cfg.get("tta", "none"))

    best_metric = float("-inf")
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
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
            grad_clip_norm=grad_clip_norm,
            update_ema_fn=update_ema_fn if ema_enabled else None,
        )
        eval_model = ema_model if ema_model is not None else model
        val_metrics = _eval_with_params(
            model=eval_model,
            loader=val_loader,
            criterion=criterion,
            runtime=runtime,
            threshold=eval_threshold,
            tta_mode=eval_tta,
            max_batches=max_eval_batches,
        )
        if scheduler is not None and not scheduler_step_per_batch:
            scheduler.step()

        rec = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "presence_bce_weight": float(effective_presence_bce_weight),
            "val_loss": float(val_metrics["loss"]),
            "val_dice": float(val_metrics["dice"]),
            "val_dice_pos": float(val_metrics["dice_pos"]),
            "val_iou_pos": float(val_metrics["iou_pos"]),
            "epoch_seconds": round(time.time() - t0, 2),
        }
        history.append(rec)
        print(rec)

        score = float(val_metrics.get(selection_metric, float("-inf")))
        if score > best_metric:
            best_metric = score
            torch.save(eval_model.state_dict(), output_dir / "best_model.pt")

    save_json(output_dir / "history.json", {"history": history, "selection_metric": selection_metric})
    save_json(output_dir / "train_config_resolved.json", config)

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))

    full_eval = bool(config.get("final_eval", {}).get("use_full_eval", True))
    final_max_batches = None if full_eval else max_eval_batches

    val_default = _eval_with_params(
        model=model,
        loader=val_loader,
        criterion=criterion,
        runtime=runtime,
        threshold=eval_threshold,
        tta_mode=eval_tta,
        max_batches=final_max_batches,
    )
    test_default = _eval_with_params(
        model=model,
        loader=test_loader,
        criterion=criterion,
        runtime=runtime,
        threshold=eval_threshold,
        tta_mode=eval_tta,
        max_batches=final_max_batches,
    )
    save_json(output_dir / "validate_metrics.json", val_default)
    save_json(output_dir / "test_metrics.json", test_default)

    tuning_cfg = config.get("threshold_tuning", {})
    tuned_payload: dict[str, Any] = {"enabled": False}
    if bool(tuning_cfg.get("enabled", True)):
        t_start = float(tuning_cfg.get("start", 0.1))
        t_stop = float(tuning_cfg.get("stop", 0.6))
        t_step = float(tuning_cfg.get("step", 0.05))
        tta_mode = str(tuning_cfg.get("tta", "none"))
        metric = str(tuning_cfg.get("metric", "dice_pos"))
        grid = _threshold_grid(t_start, t_stop, t_step)

        best_thr = grid[0]
        best_score = float("-inf")
        val_grid: list[dict[str, float]] = []
        for thr in grid:
            m = _eval_with_params(
                model=model,
                loader=val_loader,
                criterion=criterion,
                runtime=runtime,
                threshold=thr,
                tta_mode=tta_mode,
                max_batches=final_max_batches,
            )
            score = float(m.get(metric, float("-inf")))
            row = {"threshold": thr, "score": score, **m}
            val_grid.append(row)
            if score > best_score:
                best_score = score
                best_thr = thr

        val_tuned = _eval_with_params(
            model=model,
            loader=val_loader,
            criterion=criterion,
            runtime=runtime,
            threshold=best_thr,
            tta_mode=tta_mode,
            max_batches=final_max_batches,
        )
        test_tuned = _eval_with_params(
            model=model,
            loader=test_loader,
            criterion=criterion,
            runtime=runtime,
            threshold=best_thr,
            tta_mode=tta_mode,
            max_batches=final_max_batches,
        )
        save_json(output_dir / "threshold_tuning.json", {"metric": metric, "tta_mode": tta_mode, "grid": val_grid, "best_threshold": best_thr, "best_score": best_score})
        save_json(output_dir / "validate_metrics_tuned.json", val_tuned)
        save_json(output_dir / "test_metrics_tuned.json", test_tuned)
        tuned_payload = {
            "enabled": True,
            "metric": metric,
            "tta_mode": tta_mode,
            "best_threshold": best_thr,
            "best_score": best_score,
            "val": val_tuned,
            "test": test_tuned,
        }

    summary = {
        "output_dir": str(output_dir),
        "best_epoch_metric": best_metric,
        "selection_metric": selection_metric,
        "default_val_dice_pos": float(val_default.get("dice_pos", 0.0)),
        "default_test_dice_pos": float(test_default.get("dice_pos", 0.0)),
        "default_val_iou_pos": float(val_default.get("iou_pos", 0.0)),
        "default_test_iou_pos": float(test_default.get("iou_pos", 0.0)),
        "tuned": tuned_payload,
    }
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
