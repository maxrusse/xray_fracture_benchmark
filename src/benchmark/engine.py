from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from benchmark.data import FracAtlasSegDataset
from benchmark.losses import build_loss
from benchmark.metrics import summarize_segmentation_metrics
from benchmark.model import build_model_from_config


@dataclasses.dataclass
class RuntimeContext:
    device: torch.device
    amp: bool


def resolve_device(device_hint: str) -> torch.device:
    if device_hint == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(config: dict[str, Any]) -> nn.Module:
    return build_model_from_config(config.get("model", {}))


def build_dataloaders(config: dict[str, Any], repo_root: pathlib.Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    image_size = int(config["input"]["image_size"])
    manifest_path = (repo_root / dataset_cfg["manifest"]).resolve()

    train_ds = FracAtlasSegDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=image_size,
        base_dir=repo_root,
        augment=bool(training_cfg.get("augment", False)),
    )
    val_ds = FracAtlasSegDataset(manifest_path=manifest_path, split="val", image_size=image_size, base_dir=repo_root)
    test_ds = FracAtlasSegDataset(manifest_path=manifest_path, split="test", image_size=image_size, base_dir=repo_root)

    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 0))

    train_loader_kwargs = {
        "dataset": train_ds,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if bool(training_cfg.get("balanced_sampling", False)):
        labels = torch.tensor(train_ds.fractured_labels, dtype=torch.long)
        class_counts = torch.bincount(labels, minlength=2).float()
        class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader_kwargs["sampler"] = sampler
        train_loader_kwargs["shuffle"] = False
    else:
        train_loader_kwargs["shuffle"] = True

    train_loader = DataLoader(**train_loader_kwargs)
    eval_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    val_loader = DataLoader(val_ds, **eval_loader_kwargs)
    test_loader = DataLoader(test_ds, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"].to(device, non_blocking=True)
    masks = batch["mask"].to(device, non_blocking=True)
    return images, masks


def _extract_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        if "out" not in output:
            raise ValueError("Model output dict is missing 'out' key.")
        return output["out"]
    return output


def _align_logits_to_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def build_criterion(config: dict[str, Any]) -> nn.Module:
    return build_loss(config)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    runtime: RuntimeContext,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_batches: int | None = None,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        if max_batches is not None and step > max_batches:
            break

        images, masks = _to_device(batch, runtime.device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = runtime.amp and runtime.device.type == "cuda"
        with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
            output = model(images)
            logits = _extract_logits(output)
            logits = _align_logits_to_target(logits, masks)
            loss = criterion(logits, masks)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        loss_sum += float(loss.item())
        count += 1

    return loss_sum / max(count, 1)


def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    runtime: RuntimeContext,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    count = 0

    sum_dice = 0.0
    sum_iou = 0.0
    sum_dice_pos = 0.0
    sum_iou_pos = 0.0
    count_samples = 0.0
    count_pos = 0.0
    empty_correct = 0.0
    empty_total = 0.0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
            if max_batches is not None and step > max_batches:
                break

            images, masks = _to_device(batch, runtime.device)
            use_amp = runtime.amp and runtime.device.type == "cuda"
            with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
                output = model(images)
                logits = _extract_logits(output)
                logits = _align_logits_to_target(logits, masks)
                loss = criterion(logits, masks)
            loss_sum += float(loss.item())
            count += 1

            stats = summarize_segmentation_metrics(logits, masks)
            batch_total = float(stats["num_total"].item())
            batch_pos = float(stats["num_pos"].item())
            batch_neg = batch_total - batch_pos

            sum_dice += float(stats["dice"].item()) * batch_total
            sum_iou += float(stats["iou"].item()) * batch_total
            if batch_pos > 0:
                sum_dice_pos += float(stats["dice_pos"].item()) * batch_pos
                sum_iou_pos += float(stats["iou_pos"].item()) * batch_pos
            count_samples += batch_total
            count_pos += batch_pos

            empty_correct += float(stats["empty_correct"].item())
            empty_total += float(stats["empty_total"].item())
            tp += float(stats["tp"].item())
            fp += float(stats["fp"].item())
            fn += float(stats["fn"].item())

    precision_pos = tp / (tp + fp + 1e-6)
    recall_pos = tp / (tp + fn + 1e-6)
    return {
        "loss": loss_sum / max(count, 1),
        "dice": float(sum_dice / max(count_samples, 1.0)),
        "iou": float(sum_iou / max(count_samples, 1.0)),
        "dice_pos": float(sum_dice_pos / max(count_pos, 1.0)),
        "iou_pos": float(sum_iou_pos / max(count_pos, 1.0)),
        "empty_correct_rate": float(empty_correct / max(empty_total, 1.0)),
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "num_samples": float(count_samples),
        "num_positive": float(count_pos),
    }
