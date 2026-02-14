from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from benchmark.data import FracAtlasSegDataset
from benchmark.metrics import batch_confusion, dice_from_confusion, iou_from_confusion
from benchmark.model import SimpleUNet


@dataclasses.dataclass
class RuntimeContext:
    device: torch.device
    amp: bool


def resolve_device(device_hint: str) -> torch.device:
    if device_hint == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(config: dict[str, Any]) -> nn.Module:
    model_cfg = config.get("model", {})
    if model_cfg.get("name", "simple_unet") != "simple_unet":
        raise ValueError("Only model.name=simple_unet is supported in this baseline.")
    return SimpleUNet(
        in_channels=int(model_cfg.get("in_channels", 3)),
        out_channels=int(model_cfg.get("out_channels", 1)),
    )


def build_dataloaders(config: dict[str, Any], repo_root: pathlib.Path) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_cfg = config["dataset"]
    training_cfg = config["training"]
    image_size = int(config["input"]["image_size"])
    manifest_path = (repo_root / dataset_cfg["manifest"]).resolve()

    train_ds = FracAtlasSegDataset(manifest_path=manifest_path, split="train", image_size=image_size, base_dir=repo_root)
    val_ds = FracAtlasSegDataset(manifest_path=manifest_path, split="val", image_size=image_size, base_dir=repo_root)
    test_ds = FracAtlasSegDataset(manifest_path=manifest_path, split="test", image_size=image_size, base_dir=repo_root)

    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(training_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
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


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    runtime: RuntimeContext,
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

        logits = model(images)
        loss = criterion(logits, masks)
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

    all_dice = []
    all_iou = []
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
            if max_batches is not None and step > max_batches:
                break

            images, masks = _to_device(batch, runtime.device)
            logits = model(images)
            loss = criterion(logits, masks)
            loss_sum += float(loss.item())
            count += 1

            intersection, pred_sum, target_sum = batch_confusion(logits, masks)
            dice = dice_from_confusion(intersection, pred_sum, target_sum)
            iou = iou_from_confusion(intersection, pred_sum, target_sum)
            all_dice.extend(dice.detach().cpu().tolist())
            all_iou.extend(iou.detach().cpu().tolist())

    return {
        "loss": loss_sum / max(count, 1),
        "dice": float(sum(all_dice) / max(len(all_dice), 1)),
        "iou": float(sum(all_iou) / max(len(all_iou), 1)),
    }

