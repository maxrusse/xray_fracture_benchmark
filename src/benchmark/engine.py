from __future__ import annotations

import dataclasses
import pathlib
from typing import Any, Callable

import numpy as np
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
    input_cfg = config["input"]
    image_size = int(input_cfg["image_size"])
    preserve_aspect = bool(input_cfg.get("preserve_aspect", False))
    patch_cfg = training_cfg.get("patch", {})
    augmentation_cfg = training_cfg.get("augmentation", {})
    patch_enabled = bool(patch_cfg.get("enabled", False))
    patch_size = int(patch_cfg.get("size", image_size)) if patch_enabled else None
    manifest_path = (repo_root / dataset_cfg["manifest"]).resolve()

    train_ds = FracAtlasSegDataset(
        manifest_path=manifest_path,
        split="train",
        image_size=image_size,
        base_dir=repo_root,
        augment=bool(training_cfg.get("augment", False)),
        preserve_aspect=preserve_aspect,
        patch_size=patch_size,
        positive_patch_prob=float(patch_cfg.get("positive_prob", 0.7)),
        hard_negative_prob=float(patch_cfg.get("hard_negative_prob", 0.6)),
        hard_negative_quantile=float(patch_cfg.get("hard_negative_quantile", 0.90)),
        augmentation_cfg=augmentation_cfg,
    )
    val_ds = FracAtlasSegDataset(
        manifest_path=manifest_path,
        split="val",
        image_size=image_size,
        base_dir=repo_root,
        preserve_aspect=preserve_aspect,
    )
    test_ds = FracAtlasSegDataset(
        manifest_path=manifest_path,
        split="test",
        image_size=image_size,
        base_dir=repo_root,
        preserve_aspect=preserve_aspect,
    )

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


def _extract_cls_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor | None:
    if not isinstance(output, dict):
        return None
    if "cls" in output:
        cls = output["cls"]
    elif "classification" in output:
        cls = output["classification"]
    else:
        return None

    if cls.dim() == 2 and cls.shape[1] == 1:
        cls = cls[:, 0]
    elif cls.dim() != 1:
        raise ValueError(f"Unsupported cls logits shape: {tuple(cls.shape)}")
    return cls


def _forward_outputs(model: nn.Module, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    output = model(images)
    return _extract_logits(output), _extract_cls_logits(output)


def _forward_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    logits, _ = _forward_outputs(model, images)
    return logits


def _normalize_tta_mode(tta_mode: str) -> str:
    mode = str(tta_mode).lower().strip()
    if mode in {"", "none"}:
        return "none"
    valid = {"h", "v"}
    selected = "".join(ch for ch in mode if ch in valid)
    if not selected:
        return "none"
    if "h" in selected and "v" in selected:
        return "hv"
    return "h" if "h" in selected else "v"


def _predict_logits(model: nn.Module, images: torch.Tensor, tta_mode: str = "none") -> torch.Tensor:
    logits, _ = _predict_outputs(model, images, tta_mode=tta_mode)
    return logits


def _predict_outputs(
    model: nn.Module,
    images: torch.Tensor,
    tta_mode: str = "none",
) -> tuple[torch.Tensor, torch.Tensor | None]:
    mode = _normalize_tta_mode(tta_mode)
    if mode == "none":
        return _forward_outputs(model, images)

    logits_base, cls_base = _forward_outputs(model, images)
    probs: list[torch.Tensor] = [torch.nan_to_num(torch.sigmoid(logits_base), nan=0.5, posinf=1.0, neginf=0.0)]
    cls_probs: list[torch.Tensor] | None = (
        [torch.nan_to_num(torch.sigmoid(cls_base), nan=0.5, posinf=1.0, neginf=0.0)] if cls_base is not None else None
    )
    if "h" in mode:
        logits_h, cls_h = _forward_outputs(model, torch.flip(images, dims=[3]))
        probs.append(torch.nan_to_num(torch.flip(torch.sigmoid(logits_h), dims=[3]), nan=0.5, posinf=1.0, neginf=0.0))
        if cls_probs is not None and cls_h is not None:
            cls_probs.append(torch.nan_to_num(torch.sigmoid(cls_h), nan=0.5, posinf=1.0, neginf=0.0))
    if "v" in mode:
        logits_v, cls_v = _forward_outputs(model, torch.flip(images, dims=[2]))
        probs.append(torch.nan_to_num(torch.flip(torch.sigmoid(logits_v), dims=[2]), nan=0.5, posinf=1.0, neginf=0.0))
        if cls_probs is not None and cls_v is not None:
            cls_probs.append(torch.nan_to_num(torch.sigmoid(cls_v), nan=0.5, posinf=1.0, neginf=0.0))

    mean_prob = torch.stack(probs, dim=0).mean(dim=0).float()
    mean_prob = mean_prob.clamp(min=1e-4, max=1.0 - 1e-4)
    seg_logits = torch.logit(mean_prob)

    cls_logits: torch.Tensor | None = None
    if cls_probs:
        mean_cls_prob = torch.stack(cls_probs, dim=0).mean(dim=0).float().clamp(min=1e-4, max=1.0 - 1e-4)
        cls_logits = torch.logit(mean_cls_prob)

    return seg_logits, cls_logits


def _align_logits_to_target(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != targets.shape[-2:]:
        logits = F.interpolate(logits, size=targets.shape[-2:], mode="bilinear", align_corners=False)
    return logits


def _presence_score_from_probs(
    probs: torch.Tensor, mode: str, topk_frac: float, pred_threshold: float
) -> torch.Tensor:
    flat = probs.view(probs.shape[0], -1)
    score_mode = str(mode).lower().strip()
    if score_mode == "max":
        return flat.max(dim=1).values
    if score_mode == "mean":
        return flat.mean(dim=1)
    if score_mode == "pred_area":
        return (flat >= pred_threshold).float().mean(dim=1)
    if score_mode != "topk_mean":
        raise ValueError(f"Unsupported presence score mode: {mode}")
    if not (0.0 < topk_frac <= 1.0):
        raise ValueError("presence_topk_frac must be in (0,1].")
    k = max(1, int(round(flat.shape[1] * float(topk_frac))))
    return flat.topk(k=k, dim=1).values.mean(dim=1)


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    n = len(order)
    while i < n:
        j = i
        score_i = y_score[order[i]]
        while j + 1 < n and y_score[order[j + 1]] == score_i:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = precision[y_sorted == 1].sum() / n_pos
    return float(ap)


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(y_score.astype(np.float64))
    if thresholds.size == 0:
        return 0.0, 0.5
    best_f1 = -1.0
    best_thr = float(thresholds[0])
    for thr in thresholds:
        pred = (y_score >= thr).astype(np.int64)
        tp = int(np.logical_and(pred == 1, y_true == 1).sum())
        fp = int(np.logical_and(pred == 1, y_true == 0).sum())
        fn = int(np.logical_and(pred == 0, y_true == 1).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return float(best_f1), float(best_thr)


def build_criterion(config: dict[str, Any]) -> nn.Module:
    return build_loss(config)


def run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    runtime: RuntimeContext,
    scaler: torch.cuda.amp.GradScaler | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_step_per_batch: bool = False,
    max_batches: int | None = None,
    presence_bce_weight: float = 0.0,
    grad_clip_norm: float | None = None,
    update_ema_fn: Callable[[nn.Module], None] | None = None,
) -> float:
    model.train()
    loss_sum = 0.0
    count = 0
    presence_bce = nn.BCEWithLogitsLoss() if float(presence_bce_weight) > 0.0 else None

    for step, batch in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        if max_batches is not None and step > max_batches:
            break

        images, masks = _to_device(batch, runtime.device)
        optimizer.zero_grad(set_to_none=True)

        use_amp = runtime.amp and runtime.device.type == "cuda"
        with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
            output = model(images)
            logits = _extract_logits(output)
            cls_logits = _extract_cls_logits(output)
            logits = _align_logits_to_target(logits, masks)
        loss = criterion(logits.float(), masks.float())
        if torch.isnan(loss):
            continue
        if presence_bce is not None:
            presence_logits = cls_logits if cls_logits is not None else logits.amax(dim=(2, 3)).squeeze(1)
            presence_targets = (masks.view(masks.shape[0], -1).sum(dim=1) > 0).float()
            presence_loss = presence_bce(presence_logits.float(), presence_targets.float())
            if not torch.isnan(presence_loss):
                loss = loss + float(presence_bce_weight) * presence_loss
        stepped = False
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            stepped = scale_after >= scale_before
        else:
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip_norm))
            optimizer.step()
            stepped = True
        if scheduler is not None and scheduler_step_per_batch and stepped:
            scheduler.step()
        if update_ema_fn is not None and stepped:
            update_ema_fn(model)

        loss_sum += float(loss.item())
        count += 1

    return loss_sum / max(count, 1)


def run_eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    runtime: RuntimeContext,
    max_batches: int | None = None,
    threshold: float = 0.5,
    tta_mode: str = "none",
    presence_score_mode: str = "max",
    presence_topk_frac: float = 0.01,
    presence_threshold: float = 0.5,
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
    y_true_presence: list[np.ndarray] = []
    y_score_presence: list[np.ndarray] = []
    presence_source = "segmentation"
    with torch.no_grad():
        for step, batch in enumerate(tqdm(loader, desc="eval", leave=False), start=1):
            if max_batches is not None and step > max_batches:
                break

            images, masks = _to_device(batch, runtime.device)
            use_amp = runtime.amp and runtime.device.type == "cuda"
            with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
                logits, cls_logits = _predict_outputs(model, images, tta_mode=tta_mode)
                logits = _align_logits_to_target(logits, masks)
            loss = criterion(logits.float(), masks.float())
            if not torch.isnan(loss):
                loss_sum += float(loss.item())
                count += 1

            stats = summarize_segmentation_metrics(logits, masks, threshold=threshold)
            batch_total = float(stats["num_total"].item())
            batch_pos = float(stats["num_pos"].item())

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

            labels = (masks.view(masks.shape[0], -1).sum(dim=1) > 0).long()
            mode = str(presence_score_mode).lower().strip()
            if cls_logits is not None and mode in {"cls", "auto"}:
                scores = torch.sigmoid(cls_logits)
                presence_source = "cls"
            else:
                if mode == "cls":
                    mode = "max"
                probs = torch.sigmoid(logits)
                scores = _presence_score_from_probs(
                    probs=probs,
                    mode=mode,
                    topk_frac=presence_topk_frac,
                    pred_threshold=threshold,
                )
                presence_source = "segmentation"
            y_true_presence.append(labels.detach().cpu().numpy())
            y_score_presence.append(scores.detach().cpu().numpy())

    precision_pos = tp / (tp + fp + 1e-6)
    recall_pos = tp / (tp + fn + 1e-6)
    y_true_np = np.concatenate(y_true_presence, axis=0) if y_true_presence else np.array([], dtype=np.int64)
    y_score_np = np.concatenate(y_score_presence, axis=0) if y_score_presence else np.array([], dtype=np.float64)
    roc_auc_presence = _binary_roc_auc(y_true_np, y_score_np) if y_true_np.size else float("nan")
    average_precision_presence = _average_precision(y_true_np, y_score_np) if y_true_np.size else float("nan")
    best_f1_presence, best_f1_threshold_presence = _best_f1_threshold(y_true_np, y_score_np) if y_true_np.size else (float("nan"), float("nan"))

    pred_presence = (y_score_np >= float(presence_threshold)).astype(np.int64) if y_true_np.size else np.array([], dtype=np.int64)
    tp_presence = float(np.logical_and(pred_presence == 1, y_true_np == 1).sum()) if y_true_np.size else 0.0
    fp_presence = float(np.logical_and(pred_presence == 1, y_true_np == 0).sum()) if y_true_np.size else 0.0
    fn_presence = float(np.logical_and(pred_presence == 0, y_true_np == 1).sum()) if y_true_np.size else 0.0
    precision_presence = tp_presence / (tp_presence + fp_presence + 1e-6)
    recall_presence = tp_presence / (tp_presence + fn_presence + 1e-6)
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
        "num_negative": float(count_samples - count_pos),
        "roc_auc_presence": float(roc_auc_presence),
        "average_precision_presence": float(average_precision_presence),
        "best_f1_presence": float(best_f1_presence),
        "best_f1_threshold_presence": float(best_f1_threshold_presence),
        "precision_presence": float(precision_presence),
        "recall_presence": float(recall_presence),
        "presence_score_mode": str(presence_score_mode),
        "presence_score_source": str(presence_source),
        "presence_topk_frac": float(presence_topk_frac),
        "presence_threshold": float(presence_threshold),
    }
