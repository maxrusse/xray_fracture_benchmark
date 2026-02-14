from __future__ import annotations

import torch


def batch_confusion(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (preds * targets).sum(dim=1)
    pred_sum = preds.sum(dim=1)
    target_sum = targets.sum(dim=1)
    return intersection, pred_sum, target_sum


def dice_from_confusion(intersection: torch.Tensor, pred_sum: torch.Tensor, target_sum: torch.Tensor) -> torch.Tensor:
    return (2.0 * intersection + 1e-6) / (pred_sum + target_sum + 1e-6)


def iou_from_confusion(intersection: torch.Tensor, pred_sum: torch.Tensor, target_sum: torch.Tensor) -> torch.Tensor:
    union = pred_sum + target_sum - intersection
    return (intersection + 1e-6) / (union + 1e-6)


def summarize_segmentation_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
) -> dict[str, torch.Tensor]:
    intersection, pred_sum, target_sum = batch_confusion(logits, targets, threshold=threshold)
    dice = dice_from_confusion(intersection, pred_sum, target_sum)
    iou = iou_from_confusion(intersection, pred_sum, target_sum)

    positive_mask = target_sum > 0
    negative_mask = ~positive_mask
    pred_empty = pred_sum == 0

    preds = (torch.sigmoid(logits) >= threshold).float()
    preds = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)
    tp = (preds * targets_flat).sum()
    fp = (preds * (1.0 - targets_flat)).sum()
    fn = ((1.0 - preds) * targets_flat).sum()

    return {
        "dice": dice.mean(),
        "iou": iou.mean(),
        "dice_pos": dice[positive_mask].mean() if positive_mask.any() else torch.tensor(0.0, device=logits.device),
        "iou_pos": iou[positive_mask].mean() if positive_mask.any() else torch.tensor(0.0, device=logits.device),
        "empty_correct": (pred_empty & negative_mask).sum().float(),
        "empty_total": negative_mask.sum().float(),
        "num_pos": positive_mask.sum().float(),
        "num_total": torch.tensor(float(dice.numel()), device=logits.device),
        "tp": tp.float(),
        "fp": fp.float(),
        "fn": fn.float(),
    }
