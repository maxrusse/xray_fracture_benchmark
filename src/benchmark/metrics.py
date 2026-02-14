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

