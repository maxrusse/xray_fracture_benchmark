from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6, positive_only: bool = False) -> None:
        super().__init__()
        self.smooth = smooth
        self.positive_only = positive_only

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        intersection = (probs * targets).sum(dim=dims)
        denom = probs.sum(dim=dims) + targets.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth)

        if self.positive_only:
            positive_mask = targets.sum(dim=dims) > 0
            if positive_mask.any():
                dice = dice[positive_mask]
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, dice_positive_only: bool = False) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = SoftDiceLoss(positive_only=dice_positive_only)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)


def build_loss(config: dict[str, Any]) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = str(loss_cfg.get("name", "bce_dice")).lower()

    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "dice":
        return SoftDiceLoss(positive_only=bool(loss_cfg.get("dice_positive_only", False)))
    if name == "bce_dice":
        return BCEDiceLoss(
            bce_weight=float(loss_cfg.get("bce_weight", 0.5)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.5)),
            dice_positive_only=bool(loss_cfg.get("dice_positive_only", True)),
        )
    raise ValueError(f"Unsupported loss name: {name}")

