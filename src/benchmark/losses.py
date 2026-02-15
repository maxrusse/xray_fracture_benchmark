from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        dice_positive_only: bool = False,
        pos_weight: float | None = None,
    ) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = float(pos_weight) if pos_weight is not None else None
        self.dice = SoftDiceLoss(positive_only=dice_positive_only)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = (
            torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            if self.pos_weight is not None
            else None
        )
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
        return self.bce_weight * bce + self.dice_weight * self.dice(logits, targets)


class FocalBCELoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None, pos_weight: float | None = None) -> None:
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha) if alpha is not None else None
        self.pos_weight = float(pos_weight) if pos_weight is not None else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pos_weight = (
            torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
            if self.pos_weight is not None
            else None
        )
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).clamp(min=0.0).pow(self.gamma)
        loss = focal_factor * bce
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


class FocalTverskyLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.0,
        smooth: float = 1e-6,
        positive_only: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.smooth = float(smooth)
        self.positive_only = bool(positive_only)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        tp = (probs * targets).sum(dim=dims)
        fp = (probs * (1.0 - targets)).sum(dim=dims)
        fn = ((1.0 - probs) * targets).sum(dim=dims)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = (1.0 - tversky).pow(self.gamma)
        if self.positive_only:
            positive_mask = targets.sum(dim=dims) > 0
            if positive_mask.any():
                loss = loss[positive_mask]
        return loss.mean()


class FocalDiceTverskyLoss(nn.Module):
    def __init__(
        self,
        focal_weight: float = 0.35,
        dice_weight: float = 0.35,
        tversky_weight: float = 0.30,
        focal_gamma: float = 2.0,
        focal_alpha: float | None = None,
        pos_weight: float | None = None,
        dice_positive_only: bool = True,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        tversky_gamma: float = 1.0,
    ) -> None:
        super().__init__()
        self.focal_weight = float(focal_weight)
        self.dice_weight = float(dice_weight)
        self.tversky_weight = float(tversky_weight)
        self.focal = FocalBCELoss(gamma=focal_gamma, alpha=focal_alpha, pos_weight=pos_weight)
        self.dice = SoftDiceLoss(positive_only=dice_positive_only)
        self.tversky = FocalTverskyLoss(
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            positive_only=dice_positive_only,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        loss += self.focal_weight * self.focal(logits, targets)
        loss += self.dice_weight * self.dice(logits, targets)
        loss += self.tversky_weight * self.tversky(logits, targets)
        return loss


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
            pos_weight=float(loss_cfg["pos_weight"]) if "pos_weight" in loss_cfg else None,
        )
    if name == "focal_dice_tversky":
        return FocalDiceTverskyLoss(
            focal_weight=float(loss_cfg.get("focal_weight", 0.35)),
            dice_weight=float(loss_cfg.get("dice_weight", 0.35)),
            tversky_weight=float(loss_cfg.get("tversky_weight", 0.30)),
            focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
            focal_alpha=float(loss_cfg["focal_alpha"]) if "focal_alpha" in loss_cfg else None,
            pos_weight=float(loss_cfg["pos_weight"]) if "pos_weight" in loss_cfg else None,
            dice_positive_only=bool(loss_cfg.get("dice_positive_only", True)),
            tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
            tversky_beta=float(loss_cfg.get("tversky_beta", 0.7)),
            tversky_gamma=float(loss_cfg.get("tversky_gamma", 1.0)),
        )
    raise ValueError(f"Unsupported loss name: {name}")
