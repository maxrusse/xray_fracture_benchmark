from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    DeepLabV3_ResNet101_Weights,
    deeplabv3_resnet50,
    deeplabv3_resnet101,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_channels)
        self.enc2 = ConvBlock(base_channels, base_channels * 2)
        self.enc3 = ConvBlock(base_channels * 2, base_channels * 4)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bottleneck = ConvBlock(base_channels * 4, base_channels * 8)

        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_channels * 2, base_channels)

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        return e1, e2, e3, b

    def _decode(self, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return d1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, b = self._encode(x)
        d1 = self._decode(e1, e2, e3, b)
        return self.out_conv(d1)


class SimpleUNetDualHead(SimpleUNet):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        base_channels: int = 32,
        cls_hidden: int = 128,
        cls_dropout: float = 0.2,
    ) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, base_channels=base_channels)
        bottleneck_channels = base_channels * 8
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(bottleneck_channels, cls_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cls_dropout)),
            nn.Linear(cls_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1, e2, e3, b = self._encode(x)
        d1 = self._decode(e1, e2, e3, b)
        seg_logits = self.out_conv(d1)
        cls_logits = self.cls_head(torch.flatten(self.cls_pool(b), start_dim=1)).squeeze(1)
        return {"out": seg_logits, "cls": cls_logits}


class DeepLabV3DualHead(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet50",
        out_channels: int = 1,
        pretrained: bool = False,
        pretrained_backbone: bool = False,
        cls_hidden: int = 256,
        cls_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        backbone_name = str(backbone).lower().strip()
        if backbone_name == "resnet50":
            weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
            backbone_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            base = deeplabv3_resnet50(weights=weights, weights_backbone=backbone_weights, aux_loss=False)
        elif backbone_name == "resnet101":
            weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
            backbone_weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained_backbone else None
            base = deeplabv3_resnet101(weights=weights, weights_backbone=backbone_weights, aux_loss=False)
        else:
            raise ValueError(f"Unsupported DeepLab dual backbone: {backbone}")

        in_features = base.classifier[-1].in_channels
        base.classifier[-1] = nn.Conv2d(in_features, int(out_channels), kernel_size=1)

        self.backbone = base.backbone
        self.classifier = base.classifier
        self.cls_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(2048, int(cls_hidden)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cls_dropout)),
            nn.Linear(int(cls_hidden), 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        feat = features["out"]
        seg_logits = self.classifier(feat)
        seg_logits = F.interpolate(seg_logits, size=input_shape, mode="bilinear", align_corners=False)
        cls_logits = self.cls_head(torch.flatten(self.cls_pool(feat), start_dim=1)).squeeze(1)
        return {"out": seg_logits, "cls": cls_logits}


def build_model_from_config(model_cfg: dict[str, Any]) -> nn.Module:
    name = str(model_cfg.get("name", "simple_unet")).lower()
    if name == "simple_unet":
        return SimpleUNet(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 32)),
        )

    if name in {"simple_unet_dual_head", "simple_unet_dual"}:
        return SimpleUNetDualHead(
            in_channels=int(model_cfg.get("in_channels", 3)),
            out_channels=int(model_cfg.get("out_channels", 1)),
            base_channels=int(model_cfg.get("base_channels", 32)),
            cls_hidden=int(model_cfg.get("cls_hidden", 128)),
            cls_dropout=float(model_cfg.get("cls_dropout", 0.2)),
        )

    if name == "deeplabv3_resnet50":
        pretrained = bool(model_cfg.get("pretrained", False))
        pretrained_backbone = bool(model_cfg.get("pretrained_backbone", False))
        weights = DeepLabV3_ResNet50_Weights.DEFAULT if pretrained else None
        backbone_weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None

        model = deeplabv3_resnet50(
            weights=weights,
            weights_backbone=backbone_weights,
            aux_loss=False,
        )
        out_channels = int(model_cfg.get("out_channels", 1))
        in_features = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_features, out_channels, kernel_size=1)
        return model

    if name == "deeplabv3_resnet101":
        pretrained = bool(model_cfg.get("pretrained", False))
        pretrained_backbone = bool(model_cfg.get("pretrained_backbone", False))
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
        backbone_weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained_backbone else None

        model = deeplabv3_resnet101(
            weights=weights,
            weights_backbone=backbone_weights,
            aux_loss=False,
        )
        out_channels = int(model_cfg.get("out_channels", 1))
        in_features = model.classifier[-1].in_channels
        model.classifier[-1] = nn.Conv2d(in_features, out_channels, kernel_size=1)
        return model

    if name in {"deeplabv3_resnet50_dual_head", "deeplabv3_resnet50_dual"}:
        return DeepLabV3DualHead(
            backbone="resnet50",
            out_channels=int(model_cfg.get("out_channels", 1)),
            pretrained=bool(model_cfg.get("pretrained", False)),
            pretrained_backbone=bool(model_cfg.get("pretrained_backbone", False)),
            cls_hidden=int(model_cfg.get("cls_hidden", 256)),
            cls_dropout=float(model_cfg.get("cls_dropout", 0.2)),
        )

    if name in {"deeplabv3_resnet101_dual_head", "deeplabv3_resnet101_dual"}:
        return DeepLabV3DualHead(
            backbone="resnet101",
            out_channels=int(model_cfg.get("out_channels", 1)),
            pretrained=bool(model_cfg.get("pretrained", False)),
            pretrained_backbone=bool(model_cfg.get("pretrained_backbone", False)),
            cls_hidden=int(model_cfg.get("cls_hidden", 256)),
            cls_dropout=float(model_cfg.get("cls_dropout", 0.2)),
        )

    raise ValueError(f"Unsupported model.name: {name}")
