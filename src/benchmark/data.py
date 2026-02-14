from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _resolve_path(base_dir: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _resize_with_aspect_and_pad(image: Image.Image, target_size: int, interpolation: InterpolationMode) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size.")

    scale = min(target_size / width, target_size / height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    image = TF.resize(image, [new_h, new_w], interpolation=interpolation)

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    return TF.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)


class FracAtlasSegDataset(Dataset):
    def __init__(
        self,
        manifest_path: pathlib.Path,
        split: str,
        image_size: int,
        base_dir: pathlib.Path | None = None,
        augment: bool = False,
        preserve_aspect: bool = False,
        patch_size: int | None = None,
        positive_patch_prob: float = 0.7,
        hard_negative_prob: float = 0.6,
        hard_negative_quantile: float = 0.90,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.base_dir = base_dir.resolve() if base_dir else self.manifest_path.parents[4]
        self.image_size = image_size
        self.augment = augment
        self.preserve_aspect = preserve_aspect
        self.patch_size = patch_size
        self.positive_patch_prob = positive_patch_prob
        self.hard_negative_prob = hard_negative_prob
        self.hard_negative_quantile = hard_negative_quantile

        df = pd.read_csv(self.manifest_path)
        df = df[df["split"] == split].copy()
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} in {self.manifest_path}")
        self.rows = df.to_dict(orient="records")
        self.fractured_labels = [int(row["fractured"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def _sample_random_center(self, width: int, height: int) -> tuple[int, int]:
        cx = int(torch.randint(low=0, high=width, size=(1,)).item())
        cy = int(torch.randint(low=0, high=height, size=(1,)).item())
        return cx, cy

    def _sample_hard_negative_center(self, image: Image.Image, mask: Image.Image) -> tuple[int, int] | None:
        gray = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        mask_arr = np.asarray(mask, dtype=np.uint8) > 0

        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        grad = gx + gy
        grad[mask_arr] = 0.0

        if not np.any(grad > 0):
            return None

        threshold = float(np.quantile(grad, self.hard_negative_quantile))
        candidates = np.argwhere(grad >= threshold)
        if len(candidates) == 0:
            candidates = np.argwhere(grad > 0)
        if len(candidates) == 0:
            return None

        idx = int(torch.randint(low=0, high=len(candidates), size=(1,)).item())
        cy, cx = candidates[idx]
        return int(cx), int(cy)

    def _sample_positive_center(self, mask: Image.Image) -> tuple[int, int] | None:
        mask_arr = np.asarray(mask, dtype=np.uint8) > 0
        ys, xs = np.where(mask_arr)
        if len(xs) == 0:
            return None
        idx = int(torch.randint(low=0, high=len(xs), size=(1,)).item())
        return int(xs[idx]), int(ys[idx])

    def _crop_around_center(self, image: Image.Image, mask: Image.Image, center: tuple[int, int], patch_size: int) -> tuple[Image.Image, Image.Image]:
        width, height = image.size
        patch_w = min(patch_size, width)
        patch_h = min(patch_size, height)
        cx, cy = center

        x0 = max(0, min(cx - patch_w // 2, width - patch_w))
        y0 = max(0, min(cy - patch_h // 2, height - patch_h))
        x1 = x0 + patch_w
        y1 = y0 + patch_h

        return image.crop((x0, y0, x1, y1)), mask.crop((x0, y0, x1, y1))

    def _apply_augment(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if not self.augment:
            return image, mask
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if torch.rand(1).item() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        if torch.rand(1).item() < 0.3:
            angle = float(torch.empty(1).uniform_(-12.0, 12.0).item())
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        return image, mask

    def _resize_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        if self.preserve_aspect:
            image = _resize_with_aspect_and_pad(image, self.image_size, interpolation=InterpolationMode.BILINEAR)
            mask = _resize_with_aspect_and_pad(mask, self.image_size, interpolation=InterpolationMode.NEAREST)
            return image, mask
        image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _maybe_patch_sample(self, image: Image.Image, mask: Image.Image, fractured: int) -> tuple[Image.Image, Image.Image]:
        if self.patch_size is None:
            return image, mask

        width, height = image.size
        center: tuple[int, int] | None = None

        if fractured == 1 and torch.rand(1).item() < self.positive_patch_prob:
            center = self._sample_positive_center(mask)

        if center is None and torch.rand(1).item() < self.hard_negative_prob:
            center = self._sample_hard_negative_center(image, mask)

        if center is None:
            center = self._sample_random_center(width, height)

        return self._crop_around_center(image, mask, center, self.patch_size)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image_path = _resolve_path(self.base_dir, row["image_path"])
        mask_path = _resolve_path(self.base_dir, row["mask_path"])
        fractured = int(row["fractured"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            with Image.open(mask_path) as mask:
                mask = mask.convert("L")

                image, mask = self._maybe_patch_sample(image, mask, fractured)
                image, mask = self._apply_augment(image, mask)
                image, mask = self._resize_pair(image, mask)

                image_tensor = TF.to_tensor(image)
                mask_tensor = TF.to_tensor(mask)
                mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }
