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
        augmentation_cfg: dict[str, Any] | None = None,
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
        self.augmentation_cfg = augmentation_cfg or {}

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

    def _scaled_aug_prob(self, base_prob: float, fractured: int) -> float:
        if fractured != 1:
            return float(base_prob)
        mult = float(self.augmentation_cfg.get("positive_aug_multiplier", 1.0))
        return float(min(1.0, max(0.0, base_prob * mult)))

    def _apply_augment(self, image: Image.Image, mask: Image.Image, fractured: int) -> tuple[Image.Image, Image.Image]:
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

        affine_prob = self._scaled_aug_prob(float(self.augmentation_cfg.get("affine_prob", 0.0)), fractured)
        if torch.rand(1).item() < affine_prob:
            max_deg = float(self.augmentation_cfg.get("affine_max_degrees", 10.0))
            translate_frac = float(self.augmentation_cfg.get("affine_translate_frac", 0.06))
            scale_min = float(self.augmentation_cfg.get("affine_scale_min", 0.9))
            scale_max = float(self.augmentation_cfg.get("affine_scale_max", 1.1))
            width, height = image.size
            angle = float(torch.empty(1).uniform_(-max_deg, max_deg).item())
            tx = int(round(float(torch.empty(1).uniform_(-translate_frac, translate_frac).item()) * width))
            ty = int(round(float(torch.empty(1).uniform_(-translate_frac, translate_frac).item()) * height))
            scale = float(torch.empty(1).uniform_(scale_min, max(scale_min, scale_max)).item())
            image = TF.affine(
                image,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=[tx, ty],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.NEAREST,
                fill=0,
            )

        jitter_prob = self._scaled_aug_prob(float(self.augmentation_cfg.get("jitter_prob", 0.0)), fractured)
        if torch.rand(1).item() < jitter_prob:
            brightness = float(self.augmentation_cfg.get("brightness", 0.0))
            contrast = float(self.augmentation_cfg.get("contrast", 0.0))
            if brightness > 0:
                factor = float(torch.empty(1).uniform_(1.0 - brightness, 1.0 + brightness).item())
                image = TF.adjust_brightness(image, factor)
            if contrast > 0:
                factor = float(torch.empty(1).uniform_(1.0 - contrast, 1.0 + contrast).item())
                image = TF.adjust_contrast(image, factor)

        gamma_prob = self._scaled_aug_prob(float(self.augmentation_cfg.get("gamma_prob", 0.0)), fractured)
        if torch.rand(1).item() < gamma_prob:
            gamma_min = max(0.05, float(self.augmentation_cfg.get("gamma_min", 0.8)))
            gamma_max = max(gamma_min, float(self.augmentation_cfg.get("gamma_max", 1.2)))
            gamma = float(torch.empty(1).uniform_(gamma_min, gamma_max).item())
            image = TF.adjust_gamma(image, gamma=gamma)

        autocontrast_prob = self._scaled_aug_prob(float(self.augmentation_cfg.get("autocontrast_prob", 0.0)), fractured)
        if torch.rand(1).item() < autocontrast_prob:
            image = TF.autocontrast(image)

        blur_prob = self._scaled_aug_prob(float(self.augmentation_cfg.get("blur_prob", 0.0)), fractured)
        if torch.rand(1).item() < blur_prob:
            kernel = int(self.augmentation_cfg.get("blur_kernel", 5))
            if kernel < 3:
                kernel = 3
            if kernel % 2 == 0:
                kernel += 1
            sigma_min = max(0.05, float(self.augmentation_cfg.get("blur_sigma_min", 0.1)))
            sigma_max = max(sigma_min, float(self.augmentation_cfg.get("blur_sigma_max", 1.5)))
            sigma = float(torch.empty(1).uniform_(sigma_min, sigma_max).item())
            image = TF.gaussian_blur(image, kernel_size=kernel, sigma=[sigma, sigma])
        return image, mask

    def _apply_tensor_augment(self, image_tensor: torch.Tensor, fractured: int) -> torch.Tensor:
        if not self.augment:
            return image_tensor

        noise_prob = self._scaled_aug_prob(float(self.augmentation_cfg.get("noise_prob", 0.0)), fractured)
        if torch.rand(1).item() < noise_prob:
            std_min = max(0.0, float(self.augmentation_cfg.get("noise_std_min", 0.0)))
            std_max = max(std_min, float(self.augmentation_cfg.get("noise_std_max", 0.05)))
            noise_std = float(torch.empty(1).uniform_(std_min, std_max).item())
            image_tensor = image_tensor + torch.randn_like(image_tensor) * noise_std

        cutout_prob = self._scaled_aug_prob(float(self.augmentation_cfg.get("cutout_prob", 0.0)), fractured)
        if torch.rand(1).item() < cutout_prob:
            frac_min = max(0.01, float(self.augmentation_cfg.get("cutout_frac_min", 0.08)))
            frac_max = max(frac_min, float(self.augmentation_cfg.get("cutout_frac_max", 0.20)))
            frac = float(torch.empty(1).uniform_(frac_min, frac_max).item())
            _, h, w = image_tensor.shape
            cut_h = max(1, min(h, int(round(h * frac))))
            cut_w = max(1, min(w, int(round(w * frac))))
            y0 = int(torch.randint(low=0, high=max(1, h - cut_h + 1), size=(1,)).item())
            x0 = int(torch.randint(low=0, high=max(1, w - cut_w + 1), size=(1,)).item())
            image_tensor[:, y0 : y0 + cut_h, x0 : x0 + cut_w] = 0.0

        return image_tensor.clamp_(0.0, 1.0)

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
                image, mask = self._apply_augment(image, mask, fractured=fractured)
                image, mask = self._resize_pair(image, mask)

                image_tensor = TF.to_tensor(image)
                image_tensor = self._apply_tensor_augment(image_tensor, fractured=fractured)
                mask_tensor = TF.to_tensor(mask)
                mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }
