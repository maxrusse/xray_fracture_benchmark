from __future__ import annotations

import pathlib

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


class FracAtlasSegDataset(Dataset):
    def __init__(
        self,
        manifest_path: pathlib.Path,
        split: str,
        image_size: int,
        base_dir: pathlib.Path | None = None,
        augment: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.base_dir = base_dir.resolve() if base_dir else self.manifest_path.parents[4]
        self.image_size = image_size
        self.augment = augment

        df = pd.read_csv(self.manifest_path)
        df = df[df["split"] == split].copy()
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} in {self.manifest_path}")
        self.rows = df.to_dict(orient="records")
        self.fractured_labels = [int(row["fractured"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        image_path = _resolve_path(self.base_dir, row["image_path"])
        mask_path = _resolve_path(self.base_dir, row["mask_path"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=InterpolationMode.BILINEAR)
            image_tensor = TF.to_tensor(image)

        with Image.open(mask_path) as mask:
            mask = mask.convert("L")
            mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)

            if self.augment:
                if torch.rand(1).item() < 0.5:
                    image = TF.hflip(image)
                    mask = TF.hflip(mask)
                if torch.rand(1).item() < 0.5:
                    image = TF.vflip(image)
                    mask = TF.vflip(mask)
                if torch.rand(1).item() < 0.3:
                    angle = float(torch.empty(1).uniform_(-10.0, 10.0).item())
                    image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
                    mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

            image_tensor = TF.to_tensor(image)
            mask_tensor = TF.to_tensor(mask)
            mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }
