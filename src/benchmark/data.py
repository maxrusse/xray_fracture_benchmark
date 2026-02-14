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
    ) -> None:
        self.manifest_path = manifest_path.resolve()
        self.base_dir = base_dir.resolve() if base_dir else self.manifest_path.parents[4]
        self.image_size = image_size

        df = pd.read_csv(self.manifest_path)
        df = df[df["split"] == split].copy()
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} in {self.manifest_path}")
        self.rows = df.to_dict(orient="records")

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
            mask_tensor = TF.to_tensor(mask)
            mask_tensor = (mask_tensor > 0.5).float()

        return {
            "image": image_tensor,
            "mask": mask_tensor,
        }
