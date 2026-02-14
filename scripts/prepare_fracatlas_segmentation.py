from __future__ import annotations

import argparse
import csv
import json
import pathlib
import random
from collections import Counter

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare FracAtlas segmentation masks and deterministic splits."
    )
    parser.add_argument("--raw-root", default="data/raw/fracatlas/FracAtlas")
    parser.add_argument("--processed-dir", default="data/processed/fracatlas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite-masks", action="store_true")
    return parser.parse_args()


def read_dataset_rows(raw_root: pathlib.Path) -> list[dict]:
    csv_path = raw_root / "dataset.csv"
    rows: list[dict] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_vgg_annotations(raw_root: pathlib.Path) -> dict:
    path = raw_root / "Annotations" / "VGG JSON" / "VGG_fracture_masks.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(images_root: pathlib.Path, image_id: str, fractured: int) -> pathlib.Path:
    preferred = [
        images_root / ("Fractured" if fractured == 1 else "Non_fractured") / image_id,
        images_root / ("Non_fractured" if fractured == 1 else "Fractured") / image_id,
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Image not found for {image_id}")


def render_mask(image_size: tuple[int, int], regions: dict | list) -> Image.Image:
    mask = Image.new("L", image_size, color=0)
    draw = ImageDraw.Draw(mask)

    if isinstance(regions, dict):
        region_items = regions.values()
    elif isinstance(regions, list):
        region_items = regions
    else:
        region_items = []

    for region in region_items:
        shape = region.get("shape_attributes", {})
        if shape.get("name") != "polygon":
            continue
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        if len(xs) < 3 or len(xs) != len(ys):
            continue
        polygon = [(int(round(x)), int(round(y))) for x, y in zip(xs, ys)]
        draw.polygon(polygon, outline=255, fill=255)

    return mask


def stratified_split(records: list[dict], seed: int, train_ratio: float, val_ratio: float) -> None:
    groups: dict[int, list[dict]] = {0: [], 1: []}
    for record in records:
        groups[int(record["fractured"])].append(record)

    rng = random.Random(seed)
    for label in (0, 1):
        label_records = groups[label]
        rng.shuffle(label_records)
        n = len(label_records)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)

        for i, record in enumerate(label_records):
            if i < n_train:
                record["split"] = "train"
            elif i < n_train + n_val:
                record["split"] = "val"
            else:
                record["split"] = "test"


def write_csv(path: pathlib.Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Invalid split ratios. Require: train_ratio > 0, val_ratio >= 0, and train+val < 1.")

    raw_root = pathlib.Path(args.raw_root)
    processed_dir = pathlib.Path(args.processed_dir)
    images_root = raw_root / "images"
    masks_dir = processed_dir / "masks"
    manifests_dir = processed_dir / "manifests"

    rows = read_dataset_rows(raw_root)
    vgg_annotations = read_vgg_annotations(raw_root)

    masks_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    missing_images: list[str] = []

    for row in rows:
        image_id = row["image_id"]
        fractured = int(row["fractured"])

        try:
            image_path = resolve_image_path(images_root, image_id, fractured)
        except FileNotFoundError:
            missing_images.append(image_id)
            continue

        mask_path = masks_dir / (pathlib.Path(image_id).stem + ".png")
        if args.overwrite_masks or not mask_path.exists():
            with Image.open(image_path) as image:
                width, height = image.size
            entry = vgg_annotations.get(image_id, {})
            regions = entry.get("regions", {})
            mask = render_mask((width, height), regions)
            mask.save(mask_path)

        records.append(
            {
                "image_id": image_id,
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "fractured": fractured,
                "split": "",
            }
        )

    if missing_images:
        preview = ", ".join(missing_images[:10])
        raise RuntimeError(f"{len(missing_images)} images missing from archive. Sample: {preview}")

    stratified_split(records, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
    records.sort(key=lambda r: r["image_id"])

    fields = ["image_id", "image_path", "mask_path", "fractured", "split"]
    write_csv(manifests_dir / "all.csv", records, fields)
    write_csv(manifests_dir / "train.csv", [r for r in records if r["split"] == "train"], fields)
    write_csv(manifests_dir / "val.csv", [r for r in records if r["split"] == "val"], fields)
    write_csv(manifests_dir / "test.csv", [r for r in records if r["split"] == "test"], fields)

    split_counts = Counter(r["split"] for r in records)
    class_counts = Counter(int(r["fractured"]) for r in records)

    print("Preparation complete.")
    print(f"records={len(records)}")
    print(f"split_counts={dict(split_counts)}")
    print(f"class_counts={dict(class_counts)}")
    print(f"manifests_dir={manifests_dir.as_posix()}")
    print(f"masks_dir={masks_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
