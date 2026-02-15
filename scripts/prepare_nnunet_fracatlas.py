from __future__ import annotations

import argparse
import pathlib
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.data import _resolve_path  # noqa: E402
from benchmark.utils import save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare FracAtlas for nnU-Net v2 (2D natural image setup).")
    parser.add_argument("--manifest", default="data/processed/fracatlas/manifests/all.csv")
    parser.add_argument("--raw-root", default="data/nnunet/raw")
    parser.add_argument("--dataset-id", type=int, default=501)
    parser.add_argument("--dataset-name", default="FracAtlasXray")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files for this dataset.")
    return parser.parse_args()


def _sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", value)
    return cleaned or "Dataset"


def _to_case_id(split: str, image_id: str) -> str:
    stem = pathlib.Path(image_id).stem
    if split == "train":
        return stem
    return f"{split}_{stem}"


def _save_grayscale_png(src: pathlib.Path, dst: pathlib.Path) -> None:
    with Image.open(src) as image:
        gray = image.convert("L")
        gray.save(dst)


def _save_binary_mask(src: pathlib.Path, dst: pathlib.Path) -> None:
    with Image.open(src) as mask:
        arr = np.asarray(mask.convert("L"), dtype=np.uint8)
    bin_mask = (arr > 0).astype(np.uint8)
    Image.fromarray(bin_mask, mode="L").save(dst)


def main() -> int:
    args = parse_args()
    manifest_path = (REPO_ROOT / args.manifest).resolve()
    raw_root = (REPO_ROOT / args.raw_root).resolve()

    dataset_dir_name = f"Dataset{int(args.dataset_id):03d}_{_sanitize_name(args.dataset_name)}"
    dataset_dir = raw_root / dataset_dir_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_ts = dataset_dir / "imagesTs"
    labels_ts = dataset_dir / "labelsTs"

    if dataset_dir.exists() and not args.force:
        raise FileExistsError(f"{dataset_dir} already exists. Pass --force to overwrite.")

    for folder in [images_tr, labels_tr, images_ts, labels_ts]:
        folder.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest_path)
    required = {"image_id", "image_path", "mask_path", "split"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing columns: {sorted(missing_cols)}")

    summary = {"train": 0, "val": 0, "test": 0}
    mapping_rows: list[dict[str, str | int]] = []
    for row in df.to_dict(orient="records"):
        split = str(row["split"])
        if split not in summary:
            continue
        image_path = _resolve_path(REPO_ROOT, str(row["image_path"]))
        mask_path = _resolve_path(REPO_ROOT, str(row["mask_path"]))
        case_id = _to_case_id(split=split, image_id=str(row["image_id"]))

        if split == "train":
            out_image = images_tr / f"{case_id}_0000.png"
            out_mask = labels_tr / f"{case_id}.png"
        else:
            out_image = images_ts / f"{case_id}_0000.png"
            out_mask = labels_ts / f"{case_id}.png"

        _save_grayscale_png(image_path, out_image)
        _save_binary_mask(mask_path, out_mask)
        summary[split] += 1

        mapping_rows.append(
            {
                "case_id": case_id,
                "image_id": str(row["image_id"]),
                "split": split,
                "fractured": int(row.get("fractured", 0)),
            }
        )

    mapping_df = pd.DataFrame(mapping_rows).sort_values(["split", "case_id"]).reset_index(drop=True)
    mapping_df.to_csv(dataset_dir / "case_mapping.csv", index=False)

    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

    generate_dataset_json(
        output_folder=str(dataset_dir),
        channel_names={0: "xray"},
        labels={"background": 0, "fracture": 1},
        num_training_cases=int(summary["train"]),
        file_ending=".png",
        dataset_name=dataset_dir_name,
        overwrite_image_reader_writer="NaturalImage2DIO",
    )

    save_json(
        dataset_dir / "prep_summary.json",
        {
            "dataset_dir": str(dataset_dir),
            "dataset_id": int(args.dataset_id),
            "dataset_name": dataset_dir_name,
            "counts": summary,
        },
    )
    print(f"prepared {dataset_dir_name} at {dataset_dir}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
