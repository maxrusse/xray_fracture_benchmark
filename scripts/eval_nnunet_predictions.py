from __future__ import annotations

import argparse
import pathlib
from typing import Any

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net predictions with fracture-focused metrics.")
    parser.add_argument("--pred-dir", required=True, help="Directory with predicted masks (*.png).")
    parser.add_argument("--gt-dir", required=True, help="Directory with GT masks (*.png).")
    parser.add_argument("--prefix", default="test_", help="Only evaluate files starting with this prefix.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    return parser.parse_args()


def _read_bin(path: pathlib.Path) -> np.ndarray:
    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.uint8)
    return arr > 0


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_sum = float(pred.sum())
    gt_sum = float(gt.sum())
    inter = float(np.logical_and(pred, gt).sum())
    return (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    i = 0
    n = len(order)
    while i < n:
        j = i
        score_i = y_score[order[i]]
        while j + 1 < n and y_score[order[j + 1]] == score_i:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    sum_pos = ranks[y_true == 1].sum()
    auc = (sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(tp + fp, 1)
    ap = precision[y_sorted == 1].sum() / n_pos
    return float(ap)


def main() -> int:
    args = parse_args()
    pred_dir = pathlib.Path(args.pred_dir).resolve()
    gt_dir = pathlib.Path(args.gt_dir).resolve()

    gt_files = sorted([p for p in gt_dir.glob("*.png") if p.name.startswith(args.prefix)])
    if not gt_files:
        raise FileNotFoundError(f"No GT files found in {gt_dir} with prefix '{args.prefix}'")

    dice_all: list[float] = []
    dice_pos: list[float] = []
    cls_true: list[int] = []
    cls_score: list[float] = []
    tp = fp = fn = 0.0
    missing: list[str] = []
    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            missing.append(gt_path.name)
            continue

        gt = _read_bin(gt_path)
        pred = _read_bin(pred_path)
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch for {gt_path.name}: gt={gt.shape}, pred={pred.shape}")

        d = _dice(pred, gt)
        dice_all.append(d)
        if gt.any():
            dice_pos.append(d)
        cls_true.append(1 if gt.any() else 0)
        cls_score.append(float(pred.mean()))

        tp += float(np.logical_and(pred, gt).sum())
        fp += float(np.logical_and(pred, np.logical_not(gt)).sum())
        fn += float(np.logical_and(np.logical_not(pred), gt).sum())

    if not dice_all:
        raise RuntimeError("No comparable prediction/GT pairs found.")

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    y_true = np.asarray(cls_true, dtype=np.int64)
    y_score = np.asarray(cls_score, dtype=np.float64)
    cls_pred = (y_score > 0.0).astype(np.int64)
    tp_cls = int(np.logical_and(cls_pred == 1, y_true == 1).sum())
    fp_cls = int(np.logical_and(cls_pred == 1, y_true == 0).sum())
    fn_cls = int(np.logical_and(cls_pred == 0, y_true == 1).sum())
    tn_cls = int(np.logical_and(cls_pred == 0, y_true == 0).sum())
    result: dict[str, Any] = {
        "num_cases": len(dice_all),
        "num_positive": len(dice_pos),
        "num_negative": int((y_true == 0).sum()),
        "num_missing_predictions": len(missing),
        "missing_predictions": missing[:20],
        "dice": float(np.mean(dice_all)),
        "dice_pos": float(np.mean(dice_pos)) if dice_pos else 0.0,
        "precision_pos": float(precision),
        "recall_pos": float(recall),
        "roc_auc_presence": float(_binary_roc_auc(y_true, y_score)),
        "average_precision_presence": float(_average_precision(y_true, y_score)),
        "presence_score": "predicted_positive_area_fraction",
        "presence_threshold": 0.0,
        "tp_presence": tp_cls,
        "fp_presence": fp_cls,
        "fn_presence": fn_cls,
        "tn_presence": tn_cls,
        "prefix": args.prefix,
    }

    if args.output:
        out = pathlib.Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(__import__("json").dumps(result, indent=2), encoding="utf-8")

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
