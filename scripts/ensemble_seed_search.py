from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.engine import (  # noqa: E402
    RuntimeContext,
    build_dataloaders,
    build_model,
    resolve_device,
)
from benchmark.utils import load_yaml, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ensemble multiple seed runs and search threshold on one split.")
    parser.add_argument("--run-dirs", nargs="+", required=True, help="Run dirs containing resolved_config.yaml and best_model.pt")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--tta", default="hv", help="none|h|v|hv")
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--threshold-fixed", type=float, default=None)
    parser.add_argument("--selection-metric", default="dice_pos")
    parser.add_argument("--output", default="runs/ensemble_seed_search.json")
    return parser.parse_args()


def _normalize_tta_mode(tta_mode: str) -> str:
    mode = str(tta_mode).lower().strip()
    if mode in {"", "none"}:
        return "none"
    selected = "".join(ch for ch in mode if ch in {"h", "v"})
    if not selected:
        return "none"
    if "h" in selected and "v" in selected:
        return "hv"
    return "h" if "h" in selected else "v"


def _extract_seg_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        if "out" not in output:
            raise ValueError("Model output dict is missing 'out' key.")
        return output["out"]
    return output


def _predict_prob(model: torch.nn.Module, images: torch.Tensor, tta_mode: str) -> torch.Tensor:
    mode = _normalize_tta_mode(tta_mode)
    seg_logits = _extract_seg_logits(model(images))
    probs: list[torch.Tensor] = [torch.nan_to_num(torch.sigmoid(seg_logits), nan=0.5, posinf=1.0, neginf=0.0)]
    if mode in {"h", "hv"}:
        seg_h = _extract_seg_logits(model(torch.flip(images, dims=[3])))
        probs.append(torch.nan_to_num(torch.flip(torch.sigmoid(seg_h), dims=[3]), nan=0.5, posinf=1.0, neginf=0.0))
    if mode in {"v", "hv"}:
        seg_v = _extract_seg_logits(model(torch.flip(images, dims=[2])))
        probs.append(torch.nan_to_num(torch.flip(torch.sigmoid(seg_v), dims=[2]), nan=0.5, posinf=1.0, neginf=0.0))
    return torch.stack(probs, dim=0).mean(dim=0)


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
        score = y_score[order[i]]
        while j + 1 < n and y_score[order[j + 1]] == score:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    sum_pos = ranks[y_true == 1].sum()
    return float((sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


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
    return float(precision[y_sorted == 1].sum() / n_pos)


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def _compute_metrics(prob_np: np.ndarray, gt_np: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob_np >= threshold
    gt = gt_np.astype(bool)

    pred_flat = pred.reshape(pred.shape[0], -1)
    gt_flat = gt.reshape(gt.shape[0], -1)
    inter = np.logical_and(pred_flat, gt_flat).sum(axis=1).astype(np.float64)
    pred_sum = pred_flat.sum(axis=1).astype(np.float64)
    gt_sum = gt_flat.sum(axis=1).astype(np.float64)
    dice = (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
    iou = (inter + 1e-6) / (pred_sum + gt_sum - inter + 1e-6)

    pos_mask = gt_sum > 0
    neg_mask = np.logical_not(pos_mask)
    pred_empty = pred_sum == 0

    tp = np.logical_and(pred_flat, gt_flat).sum()
    fp = np.logical_and(pred_flat, np.logical_not(gt_flat)).sum()
    fn = np.logical_and(np.logical_not(pred_flat), gt_flat).sum()

    y_true_presence = pos_mask.astype(np.int64)
    y_score_presence = prob_np.reshape(prob_np.shape[0], -1).max(axis=1)

    return {
        "dice": float(dice.mean()),
        "iou": float(iou.mean()),
        "dice_pos": float(dice[pos_mask].mean()) if pos_mask.any() else 0.0,
        "iou_pos": float(iou[pos_mask].mean()) if pos_mask.any() else 0.0,
        "empty_correct_rate": float((pred_empty & neg_mask).sum() / max(neg_mask.sum(), 1)),
        "precision_pos": float(tp / (tp + fp + 1e-6)),
        "recall_pos": float(tp / (tp + fn + 1e-6)),
        "num_samples": float(pred.shape[0]),
        "num_positive": float(pos_mask.sum()),
        "num_negative": float(neg_mask.sum()),
        "roc_auc_presence": float(_binary_roc_auc(y_true_presence, y_score_presence)),
        "average_precision_presence": float(_average_precision(y_true_presence, y_score_presence)),
    }


def main() -> int:
    args = parse_args()
    run_dirs = [(REPO_ROOT / r).resolve() for r in args.run_dirs]
    for run_dir in run_dirs:
        if not (run_dir / "resolved_config.yaml").exists():
            raise FileNotFoundError(f"Missing resolved_config.yaml in {run_dir}")
        if not (run_dir / "best_model.pt").exists():
            raise FileNotFoundError(f"Missing best_model.pt in {run_dir}")

    base_cfg = load_yaml(run_dirs[0] / "resolved_config.yaml")
    device = resolve_device(str(base_cfg.get("runtime", {}).get("device", "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(base_cfg.get("runtime", {}).get("amp", False)))

    train_loader, val_loader, test_loader = build_dataloaders(base_cfg, REPO_ROOT)
    loader = val_loader if str(args.split) == "val" else test_loader

    models: list[torch.nn.Module] = []
    for run_dir in run_dirs:
        cfg = load_yaml(run_dir / "resolved_config.yaml")
        model = build_model(cfg).to(device)
        model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device), strict=False)
        model.eval()
        models.append(model)

    prob_all: list[np.ndarray] = []
    gt_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(runtime.device, non_blocking=True)
            masks = batch["mask"].to(runtime.device, non_blocking=True)
            use_amp = runtime.amp and runtime.device.type == "cuda"
            probs_per_model: list[torch.Tensor] = []
            with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
                for model in models:
                    probs = _predict_prob(model, images, tta_mode=str(args.tta))
                    if probs.shape[-2:] != masks.shape[-2:]:
                        probs = F.interpolate(probs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    probs_per_model.append(probs)
            mean_prob = torch.stack(probs_per_model, dim=0).mean(dim=0)
            prob_all.append(mean_prob.detach().cpu().numpy())
            gt_all.append((masks > 0.5).detach().cpu().numpy())

    prob_np = np.concatenate(prob_all, axis=0)
    gt_np = np.concatenate(gt_all, axis=0)

    if args.threshold_fixed is not None:
        thresholds = [float(args.threshold_fixed)]
    else:
        thresholds = _grid_values(float(args.threshold_start), float(args.threshold_stop), float(args.threshold_step))

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        metrics = _compute_metrics(prob_np, gt_np, threshold=float(threshold))
        rows.append({"threshold": float(threshold), **metrics})

    metric_name = str(args.selection_metric)
    best_row = max(rows, key=lambda x: (float(x.get(metric_name, float("-inf"))), float(x["roc_auc_presence"])))

    payload = {
        "split": str(args.split),
        "tta": str(args.tta),
        "selection_metric": metric_name,
        "run_dirs": [str(x) for x in run_dirs],
        "num_models": len(models),
        "best": best_row,
        "rows": rows,
    }
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
