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
    parser = argparse.ArgumentParser(description="Blend two trained models and search segmentation/presence tradeoffs.")
    parser.add_argument("--config-a", required=True)
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--config-b", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--tta", default="hv", help="none|h|v|hv")
    parser.add_argument("--threshold-start", type=float, default=0.20)
    parser.add_argument("--threshold-stop", type=float, default=0.70)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--alpha-step", type=float, default=0.1, help="Seg blend step for model A prob.")
    parser.add_argument("--beta-step", type=float, default=0.1, help="Presence blend step for model A cls score.")
    parser.add_argument("--alpha-fixed", type=float, default=None, help="If set, evaluate only this segmentation blend alpha.")
    parser.add_argument("--beta-fixed", type=float, default=None, help="If set, evaluate only this presence blend beta.")
    parser.add_argument("--threshold-fixed", type=float, default=None, help="If set, evaluate only this threshold.")
    parser.add_argument("--output", default="runs/blend_dual_presence_search.json")
    return parser.parse_args()


def _extract_seg_and_cls(output: torch.Tensor | dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, dict):
        if "out" not in output:
            raise ValueError("Model output dict is missing 'out' key.")
        seg = output["out"]
        cls: torch.Tensor | None = None
        if "cls" in output:
            cls = output["cls"]
        elif "classification" in output:
            cls = output["classification"]
        if cls is not None and cls.dim() == 2 and cls.shape[1] == 1:
            cls = cls[:, 0]
        return seg, cls
    return output, None


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


def _predict_probs_and_cls(model: torch.nn.Module, images: torch.Tensor, tta_mode: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    mode = _normalize_tta_mode(tta_mode)
    output = model(images)
    seg_logits, cls_logits = _extract_seg_and_cls(output)
    seg_probs: list[torch.Tensor] = [torch.nan_to_num(torch.sigmoid(seg_logits), nan=0.5, posinf=1.0, neginf=0.0)]
    cls_probs: list[torch.Tensor] | None = (
        [torch.nan_to_num(torch.sigmoid(cls_logits), nan=0.5, posinf=1.0, neginf=0.0)] if cls_logits is not None else None
    )
    if mode in {"h", "hv"}:
        out_h = model(torch.flip(images, dims=[3]))
        seg_h, cls_h = _extract_seg_and_cls(out_h)
        seg_probs.append(torch.nan_to_num(torch.flip(torch.sigmoid(seg_h), dims=[3]), nan=0.5, posinf=1.0, neginf=0.0))
        if cls_probs is not None and cls_h is not None:
            cls_probs.append(torch.nan_to_num(torch.sigmoid(cls_h), nan=0.5, posinf=1.0, neginf=0.0))
    if mode in {"v", "hv"}:
        out_v = model(torch.flip(images, dims=[2]))
        seg_v, cls_v = _extract_seg_and_cls(out_v)
        seg_probs.append(torch.nan_to_num(torch.flip(torch.sigmoid(seg_v), dims=[2]), nan=0.5, posinf=1.0, neginf=0.0))
        if cls_probs is not None and cls_v is not None:
            cls_probs.append(torch.nan_to_num(torch.sigmoid(cls_v), nan=0.5, posinf=1.0, neginf=0.0))

    seg_prob = torch.stack(seg_probs, dim=0).mean(dim=0)
    cls_prob = torch.stack(cls_probs, dim=0).mean(dim=0) if cls_probs else None
    return seg_prob, cls_prob


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
        s = y_score[order[i]]
        while j + 1 < n and y_score[order[j + 1]] == s:
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
    vals: list[float] = []
    x = start
    while x <= stop + 1e-12:
        vals.append(round(x, 6))
        x += step
    return vals


def main() -> int:
    args = parse_args()
    cfg_a = load_yaml((REPO_ROOT / args.config_a).resolve())
    cfg_b = load_yaml((REPO_ROOT / args.config_b).resolve())
    ckpt_a = (REPO_ROOT / args.checkpoint_a).resolve()
    ckpt_b = (REPO_ROOT / args.checkpoint_b).resolve()
    out_path = (REPO_ROOT / args.output).resolve()

    device = resolve_device(str(cfg_a.get("runtime", {}).get("device", "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(cfg_a.get("runtime", {}).get("amp", False)))

    train_loader, val_loader, test_loader = build_dataloaders(cfg_a, REPO_ROOT)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[str(args.split)]

    model_a = build_model(cfg_a).to(device)
    model_b = build_model(cfg_b).to(device)
    model_a.load_state_dict(torch.load(ckpt_a, map_location=device), strict=False)
    model_b.load_state_dict(torch.load(ckpt_b, map_location=device), strict=False)
    model_a.eval()
    model_b.eval()

    seg_a_all: list[np.ndarray] = []
    seg_b_all: list[np.ndarray] = []
    y_true_mask_all: list[np.ndarray] = []
    y_true_presence_all: list[np.ndarray] = []
    score_a_all: list[np.ndarray] = []
    score_b_all: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(runtime.device, non_blocking=True)
            masks = batch["mask"].to(runtime.device, non_blocking=True)
            use_amp = runtime.amp and runtime.device.type == "cuda"
            with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
                prob_a, cls_a = _predict_probs_and_cls(model_a, images, tta_mode=str(args.tta))
                prob_b, _ = _predict_probs_and_cls(model_b, images, tta_mode=str(args.tta))

            if prob_a.shape[-2:] != masks.shape[-2:]:
                prob_a = F.interpolate(prob_a, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            if prob_b.shape[-2:] != masks.shape[-2:]:
                prob_b = F.interpolate(prob_b, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            presence_true = (masks.view(masks.shape[0], -1).sum(dim=1) > 0).long().cpu().numpy()
            score_a = torch.sigmoid(cls_a).detach().cpu().numpy() if cls_a is not None else prob_a.view(prob_a.shape[0], -1).max(dim=1).values.detach().cpu().numpy()
            score_b = prob_b.view(prob_b.shape[0], -1).max(dim=1).values.detach().cpu().numpy()

            seg_a_all.append(prob_a.detach().cpu().numpy())
            seg_b_all.append(prob_b.detach().cpu().numpy())
            y_true_mask_all.append((masks > 0.5).detach().cpu().numpy())
            y_true_presence_all.append(presence_true)
            score_a_all.append(score_a)
            score_b_all.append(score_b)

    seg_a_np = np.concatenate(seg_a_all, axis=0)
    seg_b_np = np.concatenate(seg_b_all, axis=0)
    y_true_mask_np = np.concatenate(y_true_mask_all, axis=0).astype(bool)
    y_true_presence_np = np.concatenate(y_true_presence_all, axis=0).astype(np.int64)
    score_a_np = np.concatenate(score_a_all, axis=0).astype(np.float64)
    score_b_np = np.concatenate(score_b_all, axis=0).astype(np.float64)

    thresholds = (
        [float(args.threshold_fixed)]
        if args.threshold_fixed is not None
        else _grid_values(float(args.threshold_start), float(args.threshold_stop), float(args.threshold_step))
    )
    alphas = [float(args.alpha_fixed)] if args.alpha_fixed is not None else _grid_values(0.0, 1.0, float(args.alpha_step))
    betas = [float(args.beta_fixed)] if args.beta_fixed is not None else _grid_values(0.0, 1.0, float(args.beta_step))

    rows: list[dict[str, Any]] = []
    for alpha in alphas:
        seg_blend = alpha * seg_a_np + (1.0 - alpha) * seg_b_np
        pred_flat_prob = seg_blend.reshape(seg_blend.shape[0], -1)
        gt_flat = y_true_mask_np.reshape(y_true_mask_np.shape[0], -1)
        gt_pos_mask = gt_flat.sum(axis=1) > 0
        for thr in thresholds:
            pred = pred_flat_prob >= thr
            inter = np.logical_and(pred, gt_flat).sum(axis=1).astype(np.float64)
            pred_sum = pred.sum(axis=1).astype(np.float64)
            gt_sum = gt_flat.sum(axis=1).astype(np.float64)
            dice = (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
            dice_all = float(dice.mean())
            dice_pos = float(dice[gt_pos_mask].mean()) if gt_pos_mask.any() else 0.0

            tp = np.logical_and(pred, gt_flat).sum()
            fp = np.logical_and(pred, np.logical_not(gt_flat)).sum()
            fn = np.logical_and(np.logical_not(pred), gt_flat).sum()
            precision_pos = float(tp / (tp + fp + 1e-6))
            recall_pos = float(tp / (tp + fn + 1e-6))

            for beta in betas:
                score_blend = beta * score_a_np + (1.0 - beta) * score_b_np
                rows.append(
                    {
                        "alpha_seg_a": float(alpha),
                        "threshold": float(thr),
                        "beta_presence_a": float(beta),
                        "dice": dice_all,
                        "dice_pos": dice_pos,
                        "precision_pos": precision_pos,
                        "recall_pos": recall_pos,
                        "roc_auc_presence": float(_binary_roc_auc(y_true_presence_np, score_blend)),
                        "average_precision_presence": float(_average_precision(y_true_presence_np, score_blend)),
                    }
                )

    best_auc = max(rows, key=lambda x: (x["roc_auc_presence"], x["average_precision_presence"]))
    best_dice_pos = max(rows, key=lambda x: (x["dice_pos"], x["roc_auc_presence"]))
    best_joint = max(rows, key=lambda x: (x["roc_auc_presence"] + x["dice_pos"], x["average_precision_presence"]))

    payload = {
        "split": str(args.split),
        "tta": str(args.tta),
        "num_rows": len(rows),
        "search_space": {
            "alphas": alphas,
            "thresholds": thresholds,
            "betas": betas,
        },
        "best_auc": best_auc,
        "best_dice_pos": best_dice_pos,
        "best_joint_auc_plus_dice_pos": best_joint,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
