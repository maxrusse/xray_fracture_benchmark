from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models

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
    parser = argparse.ArgumentParser(description="Final fusion: segmentation blend + classifier-presence blend.")
    parser.add_argument("--config-a", required=True, help="Primary segmentation/dual-head model config.")
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--config-b", required=True, help="Aux segmentation model config (for segmax presence score).")
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--classifier-arch", default="resnet34", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
    parser.add_argument("--classifier-image-size", type=int, default=384)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--tta", default="hv", help="none|h|v|hv")
    parser.add_argument("--alpha-seg-a", type=float, default=1.0, help="Segmentation blend: alpha*a + (1-alpha)*b")
    parser.add_argument("--beta-pres-a", type=float, default=0.3, help="Base presence blend: beta*cls_a + (1-beta)*segmax_b")
    parser.add_argument("--seg-threshold", type=float, default=0.3)
    parser.add_argument("--gamma-start", type=float, default=0.0)
    parser.add_argument("--gamma-stop", type=float, default=1.0)
    parser.add_argument("--gamma-step", type=float, default=0.05)
    parser.add_argument("--gamma-fixed", type=float, default=None, help="When set, skip search and evaluate only this gamma.")
    parser.add_argument("--presence-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="runs/final_presence_fusion/result.json")
    parser.add_argument("--select-by", default="auc_ap_sum", choices=["auc_ap_sum", "auc", "ap"])
    return parser.parse_args()


def _build_classifier(arch: str) -> torch.nn.Module:
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "resnet34":
        model = models.resnet34(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 1)
        return model
    raise ValueError(f"Unsupported classifier arch: {arch}")


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


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    thresholds = np.unique(y_score.astype(np.float64))
    if thresholds.size == 0:
        return 0.0, 0.5
    best_f1 = -1.0
    best_thr = float(thresholds[0])
    for thr in thresholds:
        pred = (y_score >= thr).astype(np.int64)
        tp = int(np.logical_and(pred == 1, y_true == 1).sum())
        fp = int(np.logical_and(pred == 1, y_true == 0).sum())
        fn = int(np.logical_and(pred == 0, y_true == 1).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return float(best_f1), float(best_thr)


def _grid_values(start: float, stop: float, step: float) -> list[float]:
    values: list[float] = []
    x = start
    while x <= stop + 1e-12:
        values.append(round(x, 6))
        x += step
    return values


def _evaluate_for_gamma(
    seg_prob_np: np.ndarray,
    gt_mask_np: np.ndarray,
    y_true_presence_np: np.ndarray,
    base_score_np: np.ndarray,
    clf_score_np: np.ndarray,
    seg_threshold: float,
    gamma: float,
    presence_threshold: float,
) -> dict[str, float]:
    pred = seg_prob_np >= float(seg_threshold)
    gt = gt_mask_np.astype(bool)

    flat_pred = pred.reshape(pred.shape[0], -1)
    flat_gt = gt.reshape(gt.shape[0], -1)

    inter = np.logical_and(flat_pred, flat_gt).sum(axis=1).astype(np.float64)
    pred_sum = flat_pred.sum(axis=1).astype(np.float64)
    gt_sum = flat_gt.sum(axis=1).astype(np.float64)
    dice = (2.0 * inter + 1e-6) / (pred_sum + gt_sum + 1e-6)
    iou = (inter + 1e-6) / (pred_sum + gt_sum - inter + 1e-6)

    gt_pos_mask = gt_sum > 0
    dice_pos = float(dice[gt_pos_mask].mean()) if gt_pos_mask.any() else 0.0
    iou_pos = float(iou[gt_pos_mask].mean()) if gt_pos_mask.any() else 0.0

    empty_mask = np.logical_not(gt_pos_mask)
    empty_correct_rate = float((pred_sum[empty_mask] == 0).mean()) if empty_mask.any() else float("nan")

    tp = float(np.logical_and(flat_pred, flat_gt).sum())
    fp = float(np.logical_and(flat_pred, np.logical_not(flat_gt)).sum())
    fn = float(np.logical_and(np.logical_not(flat_pred), flat_gt).sum())
    precision_pos = tp / (tp + fp + 1e-6)
    recall_pos = tp / (tp + fn + 1e-6)

    fused_score = float(gamma) * base_score_np + (1.0 - float(gamma)) * clf_score_np
    auc = _binary_roc_auc(y_true_presence_np, fused_score)
    ap = _average_precision(y_true_presence_np, fused_score)
    best_f1, best_f1_thr = _best_f1_threshold(y_true_presence_np, fused_score)

    pred_presence = (fused_score >= float(presence_threshold)).astype(np.int64)
    tp_presence = float(np.logical_and(pred_presence == 1, y_true_presence_np == 1).sum())
    fp_presence = float(np.logical_and(pred_presence == 1, y_true_presence_np == 0).sum())
    fn_presence = float(np.logical_and(pred_presence == 0, y_true_presence_np == 1).sum())
    precision_presence = tp_presence / (tp_presence + fp_presence + 1e-6)
    recall_presence = tp_presence / (tp_presence + fn_presence + 1e-6)

    return {
        "dice": float(dice.mean()),
        "iou": float(iou.mean()),
        "dice_pos": float(dice_pos),
        "iou_pos": float(iou_pos),
        "empty_correct_rate": float(empty_correct_rate),
        "precision_pos": float(precision_pos),
        "recall_pos": float(recall_pos),
        "num_samples": float(pred.shape[0]),
        "num_positive": float(gt_pos_mask.sum()),
        "num_negative": float(pred.shape[0] - gt_pos_mask.sum()),
        "roc_auc_presence": float(auc),
        "average_precision_presence": float(ap),
        "best_f1_presence": float(best_f1),
        "best_f1_threshold_presence": float(best_f1_thr),
        "precision_presence": float(precision_presence),
        "recall_presence": float(recall_presence),
        "presence_score_mode": "blend",
        "presence_score_source": "blend(base_dual_presence,classifier)",
        "presence_topk_frac": 0.01,
        "presence_threshold": float(presence_threshold),
        "gamma_base_vs_classifier": float(gamma),
    }


def main() -> int:
    args = parse_args()
    cfg_a = load_yaml((REPO_ROOT / args.config_a).resolve())
    cfg_b = load_yaml((REPO_ROOT / args.config_b).resolve())
    ckpt_a = (REPO_ROOT / args.checkpoint_a).resolve()
    ckpt_b = (REPO_ROOT / args.checkpoint_b).resolve()
    clf_ckpt = (REPO_ROOT / args.classifier_checkpoint).resolve()
    out_path = (REPO_ROOT / args.output).resolve()

    device = resolve_device(str(cfg_a.get("runtime", {}).get("device", args.device if hasattr(args, "device") else "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(cfg_a.get("runtime", {}).get("amp", False)))

    _, val_loader, test_loader = build_dataloaders(cfg_a, REPO_ROOT)
    loader = val_loader if str(args.split) == "val" else test_loader

    model_a = build_model(cfg_a).to(device)
    model_b = build_model(cfg_b).to(device)
    model_a.load_state_dict(torch.load(ckpt_a, map_location=device), strict=False)
    model_b.load_state_dict(torch.load(ckpt_b, map_location=device), strict=False)
    model_a.eval()
    model_b.eval()

    clf_model = _build_classifier(str(args.classifier_arch)).to(device)
    clf_model.load_state_dict(torch.load(clf_ckpt, map_location=device))
    clf_model.eval()

    seg_a_all: list[np.ndarray] = []
    seg_b_all: list[np.ndarray] = []
    gt_mask_all: list[np.ndarray] = []
    y_true_presence_all: list[np.ndarray] = []
    score_base_a_all: list[np.ndarray] = []
    score_base_b_all: list[np.ndarray] = []
    score_clf_all: list[np.ndarray] = []

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

            clf_in = F.interpolate(
                images,
                size=(int(args.classifier_image_size), int(args.classifier_image_size)),
                mode="bilinear",
                align_corners=False,
            )
            with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
                clf_logits = clf_model(clf_in).view(-1)

            score_a = (
                torch.sigmoid(cls_a).detach().cpu().numpy()
                if cls_a is not None
                else prob_a.view(prob_a.shape[0], -1).max(dim=1).values.detach().cpu().numpy()
            )
            score_b = prob_b.view(prob_b.shape[0], -1).max(dim=1).values.detach().cpu().numpy()
            score_c = torch.sigmoid(clf_logits).detach().cpu().numpy()
            y_true = (masks.view(masks.shape[0], -1).sum(dim=1) > 0).long().cpu().numpy()

            seg_a_all.append(prob_a.detach().cpu().numpy())
            seg_b_all.append(prob_b.detach().cpu().numpy())
            gt_mask_all.append((masks > 0.5).detach().cpu().numpy())
            y_true_presence_all.append(y_true)
            score_base_a_all.append(score_a)
            score_base_b_all.append(score_b)
            score_clf_all.append(score_c)

    seg_a_np = np.concatenate(seg_a_all, axis=0)
    seg_b_np = np.concatenate(seg_b_all, axis=0)
    gt_mask_np = np.concatenate(gt_mask_all, axis=0)
    y_true_presence_np = np.concatenate(y_true_presence_all, axis=0).astype(np.int64)
    score_a_np = np.concatenate(score_base_a_all, axis=0).astype(np.float64)
    score_b_np = np.concatenate(score_base_b_all, axis=0).astype(np.float64)
    score_c_np = np.concatenate(score_clf_all, axis=0).astype(np.float64)

    alpha = float(args.alpha_seg_a)
    beta = float(args.beta_pres_a)
    seg_blend = alpha * seg_a_np + (1.0 - alpha) * seg_b_np
    base_presence = beta * score_a_np + (1.0 - beta) * score_b_np

    gammas = [float(args.gamma_fixed)] if args.gamma_fixed is not None else _grid_values(float(args.gamma_start), float(args.gamma_stop), float(args.gamma_step))
    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        metrics = _evaluate_for_gamma(
            seg_prob_np=seg_blend,
            gt_mask_np=gt_mask_np,
            y_true_presence_np=y_true_presence_np,
            base_score_np=base_presence,
            clf_score_np=score_c_np,
            seg_threshold=float(args.seg_threshold),
            gamma=float(gamma),
            presence_threshold=float(args.presence_threshold),
        )
        metrics["gamma"] = float(gamma)
        rows.append(metrics)

    if str(args.select_by) == "auc":
        best = max(rows, key=lambda x: (x["roc_auc_presence"], x["average_precision_presence"]))
    elif str(args.select_by) == "ap":
        best = max(rows, key=lambda x: (x["average_precision_presence"], x["roc_auc_presence"]))
    else:
        best = max(rows, key=lambda x: (x["roc_auc_presence"] + x["average_precision_presence"], x["dice_pos"]))

    result = {
        "split": str(args.split),
        "tta_mode": str(args.tta),
        "seg_threshold": float(args.seg_threshold),
        "alpha_seg_a": float(args.alpha_seg_a),
        "beta_presence_a": float(args.beta_pres_a),
        "select_by": str(args.select_by),
        "num_rows": len(rows),
        "best": best,
        "rows": rows,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(out_path, result)
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
