from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Any

import numpy as np
import torch

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
    parser = argparse.ArgumentParser(description="Evaluate image-level fracture classification from segmentation logits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--tta", default="none", help="TTA mode: none, h, v, hv.")
    parser.add_argument(
        "--score-mode",
        default="topk_mean",
        choices=["max", "mean", "topk_mean", "pred_area"],
        help="How to aggregate pixel probabilities into one per-image score.",
    )
    parser.add_argument("--topk-frac", type=float, default=0.01, help="Used for score-mode=topk_mean.")
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Probability threshold for score-mode=pred_area and for binary confusion summary.",
    )
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def _extract_logits(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        if "out" not in output:
            raise ValueError("Model output dict is missing 'out' key.")
        return output["out"]
    return output


def _forward_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    return _extract_logits(model(images))


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


def _predict_probs(model: torch.nn.Module, images: torch.Tensor, tta_mode: str) -> torch.Tensor:
    mode = _normalize_tta_mode(tta_mode)
    probs: list[torch.Tensor] = [torch.sigmoid(_forward_logits(model, images))]
    if "h" in mode:
        probs_h = torch.sigmoid(_forward_logits(model, torch.flip(images, dims=[3])))
        probs.append(torch.flip(probs_h, dims=[3]))
    if "v" in mode:
        probs_v = torch.sigmoid(_forward_logits(model, torch.flip(images, dims=[2])))
        probs.append(torch.flip(probs_v, dims=[2]))
    return torch.stack(probs, dim=0).mean(dim=0)


def _align_to_target(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.shape[-2:] != target.shape[-2:]:
        pred = torch.nn.functional.interpolate(pred, size=target.shape[-2:], mode="bilinear", align_corners=False)
    return pred


def _score_from_probs(
    probs: torch.Tensor, mode: str, topk_frac: float, mask_threshold: float
) -> np.ndarray:
    b = probs.shape[0]
    flat = probs.view(b, -1)
    if mode == "max":
        return flat.max(dim=1).values.detach().cpu().numpy()
    if mode == "mean":
        return flat.mean(dim=1).detach().cpu().numpy()
    if mode == "pred_area":
        return (flat >= mask_threshold).float().mean(dim=1).detach().cpu().numpy()

    frac = float(topk_frac)
    if not (0.0 < frac <= 1.0):
        raise ValueError("--topk-frac must be in (0,1]")
    k = max(1, int(round(flat.shape[1] * frac)))
    topk = flat.topk(k=k, dim=1).values
    return topk.mean(dim=1).detach().cpu().numpy()


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


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    y_true = y_true.astype(np.int64)
    thresholds = np.unique(y_score)
    best_f1 = -1.0
    best_thr = float(thresholds[0]) if len(thresholds) else 0.5
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


def main() -> int:
    args = parse_args()
    config = load_yaml((REPO_ROOT / args.config).resolve())
    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    output = (REPO_ROOT / args.output).resolve() if args.output else None

    device = resolve_device(str(config.get("runtime", {}).get("device", "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(config.get("runtime", {}).get("amp", False)))

    train_loader, val_loader, test_loader = build_dataloaders(config, REPO_ROOT)
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[str(args.split)]

    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    y_true_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []
    batch_counter = 0
    with torch.no_grad():
        for batch in loader:
            batch_counter += 1
            if args.max_batches is not None and batch_counter > int(args.max_batches):
                break

            images = batch["image"].to(runtime.device, non_blocking=True)
            masks = batch["mask"].to(runtime.device, non_blocking=True)
            use_amp = runtime.amp and runtime.device.type == "cuda"
            with torch.autocast(device_type=runtime.device.type, enabled=use_amp):
                probs = _predict_probs(model=model, images=images, tta_mode=str(args.tta))
                probs = _align_to_target(probs, masks)

            labels = (masks.view(masks.shape[0], -1).sum(dim=1) > 0).long().cpu().numpy()
            scores = _score_from_probs(
                probs=probs,
                mode=str(args.score_mode),
                topk_frac=float(args.topk_frac),
                mask_threshold=float(args.mask_threshold),
            )
            y_true_all.append(labels)
            y_score_all.append(scores)

    if not y_true_all:
        raise RuntimeError("No samples evaluated.")

    y_true = np.concatenate(y_true_all, axis=0)
    y_score = np.concatenate(y_score_all, axis=0)
    auc = _binary_roc_auc(y_true, y_score)
    ap = _average_precision(y_true, y_score)
    best_f1, best_thr = _best_f1_threshold(y_true, y_score)

    pred_default = (y_score >= float(args.mask_threshold)).astype(np.int64)
    tp = int(np.logical_and(pred_default == 1, y_true == 1).sum())
    fp = int(np.logical_and(pred_default == 1, y_true == 0).sum())
    fn = int(np.logical_and(pred_default == 0, y_true == 1).sum())
    tn = int(np.logical_and(pred_default == 0, y_true == 0).sum())

    result: dict[str, Any] = {
        "split": str(args.split),
        "num_samples": int(y_true.shape[0]),
        "num_positive": int((y_true == 1).sum()),
        "num_negative": int((y_true == 0).sum()),
        "prevalence": float((y_true == 1).mean()),
        "score_mode": str(args.score_mode),
        "topk_frac": float(args.topk_frac),
        "tta_mode": str(args.tta),
        "default_threshold": float(args.mask_threshold),
        "roc_auc": float(auc),
        "average_precision": float(ap),
        "best_f1": float(best_f1),
        "best_f1_threshold": float(best_thr),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        save_json(output, result)

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
