from __future__ import annotations

import argparse
import pathlib
import random
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms
from tqdm import tqdm

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.utils import save_json  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image-level fracture presence classifier.")
    parser.add_argument("--manifest", default="data/processed/fracatlas/manifests/all.csv")
    parser.add_argument("--output-dir", default="runs/presence_classifier_final")
    parser.add_argument("--arch", default="resnet34", choices=["resnet18", "resnet34", "resnet50", "efficientnet_b0"])
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve_path(base_dir: pathlib.Path, value: str) -> pathlib.Path:
    path = pathlib.Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


class PresenceDataset(Dataset):
    def __init__(
        self,
        manifest_path: pathlib.Path,
        split: str,
        base_dir: pathlib.Path,
        transform: transforms.Compose,
    ) -> None:
        df = pd.read_csv(manifest_path)
        df = df[df["split"] == split].copy()
        if len(df) == 0:
            raise ValueError(f"No samples for split={split} in {manifest_path}")
        self.rows = df.to_dict(orient="records")
        self.base_dir = base_dir
        self.transform = transform
        self.labels = [int(r["fractured"]) for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        image_path = _resolve_path(self.base_dir, str(row["image_path"]))
        label = float(row["fractured"])
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            image_tensor = self.transform(img)
        return image_tensor, torch.tensor(label, dtype=torch.float32)


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


def _build_model(arch: str, pretrained: bool) -> nn.Module:
    def _instantiate_with_fallback(build_fn: Any, weight_enum: Any) -> nn.Module:
        if not pretrained:
            return build_fn(weights=None)
        try:
            return build_fn(weights=weight_enum)
        except Exception:
            return build_fn(weights=None)

    if arch == "resnet18":
        model = _instantiate_with_fallback(models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "resnet34":
        model = _instantiate_with_fallback(models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "resnet50":
        model = _instantiate_with_fallback(models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if arch == "efficientnet_b0":
        model = _instantiate_with_fallback(models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 1)
        return model
    raise ValueError(f"Unsupported arch: {arch}")


@dataclass
class EvalOutput:
    loss: float
    roc_auc: float
    average_precision: float
    best_f1: float
    best_f1_threshold: float
    num_samples: int
    num_positive: int
    num_negative: int


def _run_eval(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp: bool,
) -> EvalOutput:
    model.eval()
    loss_sum = 0.0
    count = 0
    y_true_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="eval", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            use_amp = amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).view(-1)
            loss = criterion(logits.float(), targets.float())
            loss_sum += float(loss.item())
            count += 1
            y_true_all.append(targets.long().cpu().numpy())
            y_score_all.append(torch.sigmoid(logits).cpu().numpy())

    y_true = np.concatenate(y_true_all, axis=0)
    y_score = np.concatenate(y_score_all, axis=0)
    auc = _binary_roc_auc(y_true, y_score)
    ap = _average_precision(y_true, y_score)
    best_f1, best_thr = _best_f1_threshold(y_true, y_score)
    return EvalOutput(
        loss=loss_sum / max(count, 1),
        roc_auc=float(auc),
        average_precision=float(ap),
        best_f1=float(best_f1),
        best_f1_threshold=float(best_thr),
        num_samples=int(y_true.shape[0]),
        num_positive=int((y_true == 1).sum()),
        num_negative=int((y_true == 0).sum()),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> int:
    args = parse_args()
    _set_seed(int(args.seed))

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = (REPO_ROOT / args.manifest).resolve()

    train_tfms = transforms.Compose(
        [
            transforms.Resize((int(args.image_size), int(args.image_size))),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=10.0, translate=(0.06, 0.06), scale=(0.90, 1.10)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
            transforms.ToTensor(),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((int(args.image_size), int(args.image_size))),
            transforms.ToTensor(),
        ]
    )

    train_ds = PresenceDataset(
        manifest_path=manifest_path,
        split="train",
        base_dir=REPO_ROOT,
        transform=train_tfms,
    )
    val_ds = PresenceDataset(
        manifest_path=manifest_path,
        split="val",
        base_dir=REPO_ROOT,
        transform=eval_tfms,
    )
    test_ds = PresenceDataset(
        manifest_path=manifest_path,
        split="test",
        base_dir=REPO_ROOT,
        transform=eval_tfms,
    )

    labels = torch.tensor(train_ds.labels, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=2).float()
    class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_args = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_ds, sampler=sampler, shuffle=False, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    amp = bool(args.amp)

    model = _build_model(str(args.arch), pretrained=bool(args.pretrained)).to(device)
    pos_count = float((labels == 1).sum().item())
    neg_count = float((labels == 0).sum().item())
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs), eta_min=float(args.learning_rate) * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=amp and device.type == "cuda")

    best_key = (-1.0, -1.0, float("inf"))
    history: list[dict[str, Any]] = []
    best_epoch = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for images, targets in tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if float(args.label_smoothing) > 0:
                smooth = float(args.label_smoothing)
                targets = targets * (1.0 - smooth) + 0.5 * smooth

            optimizer.zero_grad(set_to_none=True)
            use_amp = amp and device.type == "cuda"
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images).view(-1)
                loss = criterion(logits.float(), targets.float())
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            train_loss_sum += float(loss.item())
            train_count += 1

        scheduler.step()
        val_out = _run_eval(model=model, loader=val_loader, criterion=criterion, device=device, amp=amp)
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "val_loss": val_out.loss,
            "val_roc_auc": val_out.roc_auc,
            "val_average_precision": val_out.average_precision,
            "val_best_f1": val_out.best_f1,
            "val_best_f1_threshold": val_out.best_f1_threshold,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        key = (float(val_out.roc_auc), float(val_out.average_precision), -float(val_out.loss))
        if key > best_key:
            best_key = key
            best_epoch = epoch
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    torch.save(model.state_dict(), out_dir / "last_model.pt")

    best_model = _build_model(str(args.arch), pretrained=False).to(device)
    best_model.load_state_dict(torch.load(out_dir / "best_model.pt", map_location=device))
    val_best = _run_eval(model=best_model, loader=val_loader, criterion=criterion, device=device, amp=amp)
    test_best = _run_eval(model=best_model, loader=test_loader, criterion=criterion, device=device, amp=amp)

    payload: dict[str, Any] = {
        "arch": str(args.arch),
        "image_size": int(args.image_size),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "label_smoothing": float(args.label_smoothing),
        "seed": int(args.seed),
        "best_epoch": int(best_epoch),
        "class_balance": {
            "train_positive": int((labels == 1).sum().item()),
            "train_negative": int((labels == 0).sum().item()),
            "pos_weight": float(pos_weight.item()),
        },
        "val_best": {
            "loss": val_best.loss,
            "roc_auc": val_best.roc_auc,
            "average_precision": val_best.average_precision,
            "best_f1": val_best.best_f1,
            "best_f1_threshold": val_best.best_f1_threshold,
            "num_samples": val_best.num_samples,
            "num_positive": val_best.num_positive,
            "num_negative": val_best.num_negative,
        },
        "test_best": {
            "loss": test_best.loss,
            "roc_auc": test_best.roc_auc,
            "average_precision": test_best.average_precision,
            "best_f1": test_best.best_f1,
            "best_f1_threshold": test_best.best_f1_threshold,
            "num_samples": test_best.num_samples,
            "num_positive": test_best.num_positive,
            "num_negative": test_best.num_negative,
        },
        "history": history,
    }
    save_json(out_dir / "metrics.json", payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
