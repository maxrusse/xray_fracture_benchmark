from __future__ import annotations

import argparse
import pathlib
import sys

import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.engine import (  # noqa: E402
    RuntimeContext,
    build_criterion,
    build_dataloaders,
    build_model,
    resolve_device,
    run_eval_epoch,
)
from benchmark.utils import load_yaml, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune segmentation threshold on validation split.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/threshold_tuning.json")
    parser.add_argument("--metric", default="dice_pos")
    parser.add_argument("--tta", default="none", help="TTA mode: none, h, v, hv.")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--stop", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    return parser.parse_args()


def build_thresholds(start: float, stop: float, step: float) -> list[float]:
    if not (0.0 < start < 1.0):
        raise ValueError("--start must be in (0,1)")
    if not (0.0 < stop < 1.0):
        raise ValueError("--stop must be in (0,1)")
    if stop < start:
        raise ValueError("--stop must be >= --start")
    if step <= 0:
        raise ValueError("--step must be > 0")

    values: list[float] = []
    current = start
    while current <= stop + 1e-12:
        values.append(round(current, 6))
        current += step
    return values


def main() -> int:
    args = parse_args()
    config = load_yaml((REPO_ROOT / args.config).resolve())
    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    output = (REPO_ROOT / args.output).resolve()

    device = resolve_device(str(config.get("runtime", {}).get("device", "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(config.get("runtime", {}).get("amp", False)))

    _, val_loader, _ = build_dataloaders(config, REPO_ROOT)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    criterion = build_criterion(config)
    max_eval_batches = config.get("training", {}).get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None

    thresholds = build_thresholds(args.start, args.stop, args.step)
    history: list[dict[str, float]] = []

    best_threshold = thresholds[0]
    best_score = float("-inf")
    metric_name = str(args.metric)
    for threshold in thresholds:
        metrics = run_eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            runtime=runtime,
            max_batches=max_eval_batches,
            threshold=threshold,
            tta_mode=str(args.tta),
        )
        score = float(metrics.get(metric_name, float("-inf")))
        row = {"threshold": float(threshold), metric_name: score, **metrics}
        history.append(row)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    payload = {
        "metric": metric_name,
        "tta_mode": str(args.tta),
        "best_threshold": best_threshold,
        "best_score": best_score,
        "history": history,
    }
    save_json(output, payload)
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
