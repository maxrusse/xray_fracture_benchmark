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
    parser = argparse.ArgumentParser(description="Run locked test evaluation for a trained baseline model.")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/test_metrics.json")
    parser.add_argument("--threshold", type=float, default=None, help="Override binarization threshold for metrics.")
    parser.add_argument("--tta", default=None, help="TTA mode: none, h, v, hv.")
    parser.add_argument("--presence-score-mode", default=None, help="Image-level presence score: max, mean, topk_mean, pred_area.")
    parser.add_argument("--presence-topk-frac", type=float, default=None, help="Top-k fraction for presence-score-mode=topk_mean.")
    parser.add_argument("--presence-threshold", type=float, default=None, help="Binary threshold over image-level presence score.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml((REPO_ROOT / args.config).resolve())
    checkpoint = (REPO_ROOT / args.checkpoint).resolve()
    output = (REPO_ROOT / args.output).resolve()

    device = resolve_device(str(config.get("runtime", {}).get("device", "cuda")))
    runtime = RuntimeContext(device=device, amp=bool(config.get("runtime", {}).get("amp", False)))

    _, _, test_loader = build_dataloaders(config, REPO_ROOT)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))

    criterion = build_criterion(config)
    eval_cfg = config.get("evaluation", {})
    threshold = float(args.threshold) if args.threshold is not None else float(eval_cfg.get("threshold", 0.5))
    tta_mode = str(args.tta) if args.tta is not None else str(eval_cfg.get("tta", "none"))
    presence_score_mode = (
        str(args.presence_score_mode)
        if args.presence_score_mode is not None
        else str(eval_cfg.get("presence_score_mode", "max"))
    )
    presence_topk_frac = (
        float(args.presence_topk_frac)
        if args.presence_topk_frac is not None
        else float(eval_cfg.get("presence_topk_frac", 0.01))
    )
    presence_threshold = (
        float(args.presence_threshold)
        if args.presence_threshold is not None
        else float(eval_cfg.get("presence_threshold", 0.5))
    )
    max_eval_batches = config.get("training", {}).get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    metrics = run_eval_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        runtime=runtime,
        max_batches=max_eval_batches,
        threshold=threshold,
        tta_mode=tta_mode,
        presence_score_mode=presence_score_mode,
        presence_topk_frac=presence_topk_frac,
        presence_threshold=presence_threshold,
    )
    metrics["threshold"] = threshold
    metrics["tta_mode"] = tta_mode
    save_json(output, metrics)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
