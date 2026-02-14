from __future__ import annotations

import argparse
import pathlib
import sys

import torch
import torch.nn as nn

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.engine import (  # noqa: E402
    RuntimeContext,
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

    criterion = nn.BCEWithLogitsLoss()
    max_eval_batches = config.get("training", {}).get("max_eval_batches")
    max_eval_batches = int(max_eval_batches) if max_eval_batches is not None else None
    metrics = run_eval_epoch(model=model, loader=test_loader, criterion=criterion, runtime=runtime, max_batches=max_eval_batches)
    save_json(output, metrics)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
