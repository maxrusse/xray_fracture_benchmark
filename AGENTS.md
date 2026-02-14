# AGENTS.md - xray_fracture_benchmark

## Mission
Build a clean, reproducible local benchmark for 2D fracture X-ray segmentation with fair model comparison.

## Scope
- Primary task: segmentation on fracture X-rays.
- Optional derived tasks: detection/classification from predicted masks.
- Do not hard-code one model family; keep benchmark model-agnostic.

## Non-Negotiables
- Never commit dataset or run artifacts.
- Use the same data split and metric implementation for all methods.
- Keep runs reproducible with fixed seeds and saved resolved configs.
- Prefer config-driven changes over ad-hoc code edits.

## Repository Rules
- Code in `src/`, runnable entrypoints in `scripts/`, configs in `configs/`.
- Keep `data/`, `runs/`, `results/`, `artifacts/` local-only (gitignored).
- Update docs when behavior/CLI/contracts change.
- Do not introduce breaking interface changes without updating tests/docs.

## Evaluation Rules
- Required segmentation metrics: Dice and IoU.
- Report confidence intervals for primary metrics when possible.
- Test set must be treated as locked; tune only on train/val.
- Comparison claims must reference identical split, preprocessing, and metrics.

## Engineering Rules
- Keep code simple and explicit; avoid hidden global state.
- Add tests for data split determinism and metric correctness.
- Keep CLI scripts idempotent where possible.
- Record environment details for each benchmark run.

## Quick Commands
- Create/activate venv:
  - `python -m venv C:\Users\Max\code\xray_fracture_benchmark_venv`
  - `C:\Users\Max\code\xray_fracture_benchmark_venv\Scripts\Activate.ps1`
- Install deps:
  - `.\scripts\setup_env.ps1`
- Download/prepare dataset:
  - `python scripts/download_fracatlas.py`
  - `python scripts/prepare_fracatlas_segmentation.py`
- Local dataset guard:
  - `python scripts/check_no_dataset_files.py`
