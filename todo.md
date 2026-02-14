# TODO - Fracture X-Ray Segmentation Benchmark (GitHub-Ready)

Goal: create a clean, reproducible benchmark repo where different agents/tools can implement and compare approaches under the same rules.

## 1) Repository Foundation
- [ ] Rename/scope project docs to fracture segmentation benchmark (not pneumonia classification).
- [ ] Create clear repo structure:
  - [ ] `src/` (framework code)
  - [ ] `configs/` (experiment configs only)
  - [ ] `scripts/` (CLI entrypoints)
  - [ ] `data/` (ignored, local only)
  - [ ] `artifacts/` or `runs/` (ignored)
  - [ ] `tests/`
  - [ ] `docs/`
- [ ] Add `LICENSE` (pick one: MIT/Apache-2.0).
- [ ] Add `CONTRIBUTING.md` with PR/test rules.
- [ ] Add `CHANGELOG.md`.
- [ ] Add `.editorconfig`, `pre-commit` config, and formatter/linter config.

## 2) Dataset Access Layer
- [ ] Add `docs/dataset.md` with source links, citation, license, and usage constraints.
- [ ] Implement `scripts/download_dataset.py` (manual URL/token support if required).
- [ ] Implement checksum validation for downloaded archives.
- [ ] Implement deterministic extraction to `data/raw/`.
- [ ] Add `scripts/prepare_dataset.py` to build:
  - [ ] `data/processed/images/`
  - [ ] `data/processed/masks/`
  - [ ] split files (`train.csv`, `val.csv`, `test.csv`)
- [ ] Enforce patient-level split logic to prevent leakage.
- [ ] Log dataset stats (images, positive/negative, mask coverage).

## 3) Benchmark Contract (Model-Agnostic)
- [ ] Define a strict model interface in `src/benchmark/interfaces.py`:
  - [ ] `fit(train, val, config)`
  - [ ] `predict(images) -> masks/probabilities`
  - [ ] `save(path)` / `load(path)`
- [ ] Define input size, mask format, and allowed preprocessing contract.
- [ ] Define output contract for predictions and metadata.
- [ ] Add `docs/benchmark_rules.md` with:
  - [ ] allowed data
  - [ ] allowed external pretrained weights (yes/no)
  - [ ] time/resource reporting requirements
  - [ ] seed/reproducibility requirements

## 4) Training + Evaluation Pipeline
- [ ] Implement `scripts/train.py` using only config-driven settings.
- [ ] Implement `scripts/validate.py`.
- [ ] Implement `scripts/test.py` with locked test evaluation mode.
- [ ] Implement metrics module:
  - [ ] Dice
  - [ ] IoU
  - [ ] pixel precision/recall
  - [ ] optional detection-from-mask metrics
- [ ] Implement bootstrap confidence intervals for primary metrics.
- [ ] Save standardized artifacts:
  - [ ] `metrics.json`
  - [ ] `config_resolved.yaml`
  - [ ] `environment.txt`
  - [ ] `predictions/`

## 5) Baseline + Comparison Tracks
- [ ] Add one minimal sanity baseline (for pipeline verification only).
- [ ] Add nnU-Net comparison track script (`scripts/run_nnunet_baseline.py`).
- [ ] Ensure same split and same metric code is used for all methods.
- [ ] Add `docs/comparison_protocol.md` for fair method comparison.

## 6) Experiment Management
- [ ] Implement run naming convention (`<date>-<method>-<seed>`).
- [ ] Add experiment registry table (CSV/JSON) auto-updated after each run.
- [ ] Add config templates:
  - [ ] `configs/default.yaml`
  - [ ] `configs/fast_dev.yaml`
  - [ ] `configs/repro_strict.yaml`
- [ ] Add one-command smoke run (`scripts/smoke_test.py`) on tiny subset.

## 7) Testing + Quality Gates
- [ ] Add unit tests for:
  - [ ] data loading/transforms
  - [ ] metric correctness on known toy examples
  - [ ] split determinism
- [ ] Add integration test: one mini end-to-end training/eval pass.
- [ ] Add CI workflow:
  - [ ] lint
  - [ ] unit tests
  - [ ] smoke integration test
- [ ] Set merge gate: CI must pass.

## 8) Reproducibility + Environment
- [ ] Pin dependencies (`requirements.txt` + optional `requirements-lock.txt`).
- [ ] Add `scripts/export_env.py` to capture package/hardware info.
- [ ] Add deterministic seed utility used by all entrypoints.
- [ ] Document venv creation/activation for this repo.
- [ ] Add optional Dockerfile for fully portable runs.

## 9) Reporting + GitHub Readiness
- [ ] Rewrite `README.md` with quickstart in <10 minutes.
- [ ] Add benchmark table template (empty, to fill from real runs).
- [ ] Add `docs/results_schema.md` describing required result fields.
- [ ] Add issue templates: bug report, experiment report.
- [ ] Add PR template requiring metrics + config + seed disclosure.

## 10) Definition Of Done (v1.0 Baseline)
- [ ] Fresh clone + venv setup works.
- [ ] Dataset download + prep works from CLI.
- [ ] `fast_dev` run completes end-to-end locally.
- [ ] Test evaluation produces standardized artifacts.
- [ ] CI passes on default branch.
- [ ] Repo is publishable to GitHub with clear docs and license.

## Immediate Next 5 Tasks
- [ ] Task 1: align README and scope to fracture benchmark.
- [ ] Task 2: implement dataset download/prepare scripts.
- [ ] Task 3: finalize benchmark interface + metric module.
- [ ] Task 4: implement `train/validate/test` CLI pipeline.
- [ ] Task 5: add tests + CI + smoke run.

