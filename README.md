# xray_fracture_benchmark

Local benchmark scaffold for fracture X-ray segmentation/classification experiments with strict reproducibility.

## Dataset (Step 1)

Primary dataset: **FracAtlas** (public, CC BY 4.0).
- 4,083 musculoskeletal X-rays
- fractured / non-fractured labels
- polygon annotations (VGG JSON) that can be rasterized to masks
- canonical source: https://doi.org/10.6084/m9.figshare.22363012.v6

## Quickstart

```powershell
cd C:\Users\Max\code\xray_fracture_benchmark
C:\Users\Max\code\xray_fracture_benchmark_venv\Scripts\Activate.ps1
.\scripts\setup_env.ps1
```

`setup_env.ps1` installs:
- CUDA PyTorch (`cu128`)
- base project dependencies
- and runs a CUDA verification script

## Download + Prepare Data

```powershell
python .\scripts\download_fracatlas.py
python .\scripts\prepare_fracatlas_segmentation.py
```

Outputs:
- raw archive and extraction under `data/raw/`
- binary masks + manifests under `data/processed/fracatlas/`
- deterministic split CSV files (`train.csv`, `val.csv`, `test.csv`)

## Notes

- Dataset and artifacts are intentionally ignored by git.
- A CI guard and pre-commit hook block accidental dataset commits.
- Baseline training/evaluation scripts:
  - `python .\scripts\train.py --config .\configs\fast_dev.yaml --output-dir .\runs\fast_dev`
  - `python .\scripts\validate.py --config .\configs\fast_dev.yaml --checkpoint .\runs\fast_dev\best_model.pt --output .\runs\fast_dev\validate_metrics.json`
  - `python .\scripts\test.py --config .\configs\fast_dev.yaml --checkpoint .\runs\fast_dev\best_model.pt --output .\runs\fast_dev\test_metrics.json`
- Primary model selection metric is `dice_pos` (fractured-only Dice), not all-sample Dice.
- Stronger model track (DeepLabV3-ResNet50):
  - Fast check: `python .\scripts\train.py --config .\configs\deeplabv3_fast.yaml --output-dir .\runs\deeplab_fast`
  - Full run: `python .\scripts\train.py --config .\configs\deeplabv3_resnet50.yaml --output-dir .\runs\deeplab_full`
