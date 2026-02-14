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
- Challenge track (aspect-preserving + patch sampling + hard negatives):
  - Fast check: `python .\scripts\train.py --config .\configs\challenge_patch_hardneg_fast.yaml --output-dir .\runs\challenge_fast`
  - Full run: `python .\scripts\train.py --config .\configs\challenge_patch_hardneg.yaml --output-dir .\runs\challenge_full`
  - Warm-start finetune: `python .\scripts\train.py --config .\configs\challenge_patch_hardneg_finetune.yaml --output-dir .\runs\challenge_finetune_v1 --init-checkpoint .\runs\deeplab_full\best_model.pt`
- Inference-time optimization track:
  - Tune threshold on validation with TTA: `python .\scripts\tune_threshold.py --config .\configs\deeplabv3_resnet50.yaml --checkpoint .\runs\deeplab_full\best_model.pt --tta hv --output .\runs\deeplab_full\threshold_tuning_hv.json --start 0.05 --stop 0.50 --step 0.05`
  - Validate with tuned settings: `python .\scripts\validate.py --config .\configs\deeplabv3_resnet50.yaml --checkpoint .\runs\deeplab_full\best_model.pt --tta hv --threshold 0.25 --output .\runs\deeplab_full\validate_metrics_tta_hv_thr025.json`
  - Test with tuned settings: `python .\scripts\test.py --config .\configs\deeplabv3_resnet50.yaml --checkpoint .\runs\deeplab_full\best_model.pt --tta hv --threshold 0.25 --output .\runs\deeplab_full\test_metrics_tta_hv_thr025.json`
- Latest local reference numbers:
  - `runs/deeplab_full/test_metrics.json`: `dice_pos=0.35627` (baseline, threshold `0.5`, no TTA)
  - `runs/deeplab_full/test_metrics_tta_hv_thr025.json`: `dice_pos=0.36710` (tuned threshold + TTA)
