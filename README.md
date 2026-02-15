# xray_fracture_benchmark

Local benchmark scaffold for fracture X-ray segmentation/classification experiments with strict reproducibility.

Current best model: `runs/final_presence_fusion/test_metrics.json` (`dice_pos=0.3837`, `roc_auc_presence=0.9078`, `average_precision_presence=0.7512`; locked val-tuned, `tta=hv`).

Final reporting docs: `docs/final_results_clean.md`, `docs/final_last_chance_summary.md`, `docs/collaboration_journey.md`.

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
- Evaluation outputs now also include image-level fracture classification metrics derived from segmentation outputs:
  - `roc_auc_presence`
  - `average_precision_presence`
  - `best_f1_presence`
- Stronger model track (DeepLabV3-ResNet50):
  - Fast check: `python .\scripts\train.py --config .\configs\deeplabv3_fast.yaml --output-dir .\runs\deeplab_fast`
  - Full run: `python .\scripts\train.py --config .\configs\deeplabv3_resnet50.yaml --output-dir .\runs\deeplab_full`
- Challenge track (aspect-preserving + patch sampling + hard negatives):
  - Fast check: `python .\scripts\train.py --config .\configs\challenge_patch_hardneg_fast.yaml --output-dir .\runs\challenge_fast`
  - Full run: `python .\scripts\train.py --config .\configs\challenge_patch_hardneg.yaml --output-dir .\runs\challenge_full`
  - Warm-start finetune: `python .\scripts\train.py --config .\configs\challenge_patch_hardneg_finetune.yaml --output-dir .\runs\challenge_finetune_v1 --init-checkpoint .\runs\deeplab_full\best_model.pt`
- Presence-aware challenge track (auxiliary image-level loss, tuned for fracture detection AUC):
  - Train: `python .\scripts\train.py --config .\configs\challenge_presence_aux.yaml --output-dir .\runs\challenge_presence_aux_v1 --init-checkpoint .\runs\baseline_posmetric\best_model.pt`
  - Test: `python .\scripts\test.py --config .\runs\challenge_presence_aux_v1\resolved_config.yaml --checkpoint .\runs\challenge_presence_aux_v1\best_model.pt --tta hv --output .\runs\challenge_presence_aux_v1\test_metrics_presence_hv.json`
- Dual-head challenge track (shared encoder with segmentation + fracture classification heads):
  - Train: `python .\scripts\train.py --config .\configs\challenge_dual_head_v1.yaml --output-dir .\runs\challenge_dual_head_v1 --init-checkpoint .\runs\baseline_posmetric\best_model.pt`
  - Test: `python .\scripts\test.py --config .\runs\challenge_dual_head_v1\resolved_config.yaml --checkpoint .\runs\challenge_dual_head_v1\best_model.pt --tta hv --presence-score-mode cls --output .\runs\challenge_dual_head_v1\test_metrics_cls_hv.json`
- Clear DG track (no metadata, no dataset edits, no test tuning):
  - Multi-seed train with stronger image-only DG augmentations:
    - `python .\scripts\train_seed_sweep.py --config .\configs\challenge_dg_aug.yaml --output-root .\runs\dg_aug_seeds --seeds 42,1337,2025 --init-checkpoint .\runs\baseline_posmetric\best_model.pt`
  - Select threshold on `val` for 3-seed ensemble:
    - `python .\scripts\ensemble_seed_search.py --run-dirs .\runs\dg_aug_seeds\seed_42 .\runs\dg_aug_seeds\seed_1337 .\runs\dg_aug_seeds\seed_2025 --split val --tta hv --selection-metric dice_pos --output .\runs\dg_aug_seeds\ensemble_val_search.json`
  - Freeze that threshold and evaluate once on `test`:
    - `python .\scripts\ensemble_seed_search.py --run-dirs .\runs\dg_aug_seeds\seed_42 .\runs\dg_aug_seeds\seed_1337 .\runs\dg_aug_seeds\seed_2025 --split test --tta hv --threshold-fixed <VAL_BEST_THRESHOLD> --output .\runs\dg_aug_seeds\ensemble_test_frozen.json`
- Blend search (dual-head + presence-aware):
  - Validation sweep: `python .\scripts\blend_search_dual_presence.py --config-a .\runs\challenge_dual_head_v1\resolved_config.yaml --checkpoint-a .\runs\challenge_dual_head_v1\best_model.pt --config-b .\runs\challenge_presence_aux_v1\resolved_config.yaml --checkpoint-b .\runs\challenge_presence_aux_v1\best_model.pt --split val --tta hv --output .\runs\blend_dual_presence_search_val_hv.json`
  - Evaluate val-selected setting on test: `python .\scripts\blend_search_dual_presence.py --config-a .\runs\challenge_dual_head_v1\resolved_config.yaml --checkpoint-a .\runs\challenge_dual_head_v1\best_model.pt --config-b .\runs\challenge_presence_aux_v1\resolved_config.yaml --checkpoint-b .\runs\challenge_presence_aux_v1\best_model.pt --split test --tta hv --alpha-fixed 1.0 --threshold-fixed 0.3 --beta-fixed 0.3 --output .\runs\blend_dual_presence_eval_test_valjoint_hv.json`
- Inference-time optimization track:
  - Tune threshold on validation with TTA: `python .\scripts\tune_threshold.py --config .\configs\deeplabv3_resnet50.yaml --checkpoint .\runs\deeplab_full\best_model.pt --tta hv --output .\runs\deeplab_full\threshold_tuning_hv.json --start 0.05 --stop 0.50 --step 0.05`
  - Validate with tuned settings: `python .\scripts\validate.py --config .\configs\deeplabv3_resnet50.yaml --checkpoint .\runs\deeplab_full\best_model.pt --tta hv --threshold 0.25 --output .\runs\deeplab_full\validate_metrics_tta_hv_thr025.json`
  - Test with tuned settings: `python .\scripts\test.py --config .\configs\deeplabv3_resnet50.yaml --checkpoint .\runs\deeplab_full\best_model.pt --tta hv --threshold 0.25 --output .\runs\deeplab_full\test_metrics_tta_hv_thr025.json`
- Latest local reference numbers:
  - `runs/deeplab_full/test_metrics.json`: `dice_pos=0.35627` (baseline, threshold `0.5`, no TTA)
  - `runs/deeplab_full/test_metrics_tta_hv_thr025.json`: `dice_pos=0.36710` (tuned threshold + TTA)
