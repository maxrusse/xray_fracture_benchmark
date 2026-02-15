# Collaboration Journey: Final Challenge Push

## Shared Goal

Build a local, reproducible X-ray fracture pipeline that can outperform the nnUNet reference on both:
- segmentation quality (`dice_pos`)
- fracture-presence classification (`roc_auc_presence`, `average_precision_presence`)

while staying fair:
- no dataset edits after split creation
- no metadata shortcuts
- no test-time threshold tuning

## Timeline (Condensed)

1. Baseline setup and reproducibility hardening
- Project structure, venv, scripts, and deterministic data workflow were stabilized.
- Early results established a reproducible baseline and a metric protocol.

2. Competitive model development
- Multiple tracks were tested: baseline U-Net, DeepLab variants, dual-head models, DG augment, Swin-adapter attempts.
- nnUNet was installed/evaluated as explicit external reference.

3. Evaluation discipline and leaderboarding
- Ranking docs were added (segmentation + classification + combined views).
- Clean filtering rules were introduced to exclude dev/smoke/scan-only artifacts.

4. Final round optimization
- A modernized stack was implemented (data-aware augment controls, loss variants, solver upgrades, EMA/scheduler options).
- Heavy modern single-model variants did not beat the existing top blend.

5. Last-chance winning move
- New idea: keep best segmentation untouched and improve only presence scoring with a separate classifier.
- Implemented:
  - `scripts/train_presence_classifier.py`
  - `scripts/final_fusion_with_classifier.py`
- Searched classifier fusion weight on **validation only**, then locked for test.

## What Worked Best

1. Preserve segmentation, improve classification separately
- Segmentation stayed at top `dice_pos` while classification improved materially.

2. Multi-branch presence fusion
- Final score blended:
  - dual-head classification score
  - presence-aux segmentation-max score
  - standalone classifier score

3. Strong protocol discipline
- Validation-driven tuning, then one locked test run.
- Clear reporting and direct nnUNet deltas.

## What Did Not Help (or Helped Less)

1. Heavy "modern" single-model rework alone
- Complex loss/model/solver combinations were not enough to beat the best blend by themselves.

2. Some architecture detours
- Several adapter/3D-style experiments underperformed on this dataset split/protocol.

## Roles and Contributions

## User contributions (critical)

1. Competitive direction and pressure
- Repeatedly pushed to benchmark against nnUNet and pursue real improvements, not cosmetic changes.

2. Fairness constraints
- Explicitly required no metadata shortcuts and no unfair data manipulation.
- Kept the process challenge-relevant rather than overfitted to quirks.

3. Decision velocity
- Fast go/no-go calls enabled rapid iteration and narrowed focus to promising tracks.

## Assistant contributions

1. Engineering implementation
- Added/extended training, evaluation, fusion, and ranking scripts.
- Implemented classifier branch and locked val->test fusion workflow.

2. Experiment execution
- Ran training/evaluation loops, monitored GPU state, and produced comparative reports.

3. Reporting
- Produced clean final leaderboard docs and nnUNet-relative summary tables.

## Net Result

Best current model artifact:
- `runs/final_presence_fusion/test_metrics.json`

Key metrics:
- `dice_pos=0.3837`
- `roc_auc_presence=0.9078`
- `average_precision_presence=0.7512`

Relative to nnUNet final reference (`runs/nnunet_fold0_final_test_metrics.json`):
- `dice_pos`: `+0.2770` (`+259.40%`)
- `roc_auc_presence`: `+0.2229` (`+32.55%`)
- `average_precision_presence`: `+0.4478` (`+147.59%`)

## Repro Path (Final Winning Branch)

1. Train classifier
```powershell
python scripts/train_presence_classifier.py --output-dir runs/presence_classifier_final --arch resnet34 --image-size 384 --batch-size 12 --epochs 24 --learning-rate 0.0003 --weight-decay 0.0001 --label-smoothing 0.02 --amp --pretrained
```

2. Search fusion on validation
```powershell
python scripts/final_fusion_with_classifier.py --config-a runs/challenge_dual_head_v1/resolved_config.yaml --checkpoint-a runs/challenge_dual_head_v1/best_model.pt --config-b runs/challenge_presence_aux_v1/resolved_config.yaml --checkpoint-b runs/challenge_presence_aux_v1/best_model.pt --classifier-checkpoint runs/presence_classifier_final/best_model.pt --classifier-arch resnet34 --classifier-image-size 384 --split val --tta hv --alpha-seg-a 1.0 --beta-pres-a 0.3 --seg-threshold 0.3 --gamma-start 0.0 --gamma-stop 1.0 --gamma-step 0.05 --presence-threshold 0.5 --select-by auc_ap_sum --output runs/final_presence_fusion/val_search.json
```

3. Locked test evaluation
```powershell
python scripts/final_fusion_with_classifier.py --config-a runs/challenge_dual_head_v1/resolved_config.yaml --checkpoint-a runs/challenge_dual_head_v1/best_model.pt --config-b runs/challenge_presence_aux_v1/resolved_config.yaml --checkpoint-b runs/challenge_presence_aux_v1/best_model.pt --classifier-checkpoint runs/presence_classifier_final/best_model.pt --classifier-arch resnet34 --classifier-image-size 384 --split test --tta hv --alpha-seg-a 1.0 --beta-pres-a 0.3 --seg-threshold 0.3 --gamma-fixed 0.55 --presence-threshold 0.5 --select-by auc_ap_sum --output runs/final_presence_fusion/test_locked_gamma055.json
```
