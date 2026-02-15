# Final Last-Chance Run Summary

_Generated: 2026-02-15 08:14:00_

## Final Decision

- **New best overall for deployment candidate:** `runs/final_presence_fusion/test_metrics.json`
- Why: preserves top segmentation while improving classification (AUC/AP) over previous best.

## Metrics Comparison

| Metric | New Final Fusion | Previous Best Blend | nnUNet Final | New vs Previous | New vs nnUNet |
|---|---:|---:|---:|---:|---:|
| `dice_pos` | 0.3837 | 0.3837 | 0.1068 | +0.0000 (+0.00%) | +0.2770 (+259.40%) |
| `iou_pos` | 0.2808 | - | - | - (-) | - (-) |
| `roc_auc_presence` | 0.9078 | 0.8866 | 0.6849 | +0.0211 (+2.38%) | +0.2229 (+32.55%) |
| `average_precision_presence` | 0.7512 | 0.7234 | 0.3034 | +0.0278 (+3.84%) | +0.4478 (+147.59%) |
| `precision_pos` | 0.0964 | 0.0964 | 0.0113 | +0.0000 (+0.00%) | +0.0850 (+749.75%) |
| `recall_pos` | 0.5227 | 0.5227 | 0.0724 | +0.0000 (+0.00%) | +0.4503 (+622.17%) |

## Locked Fusion Settings

- Segmentation backbone: `challenge_dual_head_v1` (primary) + `challenge_presence_aux_v1` (aux presence branch)
- Added branch: standalone image-level classifier `presence_classifier_final` (ResNet34, 384px)
- Locked from validation search only: `gamma=0.55` (base blend vs classifier), `alpha_seg_a=1.0`, `beta_presence_a=0.3`, `seg_threshold=0.3`, `tta=hv`
- Test was executed once with locked settings (no test-time search).
