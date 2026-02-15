# Clean Final Results

_Generated: 2026-02-15 08:14:41_

Filtered for final comparison: excluded dev/smoke/scan artifacts and zero-Dice outputs; one best test artifact per run kept.

Best model one-liner: `final_presence_fusion` (`runs/final_presence_fusion/test_metrics.json`) with `dice_pos=0.3837`, `roc_auc_presence=0.9078`, `average_precision_presence=0.7512`.

- Valid challenger runs: **22**
- Valid test artifacts after filtering: **36**
- `challenge_fast` remains excluded because only `test_metrics_presence_scan.json` is available for that run.

## Executive Summary (Top 3 + Clinical Candidate)

Top 3 by combined final rank (Dice + ROC-AUC + AP):

1. `final_presence_fusion`
- Metrics: `dice_pos=0.3837`, `roc_auc_presence=0.9078`, `average_precision_presence=0.7512`
- Key feature: Locked val-tuned 3-way presence fusion: base dual/presence blend + standalone classifier (gamma=0.55).
2. `blend_dual_presence_valjoint`
- Metrics: `dice_pos=0.3837`, `roc_auc_presence=0.8866`, `average_precision_presence=0.7234`
- Key feature: Validation-joint tuned dual/presence blend with hv TTA and threshold 0.3.
3. `dg_aug_seeds/seed_2025`
- Metrics: `dice_pos=0.3780`, `roc_auc_presence=0.7450`, `average_precision_presence=0.3956`
- Key feature: Data-aware augmentation heavy dual-head seed with strong segmentation robustness.

Clinical testing recommendation (research pilot):
- Choose `final_presence_fusion` first. It has the strongest combined profile with locked val-only tuning.
- Keep `blend_dual_presence_valjoint` as fallback comparator in the same protocol.

nnUNet comparison reference (`nnunet_fold0_final_test_metrics`):
- Baseline: `dice_pos=0.1068`, `roc_auc_presence=0.6849`, `average_precision_presence=0.3034`

| Model | Dice_pos | Delta vs nnUNet | ROC-AUC | Delta vs nnUNet | AP | Delta vs nnUNet |
|---|---:|---:|---:|---:|---:|---:|
| `final_presence_fusion` | 0.3837 | +0.2770 (+259.40%) | 0.9078 | +0.2229 (+32.55%) | 0.7512 | +0.4478 (+147.59%) |
| `blend_dual_presence_valjoint` | 0.3837 | +0.2770 (+259.40%) | 0.8866 | +0.2018 (+29.46%) | 0.7234 | +0.4200 (+138.43%) |
| `dg_aug_seeds/seed_2025` | 0.3780 | +0.2713 (+254.06%) | 0.7450 | +0.0602 (+8.78%) | 0.3956 | +0.0922 (+30.37%) |

## 1) Final Segmentation Ranking (Dice_pos)

| Rank | Run | Dice_pos | IoU_pos | ROC-AUC | AP | Source |
|---:|---|---:|---:|---:|---:|---|
| 1 | `blend_dual_presence_valjoint` | 0.3837 | - | 0.8866 | 0.7234 | `blend_dual_presence_valjoint/test_metrics.json` |
| 2 | `final_presence_fusion` | 0.3837 | 0.2808 | 0.9078 | 0.7512 | `final_presence_fusion/test_metrics.json` |
| 3 | `dg_aug_seeds/seed_2025` | 0.3780 | 0.2741 | 0.7450 | 0.3956 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 4 | `dg_aug_seeds/seed_1337` | 0.3766 | 0.2719 | 0.7352 | 0.3807 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 5 | `dg_aug_seeds/seed_42` | 0.3759 | 0.2722 | 0.7393 | 0.3761 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 6 | `challenge_dual_head_v1` | 0.3755 | 0.2782 | 0.7974 | 0.5238 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 7 | `deeplab_full` | 0.3671 | 0.2699 | - | - | `deeplab_full/test_metrics_tta_hv_thr025.json` |
| 8 | `deeplabv3_resnet101_full_like_v2` | 0.3605 | 0.2616 | 0.6331 | 0.2401 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 9 | `baseline_posmetric` | 0.3505 | 0.2524 | - | - | `baseline_posmetric/test_metrics.json` |
| 10 | `challenge_presence_aux_v1` | 0.3242 | 0.2394 | 0.8794 | 0.7090 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 11 | `challenge_finetune_v1` | 0.3125 | 0.2296 | - | - | `challenge_finetune_v1/test_metrics.json` |
| 12 | `final_round_modern_v2` | 0.2455 | 0.1654 | 0.6878 | 0.3624 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 13 | `swin3d_adapter_baseline_like_long_v4` | 0.1551 | 0.0927 | 0.5850 | 0.2256 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 14 | `swin3d_adapter_dnpplus_fullsend_v2` | 0.1506 | 0.0897 | 0.5970 | 0.2759 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 15 | `swin3d_adapter_dnpplus_v1` | 0.1426 | 0.0844 | 0.5668 | 0.1943 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 16 | `swin3d_adapter_baseline_like_v2` | 0.1398 | 0.0815 | 0.6082 | 0.2387 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 17 | `swin3d_adapter_rgb_v3` | 0.1365 | 0.0796 | 0.5917 | 0.2033 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 18 | `swin3d_d1_adapter_20260214` | 0.1177 | 0.0672 | 0.6240 | 0.2424 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 19 | `nnunet_fold0_final_test_metrics` | 0.1068 | - | 0.6849 | 0.3034 | `nnunet_fold0_final_test_metrics.json` |
| 20 | `deeplabv3_resnet101_highpower_v1` | 0.1041 | 0.0638 | 0.5315 | 0.2662 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 21 | `nnunet_fold0_test_metrics` | 0.0979 | - | 0.6806 | 0.2844 | `nnunet_fold0_test_metrics.json` |
| 22 | `swin3d_adapter_fast_v1` | 0.0621 | 0.0334 | 0.5306 | 0.1846 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |

## 2) Final Classification Ranking (ROC-AUC)

| Rank | Run | ROC-AUC | AP | Dice_pos | Source |
|---:|---|---:|---:|---:|---|
| 1 | `final_presence_fusion` | 0.9078 | 0.7512 | 0.3837 | `final_presence_fusion/test_metrics.json` |
| 2 | `blend_dual_presence_valjoint` | 0.8866 | 0.7234 | 0.3837 | `blend_dual_presence_valjoint/test_metrics.json` |
| 3 | `challenge_presence_aux_v1` | 0.8794 | 0.7090 | 0.3242 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 4 | `challenge_dual_head_v1` | 0.7974 | 0.5238 | 0.3755 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 5 | `dg_aug_seeds/seed_2025` | 0.7450 | 0.3956 | 0.3780 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 6 | `dg_aug_seeds/seed_42` | 0.7393 | 0.3761 | 0.3759 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 7 | `dg_aug_seeds/seed_1337` | 0.7352 | 0.3807 | 0.3766 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 8 | `final_round_modern_v2` | 0.6878 | 0.3624 | 0.2455 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 9 | `nnunet_fold0_final_test_metrics` | 0.6849 | 0.3034 | 0.1068 | `nnunet_fold0_final_test_metrics.json` |
| 10 | `nnunet_fold0_test_metrics` | 0.6806 | 0.2844 | 0.0979 | `nnunet_fold0_test_metrics.json` |
| 11 | `deeplabv3_resnet101_full_like_v2` | 0.6331 | 0.2401 | 0.3605 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 12 | `swin3d_d1_adapter_20260214` | 0.6240 | 0.2424 | 0.1177 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 13 | `swin3d_adapter_baseline_like_v2` | 0.6082 | 0.2387 | 0.1398 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 14 | `swin3d_adapter_dnpplus_fullsend_v2` | 0.5970 | 0.2759 | 0.1506 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 15 | `swin3d_adapter_rgb_v3` | 0.5917 | 0.2033 | 0.1365 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 16 | `swin3d_adapter_baseline_like_long_v4` | 0.5850 | 0.2256 | 0.1551 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 17 | `swin3d_adapter_dnpplus_v1` | 0.5668 | 0.1943 | 0.1426 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 18 | `deeplabv3_resnet101_highpower_v1` | 0.5315 | 0.2662 | 0.1041 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 19 | `swin3d_adapter_fast_v1` | 0.5306 | 0.1846 | 0.0621 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |

## 3) Final Classification Ranking (AP)

| Rank | Run | AP | ROC-AUC | Dice_pos | Source |
|---:|---|---:|---:|---:|---|
| 1 | `final_presence_fusion` | 0.7512 | 0.9078 | 0.3837 | `final_presence_fusion/test_metrics.json` |
| 2 | `blend_dual_presence_valjoint` | 0.7234 | 0.8866 | 0.3837 | `blend_dual_presence_valjoint/test_metrics.json` |
| 3 | `challenge_presence_aux_v1` | 0.7090 | 0.8794 | 0.3242 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 4 | `challenge_dual_head_v1` | 0.5238 | 0.7974 | 0.3755 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 5 | `dg_aug_seeds/seed_2025` | 0.3956 | 0.7450 | 0.3780 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 6 | `dg_aug_seeds/seed_1337` | 0.3807 | 0.7352 | 0.3766 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 7 | `dg_aug_seeds/seed_42` | 0.3761 | 0.7393 | 0.3759 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 8 | `final_round_modern_v2` | 0.3624 | 0.6878 | 0.2455 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 9 | `nnunet_fold0_final_test_metrics` | 0.3034 | 0.6849 | 0.1068 | `nnunet_fold0_final_test_metrics.json` |
| 10 | `nnunet_fold0_test_metrics` | 0.2844 | 0.6806 | 0.0979 | `nnunet_fold0_test_metrics.json` |
| 11 | `swin3d_adapter_dnpplus_fullsend_v2` | 0.2759 | 0.5970 | 0.1506 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 12 | `deeplabv3_resnet101_highpower_v1` | 0.2662 | 0.5315 | 0.1041 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 13 | `swin3d_d1_adapter_20260214` | 0.2424 | 0.6240 | 0.1177 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 14 | `deeplabv3_resnet101_full_like_v2` | 0.2401 | 0.6331 | 0.3605 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 15 | `swin3d_adapter_baseline_like_v2` | 0.2387 | 0.6082 | 0.1398 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 16 | `swin3d_adapter_baseline_like_long_v4` | 0.2256 | 0.5850 | 0.1551 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 17 | `swin3d_adapter_rgb_v3` | 0.2033 | 0.5917 | 0.1365 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 18 | `swin3d_adapter_dnpplus_v1` | 0.1943 | 0.5668 | 0.1426 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 19 | `swin3d_adapter_fast_v1` | 0.1846 | 0.5306 | 0.0621 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |

## 4) Combined Final Ranking (Mean Rank across Dice + ROC-AUC + AP)

| Rank | Run | Mean Rank | Metrics Used | Dice_pos | ROC-AUC | AP | Source |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `final_presence_fusion` | 1.33 | 3 | 0.3837 | 0.9078 | 0.7512 | `final_presence_fusion/test_metrics.json` |
| 2 | `blend_dual_presence_valjoint` | 1.67 | 3 | 0.3837 | 0.8866 | 0.7234 | `blend_dual_presence_valjoint/test_metrics.json` |
| 3 | `dg_aug_seeds/seed_2025` | 4.33 | 3 | 0.3780 | 0.7450 | 0.3956 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 4 | `challenge_dual_head_v1` | 4.67 | 3 | 0.3755 | 0.7974 | 0.5238 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 5 | `challenge_presence_aux_v1` | 5.33 | 3 | 0.3242 | 0.8794 | 0.7090 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 6 | `dg_aug_seeds/seed_1337` | 5.67 | 3 | 0.3766 | 0.7352 | 0.3807 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 7 | `dg_aug_seeds/seed_42` | 6.00 | 3 | 0.3759 | 0.7393 | 0.3761 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 8 | `deeplab_full` | 7.00 | 1 | 0.3671 | - | - | `deeplab_full/test_metrics_tta_hv_thr025.json` |
| 9 | `baseline_posmetric` | 9.00 | 1 | 0.3505 | - | - | `baseline_posmetric/test_metrics.json` |
| 10 | `final_round_modern_v2` | 9.33 | 3 | 0.2455 | 0.6878 | 0.3624 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 11 | `deeplabv3_resnet101_full_like_v2` | 11.00 | 3 | 0.3605 | 0.6331 | 0.2401 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 12 | `challenge_finetune_v1` | 11.00 | 1 | 0.3125 | - | - | `challenge_finetune_v1/test_metrics.json` |
| 13 | `nnunet_fold0_final_test_metrics` | 12.33 | 3 | 0.1068 | 0.6849 | 0.3034 | `nnunet_fold0_final_test_metrics.json` |
| 14 | `swin3d_adapter_dnpplus_fullsend_v2` | 13.00 | 3 | 0.1506 | 0.5970 | 0.2759 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 15 | `nnunet_fold0_test_metrics` | 13.67 | 3 | 0.0979 | 0.6806 | 0.2844 | `nnunet_fold0_test_metrics.json` |
| 16 | `swin3d_d1_adapter_20260214` | 14.33 | 3 | 0.1177 | 0.6240 | 0.2424 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 17 | `swin3d_adapter_baseline_like_v2` | 14.67 | 3 | 0.1398 | 0.6082 | 0.2387 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 18 | `swin3d_adapter_baseline_like_long_v4` | 15.00 | 3 | 0.1551 | 0.5850 | 0.2256 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 19 | `swin3d_adapter_rgb_v3` | 16.33 | 3 | 0.1365 | 0.5917 | 0.2033 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 20 | `swin3d_adapter_dnpplus_v1` | 16.67 | 3 | 0.1426 | 0.5668 | 0.1943 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 21 | `deeplabv3_resnet101_highpower_v1` | 16.67 | 3 | 0.1041 | 0.5315 | 0.2662 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 22 | `swin3d_adapter_fast_v1` | 20.00 | 3 | 0.0621 | 0.5306 | 0.1846 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |
