# X-Ray Challenge Leaderboard

_Generated: 2026-02-15 00:38:52_

## GPU Status

- No active CUDA training/inference process detected in latest `nvidia-smi` check (only `C+G` desktop/system processes).

## Ranking (Best Comparable Test Result Per Run)

| Rank | Run | Dice_pos | IoU_pos | ROC-AUC (presence) | AP (presence) | Precision_pos | Recall_pos | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `blend_dual_presence_valjoint` | 0.3837 | - | 0.8866 | 0.7234 | 0.0964 | 0.5227 | `blend_dual_presence_valjoint/test_metrics.json` |
| 2 | `dg_aug_seeds/seed_2025` | 0.3780 | 0.2741 | 0.7450 | 0.3956 | 0.0964 | 0.5333 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 3 | `dg_aug_seeds/seed_1337` | 0.3766 | 0.2719 | 0.7352 | 0.3807 | 0.0770 | 0.5613 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 4 | `dg_aug_seeds/seed_42` | 0.3759 | 0.2722 | 0.7393 | 0.3761 | 0.0811 | 0.5421 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 5 | `challenge_dual_head_v1` | 0.3755 | 0.2782 | 0.7974 | 0.5238 | 0.1419 | 0.4514 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 6 | `deeplab_full` | 0.3671 | 0.2699 | - | - | 0.0647 | 0.4367 | `deeplab_full/test_metrics_tta_hv_thr025.json` |
| 7 | `deeplabv3_resnet101_full_like_v2` | 0.3605 | 0.2616 | 0.6331 | 0.2401 | 0.0499 | 0.5300 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 8 | `baseline_posmetric` | 0.3505 | 0.2524 | - | - | 0.0755 | 0.4080 | `baseline_posmetric/test_metrics.json` |
| 9 | `challenge_presence_aux_v1` | 0.3242 | 0.2394 | 0.8794 | 0.7090 | 0.3906 | 0.3338 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 10 | `challenge_finetune_v1` | 0.3125 | 0.2296 | - | - | 0.0296 | 0.4220 | `challenge_finetune_v1/test_metrics.json` |
| 11 | `final_round_modern_v2` | 0.2455 | 0.1654 | 0.6878 | 0.3624 | 0.0372 | 0.4702 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 12 | `swin3d_adapter_baseline_like_long_v4` | 0.1551 | 0.0927 | 0.5850 | 0.2256 | 0.0195 | 0.2957 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 13 | `swin3d_adapter_dnpplus_fullsend_v2` | 0.1506 | 0.0897 | 0.5970 | 0.2759 | 0.0089 | 0.4472 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 14 | `swin3d_adapter_dnpplus_v1` | 0.1426 | 0.0844 | 0.5668 | 0.1943 | 0.0164 | 0.3357 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 15 | `swin3d_adapter_baseline_like_v2` | 0.1398 | 0.0815 | 0.6082 | 0.2387 | 0.0209 | 0.3213 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 16 | `swin3d_adapter_rgb_v3` | 0.1365 | 0.0796 | 0.5917 | 0.2033 | 0.0188 | 0.3015 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 17 | `swin3d_d1_adapter_20260214` | 0.1177 | 0.0672 | 0.6240 | 0.2424 | 0.0193 | 0.3441 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 18 | `nnunet_fold0_final_test_metrics` | 0.1068 | - | 0.6849 | 0.3034 | 0.0113 | 0.0724 | `nnunet_fold0_final_test_metrics.json` |
| 19 | `deeplabv3_resnet101_highpower_v1` | 0.1041 | 0.0638 | 0.5315 | 0.2662 | 0.0102 | 0.2043 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 20 | `nnunet_fold0_test_metrics` | 0.0979 | - | 0.6806 | 0.2844 | 0.0108 | 0.0788 | `nnunet_fold0_test_metrics.json` |
| 21 | `swin3d_adapter_fast_v1` | 0.0621 | 0.0334 | 0.5306 | 0.1846 | 0.0049 | 0.6553 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |

## Analysis

- Best overall `dice_pos`: `blend_dual_presence_valjoint` at **0.3837**.
- Versus `nnUNet` reference (`nnunet_fold0_final_test_metrics`, 0.1068), the top run is **+0.2770 Dice_pos** (259.40% relative).
- Best fracture-presence ROC-AUC: `blend_dual_presence_valjoint` at **0.8866**.
- Best fracture-presence AP: `blend_dual_presence_valjoint` at **0.7234**.
- `dg_aug_seeds` are tightly clustered near the top on segmentation, suggesting robust gains from data-aware augmentation and seed averaging potential.
- Recent “modern” single-run attempt (`final_round_modern_v2`) improved stability but remains below the leading blend/dual-head track on this dataset.
- Current strongest direction for final push is ensembling top dual-head + dg-aug checkpoints with validation-selected thresholding/TTA (without test-time threshold search).

## Notes

- This ranking keeps one best **comparable** test artifact per run and excludes files with `presence_scan`, `regression`, or `smoke` in path/name.
- Some older runs do not report presence metrics; these cells are shown as `-`.
