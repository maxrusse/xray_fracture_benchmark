# Multi-Task Challenger Ranking

_Generated: 2026-02-15 07:45:01_

This report ranks challengers by segmentation and classification metrics from their best-per-run test artifact (selected by highest `dice_pos`).

- Runs considered: **28**

## A) Segmentation Ranking (Dice_pos)

| Rank | Run | Dice_pos | ROC-AUC | AP | IoU_pos | Precision_pos | Recall_pos | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `blend_dual_presence_valjoint` | 0.3837 | 0.8866 | 0.7234 | - | 0.0964 | 0.5227 | `blend_dual_presence_valjoint/test_metrics.json` |
| 2 | `dg_aug_seeds/seed_2025` | 0.3780 | 0.7450 | 0.3956 | 0.2741 | 0.0964 | 0.5333 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 3 | `dg_aug_seeds/seed_1337` | 0.3766 | 0.7352 | 0.3807 | 0.2719 | 0.0770 | 0.5613 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 4 | `dg_aug_seeds/seed_42` | 0.3759 | 0.7393 | 0.3761 | 0.2722 | 0.0811 | 0.5421 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 5 | `challenge_dual_head_v1` | 0.3755 | 0.7974 | 0.5238 | 0.2782 | 0.1419 | 0.4514 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 6 | `deeplab_full` | 0.3671 | - | - | 0.2699 | 0.0647 | 0.4367 | `deeplab_full/test_metrics_tta_hv_thr025.json` |
| 7 | `deeplabv3_resnet101_full_like_v2` | 0.3605 | 0.6331 | 0.2401 | 0.2616 | 0.0499 | 0.5300 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 8 | `baseline_posmetric` | 0.3505 | - | - | 0.2524 | 0.0755 | 0.4080 | `baseline_posmetric/test_metrics.json` |
| 9 | `challenge_presence_aux_v1` | 0.3242 | 0.8794 | 0.7090 | 0.2394 | 0.3906 | 0.3338 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 10 | `challenge_finetune_v1` | 0.3125 | - | - | 0.2296 | 0.0296 | 0.4220 | `challenge_finetune_v1/test_metrics.json` |
| 11 | `final_round_modern_v2` | 0.2455 | 0.6878 | 0.3624 | 0.1654 | 0.0372 | 0.4702 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 12 | `swin3d_adapter_baseline_like_long_v4` | 0.1551 | 0.5850 | 0.2256 | 0.0927 | 0.0195 | 0.2957 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 13 | `swin3d_adapter_dnpplus_fullsend_v2` | 0.1506 | 0.5970 | 0.2759 | 0.0897 | 0.0089 | 0.4472 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 14 | `swin3d_adapter_dnpplus_v1` | 0.1426 | 0.5668 | 0.1943 | 0.0844 | 0.0164 | 0.3357 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 15 | `swin3d_adapter_baseline_like_v2` | 0.1398 | 0.6082 | 0.2387 | 0.0815 | 0.0209 | 0.3213 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 16 | `swin3d_adapter_rgb_v3` | 0.1365 | 0.5917 | 0.2033 | 0.0796 | 0.0188 | 0.3015 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 17 | `challenge_fast_v2` | 0.1285 | 0.2716 | 0.0309 | 0.0865 | 0.0020 | 0.2284 | `challenge_fast_v2/test_metrics_presence_scan.json` |
| 18 | `swin3d_d1_adapter_20260214` | 0.1177 | 0.6240 | 0.2424 | 0.0672 | 0.0193 | 0.3441 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 19 | `nnunet_fold0_final_test_metrics` | 0.1068 | 0.6849 | 0.3034 | - | 0.0113 | 0.0724 | `nnunet_fold0_final_test_metrics.json` |
| 20 | `deeplabv3_resnet101_highpower_v1` | 0.1041 | 0.5315 | 0.2662 | 0.0638 | 0.0102 | 0.2043 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 21 | `nnunet_fold0_test_metrics` | 0.0979 | 0.6806 | 0.2844 | - | 0.0108 | 0.0788 | `nnunet_fold0_test_metrics.json` |
| 22 | `swin3d_adapter_fast_v1` | 0.0621 | 0.5306 | 0.1846 | 0.0334 | 0.0049 | 0.6553 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |
| 23 | `fast_dev` | 0.0000 | 0.7895 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | `fast_dev/test_metrics_presence_scan.json` |
| 24 | `fast_dev_posmetric` | 0.0000 | 0.7368 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | `fast_dev_posmetric/test_metrics_presence_scan.json` |
| 25 | `baseline_full` | 0.0000 | 0.5727 | 0.2110 | 0.0000 | 0.0000 | 0.0000 | `baseline_full/test_metrics_presence_scan.json` |
| 26 | `swin3d_adapter_dnpplus_smoke_v1` | 0.0000 | 0.0968 | 0.0345 | 0.0000 | 0.0000 | 0.0000 | `swin3d_adapter_dnpplus_smoke_v1/test_metrics.json` |
| 27 | `challenge_fast` | 0.0000 | 0.9231 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | `challenge_fast/test_metrics_presence_scan.json` |
| 28 | `deeplab_fast` | 0.0000 | 0.4737 | 0.0909 | 0.0000 | 0.0000 | 0.0000 | `deeplab_fast/test_metrics_presence_scan.json` |

## B) Classification Ranking (ROC-AUC Presence)

| Rank | Run | Dice_pos | ROC-AUC | AP | IoU_pos | Precision_pos | Recall_pos | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `challenge_fast` | 0.0000 | 0.9231 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | `challenge_fast/test_metrics_presence_scan.json` |
| 2 | `blend_dual_presence_valjoint` | 0.3837 | 0.8866 | 0.7234 | - | 0.0964 | 0.5227 | `blend_dual_presence_valjoint/test_metrics.json` |
| 3 | `challenge_presence_aux_v1` | 0.3242 | 0.8794 | 0.7090 | 0.2394 | 0.3906 | 0.3338 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 4 | `challenge_dual_head_v1` | 0.3755 | 0.7974 | 0.5238 | 0.2782 | 0.1419 | 0.4514 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 5 | `fast_dev` | 0.0000 | 0.7895 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | `fast_dev/test_metrics_presence_scan.json` |
| 6 | `dg_aug_seeds/seed_2025` | 0.3780 | 0.7450 | 0.3956 | 0.2741 | 0.0964 | 0.5333 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 7 | `dg_aug_seeds/seed_42` | 0.3759 | 0.7393 | 0.3761 | 0.2722 | 0.0811 | 0.5421 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 8 | `fast_dev_posmetric` | 0.0000 | 0.7368 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | `fast_dev_posmetric/test_metrics_presence_scan.json` |
| 9 | `dg_aug_seeds/seed_1337` | 0.3766 | 0.7352 | 0.3807 | 0.2719 | 0.0770 | 0.5613 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 10 | `final_round_modern_v2` | 0.2455 | 0.6878 | 0.3624 | 0.1654 | 0.0372 | 0.4702 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 11 | `nnunet_fold0_final_test_metrics` | 0.1068 | 0.6849 | 0.3034 | - | 0.0113 | 0.0724 | `nnunet_fold0_final_test_metrics.json` |
| 12 | `nnunet_fold0_test_metrics` | 0.0979 | 0.6806 | 0.2844 | - | 0.0108 | 0.0788 | `nnunet_fold0_test_metrics.json` |
| 13 | `deeplabv3_resnet101_full_like_v2` | 0.3605 | 0.6331 | 0.2401 | 0.2616 | 0.0499 | 0.5300 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 14 | `swin3d_d1_adapter_20260214` | 0.1177 | 0.6240 | 0.2424 | 0.0672 | 0.0193 | 0.3441 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 15 | `swin3d_adapter_baseline_like_v2` | 0.1398 | 0.6082 | 0.2387 | 0.0815 | 0.0209 | 0.3213 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 16 | `swin3d_adapter_dnpplus_fullsend_v2` | 0.1506 | 0.5970 | 0.2759 | 0.0897 | 0.0089 | 0.4472 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 17 | `swin3d_adapter_rgb_v3` | 0.1365 | 0.5917 | 0.2033 | 0.0796 | 0.0188 | 0.3015 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 18 | `swin3d_adapter_baseline_like_long_v4` | 0.1551 | 0.5850 | 0.2256 | 0.0927 | 0.0195 | 0.2957 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 19 | `baseline_full` | 0.0000 | 0.5727 | 0.2110 | 0.0000 | 0.0000 | 0.0000 | `baseline_full/test_metrics_presence_scan.json` |
| 20 | `swin3d_adapter_dnpplus_v1` | 0.1426 | 0.5668 | 0.1943 | 0.0844 | 0.0164 | 0.3357 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 21 | `deeplabv3_resnet101_highpower_v1` | 0.1041 | 0.5315 | 0.2662 | 0.0638 | 0.0102 | 0.2043 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 22 | `swin3d_adapter_fast_v1` | 0.0621 | 0.5306 | 0.1846 | 0.0334 | 0.0049 | 0.6553 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |
| 23 | `deeplab_fast` | 0.0000 | 0.4737 | 0.0909 | 0.0000 | 0.0000 | 0.0000 | `deeplab_fast/test_metrics_presence_scan.json` |
| 24 | `challenge_fast_v2` | 0.1285 | 0.2716 | 0.0309 | 0.0865 | 0.0020 | 0.2284 | `challenge_fast_v2/test_metrics_presence_scan.json` |
| 25 | `swin3d_adapter_dnpplus_smoke_v1` | 0.0000 | 0.0968 | 0.0345 | 0.0000 | 0.0000 | 0.0000 | `swin3d_adapter_dnpplus_smoke_v1/test_metrics.json` |

## C) Classification Ranking (AP Presence)

| Rank | Run | Dice_pos | ROC-AUC | AP | IoU_pos | Precision_pos | Recall_pos | Source |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `blend_dual_presence_valjoint` | 0.3837 | 0.8866 | 0.7234 | - | 0.0964 | 0.5227 | `blend_dual_presence_valjoint/test_metrics.json` |
| 2 | `challenge_presence_aux_v1` | 0.3242 | 0.8794 | 0.7090 | 0.2394 | 0.3906 | 0.3338 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 3 | `challenge_dual_head_v1` | 0.3755 | 0.7974 | 0.5238 | 0.2782 | 0.1419 | 0.4514 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 4 | `dg_aug_seeds/seed_2025` | 0.3780 | 0.7450 | 0.3956 | 0.2741 | 0.0964 | 0.5333 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 5 | `dg_aug_seeds/seed_1337` | 0.3766 | 0.7352 | 0.3807 | 0.2719 | 0.0770 | 0.5613 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 6 | `dg_aug_seeds/seed_42` | 0.3759 | 0.7393 | 0.3761 | 0.2722 | 0.0811 | 0.5421 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 7 | `final_round_modern_v2` | 0.2455 | 0.6878 | 0.3624 | 0.1654 | 0.0372 | 0.4702 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 8 | `nnunet_fold0_final_test_metrics` | 0.1068 | 0.6849 | 0.3034 | - | 0.0113 | 0.0724 | `nnunet_fold0_final_test_metrics.json` |
| 9 | `nnunet_fold0_test_metrics` | 0.0979 | 0.6806 | 0.2844 | - | 0.0108 | 0.0788 | `nnunet_fold0_test_metrics.json` |
| 10 | `swin3d_adapter_dnpplus_fullsend_v2` | 0.1506 | 0.5970 | 0.2759 | 0.0897 | 0.0089 | 0.4472 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 11 | `deeplabv3_resnet101_highpower_v1` | 0.1041 | 0.5315 | 0.2662 | 0.0638 | 0.0102 | 0.2043 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 12 | `challenge_fast` | 0.0000 | 0.9231 | 0.2500 | 0.0000 | 0.0000 | 0.0000 | `challenge_fast/test_metrics_presence_scan.json` |
| 13 | `swin3d_d1_adapter_20260214` | 0.1177 | 0.6240 | 0.2424 | 0.0672 | 0.0193 | 0.3441 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 14 | `deeplabv3_resnet101_full_like_v2` | 0.3605 | 0.6331 | 0.2401 | 0.2616 | 0.0499 | 0.5300 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 15 | `swin3d_adapter_baseline_like_v2` | 0.1398 | 0.6082 | 0.2387 | 0.0815 | 0.0209 | 0.3213 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 16 | `swin3d_adapter_baseline_like_long_v4` | 0.1551 | 0.5850 | 0.2256 | 0.0927 | 0.0195 | 0.2957 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 17 | `baseline_full` | 0.0000 | 0.5727 | 0.2110 | 0.0000 | 0.0000 | 0.0000 | `baseline_full/test_metrics_presence_scan.json` |
| 18 | `swin3d_adapter_rgb_v3` | 0.1365 | 0.5917 | 0.2033 | 0.0796 | 0.0188 | 0.3015 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 19 | `fast_dev` | 0.0000 | 0.7895 | 0.2000 | 0.0000 | 0.0000 | 0.0000 | `fast_dev/test_metrics_presence_scan.json` |
| 20 | `swin3d_adapter_dnpplus_v1` | 0.1426 | 0.5668 | 0.1943 | 0.0844 | 0.0164 | 0.3357 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 21 | `swin3d_adapter_fast_v1` | 0.0621 | 0.5306 | 0.1846 | 0.0334 | 0.0049 | 0.6553 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |
| 22 | `fast_dev_posmetric` | 0.0000 | 0.7368 | 0.1667 | 0.0000 | 0.0000 | 0.0000 | `fast_dev_posmetric/test_metrics_presence_scan.json` |
| 23 | `deeplab_fast` | 0.0000 | 0.4737 | 0.0909 | 0.0000 | 0.0000 | 0.0000 | `deeplab_fast/test_metrics_presence_scan.json` |
| 24 | `swin3d_adapter_dnpplus_smoke_v1` | 0.0000 | 0.0968 | 0.0345 | 0.0000 | 0.0000 | 0.0000 | `swin3d_adapter_dnpplus_smoke_v1/test_metrics.json` |
| 25 | `challenge_fast_v2` | 0.1285 | 0.2716 | 0.0309 | 0.0865 | 0.0020 | 0.2284 | `challenge_fast_v2/test_metrics_presence_scan.json` |

## D) Combined Rank (Mean Rank across Dice + ROC-AUC + AP)

| Rank | Run | Mean Rank | Metrics Used | Dice_pos | ROC-AUC | AP | Source |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | `blend_dual_presence_valjoint` | 1.33 | 3 | 0.3837 | 0.8866 | 0.7234 | `blend_dual_presence_valjoint/test_metrics.json` |
| 2 | `dg_aug_seeds/seed_2025` | 4.00 | 3 | 0.3780 | 0.7450 | 0.3956 | `dg_aug_seeds/seed_2025/test_metrics_tta_hv_thr025.json` |
| 3 | `challenge_dual_head_v1` | 4.00 | 3 | 0.3755 | 0.7974 | 0.5238 | `challenge_dual_head_v1/test_metrics_cls_hv.json` |
| 4 | `challenge_presence_aux_v1` | 4.67 | 3 | 0.3242 | 0.8794 | 0.7090 | `challenge_presence_aux_v1/test_metrics_presence_hv.json` |
| 5 | `dg_aug_seeds/seed_1337` | 5.67 | 3 | 0.3766 | 0.7352 | 0.3807 | `dg_aug_seeds/seed_1337/test_metrics_tta_hv_thr025.json` |
| 6 | `dg_aug_seeds/seed_42` | 5.67 | 3 | 0.3759 | 0.7393 | 0.3761 | `dg_aug_seeds/seed_42/test_metrics_tta_hv_thr025.json` |
| 7 | `deeplab_full` | 6.00 | 1 | 0.3671 | - | - | `deeplab_full/test_metrics_tta_hv_thr025.json` |
| 8 | `baseline_posmetric` | 8.00 | 1 | 0.3505 | - | - | `baseline_posmetric/test_metrics.json` |
| 9 | `final_round_modern_v2` | 9.33 | 3 | 0.2455 | 0.6878 | 0.3624 | `final_round_modern_v2/test_metrics_tta_hv_thr035_cls.json` |
| 10 | `challenge_finetune_v1` | 10.00 | 1 | 0.3125 | - | - | `challenge_finetune_v1/test_metrics.json` |
| 11 | `deeplabv3_resnet101_full_like_v2` | 11.33 | 3 | 0.3605 | 0.6331 | 0.2401 | `deeplabv3_resnet101_full_like_v2/test_metrics_tuned_hv.json` |
| 12 | `nnunet_fold0_final_test_metrics` | 12.67 | 3 | 0.1068 | 0.6849 | 0.3034 | `nnunet_fold0_final_test_metrics.json` |
| 13 | `swin3d_adapter_dnpplus_fullsend_v2` | 13.00 | 3 | 0.1506 | 0.5970 | 0.2759 | `swin3d_adapter_dnpplus_fullsend_v2/test_metrics_tuned.json` |
| 14 | `challenge_fast` | 13.33 | 3 | 0.0000 | 0.9231 | 0.2500 | `challenge_fast/test_metrics_presence_scan.json` |
| 15 | `nnunet_fold0_test_metrics` | 14.00 | 3 | 0.0979 | 0.6806 | 0.2844 | `nnunet_fold0_test_metrics.json` |
| 16 | `swin3d_adapter_baseline_like_v2` | 15.00 | 3 | 0.1398 | 0.6082 | 0.2387 | `swin3d_adapter_baseline_like_v2/test_metrics.json` |
| 17 | `swin3d_d1_adapter_20260214` | 15.00 | 3 | 0.1177 | 0.6240 | 0.2424 | `swin3d_d1_adapter_20260214/test_metrics.json` |
| 18 | `swin3d_adapter_baseline_like_long_v4` | 15.33 | 3 | 0.1551 | 0.5850 | 0.2256 | `swin3d_adapter_baseline_like_long_v4/test_metrics_tuned.json` |
| 19 | `fast_dev` | 15.67 | 3 | 0.0000 | 0.7895 | 0.2000 | `fast_dev/test_metrics_presence_scan.json` |
| 20 | `swin3d_adapter_rgb_v3` | 17.00 | 3 | 0.1365 | 0.5917 | 0.2033 | `swin3d_adapter_rgb_v3/test_metrics_tuned.json` |
| 21 | `deeplabv3_resnet101_highpower_v1` | 17.33 | 3 | 0.1041 | 0.5315 | 0.2662 | `deeplabv3_resnet101_highpower_v1/test_metrics.json` |
| 22 | `swin3d_adapter_dnpplus_v1` | 18.00 | 3 | 0.1426 | 0.5668 | 0.1943 | `swin3d_adapter_dnpplus_v1/test_metrics_tuned.json` |
| 23 | `fast_dev_posmetric` | 18.00 | 3 | 0.0000 | 0.7368 | 0.1667 | `fast_dev_posmetric/test_metrics_presence_scan.json` |
| 24 | `baseline_full` | 20.33 | 3 | 0.0000 | 0.5727 | 0.2110 | `baseline_full/test_metrics_presence_scan.json` |
| 25 | `swin3d_adapter_fast_v1` | 21.67 | 3 | 0.0621 | 0.5306 | 0.1846 | `swin3d_adapter_fast_v1/test_metrics_tuned.json` |
| 26 | `challenge_fast_v2` | 22.00 | 3 | 0.1285 | 0.2716 | 0.0309 | `challenge_fast_v2/test_metrics_presence_scan.json` |
| 27 | `deeplab_fast` | 24.67 | 3 | 0.0000 | 0.4737 | 0.0909 | `deeplab_fast/test_metrics_presence_scan.json` |
| 28 | `swin3d_adapter_dnpplus_smoke_v1` | 25.00 | 3 | 0.0000 | 0.0968 | 0.0345 | `swin3d_adapter_dnpplus_smoke_v1/test_metrics.json` |
