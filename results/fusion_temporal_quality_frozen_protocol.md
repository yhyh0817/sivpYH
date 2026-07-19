# Fusion Temporal Quality Frozen Protocol

Date: 2026-07-17

Real fusion-model metric correction: 2026-07-18

## Purpose

This file freezes the current development protocol for the RGB-T fusion video
temporal quality metric.  Any future formula change after this point should be
treated as a new development round and should not reuse the current validation
numbers as locked final evidence.

## Frozen method

The method is a dual-output source-referenced temporal quality metric:

- `artifact_detection_score`
  - calibrated positive z-score of `source_conditioned_residual`
  - threshold quantile on clean/benign calibration samples: `1.00`
- `severity_ranking_score`
  - direct calibrated severity evidence:
    - `span_residual`, weight `4.0`
    - `global_weight_drift`, weight `2.0`
    - `source_conditioned_residual`, weight `1.0`
    - `static_operator_residual`, weight `0.25`
  - final robust blend:
    - `0.80 * direct calibrated severity evidence`
    - `0.20 * proposed composite`
- `proposed`
  - calibrated composite of:
    - `source_conditioned_residual`, weight `1.0`
    - `static_operator_residual`, weight `2.0`

The `patch_lag` synthetic artifact now uses:

- one-frame local lag reference;
- local lag mixture `min(0.90, 8.0 * severity)`;
- the same local mask family used in the current source code.

## Frozen validation protocols

### VidLLVIP independent

- Calibration sources: `01, 02, 04, 05, 07, 08, 10, 14`
- Independent sources: `03, 06, 09, 11, 12, 13`
- Calibration artifact families: `global_gain`, `global_weight`
- Independent artifact families: `local_weight`, `aperiodic_gain`,
  `local_gain`, `patch_lag`
- Benign source-variation families: `source_global_gain`,
  `source_local_gain`, `source_cross_exposure`
- Severities: `0.025`, `0.06`, `0.11`
- Output file: `results/fusion_temporal_quality_independent.json`

### M3SVD 12-pair subset

- Calibration IDs: `0111_1716`, `0111_1753`, `0111_1803`, `0112_1705`,
  `0112_1707`, `0112_1722`
- Test IDs: `0112_1732`, `0113_1647`, `0113_1714`, `0114_1537`,
  `0114_1551`, `0114_1609`
- Output file: `results/fusion_temporal_quality_m3svd_subset.json`

### M3SVD 30-pair full-test split

- Calibration IDs: first 10 sorted M3SVD test IDs:
  `0111_1716`, `0111_1753`, `0111_1803`, `0112_1705`, `0112_1707`,
  `0112_1722`, `0112_1732`, `0113_1647`, `0113_1714`, `0114_1537`
- Test IDs: remaining 20 sorted M3SVD test IDs:
  `0114_1551`, `0114_1609`, `0114_1611`, `0115_1829`, `0115_1831`,
  `0115_1834`, `0115_1847`, `0117_1605`, `0117_1620`, `0118_1803`,
  `0118_1904`, `0118_1913`, `1204_1139`, `1207_1712`, `1207_1739`,
  `1208_1654`, `1208_1711`, `1208_1717`, `1230_1154`, `1230_1202`
- Output file: `results/fusion_temporal_quality_m3svd_full_test.json`

## Current locked evidence

| Protocol | Detection AUROC | Detection FPR | Severity AUROC | Severity FPR | Severity Spearman |
|---|---:|---:|---:|---:|---:|
| VidLLVIP independent | 0.9977 | 0.0056 | 0.9948 | 0.0417 | 0.9262 |
| M3SVD 12-pair subset | 0.9991 | 0.0000 | 0.9926 | 0.0833 | 0.8666 |
| M3SVD 30-pair full-test | 0.9963 | 0.0083 | 0.9715 | 0.0825 | 0.8682 |

## Measured real fusion-model evidence

| Protocol | Detection AUROC | Worst AUROC | Accuracy | Detection FPR | Severity AUROC | Overall Spearman | Positive-only Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| Real fusion-model outputs | 0.9078 | 0.8808 | 0.8889 | 0.0444 | 0.8972 | 0.8176 | 0.7589 |

These measured values were confirmed by the author on 2026-07-19 and are
recorded in `results/real_fusion_model_confirmed_metrics.json` together with the
nine-model breakdown, sample-level evaluation, and pooled confusion matrix (90
positive and 90 negative samples). They supersede the rendered-GIF pilot values
as the primary real fusion-model result.

## Baseline comparison artifacts

- `results/fusion_temporal_quality_baseline_table.md`
- `results/fusion_temporal_quality_baseline_table.csv`
- `results/temcoco_resnet_feacd_independent.json`
- `results/temcoco_searaft_flowd_independent.json`
- `results/temcoco_searaft_flowd_m3svd_full_test.json`
- `results/real_fusion_model_confirmed_metrics.json`
- `results/real_fusion_temcoco_gif_pilot.json`
- `results/real_fusion_temcoco_gif_multi_pilot.json`
- `results/framewise_fusion_metric_baselines_independent.json`

The ResNet-18 feaCD result and SEA-RAFT flowD result are local probes, not
official TemCoCo metric-script reproductions.

## Rule for future work

Future additions are allowed if they do not change the frozen scoring formula
or artifact generator.  Examples:

- adding official or faithful baseline implementations;
- adding a new external dataset;
- improving manuscript tables and explanations.

If the scoring formula, artifact generator, calibration split, or validation
split changes, the result should be labeled as a new development protocol rather
than a locked final result.
