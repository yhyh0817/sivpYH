"""Full VidLLVIP independent ResNet-18 feaCD baseline probe."""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cursor_rgbt.fusion_temporal_quality import (
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
)
from cursor_rgbt.temcoco_baselines import (
    feature_change_direction_distance,
    resnet18_features,
)
from run_fusion_temporal_quality_external import (
    BENIGN_FAMILIES,
    FUSION_METHODS,
    ROOT,
    VIDEOS,
    WINDOW_LENGTH,
    WINDOW_STARTS,
    load_video,
)


CALIBRATION_SOURCES = (1, 2, 4, 5, 7, 8, 10, 14)
INDEPENDENT_SOURCES = (3, 6, 9, 11, 12, 13)
CALIBRATION_FAMILIES = ("global_gain", "global_weight")
INDEPENDENT_FAMILIES = ("local_weight", "aperiodic_gain", "local_gain", "patch_lag")
SEVERITIES = (0.025, 0.06, 0.11)
METHOD = "resnet18_feacd_visible"


def feacd(source_features, fused) -> float:
    return feature_change_direction_distance(source_features, resnet18_features(fused))


def score_rows(source_ids, artifact_families):
    rows = []
    for source_id in source_ids:
        name = f"{source_id:02d}_0000_0005.mp4"
        visible = load_video(VIDEOS / "vi" / name)
        thermal = load_video(VIDEOS / "ir" / name)
        for start in WINDOW_STARTS:
            first = visible[start : start + WINDOW_LENGTH]
            second = thermal[start : start + WINDOW_LENGTH]
            first_features = resnet18_features(first)
            benign_feature_cache = {}
            for fusion_method in FUSION_METHODS:
                clean = fuse_frames(first, second, fusion_method)
                rows.append(
                    {
                        "source_id": source_id,
                        "start": start,
                        "fusion_method": fusion_method,
                        "family": "clean",
                        "severity": 0.0,
                        "distorted": False,
                        METHOD: feacd(first_features, clean),
                    }
                )
                for family in BENIGN_FAMILIES:
                    for severity in SEVERITIES:
                        varied_first, varied_second = inject_source_variation(
                            first, second, family, severity
                        )
                        varied_clean = fuse_frames(
                            varied_first, varied_second, fusion_method
                        )
                        cache_key = (family, severity)
                        if cache_key not in benign_feature_cache:
                            benign_feature_cache[cache_key] = resnet18_features(
                                varied_first
                            )
                        rows.append(
                            {
                                "source_id": source_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": False,
                                METHOD: feacd(
                                    benign_feature_cache[cache_key],
                                    varied_clean,
                                ),
                            }
                        )
                for family in artifact_families:
                    for severity in SEVERITIES:
                        altered = inject_temporal_artifact(
                            clean, first, second, family, severity
                        )
                        rows.append(
                            {
                                "source_id": source_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": True,
                                METHOD: feacd(first_features, altered),
                            }
                        )
        print(f"processed source {source_id:02d}")
    return rows


def thresholds(rows):
    return {
        fusion_method: float(
            np.quantile(
                [
                    row[METHOD]
                    for row in rows
                    if row["fusion_method"] == fusion_method
                    and not row["distorted"]
                ],
                0.95,
            )
        )
        for fusion_method in FUSION_METHODS
    }


def safe_spearman(labels, scores) -> float:
    statistic = float(spearmanr(labels, scores).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def summarize(rows, fitted_thresholds):
    predictions = []
    truths = []
    aucs = {}
    correlations = []
    for fusion_method in FUSION_METHODS:
        selected = [row for row in rows if row["fusion_method"] == fusion_method]
        truth = np.asarray([row["distorted"] for row in selected], dtype=bool)
        scores = np.asarray([row[METHOD] for row in selected], dtype=float)
        aucs[fusion_method] = float(roc_auc_score(truth, scores))
        predictions.extend(scores >= fitted_thresholds[fusion_method])
        truths.extend(truth)
        for family in sorted({row["family"] for row in selected if row["distorted"]}):
            family_rows = [
                row for row in selected if row["family"] in ("clean", family)
            ]
            correlations.append(
                safe_spearman(
                    [row["severity"] for row in family_rows],
                    [row[METHOD] for row in family_rows],
                )
            )
    predictions = np.asarray(predictions, dtype=bool)
    truths = np.asarray(truths, dtype=bool)
    return {
        "macro_fusion_auroc": float(np.mean(list(aucs.values()))),
        "worst_fusion_auroc": float(min(aucs.values())),
        "calibrated_accuracy": float(np.mean(predictions == truths)),
        "calibrated_false_positive_rate": float(np.mean(predictions[~truths])),
        "severity_spearman": float(np.mean(correlations)),
        "per_fusion_auroc": aucs,
    }


def main() -> None:
    calibration_rows = score_rows(CALIBRATION_SOURCES, CALIBRATION_FAMILIES)
    fitted_thresholds = thresholds(calibration_rows)
    independent_rows = score_rows(INDEPENDENT_SOURCES, INDEPENDENT_FAMILIES)
    summary = summarize(independent_rows, fitted_thresholds)
    report = {
        "status": "full_independent_probe_not_official_temcoco_reproduction",
        "metric": "ResNet-18 visible-source feaCD",
        "calibration_sources": list(CALIBRATION_SOURCES),
        "independent_sources": list(INDEPENDENT_SOURCES),
        "calibration_families": list(CALIBRATION_FAMILIES),
        "independent_families": list(INDEPENDENT_FAMILIES),
        "benign_source_variation_families": list(BENIGN_FAMILIES),
        "thresholds": fitted_thresholds,
        "summary": summary,
        "raw_rows": {
            "calibration": calibration_rows,
            "independent": independent_rows,
        },
    }
    output = ROOT / "results/temcoco_resnet_feacd_independent.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
