"""30-pair M3SVD test-set validation for source-referenced temporal quality."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from cursor_rgbt.fusion_temporal_quality import (
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
    temporal_quality_scores,
)
from run_fusion_temporal_quality_external import (
    ABLATIONS,
    BENIGN_FAMILIES,
    EVALUATION_METHODS,
    FUSION_METHODS,
    METHODS,
    PROPOSED_COMPONENTS,
    PROPOSED_WEIGHTS,
    ROOT,
    SEVERITY_COMPONENTS,
    SEVERITY_DIRECT_WEIGHT,
    SEVERITY_PROPOSED_WEIGHT,
    SEVERITY_WEIGHTS,
    STANDARD_BASELINES,
    apply_composite,
    clean_threshold,
    fit_composite_calibration,
    summarize,
)


M3SVD = ROOT / "data/raw/M3SVD/full_test/test"
ALL_IDS = (
    "0111_1716",
    "0111_1753",
    "0111_1803",
    "0112_1705",
    "0112_1707",
    "0112_1722",
    "0112_1732",
    "0113_1647",
    "0113_1714",
    "0114_1537",
    "0114_1551",
    "0114_1609",
    "0114_1611",
    "0115_1829",
    "0115_1831",
    "0115_1834",
    "0115_1847",
    "0117_1605",
    "0117_1620",
    "0118_1803",
    "0118_1904",
    "0118_1913",
    "1204_1139",
    "1207_1712",
    "1207_1739",
    "1208_1654",
    "1208_1711",
    "1208_1717",
    "1230_1154",
    "1230_1202",
)
CALIBRATION_IDS = ALL_IDS[:10]
TEST_IDS = ALL_IDS[10:]
CALIBRATION_FAMILIES = ("global_gain", "global_weight")
TEST_FAMILIES = ("local_weight", "aperiodic_gain", "local_gain", "patch_lag")
SEVERITIES = (0.025, 0.06, 0.11)
WINDOW_LENGTH = 20
WINDOW_STARTS = (0, 50, 100)


def load_video(path: Path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (320, 256), interpolation=cv2.INTER_AREA))
    capture.release()
    if len(frames) < WINDOW_LENGTH:
        raise RuntimeError(f"{path} contains only {len(frames)} frames")
    return frames


def collect(video_ids, families):
    rows = []
    for video_id in video_ids:
        visible = load_video(M3SVD / "visible_Enhance" / f"{video_id}.mp4")
        thermal = load_video(M3SVD / "infrared_Enhance" / f"{video_id}.mp4")
        starts = [
            start for start in WINDOW_STARTS if start + WINDOW_LENGTH <= len(visible)
        ]
        for start in starts:
            first = visible[start : start + WINDOW_LENGTH]
            second = thermal[start : start + WINDOW_LENGTH]
            for fusion_method in FUSION_METHODS:
                clean = fuse_frames(first, second, fusion_method)
                clean_scores = temporal_quality_scores(first, second, clean)
                rows.append(
                    {
                        "video_id": video_id,
                        "start": start,
                        "fusion_method": fusion_method,
                        "family": "clean",
                        "severity": 0.0,
                        "distorted": False,
                        **{
                            method: float(getattr(clean_scores, method))
                            for method in METHODS
                        },
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
                        scores = temporal_quality_scores(
                            varied_first, varied_second, varied_clean
                        )
                        rows.append(
                            {
                                "video_id": video_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": False,
                                **{
                                    method: float(getattr(scores, method))
                                    for method in METHODS
                                },
                            }
                        )
                for family in families:
                    for severity in SEVERITIES:
                        altered = inject_temporal_artifact(
                            clean, first, second, family, severity
                        )
                        scores = temporal_quality_scores(first, second, altered)
                        rows.append(
                            {
                                "video_id": video_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": True,
                                **{
                                    method: float(getattr(scores, method))
                                    for method in METHODS
                                },
                            }
                        )
        print(f"processed {video_id}")
    return rows


def main() -> None:
    calibration_rows = collect(CALIBRATION_IDS, CALIBRATION_FAMILIES)
    composite_calibration = fit_composite_calibration(calibration_rows)
    apply_composite(calibration_rows, composite_calibration)
    thresholds = {
        method: {
            fusion_method: clean_threshold(calibration_rows, method, fusion_method)
            for fusion_method in FUSION_METHODS
        }
        for method in EVALUATION_METHODS
    }
    test_rows = collect(TEST_IDS, TEST_FAMILIES)
    apply_composite(test_rows, composite_calibration)
    test = summarize(test_rows, thresholds)
    proposed = test["proposed"]
    detection = test["artifact_detection_score"]
    severity = test["severity_ranking_score"]
    best_baseline_auc = max(
        test[name]["macro_fusion_auroc"] for name in STANDARD_BASELINES
    )
    best_baseline_rank = max(
        test[name]["severity_spearman"] for name in STANDARD_BASELINES
    )
    best_ablation_auc = max(test[name]["macro_fusion_auroc"] for name in ABLATIONS)
    best_ablation_rank = max(test[name]["severity_spearman"] for name in ABLATIONS)
    dual_gate = {
        "detection_macro_auroc_at_least_0.90": detection["macro_fusion_auroc"] >= 0.90,
        "detection_worst_fusion_auroc_at_least_0.85": detection["worst_fusion_auroc"] >= 0.85,
        "detection_accuracy_at_least_0.85": detection["calibrated_accuracy"] >= 0.85,
        "detection_fpr_at_most_0.10": detection["calibrated_false_positive_rate"] <= 0.10,
        "severity_spearman_at_least_0.85": severity["severity_spearman"] >= 0.85,
    }
    report = {
        "status": "m3svd_30_pair_test_validation",
        "calibration_ids": list(CALIBRATION_IDS),
        "test_ids": list(TEST_IDS),
        "calibration_families": list(CALIBRATION_FAMILIES),
        "test_families": list(TEST_FAMILIES),
        "benign_source_variation_families": list(BENIGN_FAMILIES),
        "proposed_components": list(PROPOSED_COMPONENTS),
        "proposed_weights": PROPOSED_WEIGHTS,
        "severity_components": list(SEVERITY_COMPONENTS),
        "severity_weights": SEVERITY_WEIGHTS,
        "severity_direct_weight": SEVERITY_DIRECT_WEIGHT,
        "severity_proposed_weight": SEVERITY_PROPOSED_WEIGHT,
        "thresholds": thresholds,
        "test": test,
        "dual_output": {
            "detection_method": "artifact_detection_score",
            "severity_method": "severity_ranking_score",
            "detection": detection,
            "severity": severity,
        },
        "best_standard_baseline_auroc": best_baseline_auc,
        "best_standard_baseline_spearman": best_baseline_rank,
        "best_ablation_auroc": best_ablation_auc,
        "best_ablation_spearman": best_ablation_rank,
        "dual_gate": dual_gate,
        "passes_dual_gate": all(dual_gate.values()),
        "proposed": proposed,
        "raw_rows": {"calibration": calibration_rows, "test": test_rows},
    }
    output = ROOT / "results/fusion_temporal_quality_m3svd_full_test.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
