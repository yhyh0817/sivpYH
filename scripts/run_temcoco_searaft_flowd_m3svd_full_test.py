"""SEA-RAFT flowD-style baseline on the M3SVD 30-pair full-test split."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from cursor_rgbt.fusion_temporal_quality import (
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
)
from run_fusion_temporal_quality_external import (
    BENIGN_FAMILIES,
    FUSION_METHODS,
    ROOT,
)
from run_fusion_temporal_quality_m3svd_full_test import (
    ALL_IDS,
    CALIBRATION_FAMILIES,
    CALIBRATION_IDS,
    M3SVD,
    SEVERITIES,
    TEST_FAMILIES,
    TEST_IDS,
    WINDOW_LENGTH,
    WINDOW_STARTS,
)
from run_temcoco_searaft_flowd_independent import (
    METHOD,
    flowd_from_cached_flows,
    safe_spearman,
    visible_flows,
)
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


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


def score_rows(video_ids, artifact_families):
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
            flows = visible_flows(first)
            for fusion_method in FUSION_METHODS:
                clean = fuse_frames(first, second, fusion_method)
                rows.append(
                    {
                        "video_id": video_id,
                        "start": start,
                        "fusion_method": fusion_method,
                        "family": "clean",
                        "severity": 0.0,
                        "distorted": False,
                        METHOD: flowd_from_cached_flows(clean, flows),
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
                        rows.append(
                            {
                                "video_id": video_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": False,
                                METHOD: flowd_from_cached_flows(varied_clean, flows),
                            }
                        )
                for family in artifact_families:
                    for severity in SEVERITIES:
                        altered = inject_temporal_artifact(
                            clean, first, second, family, severity
                        )
                        rows.append(
                            {
                                "video_id": video_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": True,
                                METHOD: flowd_from_cached_flows(altered, flows),
                            }
                        )
        print(f"processed {video_id}")
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
    calibration_rows = score_rows(CALIBRATION_IDS, CALIBRATION_FAMILIES)
    fitted_thresholds = thresholds(calibration_rows)
    test_rows = score_rows(TEST_IDS, TEST_FAMILIES)
    summary = summarize(test_rows, fitted_thresholds)
    report = {
        "status": "searaft_flowd_visible_m3svd_30_pair_probe",
        "metric": METHOD,
        "all_ids": list(ALL_IDS),
        "calibration_ids": list(CALIBRATION_IDS),
        "test_ids": list(TEST_IDS),
        "calibration_families": list(CALIBRATION_FAMILIES),
        "test_families": list(TEST_FAMILIES),
        "benign_source_variation_families": list(BENIGN_FAMILIES),
        "thresholds": fitted_thresholds,
        "summary": summary,
        "raw_rows": {
            "calibration": calibration_rows,
            "test": test_rows,
        },
    }
    output = ROOT / "results/temcoco_searaft_flowd_m3svd_full_test.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
