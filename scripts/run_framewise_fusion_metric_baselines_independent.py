"""Framewise image-fusion metric instability baselines on VidLLVIP independent."""

from __future__ import annotations

import json

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cursor_rgbt.fusion_temporal_quality import (
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
)
from cursor_rgbt.temporal_sync import _gray
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
SIZE = (160, 128)
METHODS = (
    "entropy_instability",
    "std_instability",
    "spatial_frequency_instability",
    "mutual_information_instability",
    "edge_correlation_instability",
)


def entropy(frame: np.ndarray) -> float:
    hist, _ = np.histogram(frame.ravel(), bins=64, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64)
    prob /= prob.sum() + 1e-12
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def mutual_information(first: np.ndarray, second: np.ndarray) -> float:
    hist, _, _ = np.histogram2d(
        first.ravel(),
        second.ravel(),
        bins=32,
        range=((0.0, 1.0), (0.0, 1.0)),
    )
    joint = hist.astype(np.float64)
    joint /= joint.sum() + 1e-12
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / (expected[mask] + 1e-12))))


def spatial_frequency(frame: np.ndarray) -> float:
    row_frequency = np.sqrt(np.mean(np.diff(frame, axis=0) ** 2))
    col_frequency = np.sqrt(np.mean(np.diff(frame, axis=1) ** 2))
    return float(np.sqrt(row_frequency**2 + col_frequency**2))


def edge_correlation(source: np.ndarray, fused: np.ndarray) -> float:
    source_grad = cv2.magnitude(
        cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3),
    ).ravel()
    fused_grad = cv2.magnitude(
        cv2.Sobel(fused, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(fused, cv2.CV_32F, 0, 1, ksize=3),
    ).ravel()
    source_grad -= float(source_grad.mean())
    fused_grad -= float(fused_grad.mean())
    return float(
        np.dot(source_grad, fused_grad)
        / (np.linalg.norm(source_grad) * np.linalg.norm(fused_grad) + 1e-8)
    )


def instability(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.std(array) / (abs(float(np.mean(array))) + 1e-6))


def metric_scores(visible, thermal, fused) -> dict[str, float]:
    first = [_gray(frame, SIZE) for frame in visible]
    second = [_gray(frame, SIZE) for frame in thermal]
    output = [_gray(frame, SIZE) for frame in fused]
    entropy_values = [entropy(frame) for frame in output]
    std_values = [float(np.std(frame)) for frame in output]
    sf_values = [spatial_frequency(frame) for frame in output]
    mi_values = [
        mutual_information(one, target) + mutual_information(two, target)
        for one, two, target in zip(first, second, output)
    ]
    edge_values = [
        max(edge_correlation(one, target), edge_correlation(two, target))
        for one, two, target in zip(first, second, output)
    ]
    return {
        "entropy_instability": instability(entropy_values),
        "std_instability": instability(std_values),
        "spatial_frequency_instability": instability(sf_values),
        "mutual_information_instability": instability(mi_values),
        "edge_correlation_instability": instability(edge_values),
    }


def score_rows(source_ids, artifact_families):
    rows = []
    for source_id in source_ids:
        name = f"{source_id:02d}_0000_0005.mp4"
        visible = load_video(VIDEOS / "vi" / name)
        thermal = load_video(VIDEOS / "ir" / name)
        for start in WINDOW_STARTS:
            first = visible[start : start + WINDOW_LENGTH]
            second = thermal[start : start + WINDOW_LENGTH]
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
                        **metric_scores(first, second, clean),
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
                                "source_id": source_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": False,
                                **metric_scores(
                                    varied_first, varied_second, varied_clean
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
                                **metric_scores(first, second, altered),
                            }
                        )
        print(f"processed source {source_id:02d}")
    return rows


def thresholds(rows, method):
    return {
        fusion_method: float(
            np.quantile(
                [
                    row[method]
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
    result = {}
    for method in METHODS:
        predictions = []
        truths = []
        aucs = {}
        correlations = []
        for fusion_method in FUSION_METHODS:
            selected = [row for row in rows if row["fusion_method"] == fusion_method]
            truth = np.asarray([row["distorted"] for row in selected], dtype=bool)
            scores = np.asarray([row[method] for row in selected], dtype=float)
            aucs[fusion_method] = float(roc_auc_score(truth, scores))
            predictions.extend(scores >= fitted_thresholds[method][fusion_method])
            truths.extend(truth)
            for family in sorted(
                {row["family"] for row in selected if row["distorted"]}
            ):
                family_rows = [
                    row for row in selected if row["family"] in ("clean", family)
                ]
                correlations.append(
                    safe_spearman(
                        [row["severity"] for row in family_rows],
                        [row[method] for row in family_rows],
                    )
                )
        predictions = np.asarray(predictions, dtype=bool)
        truths = np.asarray(truths, dtype=bool)
        result[method] = {
            "macro_fusion_auroc": float(np.mean(list(aucs.values()))),
            "worst_fusion_auroc": float(min(aucs.values())),
            "calibrated_accuracy": float(np.mean(predictions == truths)),
            "calibrated_false_positive_rate": float(np.mean(predictions[~truths])),
            "severity_spearman": float(np.mean(correlations)),
            "per_fusion_auroc": aucs,
        }
    return result


def main() -> None:
    calibration_rows = score_rows(CALIBRATION_SOURCES, CALIBRATION_FAMILIES)
    fitted_thresholds = {
        method: thresholds(calibration_rows, method) for method in METHODS
    }
    independent_rows = score_rows(INDEPENDENT_SOURCES, INDEPENDENT_FAMILIES)
    summary = summarize(independent_rows, fitted_thresholds)
    report = {
        "status": "framewise_fusion_metric_instability_baselines",
        "calibration_sources": list(CALIBRATION_SOURCES),
        "independent_sources": list(INDEPENDENT_SOURCES),
        "calibration_families": list(CALIBRATION_FAMILIES),
        "independent_families": list(INDEPENDENT_FAMILIES),
        "benign_source_variation_families": list(BENIGN_FAMILIES),
        "methods": list(METHODS),
        "thresholds": fitted_thresholds,
        "summary": summary,
        "raw_rows": {
            "calibration": calibration_rows,
            "independent": independent_rows,
        },
    }
    output = ROOT / "results/framewise_fusion_metric_baselines_independent.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
