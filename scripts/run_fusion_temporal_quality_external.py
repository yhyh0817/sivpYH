"""Cross-source validation on six independent VidLLVIP source videos."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cursor_rgbt.fusion_temporal_quality import (
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
    temporal_quality_scores,
)


ROOT = Path(__file__).resolve().parents[1]
VIDEOS = ROOT / "data/raw/VidLLVIP_subset"
METHODS = (
    "proposed",
    "source_conditioned_residual",
    "reliability_gated_residual",
    "motion_operator_drift",
    "multiscale_source_residual",
    "static_operator_residual",
    "operator_instability",
    "span_residual",
    "temporal_difference",
    "adjacent_dissimilarity",
    "source_normalized_change",
    "global_weight_drift",
    "flow_warp_error",
    "temcoco_flowd_visible",
    "temcoco_feacd_sobel",
)
DERIVED_METHODS = (
    "artifact_detection_score",
    "severity_ranking_score",
)
EVALUATION_METHODS = METHODS + DERIVED_METHODS
THRESHOLD_QUANTILES = {
    "artifact_detection_score": 1.0,
}
PROPOSED_COMPONENTS = (
    "source_conditioned_residual",
    "static_operator_residual",
)
PROPOSED_WEIGHTS = {
    "source_conditioned_residual": 1.0,
    "static_operator_residual": 2.0,
}
SEVERITY_COMPONENTS = (
    "span_residual",
    "global_weight_drift",
    "source_conditioned_residual",
    "static_operator_residual",
)
SEVERITY_WEIGHTS = {
    "span_residual": 4.0,
    "global_weight_drift": 2.0,
    "source_conditioned_residual": 1.0,
    "static_operator_residual": 0.25,
}
SEVERITY_DIRECT_WEIGHT = 0.80
SEVERITY_PROPOSED_WEIGHT = 0.20
ABLATIONS = (
    "source_conditioned_residual",
    "reliability_gated_residual",
    "motion_operator_drift",
    "multiscale_source_residual",
    "static_operator_residual",
    "operator_instability",
    "span_residual",
)
STANDARD_BASELINES = (
    "temporal_difference",
    "adjacent_dissimilarity",
    "source_normalized_change",
    "global_weight_drift",
    "flow_warp_error",
    "temcoco_flowd_visible",
    "temcoco_feacd_sobel",
)
FUSION_METHODS = ("average", "maximum")
BENIGN_FAMILIES = ("source_global_gain", "source_local_gain", "source_cross_exposure")
WINDOW_STARTS = (0, 50, 100)
WINDOW_LENGTH = 20


def load_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (320, 256), interpolation=cv2.INTER_AREA))
    capture.release()
    if len(frames) != 125:
        raise RuntimeError(f"{path} contains {len(frames)} frames, expected 125")
    return frames


def collect_source(source_id: int, families, severities, benign_families=()):
    name = f"{source_id:02d}_0000_0005.mp4"
    visible = load_video(VIDEOS / "vi" / name)
    thermal = load_video(VIDEOS / "ir" / name)
    rows = []
    for start in WINDOW_STARTS:
        first = visible[start : start + WINDOW_LENGTH]
        second = thermal[start : start + WINDOW_LENGTH]
        for fusion_method in FUSION_METHODS:
            clean = fuse_frames(first, second, fusion_method)
            clean_scores = temporal_quality_scores(first, second, clean)
            rows.append(
                {
                    "source_id": source_id,
                    "start": start,
                    "fusion_method": fusion_method,
                    "family": "clean",
                    "severity": 0.0,
                    "distorted": False,
                    **{method: float(getattr(clean_scores, method)) for method in METHODS},
                }
            )
            for family in benign_families:
                for severity in severities:
                    varied_first, varied_second = inject_source_variation(
                        first, second, family, severity
                    )
                    varied_clean = fuse_frames(varied_first, varied_second, fusion_method)
                    scores = temporal_quality_scores(
                        varied_first, varied_second, varied_clean
                    )
                    rows.append(
                        {
                            "source_id": source_id,
                            "start": start,
                            "fusion_method": fusion_method,
                            "family": family,
                            "severity": severity,
                            "distorted": False,
                            **{method: float(getattr(scores, method)) for method in METHODS},
                        }
                    )
            for family in families:
                for severity in severities:
                    altered = inject_temporal_artifact(
                        clean, first, second, family, severity
                    )
                    scores = temporal_quality_scores(first, second, altered)
                    rows.append(
                        {
                            "source_id": source_id,
                            "start": start,
                            "fusion_method": fusion_method,
                            "family": family,
                            "severity": severity,
                            "distorted": True,
                            **{method: float(getattr(scores, method)) for method in METHODS},
                        }
                    )
    return rows


def collect(source_ids, families, severities, benign_families=()):
    rows = []
    for source_id in source_ids:
        rows.extend(collect_source(source_id, families, severities, benign_families))
        print(f"processed source {source_id:02d}")
    return rows


def fit_composite_calibration(rows):
    calibration = {}
    calibrated_components = tuple(
        dict.fromkeys(PROPOSED_COMPONENTS + SEVERITY_COMPONENTS)
    )
    for fusion_method in FUSION_METHODS:
        selected = [row for row in rows if row["fusion_method"] == fusion_method]
        clean = [row for row in selected if not row["distorted"]]
        calibration[fusion_method] = {}
        for component in calibrated_components:
            center = float(np.median([row[component] for row in clean]))
            values = np.asarray([row[component] for row in selected])
            first, third = np.quantile(values, (0.25, 0.75))
            scale = max(float(third - first), 1e-6)
            calibration[fusion_method][component] = {
                "center": center,
                "scale": scale,
            }
    return calibration


def apply_composite(rows, calibration):
    for row in rows:
        parameters = calibration[row["fusion_method"]]
        row["artifact_detection_score"] = max(
            0.0,
            (
                row["source_conditioned_residual"]
                - parameters["source_conditioned_residual"]["center"]
            )
            / parameters["source_conditioned_residual"]["scale"],
        )
        severity_evidence = float(
            np.average(
                [
                    max(
                        0.0,
                        (row[component] - parameters[component]["center"])
                        / parameters[component]["scale"],
                    )
                    for component in SEVERITY_COMPONENTS
                ],
                weights=[
                    SEVERITY_WEIGHTS[component] for component in SEVERITY_COMPONENTS
                ],
            )
        )
        evidence = [
            PROPOSED_WEIGHTS[component]
            * max(
                0.0,
                (row[component] - parameters[component]["center"])
                / parameters[component]["scale"],
            )
            for component in PROPOSED_COMPONENTS
        ]
        row["proposed"] = float(np.mean(evidence))
        row["severity_ranking_score"] = float(
            SEVERITY_DIRECT_WEIGHT * severity_evidence
            + SEVERITY_PROPOSED_WEIGHT * row["proposed"]
        )


def clean_threshold(rows, method, fusion_method):
    selected = [row for row in rows if row["fusion_method"] == fusion_method]
    clean_scores = np.asarray([row[method] for row in selected if not row["distorted"]])
    return float(np.quantile(clean_scores, THRESHOLD_QUANTILES.get(method, 0.95)))


def safe_spearman(labels, scores) -> float:
    statistic = float(spearmanr(labels, scores).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def summarize(rows, thresholds):
    result = {}
    for method in EVALUATION_METHODS:
        per_fusion_auc = {}
        predictions = []
        truths = []
        correlations = []
        for fusion_method in FUSION_METHODS:
            selected = [row for row in rows if row["fusion_method"] == fusion_method]
            truth = np.asarray([row["distorted"] for row in selected], dtype=bool)
            scores = np.asarray([row[method] for row in selected])
            predictions.extend(scores >= thresholds[method][fusion_method])
            truths.extend(truth)
            per_fusion_auc[fusion_method] = float(roc_auc_score(truth, scores))
            for family in sorted({row["family"] for row in selected if row["distorted"]}):
                family_rows = [
                    row for row in selected if row["family"] in ("clean", family)
                ]
                correlations.append(
                    safe_spearman(
                        [row["severity"] for row in family_rows],
                        [row[method] for row in family_rows],
                    )
                )
        prediction = np.asarray(predictions, dtype=bool)
        truth = np.asarray(truths, dtype=bool)
        result[method] = {
            "macro_fusion_auroc": float(np.mean(list(per_fusion_auc.values()))),
            "worst_fusion_auroc": float(min(per_fusion_auc.values())),
            "per_fusion_auroc": per_fusion_auc,
            "calibrated_accuracy": float(np.mean(prediction == truth)),
            "calibrated_false_positive_rate": float(np.mean(prediction[~truth])),
            "severity_spearman": float(np.mean(correlations)),
        }
    return result


def main():
    known_families = ("global_gain", "global_weight")
    unseen_families = ("local_weight", "aperiodic_gain", "patch_lag")
    severities = (0.04, 0.09, 0.16)
    fit_rows = collect((1, 2), known_families, severities, BENIGN_FAMILIES)
    composite_calibration = fit_composite_calibration(fit_rows)
    apply_composite(fit_rows, composite_calibration)
    thresholds = {
        method: {
            fusion_method: clean_threshold(fit_rows, method, fusion_method)
            for fusion_method in FUSION_METHODS
        }
        for method in EVALUATION_METHODS
    }
    known_rows = collect((4, 7), known_families, severities, BENIGN_FAMILIES)
    unseen_rows = collect((10, 14), unseen_families, severities, BENIGN_FAMILIES)
    apply_composite(known_rows, composite_calibration)
    apply_composite(unseen_rows, composite_calibration)
    known, unseen = summarize(known_rows, thresholds), summarize(unseen_rows, thresholds)
    proposed = unseen["proposed"]
    best_baseline_auc = max(
        unseen[name]["macro_fusion_auroc"] for name in STANDARD_BASELINES
    )
    best_baseline_rank = max(
        unseen[name]["severity_spearman"] for name in STANDARD_BASELINES
    )
    best_ablation_auc = max(
        unseen[name]["macro_fusion_auroc"] for name in ABLATIONS
    )
    best_ablation_rank = max(
        unseen[name]["severity_spearman"] for name in ABLATIONS
    )
    gate = {
        "unseen_macro_auroc_at_least_0.90": proposed["macro_fusion_auroc"] >= 0.90,
        "unseen_worst_fusion_auroc_at_least_0.85": proposed["worst_fusion_auroc"] >= 0.85,
        "unseen_accuracy_at_least_0.85": proposed["calibrated_accuracy"] >= 0.85,
        "unseen_fpr_at_most_0.10": proposed["calibrated_false_positive_rate"] <= 0.10,
        "unseen_spearman_at_least_0.85": proposed["severity_spearman"] >= 0.85,
        "auroc_margin_at_least_0.10": proposed["macro_fusion_auroc"] - best_baseline_auc >= 0.10,
        "rank_margin_at_least_0.10": proposed["severity_spearman"] - best_baseline_rank >= 0.10,
        "auroc_not_worse_than_ablation_by_0.02": proposed["macro_fusion_auroc"] >= best_ablation_auc - 0.02,
        "rank_not_worse_than_ablation_by_0.02": proposed["severity_spearman"] >= best_ablation_rank - 0.02,
    }
    report = {
        "status": "external_synthetic_artifact_validation_not_paper_evidence",
        "direction": "source-referenced temporal flicker assessment for RGB-T fusion video",
        "innovation_candidate": "motion-compensated source-conditioned fusion-operator consistency",
        "source_split": {"fit": [1, 2], "known": [4, 7], "unseen": [10, 14]},
        "benign_source_variation_families": list(BENIGN_FAMILIES),
        "window_starts": list(WINDOW_STARTS),
        "window_length": WINDOW_LENGTH,
        "proposed_components": list(PROPOSED_COMPONENTS),
        "proposed_weights": PROPOSED_WEIGHTS,
        "severity_components": list(SEVERITY_COMPONENTS),
        "severity_weights": SEVERITY_WEIGHTS,
        "severity_direct_weight": SEVERITY_DIRECT_WEIGHT,
        "severity_proposed_weight": SEVERITY_PROPOSED_WEIGHT,
        "composite_calibration": composite_calibration,
        "thresholds": thresholds,
        "known": known,
        "unseen": unseen,
        "gate": gate,
        "passes_all_gates": all(gate.values()),
        "raw_rows": {"fit": fit_rows, "known": known_rows, "unseen": unseen_rows},
    }
    output = ROOT / "results/fusion_temporal_quality_external.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
