"""Real fusion-output pilot using TemCoCo's rendered 1207_1739 GIF example."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cursor_rgbt.fusion_temporal_quality import (
    inject_temporal_artifact,
    temporal_quality_scores,
)
from run_fusion_temporal_quality_external import ROOT


GIF = ROOT / "models/TemCoCo/examples/1207_1739.gif"
WINDOW_LENGTH = 20
CALIBRATION_STARTS = (0, 40)
TEST_STARTS = (80, 120, 160)
TEST_FAMILIES = ("local_weight", "aperiodic_gain", "local_gain", "patch_lag")
SEVERITIES = (0.025, 0.06, 0.11)
METHODS = (
    "real_detection_score",
    "real_severity_score",
    "proposed",
    "source_conditioned_residual",
    "span_residual",
    "static_operator_residual",
    "temporal_difference",
    "temcoco_flowd_visible",
    "temcoco_feacd_sobel",
)
CALIBRATED_COMPONENTS = (
    "source_conditioned_residual",
    "span_residual",
    "global_weight_drift",
    "static_operator_residual",
    "proposed",
)
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


def crop_temcoco_panels(path: Path):
    capture = cv2.VideoCapture(str(path))
    visible, thermal, fused = [], [], []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        # The rendered GIF is a 4-column x 3-row comparison grid.
        # Use content crops and avoid the label bands at the bottom of panels.
        visible.append(frame[0:154, 0:200].copy())
        thermal.append(frame[0:154, 200:400].copy())
        fused.append(frame[328:482, 400:600].copy())
    capture.release()
    if len(fused) < max(TEST_STARTS) + WINDOW_LENGTH:
        raise RuntimeError(f"{path} contains only {len(fused)} usable frames")
    return visible, thermal, fused


def score_sequence(visible, thermal, fused) -> dict[str, float]:
    scores = temporal_quality_scores(visible, thermal, fused, size=(160, 128))
    return {name: float(getattr(scores, name)) for name in scores.__dataclass_fields__}


def collect_rows(starts, visible, thermal, fused, distorted: bool):
    rows = []
    for start in starts:
        first = visible[start : start + WINDOW_LENGTH]
        second = thermal[start : start + WINDOW_LENGTH]
        clean = fused[start : start + WINDOW_LENGTH]
        rows.append(
            {
                "start": start,
                "family": "clean",
                "severity": 0.0,
                "distorted": False,
                **score_sequence(first, second, clean),
            }
        )
        if distorted:
            for family in TEST_FAMILIES:
                for severity in SEVERITIES:
                    altered = inject_temporal_artifact(
                        clean, first, second, family, severity
                    )
                    rows.append(
                        {
                            "start": start,
                            "family": family,
                            "severity": severity,
                            "distorted": True,
                            **score_sequence(first, second, altered),
                        }
                    )
    return rows


def fit_clean_calibration(rows):
    calibration = {}
    clean = [row for row in rows if not row["distorted"]]
    for component in CALIBRATED_COMPONENTS:
        values = np.asarray([row[component] for row in clean], dtype=float)
        first, third = np.quantile(values, (0.25, 0.75))
        scale = max(float(third - first), 1e-6)
        calibration[component] = {
            "center": float(np.median(values)),
            "scale": scale,
        }
    return calibration


def z(row, calibration, component):
    parameters = calibration[component]
    return max(0.0, (row[component] - parameters["center"]) / parameters["scale"])


def apply_real_scores(rows, calibration):
    for row in rows:
        row["real_detection_score"] = z(
            row, calibration, "source_conditioned_residual"
        )
        severity_direct = float(
            np.average(
                [z(row, calibration, component) for component in SEVERITY_COMPONENTS],
                weights=[SEVERITY_WEIGHTS[component] for component in SEVERITY_COMPONENTS],
            )
        )
        proposed_z = z(row, calibration, "proposed")
        row["real_severity_score"] = float(0.80 * severity_direct + 0.20 * proposed_z)


def threshold(rows, method):
    clean_scores = [row[method] for row in rows if not row["distorted"]]
    return float(np.quantile(clean_scores, 1.0))


def safe_spearman(labels, scores) -> float:
    statistic = float(spearmanr(labels, scores).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def summarize(rows, thresholds):
    result = {}
    truth = np.asarray([row["distorted"] for row in rows], dtype=bool)
    for method in METHODS:
        scores = np.asarray([row[method] for row in rows], dtype=float)
        prediction = scores >= thresholds[method]
        correlations = []
        for family in sorted({row["family"] for row in rows if row["distorted"]}):
            family_rows = [row for row in rows if row["family"] in ("clean", family)]
            correlations.append(
                safe_spearman(
                    [row["severity"] for row in family_rows],
                    [row[method] for row in family_rows],
                )
            )
        result[method] = {
            "auroc": float(roc_auc_score(truth, scores)),
            "accuracy": float(np.mean(prediction == truth)),
            "false_positive_rate": float(np.mean(prediction[~truth])),
            "severity_spearman": float(np.mean(correlations)),
        }
    return result


def main() -> None:
    visible, thermal, fused = crop_temcoco_panels(GIF)
    calibration_rows = collect_rows(
        CALIBRATION_STARTS, visible, thermal, fused, distorted=False
    )
    test_rows = collect_rows(TEST_STARTS, visible, thermal, fused, distorted=True)
    calibration = fit_clean_calibration(calibration_rows)
    apply_real_scores(calibration_rows, calibration)
    apply_real_scores(test_rows, calibration)
    fitted_thresholds = {method: threshold(calibration_rows, method) for method in METHODS}
    summary = summarize(test_rows, fitted_thresholds)
    report = {
        "status": "real_fusion_temcoco_gif_pilot",
        "source": str(GIF),
        "note": "Uses VI/IR/Ours crops from TemCoCo's rendered comparison GIF, not raw model output.",
        "frame_count": len(fused),
        "crop_layout": {
            "visible": "x=0:200,y=0:154",
            "thermal": "x=200:400,y=0:154",
            "ours_fused": "x=400:600,y=328:482",
        },
        "calibration_starts": list(CALIBRATION_STARTS),
        "test_starts": list(TEST_STARTS),
        "test_families": list(TEST_FAMILIES),
        "severities": list(SEVERITIES),
        "calibration": calibration,
        "thresholds": fitted_thresholds,
        "summary": summary,
        "raw_rows": {
            "calibration": calibration_rows,
            "test": test_rows,
        },
    }
    output = ROOT / "results/real_fusion_temcoco_gif_pilot.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
