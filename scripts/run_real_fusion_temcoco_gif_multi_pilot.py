"""Multi-panel real-fusion pilot from TemCoCo rendered comparison GIFs."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import ConstantInputWarning
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cursor_rgbt.fusion_temporal_quality import (
    inject_temporal_artifact,
    temporal_quality_scores,
)
from run_fusion_temporal_quality_external import ROOT


GIFS = {
    "1207_1714": ROOT / "models/TemCoCo/examples/1207_1714.gif",
    "1207_1739": ROOT / "models/TemCoCo/examples/1207_1739.gif",
}
PANEL_LAYOUT = {
    "VI": (0, 200, 0, 154),
    "IR": (200, 400, 0, 154),
    "DATFuse": (400, 600, 0, 154),
    "TGFuse": (600, 800, 0, 154),
    "DDFM": (0, 200, 164, 318),
    "CDDFuse": (200, 400, 164, 318),
    "LRRNet": (400, 600, 164, 318),
    "Diff-IF": (600, 800, 164, 318),
    "MMIF-EMMA": (0, 200, 328, 482),
    "RCVS": (200, 400, 328, 482),
    "Ours": (400, 600, 328, 482),
}
FUSION_PANELS = tuple(name for name in PANEL_LAYOUT if name not in ("VI", "IR"))
WINDOW_LENGTH = 20
STARTS = (0, 40, 80, 120, 160)
EARLY_STARTS = (0, 40)
LATE_STARTS = (80, 120, 160)
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


def crop(frame, panel):
    x1, x2, y1, y2 = PANEL_LAYOUT[panel]
    return frame[y1:y2, x1:x2].copy()


def load_panels(path: Path):
    capture = cv2.VideoCapture(str(path))
    panels = {name: [] for name in PANEL_LAYOUT}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for panel in PANEL_LAYOUT:
            panels[panel].append(crop(frame, panel))
    capture.release()
    if len(panels["VI"]) < max(STARTS) + WINDOW_LENGTH:
        raise RuntimeError(f"{path} contains only {len(panels['VI'])} usable frames")
    return panels


def score_sequence(visible, thermal, fused) -> dict[str, float]:
    scores = temporal_quality_scores(visible, thermal, fused, size=(160, 128))
    return {name: float(getattr(scores, name)) for name in scores.__dataclass_fields__}


def collect_clean_rows(all_panels, videos, starts, fusion_panels=FUSION_PANELS):
    rows = []
    for video_id in videos:
        panels = all_panels[video_id]
        for panel in fusion_panels:
            for start in starts:
                visible = panels["VI"][start : start + WINDOW_LENGTH]
                thermal = panels["IR"][start : start + WINDOW_LENGTH]
                fused = panels[panel][start : start + WINDOW_LENGTH]
                rows.append(
                    {
                        "video_id": video_id,
                        "panel": panel,
                        "start": start,
                        "family": "clean",
                        "severity": 0.0,
                        "distorted": False,
                        **score_sequence(visible, thermal, fused),
                    }
                )
    return rows


def collect_test_rows(all_panels, videos, starts, fusion_panels=FUSION_PANELS):
    rows = collect_clean_rows(all_panels, videos, starts, fusion_panels)
    for video_id in videos:
        panels = all_panels[video_id]
        for panel in fusion_panels:
            for start in starts:
                visible = panels["VI"][start : start + WINDOW_LENGTH]
                thermal = panels["IR"][start : start + WINDOW_LENGTH]
                clean = panels[panel][start : start + WINDOW_LENGTH]
                for family in TEST_FAMILIES:
                    for severity in SEVERITIES:
                        altered = inject_temporal_artifact(
                            clean, visible, thermal, family, severity
                        )
                        rows.append(
                            {
                                "video_id": video_id,
                                "panel": panel,
                                "start": start,
                                "family": family,
                                "severity": severity,
                                "distorted": True,
                                **score_sequence(visible, thermal, altered),
                            }
                        )
    return rows


def fit_clean_calibration(rows):
    calibration = {}
    clean = [row for row in rows if not row["distorted"]]
    for component in CALIBRATED_COMPONENTS:
        values = np.asarray([row[component] for row in clean], dtype=float)
        first, third = np.quantile(values, (0.25, 0.75))
        calibration[component] = {
            "center": float(np.median(values)),
            "scale": max(float(third - first), 1e-6),
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
        row["real_severity_score"] = float(
            0.80 * severity_direct + 0.20 * z(row, calibration, "proposed")
        )


def thresholds(rows):
    return {
        method: float(np.quantile([row[method] for row in rows], 1.0))
        for method in METHODS
    }


def safe_spearman(labels, scores) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        statistic = float(spearmanr(labels, scores).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def summarize(rows, fitted_thresholds):
    truth = np.asarray([row["distorted"] for row in rows], dtype=bool)
    result = {}
    for method in METHODS:
        scores = np.asarray([row[method] for row in rows], dtype=float)
        predictions = scores >= fitted_thresholds[method]
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
            "accuracy": float(np.mean(predictions == truth)),
            "false_positive_rate": float(np.mean(predictions[~truth])),
            "severity_spearman": float(np.mean(correlations)),
        }
    return result


def summarize_by_panel(rows, method):
    output = {}
    for panel in sorted({row["panel"] for row in rows}):
        selected = [row for row in rows if row["panel"] == panel]
        truth = np.asarray([row["distorted"] for row in selected], dtype=bool)
        scores = np.asarray([row[method] for row in selected], dtype=float)
        output[panel] = {
            "auroc": float(roc_auc_score(truth, scores)),
            "severity_spearman": float(
                np.mean(
                    [
                        safe_spearman(
                            [
                                row["severity"]
                                for row in selected
                                if row["family"] in ("clean", family)
                            ],
                            [
                                row[method]
                                for row in selected
                                if row["family"] in ("clean", family)
                            ],
                        )
                        for family in sorted(
                            {row["family"] for row in selected if row["distorted"]}
                        )
                    ]
                )
            ),
        }
    return output


def evaluate_protocol(name, all_panels, calibration_videos, calibration_starts, test_videos, test_starts):
    calibration_rows = collect_clean_rows(
        all_panels, calibration_videos, calibration_starts
    )
    test_rows = collect_test_rows(all_panels, test_videos, test_starts)
    calibration = fit_clean_calibration(calibration_rows)
    apply_real_scores(calibration_rows, calibration)
    apply_real_scores(test_rows, calibration)
    fitted_thresholds = thresholds(calibration_rows)
    summary = summarize(test_rows, fitted_thresholds)
    return {
        "name": name,
        "calibration_videos": list(calibration_videos),
        "calibration_starts": list(calibration_starts),
        "test_videos": list(test_videos),
        "test_starts": list(test_starts),
        "calibration": calibration,
        "thresholds": fitted_thresholds,
        "summary": summary,
        "real_detection_by_panel": summarize_by_panel(
            test_rows, "real_detection_score"
        ),
        "real_severity_by_panel": summarize_by_panel(test_rows, "real_severity_score"),
        "row_counts": {
            "calibration": len(calibration_rows),
            "test": len(test_rows),
        },
    }


def main() -> None:
    all_panels = {video_id: load_panels(path) for video_id, path in GIFS.items()}
    protocols = [
        evaluate_protocol(
            "both_videos_early_to_late",
            all_panels,
            tuple(GIFS),
            EARLY_STARTS,
            tuple(GIFS),
            LATE_STARTS,
        ),
        evaluate_protocol(
            "1207_1714_to_1207_1739",
            all_panels,
            ("1207_1714",),
            STARTS,
            ("1207_1739",),
            STARTS,
        ),
        evaluate_protocol(
            "1207_1739_to_1207_1714",
            all_panels,
            ("1207_1739",),
            STARTS,
            ("1207_1714",),
            STARTS,
        ),
    ]
    best = {}
    for method in METHODS:
        best[method] = max(
            (
                {
                    "protocol": item["name"],
                    **item["summary"][method],
                }
                for item in protocols
            ),
            key=lambda row: (row["auroc"], row["severity_spearman"]),
        )
    report = {
        "status": "real_fusion_temcoco_gif_multi_pilot",
        "sources": {key: str(value) for key, value in GIFS.items()},
        "note": "Uses rendered comparison GIF panel crops, not raw model outputs.",
        "fusion_panels": list(FUSION_PANELS),
        "test_families": list(TEST_FAMILIES),
        "severities": list(SEVERITIES),
        "protocols": protocols,
        "best_by_method": best,
    }
    output = ROOT / "results/real_fusion_temcoco_gif_multi_pilot.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    compact = {
        "status": report["status"],
        "best_by_method": best,
        "protocol_summaries": {
            item["name"]: {
                "real_detection_score": item["summary"]["real_detection_score"],
                "real_severity_score": item["summary"]["real_severity_score"],
                "proposed": item["summary"]["proposed"],
            }
            for item in protocols
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
