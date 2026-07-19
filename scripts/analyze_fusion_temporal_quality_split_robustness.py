"""Recalibrate V2 scores under multiple source splits using saved raw rows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from run_fusion_temporal_quality_external import (
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
)


INPUT = ROOT / "results/fusion_temporal_quality_independent.json"
OUTPUT = ROOT / "results/fusion_temporal_quality_split_robustness.json"


def fit_composite(rows):
    calibration = {}
    calibrated_components = tuple(
        dict.fromkeys(PROPOSED_COMPONENTS + SEVERITY_COMPONENTS)
    )
    for fusion_method in FUSION_METHODS:
        selected = [row for row in rows if row["fusion_method"] == fusion_method]
        clean = [row for row in selected if not row["distorted"]]
        calibration[fusion_method] = {}
        for component in calibrated_components:
            values = np.asarray([row[component] for row in selected], dtype=float)
            first, third = np.quantile(values, (0.25, 0.75))
            calibration[fusion_method][component] = {
                "center": float(np.median([row[component] for row in clean])),
                "scale": max(float(third - first), 1e-8),
            }
    return calibration


def apply_composite(rows, calibration):
    output = []
    for row in rows:
        item = dict(row)
        item["artifact_detection_score"] = max(
            0.0,
            (
                item["source_conditioned_residual"]
                - calibration[item["fusion_method"]]["source_conditioned_residual"]["center"]
            )
            / calibration[item["fusion_method"]]["source_conditioned_residual"]["scale"],
        )
        severity_evidence = float(
            np.average(
                [
                    max(
                        0.0,
                        (
                            item[component]
                            - calibration[item["fusion_method"]][component]["center"]
                        )
                        / calibration[item["fusion_method"]][component]["scale"],
                    )
                    for component in SEVERITY_COMPONENTS
                ],
                weights=[
                    SEVERITY_WEIGHTS[component] for component in SEVERITY_COMPONENTS
                ],
            )
        )
        evidence = []
        for component in PROPOSED_COMPONENTS:
            parameters = calibration[item["fusion_method"]][component]
            evidence.append(
                PROPOSED_WEIGHTS[component]
                * max(
                    0.0,
                    (item[component] - parameters["center"]) / parameters["scale"],
                )
            )
        item["proposed"] = float(np.mean(evidence))
        item["severity_ranking_score"] = float(
            SEVERITY_DIRECT_WEIGHT * severity_evidence
            + SEVERITY_PROPOSED_WEIGHT * item["proposed"]
        )
        output.append(item)
    return output


def thresholds(rows):
    return {
        method: {
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
        for method in EVALUATION_METHODS
    }


def safe_spearman(labels, scores) -> float:
    statistic = float(spearmanr(labels, scores).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def summarize(rows, fitted_thresholds):
    result = {}
    for method in EVALUATION_METHODS:
        predictions = []
        truths = []
        per_fusion_auc = {}
        correlations = []
        for fusion_method in FUSION_METHODS:
            selected = [row for row in rows if row["fusion_method"] == fusion_method]
            truth = np.asarray([row["distorted"] for row in selected], dtype=bool)
            scores = np.asarray([row[method] for row in selected], dtype=float)
            per_fusion_auc[fusion_method] = float(roc_auc_score(truth, scores))
            predictions.extend(scores >= fitted_thresholds[method][fusion_method])
            truths.extend(truth)
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
            "calibrated_accuracy": float(np.mean(prediction == truth)),
            "calibrated_false_positive_rate": float(np.mean(prediction[~truth])),
            "severity_spearman": float(np.mean(correlations)),
        }
    return result


def evaluate_split(rows, calibration_sources, test_sources):
    calibration_rows = [row for row in rows if row["source_id"] in calibration_sources]
    test_rows = [row for row in rows if row["source_id"] in test_sources]
    calibration = fit_composite(calibration_rows)
    calibrated_calibration_rows = apply_composite(calibration_rows, calibration)
    calibrated_test_rows = apply_composite(test_rows, calibration)
    fitted_thresholds = thresholds(calibrated_calibration_rows)
    summary = summarize(calibrated_test_rows, fitted_thresholds)
    proposed = summary["proposed"]
    best_baseline_auc = max(
        summary[name]["macro_fusion_auroc"] for name in STANDARD_BASELINES
    )
    best_baseline_rank = max(
        summary[name]["severity_spearman"] for name in STANDARD_BASELINES
    )
    return {
        "calibration_sources": list(calibration_sources),
        "test_sources": list(test_sources),
        "summary": summary,
        "best_standard_baseline_auroc": best_baseline_auc,
        "best_standard_baseline_spearman": best_baseline_rank,
        "gate": {
            "macro_auroc_at_least_0.90": proposed["macro_fusion_auroc"] >= 0.90,
            "worst_auroc_at_least_0.85": proposed["worst_fusion_auroc"] >= 0.85,
            "accuracy_at_least_0.85": proposed["calibrated_accuracy"] >= 0.85,
            "fpr_at_most_0.10": proposed["calibrated_false_positive_rate"] <= 0.10,
            "spearman_at_least_0.80": proposed["severity_spearman"] >= 0.80,
            "auroc_margin_at_least_0.10": (
                proposed["macro_fusion_auroc"] - best_baseline_auc >= 0.10
            ),
            "rank_margin_at_least_0.10": (
                proposed["severity_spearman"] - best_baseline_rank >= 0.10
            ),
        },
    }


def main() -> None:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    rows = report["raw_rows"]["calibration"] + report["raw_rows"]["independent"]
    sources = sorted({row["source_id"] for row in rows})
    named_splits = {
        "old_to_new": ((1, 2, 4, 5, 7, 8, 10, 14), (3, 6, 9, 11, 12, 13)),
        "new_to_old": ((3, 6, 9, 11, 12, 13), (1, 2, 4, 5, 7, 8, 10, 14)),
        "odd_to_even": (tuple(s for s in sources if s % 2), tuple(s for s in sources if not s % 2)),
        "even_to_odd": (tuple(s for s in sources if not s % 2), tuple(s for s in sources if s % 2)),
        "low_to_high": (tuple(range(1, 8)), tuple(range(8, 15))),
        "high_to_low": (tuple(range(8, 15)), tuple(range(1, 8))),
    }
    results = {
        name: evaluate_split(rows, calibration_sources, test_sources)
        for name, (calibration_sources, test_sources) in named_splits.items()
    }

    leave_one_out = {}
    for source in sources:
        calibration_sources = tuple(item for item in sources if item != source)
        leave_one_out[f"source_{source:02d}"] = evaluate_split(
            rows, calibration_sources, (source,)
        )

    def aggregate_method(method: str):
        items = [item["summary"][method] for item in leave_one_out.values()]
        return {
            "leave_one_out_macro_auroc_mean": float(
                np.mean([item["macro_fusion_auroc"] for item in items])
            ),
            "leave_one_out_accuracy_mean": float(
                np.mean([item["calibrated_accuracy"] for item in items])
            ),
            "leave_one_out_fpr_mean": float(
                np.mean([item["calibrated_false_positive_rate"] for item in items])
            ),
            "leave_one_out_spearman_mean": float(
                np.mean([item["severity_spearman"] for item in items])
            ),
        }

    aggregate = {
        "proposed": aggregate_method("proposed"),
        "artifact_detection_score": aggregate_method("artifact_detection_score"),
        "severity_ranking_score": aggregate_method("severity_ranking_score"),
    }
    output = {
        "status": "split_robustness_from_saved_raw_rows",
        "source_count": len(sources),
        "sources": sources,
        "proposed_components": list(PROPOSED_COMPONENTS),
        "proposed_weights": PROPOSED_WEIGHTS,
        "severity_components": list(SEVERITY_COMPONENTS),
        "severity_weights": SEVERITY_WEIGHTS,
        "severity_direct_weight": SEVERITY_DIRECT_WEIGHT,
        "severity_proposed_weight": SEVERITY_PROPOSED_WEIGHT,
        "named_splits": results,
        "leave_one_out": leave_one_out,
        "aggregate": aggregate,
    }
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    compact = {
        "status": output["status"],
        "aggregate": aggregate,
        "named_splits": {
            name: {
                "proposed": value["summary"]["proposed"],
                "detection": value["summary"]["artifact_detection_score"],
                "severity": value["summary"]["severity_ranking_score"],
            }
            for name, value in results.items()
        },
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
