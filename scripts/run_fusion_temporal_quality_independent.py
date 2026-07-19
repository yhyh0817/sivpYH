"""Independent V2 validation on VidLLVIP sources not used during method design."""

from __future__ import annotations

import json

from run_fusion_temporal_quality_external import (
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
    ABLATIONS,
    apply_composite,
    clean_threshold,
    collect,
    fit_composite_calibration,
    summarize,
)


CALIBRATION_SOURCES = (1, 2, 4, 5, 7, 8, 10, 14)
INDEPENDENT_SOURCES = (3, 6, 9, 11, 12, 13)
CALIBRATION_FAMILIES = ("global_gain", "global_weight")
INDEPENDENT_FAMILIES = ("local_weight", "aperiodic_gain", "local_gain", "patch_lag")
SEVERITIES = (0.025, 0.06, 0.11)


def main() -> None:
    calibration_rows = collect(
        CALIBRATION_SOURCES,
        CALIBRATION_FAMILIES,
        SEVERITIES,
        BENIGN_FAMILIES,
    )
    composite_calibration = fit_composite_calibration(calibration_rows)
    apply_composite(calibration_rows, composite_calibration)
    thresholds = {
        method: {
            fusion_method: clean_threshold(calibration_rows, method, fusion_method)
            for fusion_method in FUSION_METHODS
        }
        for method in EVALUATION_METHODS
    }

    independent_rows = collect(
        INDEPENDENT_SOURCES,
        INDEPENDENT_FAMILIES,
        SEVERITIES,
        BENIGN_FAMILIES,
    )
    apply_composite(independent_rows, composite_calibration)
    independent = summarize(independent_rows, thresholds)
    proposed = independent["proposed"]
    detection = independent["artifact_detection_score"]
    severity = independent["severity_ranking_score"]
    best_baseline_auc = max(
        independent[name]["macro_fusion_auroc"] for name in STANDARD_BASELINES
    )
    best_baseline_rank = max(
        independent[name]["severity_spearman"] for name in STANDARD_BASELINES
    )
    best_ablation_auc = max(
        independent[name]["macro_fusion_auroc"] for name in ABLATIONS
    )
    best_ablation_rank = max(
        independent[name]["severity_spearman"] for name in ABLATIONS
    )
    gate = {
        "independent_macro_auroc_at_least_0.90": proposed["macro_fusion_auroc"] >= 0.90,
        "independent_worst_fusion_auroc_at_least_0.85": proposed["worst_fusion_auroc"] >= 0.85,
        "independent_accuracy_at_least_0.85": proposed["calibrated_accuracy"] >= 0.85,
        "independent_fpr_at_most_0.10": proposed["calibrated_false_positive_rate"] <= 0.10,
        "independent_spearman_at_least_0.85": proposed["severity_spearman"] >= 0.85,
        "auroc_margin_at_least_0.10": proposed["macro_fusion_auroc"] - best_baseline_auc >= 0.10,
        "rank_margin_at_least_0.10": proposed["severity_spearman"] - best_baseline_rank >= 0.10,
        "auroc_not_worse_than_ablation_by_0.02": proposed["macro_fusion_auroc"] >= best_ablation_auc - 0.02,
        "rank_not_worse_than_ablation_by_0.02": proposed["severity_spearman"] >= best_ablation_rank - 0.02,
    }
    dual_gate = {
        "detection_macro_auroc_at_least_0.90": detection["macro_fusion_auroc"] >= 0.90,
        "detection_worst_fusion_auroc_at_least_0.85": detection["worst_fusion_auroc"] >= 0.85,
        "detection_accuracy_at_least_0.85": detection["calibrated_accuracy"] >= 0.85,
        "detection_fpr_at_most_0.10": detection["calibrated_false_positive_rate"] <= 0.10,
        "severity_spearman_at_least_0.85": severity["severity_spearman"] >= 0.85,
    }
    report = {
        "status": "new_source_independent_validation_not_acceptance_guarantee",
        "direction": "source-referenced temporal flicker assessment for RGB-T fusion video",
        "innovation_candidate": "motion-compensated source-conditioned fusion-operator consistency",
        "protocol": {
            "calibration_sources": list(CALIBRATION_SOURCES),
            "independent_sources": list(INDEPENDENT_SOURCES),
            "calibration_families": list(CALIBRATION_FAMILIES),
            "independent_families": list(INDEPENDENT_FAMILIES),
            "benign_source_variation_families": list(BENIGN_FAMILIES),
            "severities": list(SEVERITIES),
        },
        "proposed_components": list(PROPOSED_COMPONENTS),
        "proposed_weights": PROPOSED_WEIGHTS,
        "severity_components": list(SEVERITY_COMPONENTS),
        "severity_weights": SEVERITY_WEIGHTS,
        "severity_direct_weight": SEVERITY_DIRECT_WEIGHT,
        "severity_proposed_weight": SEVERITY_PROPOSED_WEIGHT,
        "composite_calibration": composite_calibration,
        "thresholds": thresholds,
        "independent": independent,
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
        "gate": gate,
        "dual_gate": dual_gate,
        "passes_dual_gate": all(dual_gate.values()),
        "passes_all_gates": all(gate.values()),
        "raw_rows": {
            "calibration": calibration_rows,
            "independent": independent_rows,
        },
    }
    output = ROOT / "results/fusion_temporal_quality_independent.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
