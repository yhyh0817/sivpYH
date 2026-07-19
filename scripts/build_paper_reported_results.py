"""Build a machine-readable summary of the values reported in the manuscript."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "paper_reported_results.json"


def load_json(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def round4(value: float) -> float:
    return round(float(value), 4)


def metric_row(
    dual_output: dict[str, Any], positive_only: float | None
) -> dict[str, float]:
    detection = dual_output["detection"]
    severity = dual_output["severity"]
    metrics = {
        "detection_auroc": round4(detection["macro_fusion_auroc"]),
        "worst_auroc": round4(detection["worst_fusion_auroc"]),
        "accuracy": round4(detection["calibrated_accuracy"]),
        "fpr": round4(detection["calibrated_false_positive_rate"]),
        "severity_head_auroc": round4(severity["macro_fusion_auroc"]),
        "severity_srocc": round4(severity["severity_spearman"]),
    }
    if positive_only is not None:
        metrics["positive_only_srocc"] = round4(positive_only)
    return metrics


def protocol_summary(
    *,
    label: str,
    calibration_ids: list[Any],
    test_ids: list[Any],
    calibration_rows: int,
    test_rows: int,
    artifact_families: list[str],
    hard_negative_families: list[str],
    metrics: dict[str, float],
    source_files: list[str],
) -> dict[str, Any]:
    return {
        "label": label,
        "calibration_ids": calibration_ids,
        "test_ids": test_ids,
        "calibration_videos": len(calibration_ids),
        "test_videos": len(test_ids),
        "calibration_rows": calibration_rows,
        "test_rows": test_rows,
        "test_artifact_families": artifact_families,
        "hard_negative_families": hard_negative_families,
        "reported_metrics": metrics,
        "source_files": source_files,
    }


def load_baselines() -> list[dict[str, Any]]:
    path = RESULTS / "fusion_temporal_quality_baseline_table.csv"
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = {
                "dataset": row["dataset"],
                "group": row["group"],
                "method": row["method"],
            }
            for key in ("auroc", "worst_auroc", "accuracy", "fpr", "spearman"):
                converted[key] = round4(float(row[key]))
            rows.append(converted)
    return rows


def build() -> dict[str, Any]:
    vid = load_json("fusion_temporal_quality_independent.json")
    m3 = load_json("fusion_temporal_quality_m3svd_full_test.json")
    subset = load_json("fusion_temporal_quality_m3svd_subset.json")
    real = load_json("real_fusion_model_confirmed_metrics.json")
    additional = load_json("manuscript_additional_verified_metrics.json")
    loso = load_json("fusion_temporal_quality_split_robustness.json")

    vid_positive = additional["positive_only_spearman"]["VidLLVIP"]
    m3_positive = additional["positive_only_spearman"]["M3SVD"]
    independent_protocol = vid["protocol"]
    m3_test = m3["test"]

    controlled = {
        "VidLLVIP independent": protocol_summary(
            label="VidLLVIP independent",
            calibration_ids=independent_protocol["calibration_sources"],
            test_ids=independent_protocol["independent_sources"],
            calibration_rows=len(vid["raw_rows"]["calibration"]),
            test_rows=len(vid["raw_rows"]["independent"]),
            artifact_families=independent_protocol["independent_families"],
            hard_negative_families=independent_protocol["benign_source_variation_families"],
            metrics=metric_row(vid["dual_output"], vid_positive),
            source_files=[
                "results/fusion_temporal_quality_independent.json",
                "results/manuscript_additional_verified_metrics.json",
            ],
        ),
        "M3SVD held out": protocol_summary(
            label="M3SVD held out",
            calibration_ids=m3["calibration_ids"],
            test_ids=m3["test_ids"],
            calibration_rows=len(m3["raw_rows"]["calibration"]),
            test_rows=len(m3["raw_rows"]["test"]),
            artifact_families=m3["test_families"],
            hard_negative_families=m3["benign_source_variation_families"],
            metrics=metric_row(m3["dual_output"], m3_positive),
            source_files=[
                "results/fusion_temporal_quality_m3svd_full_test.json",
                "results/manuscript_additional_verified_metrics.json",
            ],
        ),
    }

    real_metrics = real["metrics"]
    real_reported = {
        "evaluation_design": real["evaluation_design"],
        "metrics": {key: round4(value) for key, value in real_metrics.items()},
        "confusion_matrix": real["confusion_matrix"],
        "derived_metrics": {
            key: round4(value) for key, value in real["derived_metrics"].items()
        },
        "per_output_group": real["per_output_group"],
    }

    loso_aggregate = loso["aggregate"]
    loso_summary = {
        "source_count": loso["source_count"],
        "detection_head": {
            "mean_auroc": round4(
                loso_aggregate["artifact_detection_score"]["leave_one_out_macro_auroc_mean"]
            ),
            "mean_accuracy": round4(
                loso_aggregate["artifact_detection_score"]["leave_one_out_accuracy_mean"]
            ),
            "mean_fpr": round4(
                loso_aggregate["artifact_detection_score"]["leave_one_out_fpr_mean"]
            ),
        },
        "severity_head": {
            "mean_auroc": round4(
                loso_aggregate["severity_ranking_score"]["leave_one_out_macro_auroc_mean"]
            ),
            "mean_accuracy": round4(
                loso_aggregate["severity_ranking_score"]["leave_one_out_accuracy_mean"]
            ),
            "mean_fpr": round4(
                loso_aggregate["severity_ranking_score"]["leave_one_out_fpr_mean"]
            ),
            "mean_srocc": round4(
                loso_aggregate["severity_ranking_score"]["leave_one_out_spearman_mean"]
            ),
        },
        "per_source": {
            source: {
                "detection_auroc": round4(
                    record["summary"]["artifact_detection_score"]["macro_fusion_auroc"]
                ),
                "severity_head_auroc": round4(
                    record["summary"]["severity_ranking_score"]["macro_fusion_auroc"]
                ),
                "severity_srocc": round4(
                    record["summary"]["severity_ranking_score"]["severity_spearman"]
                ),
                "detection_fpr": round4(
                    record["summary"]["artifact_detection_score"][
                        "calibrated_false_positive_rate"
                    ]
                ),
                "severity_fpr": round4(
                    record["summary"]["severity_ranking_score"][
                        "calibrated_false_positive_rate"
                    ]
                ),
            }
            for source, record in sorted(loso["leave_one_out"].items())
        },
    }

    output = {
        "schema_version": "1.0",
        "status": "paper_reported_results",
        "reconstructed_on": "2026-07-19",
        "scope": "Values shown in the manuscript and its supplementary material.",
        "controlled_evaluations": controlled,
        "real_fusion_output_evaluation": real_reported,
        "baselines_and_ablations": {
            "rows": load_baselines(),
            "source_file": "results/fusion_temporal_quality_baseline_table.csv",
            "note": "Values are rounded to four decimals for the paper reporting view.",
        },
        "leave_one_source_out": loso_summary,
        "additional_reported_measurements": {
            "runtime": additional["runtime_benchmark"],
            "searaft_flowd_probe": additional["searaft_flowd_probe"],
            "source_file": "results/manuscript_additional_verified_metrics.json",
        },
        "other_retained_protocol": {
            "M3SVD 12-pair subset": {
                "reported_metrics": metric_row(
                    subset["dual_output"],
                    None,
                ),
                "source_file": "results/fusion_temporal_quality_m3svd_subset.json",
                "note": "Separate retained protocol; not the full-test row used in the main manuscript table.",
            }
        },
        "historical_pilot_files": {
            "included": False,
            "files": [
                "results/real_fusion_temcoco_gif_pilot.json",
                "results/real_fusion_temcoco_gif_multi_pilot.json",
            ],
            "reason": "Different exploratory protocols and not part of the final paper evaluation.",
        },
    }

    # Guard against silently drifting away from the reported real-output table.
    group_values = real["per_output_group"]
    for key in ("detection_auroc", "severity_auroc", "severity_spearman"):
        group_mean = sum(float(row[key]) for row in group_values) / len(group_values)
        assert round4(group_mean) == round4(real_metrics[
            {
                "detection_auroc": "detection_auroc",
                "severity_auroc": "severity_auroc",
                "severity_spearman": "severity_spearman",
            }[key]
        ])

    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    result = build()
    print(f"Wrote {OUTPUT}")
    print(f"Controlled evaluations: {len(result['controlled_evaluations'])}")
    print(f"Baseline rows: {len(result['baselines_and_ablations']['rows'])}")
    print(f"LOSO sources: {result['leave_one_source_out']['source_count']}")
