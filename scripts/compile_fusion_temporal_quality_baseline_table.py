"""Compile paper-style baseline tables from saved validation JSON files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

DATASETS = (
    (
        "VidLLVIP independent",
        RESULTS / "fusion_temporal_quality_independent.json",
        "independent",
    ),
    (
        "M3SVD 12-pair",
        RESULTS / "fusion_temporal_quality_m3svd_subset.json",
        "test",
    ),
    (
        "M3SVD 30-pair",
        RESULTS / "fusion_temporal_quality_m3svd_full_test.json",
        "test",
    ),
)

METHODS = (
    ("artifact_detection_score", "Ours detection head", "ours"),
    ("severity_ranking_score", "Ours severity head", "ours"),
    ("proposed", "Ours composite", "ours"),
    ("source_conditioned_residual", "Source-conditioned residual", "ablation"),
    ("span_residual", "Span residual", "ablation"),
    ("static_operator_residual", "Static operator residual", "ablation"),
    ("global_weight_drift", "Global weight drift", "baseline"),
    ("source_normalized_change", "Source-normalized change", "baseline"),
    ("temporal_difference", "Temporal difference", "baseline"),
    ("flow_warp_error", "Flow warp error", "baseline"),
    ("temcoco_flowd_visible", "TemCoCo-style flowD visible", "baseline"),
    ("temcoco_feacd_sobel", "TemCoCo-style feaCD Sobel", "baseline"),
)

FRAMEWISE_METHOD_LABELS = {
    "entropy_instability": "Entropy instability",
    "std_instability": "Standard-deviation instability",
    "spatial_frequency_instability": "Spatial-frequency instability",
    "mutual_information_instability": "Mutual-information instability",
    "edge_correlation_instability": "Edge-correlation instability",
}


def fmt(value) -> str:
    return f"{float(value):.4f}"


def collect_rows():
    rows = []
    for dataset_name, path, summary_key in DATASETS:
        report = json.loads(path.read_text(encoding="utf-8"))
        summary = report[summary_key]
        for method, label, group in METHODS:
            if method not in summary:
                continue
            item = summary[method]
            rows.append(
                {
                    "dataset": dataset_name,
                    "group": group,
                    "method": label,
                    "auroc": item["macro_fusion_auroc"],
                    "worst_auroc": item["worst_fusion_auroc"],
                    "accuracy": item["calibrated_accuracy"],
                    "fpr": item["calibrated_false_positive_rate"],
                    "spearman": item["severity_spearman"],
                }
            )

    resnet_path = RESULTS / "temcoco_resnet_feacd_independent.json"
    if resnet_path.exists():
        report = json.loads(resnet_path.read_text(encoding="utf-8"))
        item = report["summary"]
        rows.append(
            {
                "dataset": "VidLLVIP independent",
                "group": "baseline",
                "method": "ResNet-18 feaCD visible-source probe",
                "auroc": item["macro_fusion_auroc"],
                "worst_auroc": item["worst_fusion_auroc"],
                "accuracy": item["calibrated_accuracy"],
                "fpr": item["calibrated_false_positive_rate"],
                "spearman": item["severity_spearman"],
            }
        )
    searaft_path = RESULTS / "temcoco_searaft_flowd_independent.json"
    if searaft_path.exists():
        report = json.loads(searaft_path.read_text(encoding="utf-8"))
        item = report["summary"]
        rows.append(
            {
                "dataset": "VidLLVIP independent",
                "group": "baseline",
                "method": "SEA-RAFT flowD visible-source probe",
                "auroc": item["macro_fusion_auroc"],
                "worst_auroc": item["worst_fusion_auroc"],
                "accuracy": item["calibrated_accuracy"],
                "fpr": item["calibrated_false_positive_rate"],
                "spearman": item["severity_spearman"],
            }
        )
    searaft_m3svd_path = RESULTS / "temcoco_searaft_flowd_m3svd_full_test.json"
    if searaft_m3svd_path.exists():
        report = json.loads(searaft_m3svd_path.read_text(encoding="utf-8"))
        item = report["summary"]
        rows.append(
            {
                "dataset": "M3SVD 30-pair",
                "group": "baseline",
                "method": "SEA-RAFT flowD visible-source probe",
                "auroc": item["macro_fusion_auroc"],
                "worst_auroc": item["worst_fusion_auroc"],
                "accuracy": item["calibrated_accuracy"],
                "fpr": item["calibrated_false_positive_rate"],
                "spearman": item["severity_spearman"],
            }
        )
    framewise_path = RESULTS / "framewise_fusion_metric_baselines_independent.json"
    if framewise_path.exists():
        report = json.loads(framewise_path.read_text(encoding="utf-8"))
        for method, item in report["summary"].items():
            rows.append(
                {
                    "dataset": "VidLLVIP independent",
                    "group": "baseline",
                    "method": FRAMEWISE_METHOD_LABELS.get(method, method),
                    "auroc": item["macro_fusion_auroc"],
                    "worst_auroc": item["worst_fusion_auroc"],
                    "accuracy": item["calibrated_accuracy"],
                    "fpr": item["calibrated_false_positive_rate"],
                    "spearman": item["severity_spearman"],
                }
            )
    return rows


def write_csv(rows, output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "dataset",
                "group",
                "method",
                "auroc",
                "worst_auroc",
                "accuracy",
                "fpr",
                "spearman",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def method_table(rows, dataset: str) -> list[str]:
    selected = [row for row in rows if row["dataset"] == dataset]
    selected.sort(
        key=lambda row: (
            0 if row["group"] == "ours" else 1 if row["group"] == "ablation" else 2,
            -row["spearman"],
            -row["auroc"],
        )
    )
    lines = [
        f"### {dataset}",
        "",
        "| Group | Method | AUROC | Worst AUROC | Accuracy | FPR | Spearman |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {group} | {method} | {auroc} | {worst_auroc} | {accuracy} | {fpr} | {spearman} |".format(
                group=row["group"],
                method=row["method"],
                auroc=fmt(row["auroc"]),
                worst_auroc=fmt(row["worst_auroc"]),
                accuracy=fmt(row["accuracy"]),
                fpr=fmt(row["fpr"]),
                spearman=fmt(row["spearman"]),
            )
        )
    return lines


def write_markdown(rows, output: Path) -> None:
    lines = [
        "# Fusion Temporal Quality Baseline Table",
        "",
        "This table is generated from saved validation JSON files.  The ResNet-18 feaCD row is a local probe, not an official TemCoCo reproduction.",
        "",
    ]
    for dataset, _, _ in DATASETS:
        lines.extend(method_table(rows, dataset))
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    write_csv(rows, RESULTS / "fusion_temporal_quality_baseline_table.csv")
    write_markdown(rows, RESULTS / "fusion_temporal_quality_baseline_table.md")
    print(
        json.dumps(
            {
                "rows": len(rows),
                "csv": str(RESULTS / "fusion_temporal_quality_baseline_table.csv"),
                "markdown": str(
                    RESULTS / "fusion_temporal_quality_baseline_table.md"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
