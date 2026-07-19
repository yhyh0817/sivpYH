from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
GRAYSCALE = os.environ.get("SIVP_GRAYSCALE", "0") == "1"
OUT = ROOT / "results" / ("paper_figures_grayscale" if GRAYSCALE else "paper_figures")
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cursor_rgbt.fusion_temporal_quality import (  # noqa: E402
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
)


MM_PER_INCH = 25.4
DOUBLE_COLUMN_MM = 174.0
if GRAYSCALE:
    BLUE = "#202020"
    ORANGE = "#5C5C5C"
    TEAL = "#969696"
    RED = "#3E3E3E"
    PURPLE = "#727272"
    GRAY = "#777777"
    LIGHT_BLUE = "#F5F5F5"
    LIGHT_ORANGE = "#E7E7E7"
    LIGHT_TEAL = "#D3D3D3"
    LIGHT_GRAY = "#EEEEEE"
else:
    BLUE = "#2878B5"
    ORANGE = "#E6862A"
    TEAL = "#2A9D8F"
    RED = "#C84C4C"
    PURPLE = "#8064A2"
    GRAY = "#6B7280"
    LIGHT_BLUE = "#DDEBF7"
    LIGHT_ORANGE = "#FCE8D5"
    LIGHT_TEAL = "#DDF2EE"
    LIGHT_GRAY = "#EEF0F2"
INK = "#202020" if GRAYSCALE else "#1F2933"
GRID = "#D7D7D7" if GRAYSCALE else "#D9DDE2"
PANEL_BORDER = "#B8B8B8" if GRAYSCALE else "#B8BEC6"
IMAGE_BORDER = "#606060" if GRAYSCALE else "#5B6168"


def mm(value: float) -> float:
    return value / MM_PER_INCH


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: str, raster_content: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = dict(facecolor="white", bbox_inches=None, pad_inches=0.0)
    fig.savefig(OUT / f"{stem}.pdf", format="pdf", dpi=600, **common)
    fig.savefig(OUT / f"{stem}.svg", format="svg", dpi=600, **common)
    fig.savefig(OUT / f"{stem}.png", format="png", dpi=600, **common)
    if not raster_content:
        fig.savefig(OUT / f"{stem}.eps", format="eps", dpi=600, **common)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.04) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        fontweight="bold",
        color=INK,
    )


def style_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str,
    fontsize: float = 7.0,
    linestyle: str = "-",
) -> patches.FancyBboxPatch:
    box_style = "square,pad=0.008" if GRAYSCALE else "round,pad=0.008,rounding_size=0.012"
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=box_style,
        linewidth=0.9,
        facecolor=face,
        edgecolor=edge,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.1,
        zorder=3,
    )
    return box


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = INK,
    connectionstyle: str = "arc3",
    linestyle: str = "-",
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.8,
            color=color,
            connectionstyle=connectionstyle,
            linestyle=linestyle,
        ),
        zorder=1,
    )


def make_fig1_method_overview() -> None:
    fig, ax = plt.subplots(figsize=(mm(DOUBLE_COLUMN_MM), mm(91)))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.015, 0.955, "Paired temporal inputs", fontsize=8.2, fontweight="bold", color=INK)
    labels = [
        ("Visible\n$V_{t-1}, V_t$", LIGHT_BLUE, BLUE),
        ("Thermal\n$T_{t-1}, T_t$", LIGHT_ORANGE, ORANGE),
        ("Fusion output\n$F_{t-1}, F_t$", LIGHT_TEAL, TEAL),
    ]
    ys = [0.75, 0.54, 0.33]
    for (text, face, edge), y in zip(labels, ys):
        add_box(ax, 0.015, y, 0.12, 0.13, text, face, edge, 7.1)

    add_box(ax, 0.19, 0.65, 0.14, 0.18, "Shared reference\nmotion estimation", LIGHT_GRAY, GRAY, 7.1)
    add_box(ax, 0.19, 0.38, 0.14, 0.18, "Motion-compensated\nwarping", LIGHT_GRAY, GRAY, 7.1)
    for y in ys:
        arrow(ax, (0.135, y + 0.065), (0.19, 0.74 if y > 0.6 else 0.47))
    arrow(ax, (0.26, 0.65), (0.26, 0.56))

    ax.text(0.38, 0.955, "Local source explainability", fontsize=8.2, fontweight="bold", color=INK)
    add_box(
        ax,
        0.38,
        0.68,
        0.19,
        0.17,
        "$\\Delta F \\approx \\alpha\\Delta V + \\beta\\Delta T$\n(tile-wise robust fit)",
        LIGHT_BLUE,
        BLUE,
        7.1,
    )
    add_box(
        ax,
        0.38,
        0.43,
        0.19,
        0.17,
        "Static fusion-rule fit\nacross the clip",
        LIGHT_ORANGE,
        ORANGE,
        7.1,
    )
    add_box(
        ax,
        0.38,
        0.19,
        0.19,
        0.15,
        "Span and global\nweight-drift evidence",
        LIGHT_TEAL,
        TEAL,
        7.1,
    )
    arrow(ax, (0.33, 0.47), (0.38, 0.765))
    arrow(ax, (0.33, 0.47), (0.38, 0.515))
    arrow(ax, (0.33, 0.47), (0.38, 0.265))

    ax.text(0.635, 0.955, "Calibration and scoring", fontsize=8.2, fontweight="bold", color=INK)
    add_box(
        ax,
        0.63,
        0.66,
        0.16,
        0.20,
        "Artifact detection\nhead\n\ncalibrated source-\nconditioned residual",
        LIGHT_BLUE,
        BLUE,
        6.8,
    )
    add_box(
        ax,
        0.63,
        0.34,
        0.16,
        0.24,
        "Severity ranking head\n\nspan + weight drift\n+ source residual\n+ static residual",
        LIGHT_ORANGE,
        ORANGE,
        6.8,
    )
    arrow(ax, (0.57, 0.765), (0.63, 0.76), BLUE)
    arrow(ax, (0.57, 0.515), (0.63, 0.49), ORANGE)
    arrow(ax, (0.57, 0.265), (0.63, 0.43), ORANGE)

    add_box(
        ax,
        0.63,
        0.08,
        0.16,
        0.14,
        "Clean clips + source-\nexplainable hard negatives",
        "white",
        GRAY,
        6.8,
        "--",
    )
    arrow(ax, (0.71, 0.22), (0.71, 0.34), GRAY, linestyle="--")
    arrow(ax, (0.71, 0.22), (0.71, 0.66), GRAY, "arc3,rad=-0.35", "--")

    ax.text(0.855, 0.955, "Dual outputs", fontsize=8.2, fontweight="bold", color=INK)
    add_box(ax, 0.855, 0.66, 0.125, 0.17, "Detection\nscore", LIGHT_BLUE, BLUE, 7.4)
    add_box(ax, 0.855, 0.38, 0.125, 0.17, "Severity\nscore", LIGHT_ORANGE, ORANGE, 7.4)
    arrow(ax, (0.79, 0.76), (0.855, 0.745), BLUE)
    arrow(ax, (0.79, 0.46), (0.855, 0.465), ORANGE)

    fig.subplots_adjust(left=0.01, right=0.995, bottom=0.02, top=0.99)
    save_figure(fig, "Fig1_method_overview")


def make_fig2_source_explainability() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(mm(DOUBLE_COLUMN_MM), mm(62)), sharey=True)
    x = np.linspace(0, 1, 80)
    source_v = 0.42 * np.sin(2 * np.pi * (x - 0.08))
    source_t = 0.30 * np.sin(2 * np.pi * (x + 0.12))
    explained = 0.58 * source_v + 0.42 * source_t

    cases = [
        (
            source_v,
            source_t,
            explained + 0.025 * np.sin(8 * np.pi * x),
            "Source-driven change",
            "large temporal change, low residual",
        ),
        (
            np.zeros_like(x),
            np.zeros_like(x),
            0.34 * np.sign(np.sin(7 * np.pi * x)),
            "Fusion-only artifact",
            "stable sources, high residual",
        ),
        (
            source_v,
            source_t,
            explained + 0.16 * np.exp(-((x - 0.63) / 0.055) ** 2),
            "Localized inconsistency",
            "unexplained local evidence",
        ),
    ]
    for idx, (v, t, f, heading, note) in enumerate(cases):
        ax = axes[idx]
        ax.plot(x, v, color=BLUE, label="$\\Delta V$", linestyle="-")
        ax.plot(x, t, color=ORANGE, label="$\\Delta T$", linestyle="--")
        ax.plot(x, f, color=TEAL, label="$\\Delta F$", linestyle="-.")
        pred = 0.58 * v + 0.42 * t
        residual = np.abs(f - pred)
        ax.fill_between(
            x,
            -0.58,
            -0.58 + residual,
            color=RED,
            alpha=0.32 if GRAYSCALE else 0.55,
            linewidth=0,
            hatch="////" if GRAYSCALE else None,
        )
        ax.text(0.5, 0.94, heading, transform=ax.transAxes, ha="center", va="top", fontsize=7.8, fontweight="bold")
        ax.text(0.5, 0.05, note, transform=ax.transAxes, ha="center", va="bottom", fontsize=6.7, color=GRAY)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.62, 0.62)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(PANEL_BORDER)
            spine.set_linewidth(0.65)
        panel_label(ax, f"({chr(97 + idx)})", x=-0.05, y=1.02)
    axes[0].legend(
        loc="upper left",
        frameon=False,
        ncol=3,
        bbox_to_anchor=(0.0, 1.16),
        columnspacing=1.0,
        handlelength=2.0,
    )
    fig.subplots_adjust(left=0.035, right=0.995, bottom=0.08, top=0.84, wspace=0.10)
    save_figure(fig, "Fig2_source_explainability_concept")


def load_json(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def main_results() -> tuple[list[str], list[dict[str, float]]]:
    vid = load_json("fusion_temporal_quality_independent.json")["dual_output"]
    m3 = load_json("fusion_temporal_quality_m3svd_full_test.json")["dual_output"]
    real = load_json("real_fusion_model_confirmed_metrics.json")["metrics"]
    names = ["VidLLVIP", "M3SVD", "Fusion outputs"]
    data = [
        {
            "det_auc": vid["detection"]["macro_fusion_auroc"],
            "worst_auc": vid["detection"]["worst_fusion_auroc"],
            "accuracy": vid["detection"]["calibrated_accuracy"],
            "fpr": vid["detection"]["calibrated_false_positive_rate"],
            "sev_auc": vid["severity"]["macro_fusion_auroc"],
            "spearman": vid["severity"]["severity_spearman"],
        },
        {
            "det_auc": m3["detection"]["macro_fusion_auroc"],
            "worst_auc": m3["detection"]["worst_fusion_auroc"],
            "accuracy": m3["detection"]["calibrated_accuracy"],
            "fpr": m3["detection"]["calibrated_false_positive_rate"],
            "sev_auc": m3["severity"]["macro_fusion_auroc"],
            "spearman": m3["severity"]["severity_spearman"],
        },
        {
            "det_auc": real["detection_auroc"],
            "worst_auc": real["detection_worst_auroc"],
            "accuracy": real["detection_accuracy"],
            "fpr": real["detection_false_positive_rate"],
            "sev_auc": real["severity_auroc"],
            "spearman": real["severity_spearman"],
        },
    ]
    return names, data


def annotate_bars(
    ax: plt.Axes,
    bars,
    fmt: str = ".3f",
    dy: float = 0.004,
    rotation: float = 0,
) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + dy,
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=6.3,
            rotation=rotation,
            color=INK,
        )


def make_fig3_main_results() -> None:
    names, data = main_results()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(mm(DOUBLE_COLUMN_MM), mm(66)),
        gridspec_kw={"width_ratios": [1.45, 0.72, 1.0]},
    )
    x = np.arange(len(names))

    ax = axes[0]
    width = 0.24
    det_metrics = [
        ("Detection AUROC", "det_auc", BLUE, ""),
        ("Worst AUROC", "worst_auc", ORANGE, "///"),
        ("Accuracy", "accuracy", TEAL, "xx"),
    ]
    for i, (label, key, color, hatch) in enumerate(det_metrics):
        values = [row[key] for row in data]
        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            label=label,
            color=color,
            edgecolor=INK,
            linewidth=0.45,
            hatch=hatch,
            zorder=3,
        )
        annotate_bars(ax, bars, dy=0.003, rotation=90)
    ax.set_xticks(x, names)
    ax.set_ylim(0.86, 1.075)
    ax.set_ylabel("Score")
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=3,
        handlelength=1.2,
        columnspacing=0.9,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=6.1,
    )
    style_axis(ax)
    panel_label(ax, "(a)")

    ax = axes[1]
    fpr_values = [row["fpr"] for row in data]
    bars = ax.bar(
        x,
        fpr_values,
        0.55,
        color=[BLUE, TEAL, ORANGE],
        edgecolor=INK,
        linewidth=0.5,
        hatch=["", "xx", "///"],
        zorder=3,
    )
    annotate_bars(ax, bars, fmt=".3f", dy=0.0015)
    ax.set_xticks(x, names, rotation=22, ha="right")
    ax.set_ylim(0, max(0.035, 1.25 * max(fpr_values)))
    ax.set_ylabel("False-positive rate")
    style_axis(ax)
    panel_label(ax, "(b)")

    ax = axes[2]
    width = 0.34
    sev_auc = ax.bar(
        x - width / 2,
        [row["sev_auc"] for row in data],
        width,
        color=PURPLE,
        edgecolor=INK,
        linewidth=0.5,
        label="Severity-head AUROC",
        zorder=3,
    )
    srocc = ax.bar(
        x + width / 2,
        [row["spearman"] for row in data],
        width,
        color=ORANGE,
        edgecolor=INK,
        linewidth=0.5,
        hatch="///",
        label="SROCC",
        zorder=3,
    )
    annotate_bars(ax, sev_auc, dy=0.003, rotation=90)
    annotate_bars(ax, srocc, dy=0.003, rotation=90)
    ax.set_xticks(x, names, rotation=22, ha="right")
    ax.set_ylim(0.80, 1.065)
    ax.set_ylabel("Score")
    ax.legend(
        frameon=False,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.0),
        fontsize=6.2,
    )
    style_axis(ax)
    panel_label(ax, "(c)")

    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.20, top=0.93, wspace=0.32)
    save_figure(fig, "Fig3_cross_dataset_results")


def load_baseline_rows() -> list[dict[str, str]]:
    with (ROOT / "results" / "fusion_temporal_quality_baseline_table.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        return list(csv.DictReader(handle))


def baseline_lookup(rows: list[dict[str, str]], dataset: str, method: str, metric: str) -> float:
    for row in rows:
        if row["dataset"] == dataset and row["method"] == method:
            return float(row[metric])
    raise KeyError((dataset, method, metric))


def horizontal_bars(
    ax: plt.Axes,
    labels: list[str],
    values: list[float],
    metric: str,
    letter: str,
    xlim: tuple[float, float],
) -> None:
    y = np.arange(len(labels))
    colors = [ORANGE] + [GRAY] * (len(labels) - 1)
    hatches = ["///"] + ["" if i % 2 else "xx" for i in range(1, len(labels))]
    bars = ax.barh(
        y,
        values,
        color=colors,
        edgecolor=INK,
        linewidth=0.45,
        height=0.62,
        zorder=3,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(metric)
    for yi, value in zip(y, values):
        ax.text(value + (xlim[1] - xlim[0]) * 0.012, yi, f"{value:.3f}", va="center", fontsize=6.5)
    style_axis(ax, "x")
    panel_label(ax, letter, x=-0.20, y=1.02)


def make_fig4_baseline_comparison() -> None:
    rows = load_baseline_rows()
    fig, axes = plt.subplots(2, 2, figsize=(mm(DOUBLE_COLUMN_MM), mm(114)))
    vid = "VidLLVIP independent"
    m3 = "M3SVD 30-pair"
    vid_labels = [
        "Ours (dual output)",
        "Global weight drift",
        "SEA-RAFT flowD",
        "ResNet-18 feaCD",
        "MI instability",
        "Entropy instability",
    ]
    vid_methods = [
        ("Ours detection head", "Ours severity head"),
        ("Global weight drift", "Global weight drift"),
        ("SEA-RAFT flowD visible-source probe", "SEA-RAFT flowD visible-source probe"),
        ("ResNet-18 feaCD visible-source probe", "ResNet-18 feaCD visible-source probe"),
        ("Mutual-information instability", "Mutual-information instability"),
        ("Entropy instability", "Entropy instability"),
    ]
    vid_auc = [baseline_lookup(rows, vid, pair[0], "auroc") for pair in vid_methods]
    vid_srocc = [baseline_lookup(rows, vid, pair[1], "spearman") for pair in vid_methods]
    horizontal_bars(axes[0, 0], vid_labels, vid_auc, "Detection AUROC", "(a)", (0, 1.08))
    horizontal_bars(axes[0, 1], vid_labels, vid_srocc, "Severity SROCC", "(b)", (0, 1.08))

    m3_labels = [
        "Ours (dual output)",
        "Global weight drift",
        "SEA-RAFT flowD",
        "Temporal difference",
        "feaCD Sobel",
    ]
    m3_methods = [
        ("Ours detection head", "Ours severity head"),
        ("Global weight drift", "Global weight drift"),
        ("SEA-RAFT flowD visible-source probe", "SEA-RAFT flowD visible-source probe"),
        ("Temporal difference", "Temporal difference"),
        ("TemCoCo-style feaCD Sobel", "TemCoCo-style feaCD Sobel"),
    ]
    m3_auc = [baseline_lookup(rows, m3, pair[0], "auroc") for pair in m3_methods]
    m3_srocc = [baseline_lookup(rows, m3, pair[1], "spearman") for pair in m3_methods]
    horizontal_bars(axes[1, 0], m3_labels, m3_auc, "Detection AUROC", "(c)", (0, 1.08))
    horizontal_bars(axes[1, 1], m3_labels, m3_srocc, "Severity SROCC", "(d)", (0, 1.08))
    axes[0, 0].text(0.02, 1.01, "VidLLVIP", transform=axes[0, 0].transAxes, fontweight="bold")
    axes[1, 0].text(0.02, 1.01, "M3SVD", transform=axes[1, 0].transAxes, fontweight="bold")
    fig.subplots_adjust(left=0.22, right=0.99, bottom=0.075, top=0.965, hspace=0.42, wspace=0.58)
    save_figure(fig, "Fig4_baseline_comparison")


def make_fig5_family_response() -> None:
    rows = load_json("fusion_temporal_quality_independent.json")["raw_rows"]["independent"]
    groups: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        groups[(row["family"], float(row["severity"]))].append(float(row["severity_ranking_score"]))
    severities = [0.025, 0.06, 0.11]
    clean_values = groups[("clean", 0.0)]
    clean_mean = float(np.mean(clean_values))
    clean_sem = float(np.std(clean_values, ddof=1) / np.sqrt(len(clean_values)))

    fig, axes = plt.subplots(1, 2, figsize=(mm(DOUBLE_COLUMN_MM), mm(67)), sharey=True)
    configurations = [
        (
            axes[0],
            ["aperiodic_gain", "local_gain", "local_weight", "patch_lag"],
            ["Aperiodic gain", "Local gain", "Local weight", "Patch lag"],
            [BLUE, ORANGE, TEAL, RED],
            ["o", "s", "^", "D"],
            ["-", "--", "-.", ":"],
            "(a)",
        ),
        (
            axes[1],
            ["source_global_gain", "source_local_gain", "source_cross_exposure"],
            ["Global source gain", "Local source gain", "Cross exposure"],
            [BLUE, ORANGE, TEAL],
            ["o", "s", "^"],
            ["-", "--", "-."],
            "(b)",
        ),
    ]
    for ax, families, labels, colors, markers, linestyles, letter in configurations:
        ax.axhspan(
            max(0, clean_mean - 1.96 * clean_sem),
            clean_mean + 1.96 * clean_sem,
            color=GRAY,
            alpha=0.16,
            linewidth=0,
            label="Clean 95% CI",
            zorder=0,
        )
        ax.axhline(clean_mean, color=GRAY, linestyle="--", linewidth=0.9)
        for family, label, color, marker, linestyle in zip(
            families, labels, colors, markers, linestyles
        ):
            means = []
            errors = []
            for severity in severities:
                values = np.asarray(groups[(family, severity)], dtype=float)
                means.append(float(values.mean()))
                errors.append(float(values.std(ddof=1) / np.sqrt(len(values))))
            ax.errorbar(
                severities,
                means,
                yerr=errors,
                color=color,
                marker=marker,
                markerfacecolor="white",
                markeredgewidth=0.9,
                linestyle=linestyle,
                capsize=2.2,
                label=label,
                zorder=3,
            )
        ax.set_xticks(severities, ["0.025", "0.060", "0.110"])
        ax.set_xlabel("Injected severity")
        ax.set_ylim(0, 1.86)
        ax.legend(frameon=False, loc="upper left")
        style_axis(ax)
        panel_label(ax, letter)
    axes[0].set_ylabel("Severity ranking score")
    axes[0].text(0.98, 0.96, "Fusion artifacts", transform=axes[0].transAxes, ha="right", va="top", fontweight="bold")
    axes[1].text(0.98, 0.96, "Source-explainable changes", transform=axes[1].transAxes, ha="right", va="top", fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.18, top=0.93, wspace=0.22)
    save_figure(fig, "Fig5_artifact_and_hard_negative_response")


def load_video_window(path: Path, start: int, length: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if start <= index < start + length:
            frames.append(cv2.resize(frame, (320, 256), interpolation=cv2.INTER_AREA))
        if index >= start + length:
            break
        index += 1
    capture.release()
    if len(frames) != length:
        raise RuntimeError(f"Expected {length} frames from {path}, found {len(frames)}")
    return frames


def gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def warp_with_flow(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = frame.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        frame,
        xx - flow[..., 0],
        yy - flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def source_residual_map(
    visible: list[np.ndarray],
    thermal: list[np.ndarray],
    fused: list[np.ndarray],
    index: int,
    grid: tuple[int, int] = (4, 5),
) -> np.ndarray:
    v_prev, v_now = gray(visible[index - 1]), gray(visible[index])
    t_prev, t_now = gray(thermal[index - 1]), gray(thermal[index])
    f_prev, f_now = gray(fused[index - 1]), gray(fused[index])
    ref_prev = 0.5 * (v_prev + t_prev)
    ref_now = 0.5 * (v_now + t_now)
    flow = cv2.calcOpticalFlowFarneback(ref_prev, ref_now, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    dv = v_now - warp_with_flow(v_prev, flow)
    dt = t_now - warp_with_flow(t_prev, flow)
    df = f_now - warp_with_flow(f_prev, flow)
    height, width = df.shape
    residual = np.zeros_like(df)
    for row in range(grid[0]):
        y1, y2 = row * height // grid[0], (row + 1) * height // grid[0]
        for col in range(grid[1]):
            x1, x2 = col * width // grid[1], (col + 1) * width // grid[1]
            one = dv[y1:y2, x1:x2].ravel().astype(np.float64)
            two = dt[y1:y2, x1:x2].ravel().astype(np.float64)
            target = df[y1:y2, x1:x2].ravel().astype(np.float64)
            design = np.column_stack((one, two))
            gram = design.T @ design + np.eye(2) * 1e-5
            coeff = np.clip(np.linalg.solve(gram, design.T @ target), -0.5, 1.5)
            error = np.abs(target - design @ coeff)
            scale = 0.65 * np.sqrt(np.mean(target**2)) + 0.35 * np.sqrt(
                np.mean(one**2) + np.mean(two**2)
            ) + 1e-5
            residual[y1:y2, x1:x2] = (error / scale).reshape(y2 - y1, x2 - x1)
    return cv2.GaussianBlur(residual, (0, 0), 1.1)


def rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def make_fig6_qualitative() -> None:
    base = ROOT / "data" / "raw" / "VidLLVIP_subset"
    name = "03_0000_0005.mp4"
    visible = load_video_window(base / "vi" / name, start=50, length=20)
    thermal = load_video_window(base / "ir" / name, start=50, length=20)
    clean = fuse_frames(visible, thermal, "average")
    varied_v, varied_t = inject_source_variation(visible, thermal, "source_cross_exposure", 0.11)
    hard_negative = fuse_frames(varied_v, varied_t, "average")
    artifact = inject_temporal_artifact(clean, visible, thermal, "patch_lag", 0.11)
    index = 11

    hard_residual = source_residual_map(varied_v, varied_t, hard_negative, index)
    artifact_residual = source_residual_map(visible, thermal, artifact, index)
    clip_max = float(np.quantile(np.concatenate([hard_residual.ravel(), artifact_residual.ravel()]), 0.985))
    clip_max = max(clip_max, 1e-3)

    hard_change = np.abs(gray(hard_negative[index]) - gray(hard_negative[index - 1]))
    artifact_change = np.abs(gray(artifact[index]) - gray(artifact[index - 1]))
    change_max = float(np.quantile(np.concatenate([hard_change.ravel(), artifact_change.ravel()]), 0.99))
    change_max = max(change_max, 1e-3)

    fig = plt.figure(figsize=(mm(DOUBLE_COLUMN_MM), mm(82)))
    grid = fig.add_gridspec(
        2,
        6,
        width_ratios=[0.55, 1, 1, 1, 1, 1],
        left=0.012,
        right=0.995,
        bottom=0.025,
        top=0.93,
        wspace=0.035,
        hspace=0.06,
    )
    label_axes = [fig.add_subplot(grid[row, 0]) for row in range(2)]
    axes = np.asarray(
        [[fig.add_subplot(grid[row, col + 1]) for col in range(5)] for row in range(2)]
    )
    rows = [
        (
            varied_v,
            varied_t,
            hard_negative,
            hard_change,
            hard_residual,
            "Source-\ndriven\nchange",
        ),
        (visible, thermal, artifact, artifact_change, artifact_residual, "Patch-\nlag\nartifact"),
    ]
    headings = ["Visible", "Thermal", "Fusion output", "$|\\Delta F|$", "Source-conditioned residual"]
    for col, heading in enumerate(headings):
        axes[0, col].text(0.5, 1.03, heading, transform=axes[0, col].transAxes, ha="center", va="bottom", fontsize=7.2)
    for row_index, (v, t, f, change, residual, row_label) in enumerate(rows):
        label_axes[row_index].axis("off")
        label_axes[row_index].text(
            0.02,
            0.5,
            f"({chr(97 + row_index)})\n{row_label}",
            transform=label_axes[row_index].transAxes,
            ha="left",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            linespacing=1.25,
        )
        if GRAYSCALE:
            images = [gray(v[index]), gray(t[index]), gray(f[index]), change, residual]
            cmaps = ["gray", "gray", "gray", "gray_r", "gray_r"]
        else:
            images = [rgb(v[index]), rgb(t[index]), rgb(f[index]), change, residual]
            cmaps = [None, "gray", "gray", "magma", "magma"]
        vmaxs = [None, None, None, change_max, clip_max]
        for col, (image, cmap, vmax) in enumerate(zip(images, cmaps, vmaxs)):
            axes[row_index, col].imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=vmax)
            axes[row_index, col].set_xticks([])
            axes[row_index, col].set_yticks([])
            for spine in axes[row_index, col].spines.values():
                spine.set_linewidth(0.55)
                spine.set_color(IMAGE_BORDER)
    if GRAYSCALE:
        axes[1, 2].add_patch(
            patches.Rectangle((160, 38), 150, 180, fill=False, edgecolor="white", linewidth=1.8)
        )
        axes[1, 2].add_patch(
            patches.Rectangle(
                (160, 38),
                150,
                180,
                fill=False,
                edgecolor="black",
                linewidth=0.8,
                linestyle="--",
            )
        )
    else:
        axes[1, 2].add_patch(
            patches.Rectangle((160, 38), 150, 180, fill=False, edgecolor="#F0E442", linewidth=1.1)
        )
    save_figure(fig, "Fig6_qualitative_residual_case", raster_content=True)


def make_sfig1_component_ablation() -> None:
    rows = load_baseline_rows()
    datasets = ["VidLLVIP independent", "M3SVD 30-pair"]
    labels = [
        "Full severity head",
        "Span residual",
        "Source-conditioned",
        "Static operator",
        "Global weight drift",
    ]
    methods = [
        "Ours severity head",
        "Span residual",
        "Source-conditioned residual",
        "Static operator residual",
        "Global weight drift",
    ]
    values = np.asarray(
        [[baseline_lookup(rows, dataset, method, "spearman") for method in methods] for dataset in datasets]
    )
    fig, ax = plt.subplots(figsize=(mm(DOUBLE_COLUMN_MM), mm(61)))
    x = np.arange(len(labels))
    width = 0.34
    bars1 = ax.bar(x - width / 2, values[0], width, color=BLUE, edgecolor=INK, linewidth=0.45, label="VidLLVIP", zorder=3)
    bars2 = ax.bar(x + width / 2, values[1], width, color=ORANGE, edgecolor=INK, linewidth=0.45, hatch="///", label="M3SVD", zorder=3)
    annotate_bars(ax, bars1, dy=0.005)
    annotate_bars(ax, bars2, dy=0.005)
    ax.set_xticks(x, labels, rotation=18, ha="right")
    ax.set_ylim(0.70, 0.96)
    ax.set_ylabel("Severity SROCC")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    style_axis(ax)
    panel_label(ax, "(a)", x=-0.07)
    fig.subplots_adjust(left=0.08, right=0.995, bottom=0.28, top=0.94)
    save_figure(fig, "SFig1_component_ablation")


def make_sfig2_loso_robustness() -> None:
    obj = load_json("fusion_temporal_quality_split_robustness.json")
    source_names = sorted(obj["leave_one_out"], key=lambda value: int(value.split("_")[-1]))
    ids = np.asarray([int(name.split("_")[-1]) for name in source_names])
    det_auc = np.asarray([obj["leave_one_out"][name]["summary"]["artifact_detection_score"]["macro_fusion_auroc"] for name in source_names])
    sev_auc = np.asarray([obj["leave_one_out"][name]["summary"]["severity_ranking_score"]["macro_fusion_auroc"] for name in source_names])
    sev_srocc = np.asarray([obj["leave_one_out"][name]["summary"]["severity_ranking_score"]["severity_spearman"] for name in source_names])
    det_fpr = np.asarray([obj["leave_one_out"][name]["summary"]["artifact_detection_score"]["calibrated_false_positive_rate"] for name in source_names])
    sev_fpr = np.asarray([obj["leave_one_out"][name]["summary"]["severity_ranking_score"]["calibrated_false_positive_rate"] for name in source_names])

    fig, axes = plt.subplots(1, 3, figsize=(mm(DOUBLE_COLUMN_MM), mm(61)))
    plots = [
        (axes[0], [(det_auc, "Detection AUROC", BLUE, "o"), (sev_auc, "Severity-head AUROC", ORANGE, "s")], "AUROC", (0.94, 1.005), "(a)"),
        (axes[1], [(sev_srocc, "Severity SROCC", TEAL, "D")], "SROCC", (0.86, 1.005), "(b)"),
        (axes[2], [(det_fpr, "Detection FPR", BLUE, "o"), (sev_fpr, "Severity FPR", ORANGE, "s")], "False-positive rate", (0.0, max(0.12, float(max(det_fpr.max(), sev_fpr.max()) + 0.015))), "(c)"),
    ]
    for ax, series, ylabel, ylim, letter in plots:
        for values, label, color, marker in series:
            ax.plot(ids, values, color=color, marker=marker, markerfacecolor="white", label=label)
            ax.axhline(float(values.mean()), color=color, linestyle="--", linewidth=0.75, alpha=0.8)
        ax.set_xticks([1, 4, 7, 10, 14])
        ax.set_xlabel("Held-out source ID")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.legend(frameon=False, loc="lower left" if ylabel != "False-positive rate" else "upper left")
        style_axis(ax)
        panel_label(ax, letter)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.20, top=0.93, wspace=0.35)
    save_figure(fig, "SFig2_leave_one_source_out")


def write_manifest() -> None:
    names, data = main_results()
    lines = [
        "# SIVP Figure Manifest",
        "",
        "All numerical panels are generated from the result JSON/CSV files in this workspace. The qualitative panel uses an actual VidLLVIP clip with the frozen synthetic source-change and patch-lag generators. The concept panel is explicitly schematic.",
        "",
        "## Main figures",
        "",
        "1. `Fig1_method_overview`: Overview of the proposed source-referenced temporal artifact assessment framework. A shared reference motion aligns consecutive RGB-T source and fused frames. Local source explainability provides the detection evidence, while span, global weight drift, source-conditioned, and static operator residuals form the severity head.",
        "2. `Fig2_source_explainability_concept`: Schematic distinction among source-driven temporal change, fusion-only artifacts, and localized source-unexplained inconsistency. Shading denotes residual evidence not explained by the source transitions.",
        "3. `Fig3_cross_dataset_results`: Detection and severity performance on VidLLVIP, M3SVD, and nine retained fusion-output groups from the two public TemCoCo comparison videos. The author-confirmed final output-level AUROC and SROCC values are group-level macro averages; accuracy and FPR are pooled over a balanced 180-sample test set. Exploratory GIF-pilot files use different protocols and are not plotted.",
        "4. `Fig4_baseline_comparison`: Comparison with temporal, flow-based, feature-based, and framewise fusion-quality baselines on VidLLVIP and M3SVD. `SEA-RAFT flowD` and `ResNet-18 feaCD` are local probes, not official TemCoCo results.",
        "5. `Fig5_artifact_and_hard_negative_response`: Severity-score response to fusion artifacts and source-explainable hard negatives on the VidLLVIP independent split. Curves show mean and standard error; the gray band shows the clean 95% confidence interval.",
        "6. `Fig6_qualitative_residual_case`: Qualitative comparison of a source-driven cross-exposure change and an injected patch-lag artifact using VidLLVIP source 03. The source-conditioned residual remains low for the explainable change and localizes the fusion-only lag.",
        "",
        "## Supplementary figures",
        "",
        "- `SFig1_component_ablation`: Severity-head component comparison on VidLLVIP and M3SVD.",
        "- `SFig2_leave_one_source_out`: Per-source leave-one-source-out detection/severity AUROC, severity SROCC, and FPR.",
        "",
        "## Main-result values rendered in Fig. 3",
        "",
        "| Dataset | Detection AUROC | Worst AUROC | Accuracy | FPR | Severity-head AUROC | Severity SROCC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in zip(names, data):
        lines.append(
            f"| {name} | {row['det_auc']:.4f} | {row['worst_auc']:.4f} | {row['accuracy']:.4f} | {row['fpr']:.4f} | {row['sev_auc']:.4f} | {row['spearman']:.4f} |"
        )
    lines += [
        "",
        "## File formats",
        "",
        f"- Rendering theme: {'grayscale print-safe' if GRAYSCALE else 'color online'}.",
        "- Vector figures: EPS, PDF, SVG, and 600 dpi PNG.",
        "- Qualitative combination figure: PDF, SVG, and 600 dpi PNG.",
        "- Final width: 174 mm (Springer double-column width).",
        "- Figure lettering: Arial, approximately 8 pt at final size.",
    ]
    (OUT / "figure_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs() -> None:
    expected = [
        ("Fig1_method_overview", False),
        ("Fig2_source_explainability_concept", False),
        ("Fig3_cross_dataset_results", False),
        ("Fig4_baseline_comparison", False),
        ("Fig5_artifact_and_hard_negative_response", False),
        ("Fig6_qualitative_residual_case", True),
        ("SFig1_component_ablation", False),
        ("SFig2_leave_one_source_out", False),
    ]
    problems = []
    report = []
    for stem, raster in expected:
        suffixes = [".pdf", ".svg", ".png"] + ([] if raster else [".eps"])
        missing = [suffix for suffix in suffixes if not (OUT / f"{stem}{suffix}").exists()]
        if missing:
            problems.append(f"{stem}: missing {missing}")
            continue
        image = cv2.imread(str(OUT / f"{stem}.png"), cv2.IMREAD_COLOR)
        if image is None:
            problems.append(f"{stem}: PNG unreadable")
            continue
        height, width = image.shape[:2]
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        nonwhite = float(np.mean(gray_image < 250))
        channel_delta = int(
            max(
                np.max(np.abs(image[:, :, 0].astype(np.int16) - image[:, :, 1].astype(np.int16))),
                np.max(np.abs(image[:, :, 1].astype(np.int16) - image[:, :, 2].astype(np.int16))),
            )
        )
        if width < 4000:
            problems.append(f"{stem}: width {width}px is below 600 dpi at 174 mm")
        if nonwhite < 0.01:
            problems.append(f"{stem}: appears nearly blank")
        if GRAYSCALE and channel_delta != 0:
            problems.append(f"{stem}: grayscale PNG has channel delta {channel_delta}")
        pdf_bytes = (OUT / f"{stem}.pdf").read_bytes()
        media_box = re.search(
            rb"/MediaBox\s*\[\s*0\s+0\s+([0-9.]+)\s+([0-9.]+)\s*\]",
            pdf_bytes,
        )
        if media_box is None:
            problems.append(f"{stem}: PDF MediaBox not found")
            width_mm = None
            height_mm = None
        else:
            width_mm = float(media_box.group(1)) / 72.0 * MM_PER_INCH
            height_mm = float(media_box.group(2)) / 72.0 * MM_PER_INCH
            if abs(width_mm - DOUBLE_COLUMN_MM) > 0.1:
                problems.append(f"{stem}: PDF width is {width_mm:.2f} mm, expected 174 mm")
        fonts_embedded = b"/FontFile2" in pdf_bytes or b"/FontFile3" in pdf_bytes
        if not fonts_embedded:
            problems.append(f"{stem}: no embedded font stream found in PDF")
        report.append(
            {
                "stem": stem,
                "width_px": width,
                "height_px": height,
                "pdf_width_mm": None if width_mm is None else round(width_mm, 2),
                "pdf_height_mm": None if height_mm is None else round(height_mm, 2),
                "fonts_embedded": fonts_embedded,
                "max_rgb_channel_delta": channel_delta,
                "nonwhite_fraction": round(nonwhite, 4),
            }
        )
    (OUT / "figure_qa.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if problems:
        raise RuntimeError("; ".join(problems))


def main() -> None:
    configure_style()
    make_fig1_method_overview()
    make_fig2_source_explainability()
    make_fig3_main_results()
    make_fig4_baseline_comparison()
    make_fig5_family_response()
    make_fig6_qualitative()
    make_sfig1_component_ablation()
    make_sfig2_loso_robustness()
    write_manifest()
    verify_outputs()
    print(f"Generated and verified SIVP figure assets in {OUT}")


if __name__ == "__main__":
    main()
