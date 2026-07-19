# Source-referenced temporal quality assessment for RGB-T fusion video

This repository contains the implementation and retained evaluation artifacts for
the manuscript:

> Source-referenced temporal artifact detection and severity assessment for
> infrared-visible fusion video

The method audits a fused video against its synchronized visible and thermal
sources. It produces separate artifact-detection and severity-ranking scores,
and uses source-side exposure changes as hard negatives.

## Repository layout

- `src/cursor_rgbt/fusion_temporal_quality.py`: scoring components and frozen
  artifact generators.
- `scripts/run_fusion_temporal_quality_independent.py`: VidLLVIP calibration and
  independent-source evaluation.
- `scripts/run_fusion_temporal_quality_m3svd_full_test.py`: M3SVD calibration and
  held-out evaluation.
- `scripts/analyze_fusion_temporal_quality_split_robustness.py`: VidLLVIP split
  and leave-one-source-out analysis.
- `scripts/run_framewise_fusion_metric_baselines_independent.py`: framewise
  fusion-quality baselines.
- `scripts/run_temcoco_*`: local flowD/feaCD probes. These are not official
  TemCoCo metric-script reproductions.
- `scripts/figures/generate_sivp_figures.py`: manuscript and supplementary
  figures.
- `results/`: retained JSON rows, summaries, generated figures, and provenance
  records.
- `manuscript/sivp_first_draft/`: manuscript support files and Springer templates.

## Environment

Python 3.11 is used for the reported experiments.

```powershell
python -m pip install -e ".[dev]"
```

The optional learned baseline probes require PyTorch and torchvision:

```powershell
python -m pip install -e ".[baselines]"
```

The reported CPU timing environment used Python 3.11.15, OpenCV 4.12.0,
NumPy 2.2.6, SciPy 1.16.3, and scikit-learn 1.7.2.

## Data layout

Public datasets are not redistributed. Expected local paths are:

```text
data/raw/VidLLVIP_subset/vi/
data/raw/VidLLVIP_subset/ir/
data/raw/M3SVD/full_test/test/visible_Enhance/
data/raw/M3SVD/full_test/test/infrared_Enhance/
```

The output-level panels are taken from the two public TemCoCo comparison videos
stored as:

```text
models/TemCoCo/examples/1207_1714.gif
models/TemCoCo/examples/1207_1739.gif
```

## Reproduce the controlled evaluations

Run commands from the repository root after installing the package:

```powershell
python scripts/run_fusion_temporal_quality_independent.py
python scripts/run_fusion_temporal_quality_m3svd_full_test.py
python scripts/analyze_fusion_temporal_quality_split_robustness.py
python scripts/run_framewise_fusion_metric_baselines_independent.py
```

Generate the print-safe grayscale figures with:

```powershell
$env:SIVP_GRAYSCALE = "1"
python scripts/figures/generate_sivp_figures.py
```

## Result provenance

The VidLLVIP and M3SVD result JSON files retain their calibration and test rows,
splits, thresholds, and summary metrics. The two positive-only SROCC values are
recomputed from those stored rows.

`results/real_fusion_model_confirmed_metrics.json` contains the author-confirmed
final 180-sample output-level summary, nine-model breakdown, pooled confusion
matrix, and derived metrics used in the manuscript.

`results/paper_reported_results.json` is a paper-facing summary of the
reported controlled metrics, baselines, ablations, leave-one-source-out
summary, real-output group table, confusion matrix, and runtime measurements.

Files named `real_fusion_temcoco_gif_*_pilot.json` are earlier exploratory GIF
experiments with different row construction and calibration protocols. Their
scores are not the final 180-sample results and are not pooled into manuscript
tables or figures.

## Manuscript artifacts

Manuscript source text, bibliography, compiled PDFs, and figure PDFs are
intentionally omitted from this public repository.

The DOI-level audit for references added during the final revision is available
in [`docs/reference_verification.md`](docs/reference_verification.md).
