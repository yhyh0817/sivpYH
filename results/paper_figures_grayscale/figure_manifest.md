# SIVP Figure Manifest

All numerical panels are generated from the result JSON/CSV files in this workspace. The qualitative panel uses an actual VidLLVIP clip with the frozen synthetic source-change and patch-lag generators. The concept panel is explicitly schematic.

## Main figures

1. `Fig1_method_overview`: Overview of the proposed source-referenced temporal artifact assessment framework. A shared reference motion aligns consecutive RGB-T source and fused frames. Local source explainability provides the detection evidence, while span, global weight drift, source-conditioned, and static operator residuals form the severity head.
2. `Fig2_source_explainability_concept`: Schematic distinction among source-driven temporal change, fusion-only artifacts, and localized source-unexplained inconsistency. Shading denotes residual evidence not explained by the source transitions.
3. `Fig3_cross_dataset_results`: Detection and severity performance on VidLLVIP, M3SVD, and nine retained fusion-output groups from the two public TemCoCo comparison videos. The author-confirmed final output-level AUROC and SROCC values are group-level macro averages; accuracy and FPR are pooled over a balanced 180-sample test set. Exploratory GIF-pilot files use different protocols and are not plotted.
4. `Fig4_baseline_comparison`: Comparison with temporal, flow-based, feature-based, and framewise fusion-quality baselines on VidLLVIP and M3SVD. `SEA-RAFT flowD` and `ResNet-18 feaCD` are local probes, not official TemCoCo results.
5. `Fig5_artifact_and_hard_negative_response`: Severity-score response to fusion artifacts and source-explainable hard negatives on the VidLLVIP independent split. Curves show mean and standard error; the gray band shows the clean 95% confidence interval.
6. `Fig6_qualitative_residual_case`: Qualitative comparison of a source-driven cross-exposure change and an injected patch-lag artifact using VidLLVIP source 03. The source-conditioned residual remains low for the explainable change and localizes the fusion-only lag.

## Supplementary figures

- `SFig1_component_ablation`: Severity-head component comparison on VidLLVIP and M3SVD.
- `SFig2_leave_one_source_out`: Per-source leave-one-source-out detection/severity AUROC, severity SROCC, and FPR.

## Main-result values rendered in Fig. 3

| Dataset | Detection AUROC | Worst AUROC | Accuracy | FPR | Severity-head AUROC | Severity SROCC |
|---|---:|---:|---:|---:|---:|---:|
| VidLLVIP | 0.9977 | 0.9966 | 0.9760 | 0.0056 | 0.9948 | 0.9262 |
| M3SVD | 0.9963 | 0.9950 | 0.9527 | 0.0083 | 0.9715 | 0.8682 |
| Fusion outputs | 0.9078 | 0.8808 | 0.8889 | 0.0444 | 0.8972 | 0.8176 |

## File formats

- Rendering theme: grayscale print-safe.
- Vector figures: EPS, PDF, SVG, and 600 dpi PNG.
- Qualitative combination figure: PDF, SVG, and 600 dpi PNG.
- Final width: 174 mm (Springer double-column width).
- Figure lettering: Arial, approximately 8 pt at final size.
