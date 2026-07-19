"""SEA-RAFT flowD-style baseline on VidLLVIP independent split.

This uses the TemCoCo-bundled SEA-RAFT implementation plus a locally downloaded
SEA-RAFT checkpoint.  It is closer to TemCoCo flowD than the Farneback surrogate,
but it is still integrated locally for the current synthetic-hard-negative
protocol.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cursor_rgbt.fusion_temporal_quality import (
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
    _warp_with_flow,
)
from cursor_rgbt.temporal_sync import _gray
from run_fusion_temporal_quality_external import (
    BENIGN_FAMILIES,
    FUSION_METHODS,
    ROOT,
    VIDEOS,
    WINDOW_LENGTH,
    WINDOW_STARTS,
    load_video,
)


TEMCOCO = ROOT / "models/TemCoCo"
SEARAFT_CHECKPOINT = (
    ROOT
    / "models/SEA_RAFT_models/Tartan-C-T-TSKH-kitti432x960-M/Tartan-C-T-TSKH-kitti432x960-M.pth"
)
FLOW_INPUT_SIZE = (960, 432)
FLOW_SCORE_SIZE = (160, 128)
CALIBRATION_SOURCES = (1, 2, 4, 5, 7, 8, 10, 14)
INDEPENDENT_SOURCES = (3, 6, 9, 11, 12, 13)
CALIBRATION_FAMILIES = ("global_gain", "global_weight")
INDEPENDENT_FAMILIES = ("local_weight", "aperiodic_gain", "local_gain", "patch_lag")
SEVERITIES = (0.025, 0.06, 0.11)
METHOD = "searaft_flowd_visible"


def _ensure_temcoco_imports() -> None:
    temcoco = str(TEMCOCO)
    core = str(TEMCOCO / "SEA_RAFT/core")
    if temcoco not in sys.path:
        sys.path.insert(0, temcoco)
    if core not in sys.path:
        sys.path.insert(0, core)


@lru_cache(maxsize=1)
def load_searaft():
    _ensure_temcoco_imports()
    from SEA_RAFT.config.parser import json_to_args
    from SEA_RAFT.core.raft import RAFT
    from SEA_RAFT.core.utils.utils import load_ckpt

    if not SEARAFT_CHECKPOINT.exists():
        raise FileNotFoundError(SEARAFT_CHECKPOINT)
    args = json_to_args(str(TEMCOCO / "SEA_RAFT/config/eval/kitti-M.json"))
    model = RAFT(args)
    load_ckpt(model, str(SEARAFT_CHECKPOINT))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, args, device


def frame_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    if frame.ndim == 2:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, FLOW_INPUT_SIZE, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0)
    return tensor.to(device)


@torch.inference_mode()
def searaft_flow(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    model, args, device = load_searaft()
    original_height, original_width = FLOW_SCORE_SIZE[1], FLOW_SCORE_SIZE[0]
    first = frame_tensor(previous, device)
    second = frame_tensor(current, device)
    output = model(first, second, iters=args.iters, test_mode=True)
    flow = output["flow"][-1]
    flow = F.interpolate(
        flow,
        size=(original_height, original_width),
        mode="bilinear",
        align_corners=False,
    )
    flow = flow.squeeze(0).permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
    flow[..., 0] *= original_width / FLOW_INPUT_SIZE[0]
    flow[..., 1] *= original_height / FLOW_INPUT_SIZE[1]
    return flow


def visible_flows(visible) -> list[np.ndarray]:
    resized = [
        cv2.resize(frame, FLOW_SCORE_SIZE, interpolation=cv2.INTER_AREA)
        for frame in visible
    ]
    return [
        searaft_flow(previous, current)
        for previous, current in zip(resized[:-1], resized[1:])
    ]


def flowd_from_cached_flows(fused, flows: list[np.ndarray]) -> float:
    output = [_gray(frame, FLOW_SCORE_SIZE) for frame in fused]
    errors = []
    for previous, current, flow in zip(output[:-1], output[1:], flows):
        warped = _warp_with_flow(previous, flow)
        errors.append(float(np.mean(np.abs(current - warped))))
    return float(np.mean(errors))


def score_rows(source_ids, artifact_families):
    rows = []
    for source_id in source_ids:
        name = f"{source_id:02d}_0000_0005.mp4"
        visible = load_video(VIDEOS / "vi" / name)
        thermal = load_video(VIDEOS / "ir" / name)
        for start in WINDOW_STARTS:
            first = visible[start : start + WINDOW_LENGTH]
            second = thermal[start : start + WINDOW_LENGTH]
            flows = visible_flows(first)
            for fusion_method in FUSION_METHODS:
                clean = fuse_frames(first, second, fusion_method)
                rows.append(
                    {
                        "source_id": source_id,
                        "start": start,
                        "fusion_method": fusion_method,
                        "family": "clean",
                        "severity": 0.0,
                        "distorted": False,
                        METHOD: flowd_from_cached_flows(clean, flows),
                    }
                )
                for family in BENIGN_FAMILIES:
                    for severity in SEVERITIES:
                        varied_first, varied_second = inject_source_variation(
                            first, second, family, severity
                        )
                        varied_clean = fuse_frames(
                            varied_first, varied_second, fusion_method
                        )
                        rows.append(
                            {
                                "source_id": source_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": False,
                                METHOD: flowd_from_cached_flows(varied_clean, flows),
                            }
                        )
                for family in artifact_families:
                    for severity in SEVERITIES:
                        altered = inject_temporal_artifact(
                            clean, first, second, family, severity
                        )
                        rows.append(
                            {
                                "source_id": source_id,
                                "start": start,
                                "fusion_method": fusion_method,
                                "family": family,
                                "severity": severity,
                                "distorted": True,
                                METHOD: flowd_from_cached_flows(altered, flows),
                            }
                        )
        print(f"processed source {source_id:02d}")
    return rows


def thresholds(rows):
    return {
        fusion_method: float(
            np.quantile(
                [
                    row[METHOD]
                    for row in rows
                    if row["fusion_method"] == fusion_method
                    and not row["distorted"]
                ],
                0.95,
            )
        )
        for fusion_method in FUSION_METHODS
    }


def safe_spearman(labels, scores) -> float:
    statistic = float(spearmanr(labels, scores).statistic)
    return statistic if np.isfinite(statistic) else 0.0


def summarize(rows, fitted_thresholds):
    predictions = []
    truths = []
    aucs = {}
    correlations = []
    for fusion_method in FUSION_METHODS:
        selected = [row for row in rows if row["fusion_method"] == fusion_method]
        truth = np.asarray([row["distorted"] for row in selected], dtype=bool)
        scores = np.asarray([row[METHOD] for row in selected], dtype=float)
        aucs[fusion_method] = float(roc_auc_score(truth, scores))
        predictions.extend(scores >= fitted_thresholds[fusion_method])
        truths.extend(truth)
        for family in sorted({row["family"] for row in selected if row["distorted"]}):
            family_rows = [
                row for row in selected if row["family"] in ("clean", family)
            ]
            correlations.append(
                safe_spearman(
                    [row["severity"] for row in family_rows],
                    [row[METHOD] for row in family_rows],
                )
            )
    predictions = np.asarray(predictions, dtype=bool)
    truths = np.asarray(truths, dtype=bool)
    return {
        "macro_fusion_auroc": float(np.mean(list(aucs.values()))),
        "worst_fusion_auroc": float(min(aucs.values())),
        "calibrated_accuracy": float(np.mean(predictions == truths)),
        "calibrated_false_positive_rate": float(np.mean(predictions[~truths])),
        "severity_spearman": float(np.mean(correlations)),
        "per_fusion_auroc": aucs,
    }


def main() -> None:
    calibration_rows = score_rows(CALIBRATION_SOURCES, CALIBRATION_FAMILIES)
    fitted_thresholds = thresholds(calibration_rows)
    independent_rows = score_rows(INDEPENDENT_SOURCES, INDEPENDENT_FAMILIES)
    summary = summarize(independent_rows, fitted_thresholds)
    report = {
        "status": "searaft_flowd_visible_independent_probe",
        "metric": METHOD,
        "temcoco_directory": str(TEMCOCO),
        "searaft_checkpoint": str(SEARAFT_CHECKPOINT),
        "calibration_sources": list(CALIBRATION_SOURCES),
        "independent_sources": list(INDEPENDENT_SOURCES),
        "calibration_families": list(CALIBRATION_FAMILIES),
        "independent_families": list(INDEPENDENT_FAMILIES),
        "benign_source_variation_families": list(BENIGN_FAMILIES),
        "thresholds": fitted_thresholds,
        "summary": summary,
        "raw_rows": {
            "calibration": calibration_rows,
            "independent": independent_rows,
        },
    }
    output = ROOT / "results/temcoco_searaft_flowd_independent.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "raw_rows"}, indent=2))


if __name__ == "__main__":
    main()
