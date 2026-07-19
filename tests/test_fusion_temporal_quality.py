import cv2
import numpy as np
import pytest

from cursor_rgbt.fusion_temporal_quality import (
    fuse_frames,
    inject_source_variation,
    inject_temporal_artifact,
    temporal_quality_scores,
)


def paired_sequence(count=10):
    visible, thermal = [], []
    for index in range(count):
        first = np.zeros((96, 128, 3), dtype=np.uint8)
        second = np.zeros_like(first)
        cv2.rectangle(first, (8 + index * 5, 22), (30 + index * 5, 68), (180, 180, 180), -1)
        cv2.circle(second, (20 + index * 5, 45), 13, (230, 230, 230), -1)
        visible.append(first)
        thermal.append(second)
    return visible, thermal


def test_unexplainable_flicker_increases_score():
    visible, thermal = paired_sequence()
    clean = fuse_frames(visible, thermal)
    flicker = inject_temporal_artifact(clean, visible, thermal, "global_gain", 0.12)
    clean_score = temporal_quality_scores(visible, thermal, clean, grid=(2, 2))
    flicker_score = temporal_quality_scores(visible, thermal, flicker, grid=(2, 2))
    assert flicker_score.proposed > clean_score.proposed
    assert clean_score.source_conditioned_residual >= 0.0
    assert clean_score.reliability_gated_residual >= 0.0
    assert clean_score.motion_operator_drift >= 0.0
    assert clean_score.multiscale_source_residual >= 0.0
    assert clean_score.temcoco_flowd_visible >= 0.0
    assert clean_score.temcoco_feacd_sobel >= 0.0


def test_local_gain_flicker_increases_score():
    visible, thermal = paired_sequence()
    clean = fuse_frames(visible, thermal)
    flicker = inject_temporal_artifact(clean, visible, thermal, "local_gain", 0.12)
    assert temporal_quality_scores(visible, thermal, flicker, grid=(2, 2)).proposed > temporal_quality_scores(visible, thermal, clean, grid=(2, 2)).proposed


def test_patch_lag_severity_is_ordered():
    visible, thermal = paired_sequence(count=12)
    clean = fuse_frames(visible, thermal)
    scores = [
        temporal_quality_scores(
            visible,
            thermal,
            inject_temporal_artifact(clean, visible, thermal, "patch_lag", severity),
            grid=(2, 2),
        ).proposed
        for severity in (0.025, 0.06, 0.11)
    ]
    assert scores[0] < scores[1] < scores[2]


def test_source_explainable_gain_is_less_penalized_than_fused_only_gain():
    visible, thermal = paired_sequence()
    explainable_visible, explainable_thermal = inject_source_variation(
        visible, thermal, "source_global_gain", 0.12
    )
    explainable = fuse_frames(explainable_visible, explainable_thermal)
    clean = fuse_frames(visible, thermal)
    fused_only = inject_temporal_artifact(clean, visible, thermal, "global_gain", 0.12)

    explainable_score = temporal_quality_scores(
        explainable_visible, explainable_thermal, explainable, grid=(2, 2)
    )
    fused_only_score = temporal_quality_scores(visible, thermal, fused_only, grid=(2, 2))
    assert fused_only_score.proposed > explainable_score.proposed


def test_stream_length_mismatch():
    visible, thermal = paired_sequence()
    with pytest.raises(ValueError):
        temporal_quality_scores(visible, thermal[:-1], visible)


def test_unknown_fusion_method():
    visible, thermal = paired_sequence()
    with pytest.raises(ValueError):
        fuse_frames(visible, thermal, "unknown")
