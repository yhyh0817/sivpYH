import cv2
import numpy as np
import pytest

from cursor_rgbt.temporal_sync import (
    _overlap,
    estimate_temporal_lag,
    structural_change_fingerprint,
)


def moving_frames(count=24):
    frames = []
    for index in range(count):
        image = np.zeros((96, 120, 3), dtype=np.uint8)
        x = 5 + 3 * index
        cv2.rectangle(image, (x, 25), (x + 16, 55), (180, 180, 180), -1)
        if index >= 10:
            cv2.circle(image, (80, 10 + 2 * index), 8, (255, 255, 255), -1)
        frames.append(image)
    return frames


def test_overlap_lag_convention():
    first = np.arange(8)
    second = np.arange(8) - 2
    one, two = _overlap(first, second, 2)
    assert np.array_equal(one, two)


def test_fingerprint_dimensions():
    features = structural_change_fingerprint(moving_frames(12), grid=(3, 4))
    assert features.shape == (11, 12)
    assert np.isfinite(features).all()


@pytest.mark.parametrize("delay", [-3, -1, 0, 2, 3])
def test_recovers_synthetic_delay(delay):
    frames = moving_frames(30)
    start, width = 4, 20
    visible = frames[start : start + width]
    thermal = frames[start - delay : start - delay + width]
    estimate = estimate_temporal_lag(visible, thermal, max_lag=3)
    assert estimate.lag == delay
    assert 0.0 <= estimate.confidence <= 1.0


def test_rejects_unequal_stream_lengths():
    with pytest.raises(ValueError):
        estimate_temporal_lag(moving_frames(20), moving_frames(19), max_lag=3)
