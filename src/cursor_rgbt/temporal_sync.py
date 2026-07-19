"""Training-free temporal synchronization for cross-spectral video streams.

The estimator represents every frame transition by spatially distributed
structural-change energy.  Each jointly active tile proposes a lag, then a
confidence-weighted consensus rejects modality-specific local disturbances.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


@dataclass(frozen=True)
class LagEstimate:
    lag: int
    confidence: float
    support: NDArray[np.float64]
    tile_lags: NDArray[np.int64]
    tile_weights: NDArray[np.float64]


def _gray(frame: Frame, size: tuple[int, int]) -> NDArray[np.float32]:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


def _structure(frame: Frame, size: tuple[int, int]) -> NDArray[np.float32]:
    gray = cv2.GaussianBlur(_gray(frame, size), (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.log1p(cv2.magnitude(gx, gy))
    median = float(np.median(magnitude))
    scale = 1.4826 * float(np.median(np.abs(magnitude - median))) + 1e-5
    return np.clip((magnitude - median) / scale, -3.0, 6.0)


def structural_change_fingerprint(
    frames: list[Frame],
    grid: tuple[int, int] = (4, 5),
    size: tuple[int, int] = (240, 192),
) -> NDArray[np.float64]:
    """Return one robust structural-change signal per spatial tile.

    A tile value is the mean of its strongest quartile of temporal gradient
    changes.  This suppresses static cross-spectral appearance while retaining
    moving boundaries shared by visible and thermal imagery.
    """

    if len(frames) < 4:
        raise ValueError("at least four frames are required")
    maps = np.stack([_structure(frame, size) for frame in frames])
    changes = np.abs(np.diff(maps, axis=0))
    rows, cols = grid
    height, width = changes.shape[1:]
    features = np.empty((len(frames) - 1, rows * cols), dtype=np.float64)
    index = 0
    for row in range(rows):
        y1, y2 = row * height // rows, (row + 1) * height // rows
        for col in range(cols):
            x1, x2 = col * width // cols, (col + 1) * width // cols
            tile = changes[:, y1:y2, x1:x2].reshape(len(changes), -1)
            cutoff = np.quantile(tile, 0.75, axis=1, keepdims=True)
            strong = np.where(tile >= cutoff, tile, np.nan)
            features[:, index] = np.nanmean(strong, axis=1)
            index += 1
    return features


def _overlap(
    first: NDArray[np.float64], second: NDArray[np.float64], lag: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compare first[t] with second[t + lag]."""

    if lag > 0:
        return first[:-lag], second[lag:]
    if lag < 0:
        return first[-lag:], second[:lag]
    return first, second


def _correlation(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    if len(first) < 5:
        return -1.0
    one = (first - np.median(first)) / (
        1.4826 * np.median(np.abs(first - np.median(first))) + 1e-8
    )
    two = (second - np.median(second)) / (
        1.4826 * np.median(np.abs(second - np.median(second))) + 1e-8
    )
    one, two = np.clip(one, -4.0, 4.0), np.clip(two, -4.0, 4.0)
    if one.std() < 1e-7 or two.std() < 1e-7:
        return -1.0
    return float(np.corrcoef(one, two)[0, 1])


def lag_profile(
    first: NDArray[np.float64], second: NDArray[np.float64], max_lag: int
) -> NDArray[np.float64]:
    lags = range(-max_lag, max_lag + 1)
    return np.asarray(
        [_correlation(*_overlap(first, second, lag)) for lag in lags],
        dtype=np.float64,
    )


def estimate_temporal_lag(
    visible: list[Frame],
    thermal: list[Frame],
    max_lag: int = 5,
    grid: tuple[int, int] = (4, 5),
) -> LagEstimate:
    """Estimate the lag needed to advance through the thermal stream.

    Positive ``lag`` means that thermal[t + lag] best matches visible[t].
    """

    if len(visible) != len(thermal):
        raise ValueError("streams must contain the same number of frames")
    # Fingerprints contain N-1 transitions; every tested lag must retain at
    # least five paired transitions for a defined correlation.
    if len(visible) - 1 - max_lag < 5:
        raise ValueError("stream is too short for the requested lag range")
    first = structural_change_fingerprint(visible, grid)
    second = structural_change_fingerprint(thermal, grid)
    tile_count = first.shape[1]
    profiles = np.stack(
        [lag_profile(first[:, tile], second[:, tile], max_lag) for tile in range(tile_count)]
    )
    activity = np.sqrt(np.std(first, axis=0) * np.std(second, axis=0))
    active_threshold = float(np.quantile(activity, 0.35))
    active = activity >= max(active_threshold, 1e-6)

    order = np.argsort(profiles, axis=1)
    best_indices = order[:, -1]
    best = profiles[np.arange(tile_count), best_indices]
    second_best = profiles[np.arange(tile_count), order[:, -2]]
    margins = np.maximum(best - second_best, 0.0)
    normalized_activity = activity / (np.median(activity[active]) + 1e-8)
    weights = (
        active.astype(np.float64)
        * np.clip(normalized_activity, 0.25, 4.0)
        * np.maximum(best, 0.0) ** 2
        * np.maximum(margins, 0.01)
    )
    tile_lags = best_indices.astype(np.int64) - max_lag

    candidates = np.arange(-max_lag, max_lag + 1)
    support = np.asarray(
        [np.sum(weights * np.exp(-0.5 * ((tile_lags - lag) / 0.55) ** 2)) for lag in candidates]
    )
    if float(support.sum()) <= 1e-12:
        return LagEstimate(0, 0.0, support, tile_lags, weights)
    winner = int(np.argmax(support))
    sorted_support = np.sort(support)
    concentration = float(support[winner] / (weights.sum() + 1e-12))
    separation = float(
        (sorted_support[-1] - sorted_support[-2]) / (sorted_support[-1] + 1e-12)
    )
    confidence = float(np.clip(concentration * (0.5 + 0.5 * separation), 0.0, 1.0))
    return LagEstimate(
        int(candidates[winner]), confidence, support, tile_lags, weights
    )


def intensity_signal(frames: list[Frame]) -> NDArray[np.float64]:
    return np.asarray([float(_gray(frame, (160, 128)).mean()) for frame in frames])


def motion_energy_signal(frames: list[Frame]) -> NDArray[np.float64]:
    gray = np.stack([_gray(frame, (160, 128)) for frame in frames])
    return np.mean(np.abs(np.diff(gray, axis=0)), axis=(1, 2)).astype(np.float64)


def global_structure_signal(frames: list[Frame]) -> NDArray[np.float64]:
    fingerprint = structural_change_fingerprint(frames, grid=(1, 1))
    return fingerprint[:, 0]


def estimate_signal_lag(
    first: NDArray[np.float64], second: NDArray[np.float64], max_lag: int
) -> tuple[int, float]:
    profile = lag_profile(first, second, max_lag)
    winner = int(np.argmax(profile))
    ordered = np.sort(profile)
    confidence = float(np.clip((ordered[-1] - ordered[-2]) / 2.0, 0.0, 1.0))
    return winner - max_lag, confidence
