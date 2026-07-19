"""Source-explainable temporal quality for visible-infrared fusion video."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from cursor_rgbt.temporal_sync import Frame, _gray


@dataclass(frozen=True)
class TemporalQualityScores:
    proposed: float
    source_conditioned_residual: float
    reliability_gated_residual: float
    motion_operator_drift: float
    multiscale_source_residual: float
    static_operator_residual: float
    operator_instability: float
    span_residual: float
    temporal_difference: float
    adjacent_dissimilarity: float
    source_normalized_change: float
    global_weight_drift: float
    flow_warp_error: float
    temcoco_flowd_visible: float
    temcoco_feacd_sobel: float


def _sequence(frames: list[Frame], size: tuple[int, int]) -> NDArray[np.float32]:
    if len(frames) < 5:
        raise ValueError("at least five frames are required")
    return np.stack([_gray(frame, size) for frame in frames])


def _gradient_sequence(frames: NDArray[np.float32]) -> NDArray[np.float32]:
    output = []
    for frame in frames:
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        output.append(cv2.magnitude(gx, gy))
    return np.stack(output)


def _explanation_residual(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    fused: NDArray[np.float32],
    grid: tuple[int, int],
) -> NDArray[np.float64]:
    first_change = np.diff(first, axis=0)
    second_change = np.diff(second, axis=0)
    fused_change = np.diff(fused, axis=0)
    rows, cols = grid
    height, width = first.shape[1:]
    residuals = []
    for index in range(len(fused_change)):
        for row in range(rows):
            y1, y2 = row * height // rows, (row + 1) * height // rows
            for col in range(cols):
                x1, x2 = col * width // cols, (col + 1) * width // cols
                one = first_change[index, y1:y2, x1:x2].ravel().astype(np.float64)
                two = second_change[index, y1:y2, x1:x2].ravel().astype(np.float64)
                target = fused_change[index, y1:y2, x1:x2].ravel().astype(np.float64)
                design = np.column_stack((one, two))
                gram = design.T @ design + np.eye(2) * 1e-6
                coefficients = np.linalg.solve(gram, design.T @ target)
                coefficients = np.clip(coefficients, -0.1, 1.1)
                error = target - design @ coefficients
                target_energy = np.sqrt(np.mean(target * target))
                source_energy = np.sqrt(np.mean(one * one) + np.mean(two * two))
                residuals.append(
                    np.sqrt(np.mean(error * error))
                    / (target_energy + 0.25 * source_energy + 1e-5)
                )
    return np.asarray(residuals, dtype=np.float64)


def _adjacent_dissimilarity(frames: NDArray[np.float32]) -> float:
    values = []
    for first, second in zip(frames[:-1], frames[1:]):
        mean_one, mean_two = float(first.mean()), float(second.mean())
        variance_one, variance_two = float(first.var()), float(second.var())
        covariance = float(np.mean((first - mean_one) * (second - mean_two)))
        ssim = ((2 * mean_one * mean_two + 1e-4) * (2 * covariance + 9e-4)) / (
            (mean_one**2 + mean_two**2 + 1e-4)
            * (variance_one + variance_two + 9e-4)
        )
        values.append(1.0 - ssim)
    return float(np.mean(values))


def _global_weight_drift(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    fused: NDArray[np.float32],
) -> float:
    coefficients = []
    residuals = []
    for one, two, target in zip(first, second, fused):
        design = np.column_stack((one.ravel(), two.ravel(), np.ones(one.size)))
        solution, _, _, _ = np.linalg.lstsq(design, target.ravel(), rcond=None)
        coefficients.append(solution[:2])
        residuals.append(np.sqrt(np.mean((target.ravel() - design @ solution) ** 2)))
    coefficient_drift = float(np.mean(np.std(np.asarray(coefficients), axis=0)))
    return coefficient_drift + 0.25 * float(np.std(residuals))


def _flow_warp_error(frames: NDArray[np.float32]) -> float:
    """Motion-compensated temporal error baseline using fused-video flow."""

    height, width = frames.shape[1:]
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    errors = []
    for previous, current in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0,
        )
        warped = cv2.remap(
            previous,
            xx - flow[..., 0],
            yy - flow[..., 1],
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        errors.append(float(np.mean(np.abs(current - warped))))
    return float(np.mean(errors))


def _source_flow_warp_error(
    source: NDArray[np.float32],
    fused: NDArray[np.float32],
) -> float:
    """TemCoCo-style flowD surrogate using source-video optical flow."""

    errors = []
    for source_previous, source_current, fused_previous, fused_current in zip(
        source[:-1],
        source[1:],
        fused[:-1],
        fused[1:],
    ):
        flow = _farneback_flow(source_previous, source_current)
        warped = _warp_with_flow(fused_previous, flow)
        errors.append(float(np.mean(np.abs(fused_current - warped))))
    return float(np.mean(errors))


def _feature_vector(frame: NDArray[np.float32]) -> NDArray[np.float64]:
    low = cv2.resize(frame, (24, 18), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.resize(cv2.magnitude(gx, gy), (24, 18), interpolation=cv2.INTER_AREA)
    vector = np.concatenate([low.ravel(), magnitude.ravel()]).astype(np.float64)
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector / (norm + 1e-8)


def _feature_change_direction_distance(
    source: NDArray[np.float32],
    fused: NDArray[np.float32],
) -> float:
    """TemCoCo feaCD surrogate using Sobel/low-frequency feature directions."""

    distances = []
    source_features = [_feature_vector(frame) for frame in source]
    fused_features = [_feature_vector(frame) for frame in fused]
    for previous_source, current_source, previous_fused, current_fused in zip(
        source_features[:-1],
        source_features[1:],
        fused_features[:-1],
        fused_features[1:],
    ):
        source_change = current_source - previous_source
        fused_change = current_fused - previous_fused
        similarity = float(
            np.dot(source_change, fused_change)
            / (np.linalg.norm(source_change) * np.linalg.norm(fused_change) + 1e-8)
        )
        distances.append(1.0 - np.clip(similarity, -1.0, 1.0))
    return float(np.mean(distances))


def _farneback_flow(
    previous: NDArray[np.float32],
    current: NDArray[np.float32],
) -> NDArray[np.float32]:
    return cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )


def _warp_with_flow(
    frame: NDArray[np.float32],
    flow: NDArray[np.float32],
) -> NDArray[np.float32]:
    height, width = frame.shape
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    return cv2.remap(
        frame,
        xx - flow[..., 0],
        yy - flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    )


def _tile_slices(
    shape: tuple[int, int],
    grid: tuple[int, int],
) -> list[tuple[slice, slice]]:
    rows, cols = grid
    height, width = shape
    slices = []
    for row in range(rows):
        y1, y2 = row * height // rows, (row + 1) * height // rows
        for col in range(cols):
            x1, x2 = col * width // cols, (col + 1) * width // cols
            slices.append((slice(y1, y2), slice(x1, x2)))
    return slices


def _robust_scale(values: NDArray[np.float64]) -> float:
    center = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - center))) + 1e-6


def _tile_reliability(
    reference_error: NDArray[np.float32],
    source_change: NDArray[np.float32],
    source_texture: NDArray[np.float32],
) -> float:
    motion_scale = float(np.median(reference_error)) + 1e-4
    motion_reliability = np.exp(-float(np.mean(reference_error)) / (3.0 * motion_scale))
    activity = float(np.mean(source_change))
    activity_reliability = activity / (activity + 0.015)
    texture = float(np.mean(source_texture))
    texture_reliability = texture / (texture + 0.02)
    return float(
        np.clip(
            0.20 + 0.80 * motion_reliability * max(activity_reliability, texture_reliability),
            0.05,
            1.0,
        )
    )


def _motion_source_residual_components(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    fused: NDArray[np.float32],
    grid: tuple[int, int],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Explain fused temporal changes by motion-compensated source changes.

    The residual is high only when the fused transition cannot be reconstructed
    from the visible and thermal transitions under a shared reference motion.
    """

    residuals = []
    reliabilities = []
    operator_drifts = []
    previous_coefficients: list[NDArray[np.float64] | None] = [
        None for _ in _tile_slices(first.shape[1:], grid)
    ]
    for index in range(1, len(fused)):
        reference_previous = 0.5 * (first[index - 1] + second[index - 1])
        reference_current = 0.5 * (first[index] + second[index])
        flow = _farneback_flow(reference_previous, reference_current)
        warped_first = _warp_with_flow(first[index - 1], flow)
        warped_second = _warp_with_flow(second[index - 1], flow)
        warped_fused = _warp_with_flow(fused[index - 1], flow)
        warped_reference = _warp_with_flow(reference_previous, flow)

        first_change = first[index] - warped_first
        second_change = second[index] - warped_second
        fused_change = fused[index] - warped_fused
        reference_error = np.abs(reference_current - warped_reference)
        source_change = np.abs(first_change) + np.abs(second_change)
        source_texture = _gradient_sequence(
            np.stack([reference_previous, reference_current])
        ).mean(axis=0)

        for tile_index, (ys, xs) in enumerate(_tile_slices(first.shape[1:], grid)):
            one = first_change[ys, xs].ravel().astype(np.float64)
            two = second_change[ys, xs].ravel().astype(np.float64)
            target = fused_change[ys, xs].ravel().astype(np.float64)
            design = np.column_stack((one, two))
            gram = design.T @ design + np.eye(2) * 1e-5
            solution = np.linalg.solve(gram, design.T @ target)
            coefficients = np.clip(solution, -0.5, 1.5)
            prediction = design @ coefficients
            error = target - prediction
            target_scale = np.sqrt(np.mean(target * target))
            source_scale = np.sqrt(np.mean(one * one) + np.mean(two * two))
            residuals.append(
                np.sqrt(np.mean(error * error)) / (0.65 * target_scale + 0.35 * source_scale + 1e-5)
            )
            reliabilities.append(
                _tile_reliability(
                    reference_error[ys, xs],
                    source_change[ys, xs],
                    source_texture[ys, xs],
                )
            )
            previous = previous_coefficients[tile_index]
            if previous is not None:
                drift = np.linalg.norm(coefficients - previous)
                local_scale = 1.0 + source_scale / (target_scale + 1e-5)
                operator_drifts.append(drift / local_scale)
            previous_coefficients[tile_index] = coefficients
    return (
        np.asarray(residuals, dtype=np.float64),
        np.asarray(reliabilities, dtype=np.float64),
        np.asarray(operator_drifts, dtype=np.float64),
    )


def _motion_source_residual(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    fused: NDArray[np.float32],
    grid: tuple[int, int],
) -> tuple[float, float, float]:
    residuals, reliabilities, operator_drifts = _motion_source_residual_components(
        first, second, fused, grid
    )
    if residuals.size == 0:
        return 0.0, 0.0, 0.0
    gated = residuals * reliabilities
    high = residuals >= np.quantile(residuals, 0.65)
    source_conditioned = float(np.quantile(residuals, 0.85))
    reliability_gated = float(np.mean(gated[high])) if np.any(high) else float(np.mean(gated))
    motion_operator = float(np.quantile(operator_drifts, 0.85)) if operator_drifts.size else 0.0
    return source_conditioned, reliability_gated, motion_operator


def _multiscale_motion_source_residual(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    fused: NDArray[np.float32],
    grids: tuple[tuple[int, int], ...],
) -> tuple[float, float, float, float]:
    source_scores = []
    gated_scores = []
    drift_scores = []
    for grid in grids:
        source, gated, drift = _motion_source_residual(first, second, fused, grid)
        source_scores.append(source)
        gated_scores.append(gated)
        drift_scores.append(drift)
    source_conditioned = float(np.mean(source_scores))
    reliability_gated = float(np.mean(gated_scores))
    motion_operator = float(np.mean(drift_scores))
    multiscale = float(
        0.55 * reliability_gated
        + 0.30 * source_conditioned
        + 0.15 * motion_operator
    )
    return source_conditioned, reliability_gated, motion_operator, multiscale


def _operator_instability(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    fused: NDArray[np.float32],
    grid: tuple[int, int],
) -> float:
    """Measure temporal changes of a locally fitted source-to-fusion operator."""

    rows, cols = grid
    height, width = first.shape[1:]
    coefficients = np.empty((len(fused), rows * cols, 2), dtype=np.float64)
    residuals = np.empty((len(fused), rows * cols), dtype=np.float64)
    tile = 0
    for row in range(rows):
        y1, y2 = row * height // rows, (row + 1) * height // rows
        for col in range(cols):
            x1, x2 = col * width // cols, (col + 1) * width // cols
            for index in range(len(fused)):
                one = first[index, y1:y2, x1:x2].ravel().astype(np.float64)
                two = second[index, y1:y2, x1:x2].ravel().astype(np.float64)
                target = fused[index, y1:y2, x1:x2].ravel().astype(np.float64)
                design = np.column_stack((one, two, np.ones(len(one))))
                gram = design.T @ design + np.diag((1e-5, 1e-5, 1e-7))
                solution = np.linalg.solve(gram, design.T @ target)
                coefficients[index, tile] = np.clip(solution[:2], -0.5, 1.5)
                error = target - design @ solution
                source_scale = np.std(one) + np.std(two) + np.std(target) + 1e-4
                residuals[index, tile] = np.sqrt(np.mean(error * error)) / source_scale
            tile += 1

    coefficient_change = np.linalg.norm(np.diff(coefficients, axis=0), axis=2)
    residual_change = np.abs(np.diff(residuals, axis=0))
    residual_center = np.median(residuals, axis=1, keepdims=True)
    residual_excess = np.maximum(residuals - residual_center, 0.0)[1:]
    evidence = coefficient_change + 0.45 * residual_change + 0.20 * residual_excess
    return float(np.quantile(evidence, 0.85))


def _static_operator_residual(
    first: NDArray[np.float32],
    second: NDArray[np.float32],
    fused: NDArray[np.float32],
    grid: tuple[int, int],
) -> float:
    """Test whether one local nonlinear fusion rule explains the whole clip."""

    rows, cols = grid
    height, width = first.shape[1:]
    tile_scores = []
    for row in range(rows):
        y1, y2 = row * height // rows, (row + 1) * height // rows
        for col in range(cols):
            x1, x2 = col * width // cols, (col + 1) * width // cols
            one = first[:, y1:y2, x1:x2].ravel().astype(np.float64)
            two = second[:, y1:y2, x1:x2].ravel().astype(np.float64)
            target = fused[:, y1:y2, x1:x2].ravel().astype(np.float64)
            design = np.column_stack(
                (
                    one,
                    two,
                    np.maximum(one, two),
                    np.minimum(one, two),
                    one * two,
                    one * one,
                    two * two,
                    np.ones(len(one)),
                )
            )
            gram = design.T @ design + np.eye(design.shape[1]) * 1e-5
            solution = np.linalg.solve(gram, design.T @ target)
            error = target - design @ solution
            contrast = np.std(target) + 0.25 * (np.std(one) + np.std(two)) + 1e-4
            tile_scores.append(np.sqrt(np.mean(error * error)) / contrast)
    return float(np.quantile(tile_scores, 0.85))


def temporal_quality_scores(
    visible: list[Frame],
    thermal: list[Frame],
    fused: list[Frame],
    grid: tuple[int, int] = (4, 5),
    size: tuple[int, int] = (160, 128),
) -> TemporalQualityScores:
    if not (len(visible) == len(thermal) == len(fused)):
        raise ValueError("source and fused sequences must have equal length")
    first, second, output = (
        _sequence(visible, size),
        _sequence(thermal, size),
        _sequence(fused, size),
    )
    intensity_residual = _explanation_residual(first, second, output, grid)
    gradient_residual = _explanation_residual(
        _gradient_sequence(first),
        _gradient_sequence(second),
        _gradient_sequence(output),
        grid,
    )
    combined = 0.65 * intensity_residual + 0.35 * gradient_residual
    span_residual = float(np.quantile(combined, 0.80))
    operator_instability = _operator_instability(first, second, output, grid)
    static_operator_residual = _static_operator_residual(first, second, output, grid)
    (
        source_conditioned_residual,
        reliability_gated_residual,
        motion_operator_drift,
        multiscale_source_residual,
    ) = _multiscale_motion_source_residual(
        first,
        second,
        output,
        grids=(grid, (2, 3)),
    )
    proposed = float(
        0.50 * multiscale_source_residual
        + 0.25 * static_operator_residual
        + 0.15 * operator_instability
        + 0.10 * span_residual
    )

    output_change = np.abs(np.diff(output, axis=0))
    source_change = np.abs(np.diff(first, axis=0)) + np.abs(np.diff(second, axis=0))
    return TemporalQualityScores(
        proposed=proposed,
        source_conditioned_residual=source_conditioned_residual,
        reliability_gated_residual=reliability_gated_residual,
        motion_operator_drift=motion_operator_drift,
        multiscale_source_residual=multiscale_source_residual,
        static_operator_residual=static_operator_residual,
        operator_instability=operator_instability,
        span_residual=span_residual,
        temporal_difference=float(output_change.mean()),
        adjacent_dissimilarity=_adjacent_dissimilarity(output),
        source_normalized_change=float(
            output_change.mean() / (source_change.mean() + 1e-6)
        ),
        global_weight_drift=_global_weight_drift(first, second, output),
        flow_warp_error=_flow_warp_error(output),
        temcoco_flowd_visible=_source_flow_warp_error(first, output),
        temcoco_feacd_sobel=_feature_change_direction_distance(first, output),
    )


def fuse_frames(
    visible: list[Frame], thermal: list[Frame], method: str = "average"
) -> list[Frame]:
    if len(visible) != len(thermal):
        raise ValueError("paired source streams are required")
    output = []
    for first, second in zip(visible, thermal):
        if first.shape[:2] != second.shape[:2]:
            second = cv2.resize(second, (first.shape[1], first.shape[0]))
        one = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        two = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if method == "average":
            fused = 0.55 * one + 0.45 * two
        elif method == "maximum":
            fused = np.maximum(one, two)
        else:
            raise ValueError(method)
        gray = np.clip(fused * 255, 0, 255).astype(np.uint8)
        output.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return output


def inject_source_variation(
    visible: list[Frame],
    thermal: list[Frame],
    family: str,
    severity: float,
) -> tuple[list[Frame], list[Frame]]:
    """Create source-driven temporal changes that should not count as fusion faults."""

    if len(visible) != len(thermal):
        raise ValueError("paired source streams are required")
    first_output, second_output = [], []
    height, width = visible[0].shape[:2]
    yy, xx = np.mgrid[:height, :width]
    local_mask = cv2.GaussianBlur(
        (
            (xx > width * 0.15)
            & (xx < width * 0.85)
            & (yy > height * 0.25)
            & (yy < height * 0.75)
        ).astype(np.float32),
        (0, 0),
        9.0,
    )[..., None]
    for index, (first, second) in enumerate(zip(visible, thermal)):
        first_float = first.astype(np.float32) / 255.0
        second_float = second.astype(np.float32) / 255.0
        alternating = -1.0 if index % 2 else 1.0
        if family == "source_global_gain":
            first_altered = first_float * (1.0 + severity * alternating)
            second_altered = second_float * (1.0 + severity * alternating)
        elif family == "source_local_gain":
            gain = 1.0 + severity * alternating * local_mask
            first_altered = first_float * gain
            second_altered = second_float * gain
        elif family == "source_cross_exposure":
            first_altered = first_float * (1.0 + severity * alternating)
            second_altered = second_float * (1.0 - severity * alternating)
        else:
            raise ValueError(family)
        first_output.append(np.clip(first_altered * 255, 0, 255).astype(np.uint8))
        second_output.append(np.clip(second_altered * 255, 0, 255).astype(np.uint8))
    return first_output, second_output


def inject_temporal_artifact(
    clean: list[Frame],
    visible: list[Frame],
    thermal: list[Frame],
    family: str,
    severity: float,
) -> list[Frame]:
    if severity == 0:
        return [frame.copy() for frame in clean]
    output = []
    height, width = clean[0].shape[:2]
    yy, xx = np.mgrid[:height, :width]
    local_mask = cv2.GaussianBlur(
        ((xx > width * 0.48) & (yy > height * 0.15) & (yy < height * 0.85)).astype(np.float32),
        (0, 0),
        7.0,
    )[..., None]
    for index, (base, first, second) in enumerate(zip(clean, visible, thermal)):
        base_float = base.astype(np.float32) / 255.0
        first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        modality_delta = (first_gray - second_gray)[..., None]
        alternating = -1.0 if index % 2 else 1.0
        if family == "global_gain":
            altered = base_float * (1.0 + severity * alternating)
        elif family == "global_weight":
            altered = base_float + severity * alternating * modality_delta
        elif family == "local_weight":
            altered = base_float + severity * alternating * modality_delta * local_mask
        elif family == "local_gain":
            altered = base_float * (1.0 + severity * alternating * local_mask)
        elif family == "aperiodic_gain":
            phase = np.sin(1.37 * index) + 0.45 * np.sin(0.41 * index + 0.8)
            altered = base_float * (1.0 + severity * phase)
        elif family == "patch_lag":
            previous = clean[max(index - 1, 0)].astype(np.float32) / 255.0
            lag_mix = min(0.90, 8.0 * severity)
            altered = base_float * (1.0 - lag_mix * local_mask) + previous * lag_mix * local_mask
        else:
            raise ValueError(family)
        output.append(np.clip(altered * 255, 0, 255).astype(np.uint8))
    return output
