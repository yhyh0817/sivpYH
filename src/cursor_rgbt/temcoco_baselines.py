"""TemCoCo-style temporal consistency baselines.

The official TemCoCo feaCD uses ResNet-18 feature-change direction.  This
module provides a local, reproducible implementation with torchvision
ResNet-18 weights when available.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from cursor_rgbt.temporal_sync import Frame


@lru_cache(maxsize=1)
def _resnet18_backbone():
    from torch import nn
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    backbone = nn.Sequential(*(list(model.children())[:-1]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone.to(device).eval()
    return backbone, device


def _frame_tensor(frame: Frame) -> torch.Tensor:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
    array = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def resnet18_features(frames: list[Frame], batch_size: int = 16) -> NDArray[np.float64]:
    backbone, device = _resnet18_backbone()
    features = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            batch = torch.stack([_frame_tensor(frame) for frame in frames[start : start + batch_size]])
            output = backbone(batch.to(device)).flatten(1).detach().cpu().numpy()
            features.append(output.astype(np.float64))
    return np.vstack(features)


def feature_change_direction_distance(
    source_features: NDArray[np.float64],
    fused_features: NDArray[np.float64],
) -> float:
    """Feature-change direction distance from precomputed frame features."""

    if source_features.shape != fused_features.shape:
        raise ValueError("source and fused features must have equal shape")
    if len(source_features) < 3:
        raise ValueError("at least three feature vectors are required")
    source_change = np.diff(source_features, axis=0)
    fused_change = np.diff(fused_features, axis=0)
    numerator = np.sum(source_change * fused_change, axis=1)
    denominator = (
        np.linalg.norm(source_change, axis=1)
        * np.linalg.norm(fused_change, axis=1)
        + 1e-8
    )
    similarity = np.clip(numerator / denominator, -1.0, 1.0)
    return float(np.mean(1.0 - similarity))


def resnet18_feacd(source: list[Frame], fused: list[Frame]) -> float:
    """Feature-change direction distance following TemCoCo's feaCD idea."""

    if len(source) != len(fused):
        raise ValueError("source and fused videos must have equal length")
    if len(source) < 3:
        raise ValueError("at least three frames are required")
    source_features = resnet18_features(source)
    fused_features = resnet18_features(fused)
    return feature_change_direction_distance(source_features, fused_features)
