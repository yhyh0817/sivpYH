"""Core utilities for the CURSOR-RGBT feasibility study."""

from .metrics import (
    fusion_harm_rate,
    oracle_gap_closure,
    risk_coverage_curve,
    route_regret,
)
from .utility import counterfactual_utilities, route_targets

__all__ = [
    "counterfactual_utilities",
    "route_targets",
    "fusion_harm_rate",
    "oracle_gap_closure",
    "risk_coverage_curve",
    "route_regret",
]
