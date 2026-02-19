"""Cost-loss decision support and staged alert rules."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AlertLevel(str, Enum):
    """Enumerated staged alert levels."""

    NO_ACTION = "NO_ACTION"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


@dataclass(frozen=True)
class AlertThresholds:
    """Risk thresholds for staged alerts."""

    yellow: float = 0.3
    orange: float = 0.5
    red: float = 0.7

    def validate(self) -> None:
        values = (self.yellow, self.orange, self.red)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("All alert thresholds must be in [0, 1].")
        if not (self.yellow <= self.orange <= self.red):
            raise ValueError("Thresholds must satisfy yellow <= orange <= red.")


def _as_probability_series(probabilities: pd.Series) -> pd.Series:
    if probabilities.empty:
        raise ValueError("probabilities is empty.")
    series = probabilities.astype(float).clip(0.0, 1.0)
    return series


def assign_alert_levels(
    probabilities: pd.Series,
    thresholds: AlertThresholds = AlertThresholds(),
) -> pd.Series:
    """Assign NO_ACTION/YELLOW/ORANGE/RED based on configured thresholds."""
    thresholds.validate()
    probs = _as_probability_series(probabilities)

    levels = pd.Series(AlertLevel.NO_ACTION.value, index=probs.index, dtype="object")
    levels = levels.mask(probs >= thresholds.yellow, AlertLevel.YELLOW.value)
    levels = levels.mask(probs >= thresholds.orange, AlertLevel.ORANGE.value)
    levels = levels.mask(probs >= thresholds.red, AlertLevel.RED.value)
    return levels


def optimal_action_threshold(cost: float, loss: float) -> float:
    """Compute decision threshold from cost-loss ratio."""
    if cost < 0:
        raise ValueError("cost must be >= 0")
    if loss <= 0:
        raise ValueError("loss must be > 0")
    return min(max(cost / loss, 0.0), 1.0)


def recommend_action(probabilities: pd.Series, cost: float, loss: float) -> pd.Series:
    """Return binary action recommendation based on cost-loss ratio."""
    threshold = optimal_action_threshold(cost=cost, loss=loss)
    probs = _as_probability_series(probabilities)
    return (probs >= threshold).astype(int)


def optimize_decision_threshold(
    y_true: pd.Series,
    probabilities: pd.Series,
    *,
    cost: float,
    loss: float,
    min_samples: int = 100,
    grid_size: int = 101,
    grid_min: float = 0.0,
    grid_max: float = 1.0,
) -> dict[str, float | int | bool | str]:
    """Find empirical threshold minimizing observed mean cost-loss.

    Falls back to analytical C/L threshold when sample support is insufficient.
    """
    fallback = optimal_action_threshold(cost=cost, loss=loss)
    y = y_true.astype(int)
    probs = _as_probability_series(probabilities)

    if len(y) != len(probs):
        raise ValueError("y_true and probabilities must have equal length.")
    valid = y.isin([0, 1]) & probs.notna()
    y_valid = y.loc[valid]
    p_valid = probs.loc[valid]
    sample_size = int(len(y_valid))

    if sample_size < int(min_samples) or y_valid.nunique(dropna=True) < 2:
        return {
            "threshold": float(fallback),
            "objective_cost": float("nan"),
            "sample_size": sample_size,
            "optimized": False,
            "method": "fallback_cost_loss_ratio",
        }

    clipped_min = float(min(max(grid_min, 0.0), 1.0))
    clipped_max = float(min(max(grid_max, 0.0), 1.0))
    if clipped_max < clipped_min:
        clipped_min, clipped_max = clipped_max, clipped_min
    safe_grid_size = max(int(grid_size), 2)
    candidates = [float(value) for value in np.linspace(clipped_min, clipped_max, num=safe_grid_size)]

    best_threshold = float(fallback)
    best_cost = float("inf")
    for threshold in candidates:
        trial = expected_cost_loss(
            y_true=y_valid,
            probabilities=p_valid,
            threshold=float(threshold),
            cost=cost,
            loss=loss,
        )
        if trial < best_cost:
            best_cost = float(trial)
            best_threshold = float(threshold)

    return {
        "threshold": float(best_threshold),
        "objective_cost": float(best_cost),
        "sample_size": sample_size,
        "optimized": True,
        "method": "empirical_cost_loss_oof",
    }


def expected_cost_loss(
    y_true: pd.Series,
    probabilities: pd.Series,
    threshold: float,
    cost: float,
    loss: float,
) -> float:
    """Compute mean expected decision cost under binary action policy."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    if cost < 0 or loss < 0:
        raise ValueError("cost and loss must be >= 0.")

    y = y_true.astype(int)
    if not y.isin([0, 1]).all():
        raise ValueError("y_true must be binary 0/1.")

    probs = _as_probability_series(probabilities)
    if len(y) != len(probs):
        raise ValueError("y_true and probabilities must have equal length.")

    action = (probs >= threshold).astype(int)
    misses = ((y == 1) & (action == 0)).astype(int)
    total_cost = (action * cost) + (misses * loss)
    return float(total_cost.mean())


def staged_expected_value(
    y_true: pd.Series,
    probabilities: pd.Series,
    alert_costs: Mapping[str, float],
    miss_loss: float,
    thresholds: AlertThresholds = AlertThresholds(),
) -> pd.DataFrame:
    """Summarize expected value under staged alerts.

    `alert_costs` must include all `AlertLevel` keys as strings.
    """
    if miss_loss < 0:
        raise ValueError("miss_loss must be >= 0.")
    thresholds.validate()

    required = {level.value for level in AlertLevel}
    missing = required.difference(alert_costs.keys())
    if missing:
        raise ValueError(f"alert_costs is missing keys: {sorted(missing)}")

    y = y_true.astype(int)
    probs = _as_probability_series(probabilities)
    if len(y) != len(probs):
        raise ValueError("y_true and probabilities must have equal length.")

    levels = assign_alert_levels(probs, thresholds=thresholds)
    level_cost = levels.map(alert_costs).astype(float)
    miss_penalty = ((y == 1) & (levels == AlertLevel.NO_ACTION.value)).astype(float) * miss_loss
    total = level_cost + miss_penalty

    summary = (
        pd.DataFrame({"alert_level": levels, "cost": level_cost, "miss_penalty": miss_penalty, "total_cost": total})
        .groupby("alert_level", as_index=False)
        .agg(samples=("total_cost", "size"), avg_total_cost=("total_cost", "mean"))
        .sort_values("avg_total_cost")
    )
    logger.info("Computed staged expected value summary for %d samples", len(y))
    return summary
