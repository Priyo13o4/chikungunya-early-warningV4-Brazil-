"""Tests for decision layer cost-loss function."""

from __future__ import annotations

import pandas as pd

from src.decision_layer.cost_loss import AlertLevel, assign_alert_levels, optimize_decision_threshold, recommend_action


def test_recommend_action_thresholding() -> None:
    probs = pd.Series([0.1, 0.4, 0.9])
    actions = recommend_action(probs, cost=0.2, loss=1.0)
    assert actions.tolist() == [0, 1, 1]


def test_assign_alert_levels_uses_expected_threshold_bands() -> None:
    probs = pd.Series([0.0, 0.3, 0.5, 0.7, 0.9])
    levels = assign_alert_levels(probs)
    assert levels.tolist() == [
        AlertLevel.NO_ACTION.value,
        AlertLevel.YELLOW.value,
        AlertLevel.ORANGE.value,
        AlertLevel.RED.value,
        AlertLevel.RED.value,
    ]


def test_optimize_decision_threshold_uses_grid_and_fallback_by_sample_size() -> None:
    y_true = pd.Series([0, 0, 1, 1, 1, 0, 1, 0])
    probabilities = pd.Series([0.1, 0.2, 0.7, 0.9, 0.8, 0.3, 0.6, 0.4])

    optimized = optimize_decision_threshold(
        y_true=y_true,
        probabilities=probabilities,
        cost=0.2,
        loss=1.0,
        min_samples=4,
        grid_size=11,
        grid_min=0.0,
        grid_max=1.0,
    )
    assert optimized["optimized"] is True
    assert optimized["method"] == "empirical_cost_loss_oof"
    assert 0.0 <= float(optimized["threshold"]) <= 1.0

    fallback = optimize_decision_threshold(
        y_true=y_true,
        probabilities=probabilities,
        cost=0.2,
        loss=1.0,
        min_samples=1000,
    )
    assert fallback["optimized"] is False
    assert fallback["method"] == "fallback_cost_loss_ratio"
    assert abs(float(fallback["threshold"]) - (0.2 / 1.2)) < 1e-9
