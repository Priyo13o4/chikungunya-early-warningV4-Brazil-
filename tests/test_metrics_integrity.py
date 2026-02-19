"""Targeted integrity tests for leakage and lead-time evaluation semantics."""

from __future__ import annotations

from src.evaluation.metrics_baselines import lead_time_steps, simple_lead_time_utility


def test_lead_time_stale_alert_gets_no_credit_beyond_horizon() -> None:
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 0]

    bounded = lead_time_steps(y_true, y_pred, max_lookback_steps=2)
    unbounded = lead_time_steps(y_true, y_pred, max_lookback_steps=None)

    assert bounded.tolist() == [0]
    assert unbounded.tolist() == [4]
    assert simple_lead_time_utility(y_true, y_pred, max_lookback_steps=2) == 0.0
