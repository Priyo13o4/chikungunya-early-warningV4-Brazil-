"""Targeted integrity tests for leakage and lead-time evaluation semantics."""

from __future__ import annotations

from src.evaluation.metrics_baselines import lead_time_steps, simple_lead_time_utility
from src.pipeline_runtime.phases_bayesian import _resolve_convergence_failure_mode


def test_lead_time_stale_alert_gets_no_credit_beyond_horizon() -> None:
    y_true = [0, 0, 0, 0, 1]
    y_pred = [1, 0, 0, 0, 0]

    bounded = lead_time_steps(y_true, y_pred, max_lookback_steps=2)
    unbounded = lead_time_steps(y_true, y_pred, max_lookback_steps=None)

    assert bounded.tolist() == [0]
    assert unbounded.tolist() == [4]
    assert simple_lead_time_utility(y_true, y_pred, max_lookback_steps=2) == 0.0


def test_convergence_failure_mode_defaults_to_strict_in_strict_or_full_mode() -> None:
    mode, explicit, reason = _resolve_convergence_failure_mode(
        bayesian_settings={"draws": 800, "chains": 2},
        strict_or_full_bayesian_mode=True,
    )

    assert mode == "strict"
    assert explicit is False
    assert reason is None


def test_convergence_failure_mode_auto_downgrades_to_warn_for_low_sample_runs() -> None:
    mode, explicit, reason = _resolve_convergence_failure_mode(
        bayesian_settings={"draws": 20, "chains": 1},
        strict_or_full_bayesian_mode=True,
    )

    assert mode == "warn"
    assert explicit is False
    assert "auto-downgraded to warn" in str(reason)
