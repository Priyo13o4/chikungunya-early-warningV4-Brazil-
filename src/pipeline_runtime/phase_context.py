from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SharedPhaseState:
    run_id: str
    artifacts: dict[str, Path] = field(default_factory=dict)
    degraded_reasons: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class BaselinePhaseResult:
    model_input_df: pd.DataFrame
    target: pd.Series
    bayesian_count_target: pd.Series
    bayesian_threshold_series: pd.Series
    temporal_index: pd.Series | None
    district_index: pd.Series | None
    baseline_score: pd.Series | None
    baseline_oof_score: pd.Series | None
    baseline_oof_predictions: pd.DataFrame | None
    baseline_metrics: dict[str, float] | None
    baseline_metrics_fullfit: dict[str, float] | None
    baseline_model_metrics: pd.DataFrame | None
    baseline_models: dict[str, Any]
    baseline_headline_eligible: bool
    baseline_evaluated_fold_count: int


@dataclass
class BayesianPhaseResult:
    bayesian_score: pd.Series | None
    bayesian_risk_frame: pd.DataFrame | None
    bayesian_oof_score: pd.Series | None
    bayesian_idata: Any | None
    bayesian_sampling_diagnostics: dict[str, Any]
    bayesian_metrics: dict[str, float] | None
    bayesian_metrics_fullfit: dict[str, float] | None
    bayesian_headline_eligible: bool
    bayesian_convergence_payload: dict[str, Any] | None
    bayesian_converged: bool | None


@dataclass
class EvalDecisionPhaseResult:
    comparison_table: pd.DataFrame | None
    decision_frame: pd.DataFrame
    suppress_headline_comparison: bool
    degraded_run_payload: dict[str, Any]
    bayesian_headline_effective: bool
    decision_threshold_used: float
    threshold_opt_payload: dict[str, Any]
