"""Evaluation metrics and calibration utilities for Bayesian risk outputs."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score

from src.evaluation.metrics_baselines import false_alarm_rate, lead_time_steps, simple_lead_time_utility

logger = logging.getLogger(__name__)


def _as_binary_series(values: Sequence[int] | pd.Series, name: str) -> pd.Series:
    series = pd.Series(values, copy=False).astype(int).reset_index(drop=True)
    if series.empty:
        raise ValueError(f"{name} is empty.")
    invalid = ~series.isin([0, 1])
    if invalid.any():
        raise ValueError(f"{name} must contain only binary values 0/1.")
    return series


def _as_probability_series(values: Sequence[float] | pd.Series, name: str) -> pd.Series:
    series = pd.Series(values, copy=False).astype(float).reset_index(drop=True)
    if series.empty:
        raise ValueError(f"{name} is empty.")
    clipped = series.clip(0.0, 1.0)
    if not np.allclose(series.fillna(0.0), clipped.fillna(0.0)):
        logger.warning("Values in %s were clipped to [0, 1].", name)
    return clipped


def bayesian_brier_score(y_true: Sequence[int] | pd.Series, risk: Sequence[float] | pd.Series) -> float:
    """Compute Brier score for probabilistic outbreak risks."""
    y_true_series = _as_binary_series(y_true, "y_true")
    risk_series = _as_probability_series(risk, "risk")
    if len(y_true_series) != len(risk_series):
        raise ValueError("y_true and risk must have equal length.")
    return float(brier_score_loss(y_true_series, risk_series))


def reliability_bins(
    y_true: Sequence[int] | pd.Series,
    risk: Sequence[float] | pd.Series,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Create reliability bins for calibration diagnostics."""
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be either 'quantile' or 'uniform'.")

    y_true_series = _as_binary_series(y_true, "y_true")
    risk_series = _as_probability_series(risk, "risk")
    if len(y_true_series) != len(risk_series):
        raise ValueError("y_true and risk must have equal length.")

    frame = pd.DataFrame({"y_true": y_true_series, "risk": risk_series}).dropna(subset=["y_true", "risk"])
    if frame.empty:
        return pd.DataFrame(columns=["bin", "count", "avg_pred", "obs_rate", "bin_lower", "bin_upper"])

    if strategy == "quantile":
        frame["bin"] = pd.qcut(frame["risk"], q=n_bins, duplicates="drop")
    else:
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        frame["bin"] = pd.cut(frame["risk"], bins=bins, include_lowest=True)

    grouped = frame.groupby("bin", observed=True)
    result = grouped.agg(count=("y_true", "size"), avg_pred=("risk", "mean"), obs_rate=("y_true", "mean")).reset_index()
    result["bin_lower"] = result["bin"].apply(lambda interval: float(interval.left))
    result["bin_upper"] = result["bin"].apply(lambda interval: float(interval.right))
    return result[["bin", "count", "avg_pred", "obs_rate", "bin_lower", "bin_upper"]]


def ci_coverage_utility(
    y_true: Sequence[float] | pd.Series,
    ci_lower: Sequence[float] | pd.Series,
    ci_upper: Sequence[float] | pd.Series,
) -> float:
    """Compute empirical confidence-interval coverage fraction."""
    y_true_series = pd.Series(y_true, copy=False).astype(float).reset_index(drop=True)
    lower_series = pd.Series(ci_lower, copy=False).astype(float).reset_index(drop=True)
    upper_series = pd.Series(ci_upper, copy=False).astype(float).reset_index(drop=True)

    if not (len(y_true_series) == len(lower_series) == len(upper_series)):
        raise ValueError("y_true, ci_lower and ci_upper must have equal length.")
    covered = (y_true_series >= lower_series) & (y_true_series <= upper_series)
    return float(covered.mean())


def evaluate_bayesian_predictions(
    y_true: Sequence[int] | pd.Series,
    risk: Sequence[float] | pd.Series,
    threshold: float = 0.5,
    *,
    max_lookback_steps: int | None = 8,
    temporal_index: Sequence[object] | pd.Series | None = None,
    district: Sequence[object] | pd.Series | None = None,
) -> Dict[str, float]:
    """Compute core Bayesian-track predictive and calibration metrics."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1].")

    y_true_series = _as_binary_series(y_true, "y_true")
    risk_series = _as_probability_series(risk, "risk")
    if len(y_true_series) != len(risk_series):
        raise ValueError("y_true and risk must have equal length.")

    y_pred = (risk_series >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_series, y_pred)),
        "brier": bayesian_brier_score(y_true_series, risk_series),
        "false_alarm_rate": false_alarm_rate(y_true_series, y_pred),
    }

    if y_true_series.nunique(dropna=True) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_series, risk_series))
        metrics["pr_auc"] = float(average_precision_score(y_true_series, risk_series))
    else:
        logger.warning("ROC-AUC and PR-AUC are undefined for a single-class target.")
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    lead = lead_time_steps(
        y_true_series,
        y_pred,
        max_lookback_steps=max_lookback_steps,
        temporal_index=temporal_index,
        district=district,
    )
    metrics["lead_time_mean"] = float(lead.mean()) if not lead.empty else 0.0
    metrics["lead_time_utility"] = simple_lead_time_utility(
        y_true_series,
        y_pred,
        max_lookback_steps=max_lookback_steps,
        temporal_index=temporal_index,
        district=district,
    )
    return metrics
