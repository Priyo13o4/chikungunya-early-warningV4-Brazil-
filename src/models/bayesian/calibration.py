"""Calibration helpers for Bayesian probabilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_bayesian_risk(risk: pd.Series) -> pd.Series:
    """Apply bounded calibration to risk scores."""
    return pd.to_numeric(risk, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def brier_score_bayesian(y_true: pd.Series, risk: pd.Series) -> float:
    """Compute Brier score for Bayesian probabilities."""
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    probs = calibrate_bayesian_risk(risk).to_numpy(dtype=float)
    if labels.size == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))


def reliability_bins(
    y_true: pd.Series,
    risk: pd.Series,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability diagram bins for Bayesian risk predictions."""
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    probs = calibrate_bayesian_risk(risk)

    if len(probs) == 0:
        return pd.DataFrame(
            columns=["bin", "bin_lower", "bin_upper", "count", "pred_mean", "obs_rate", "abs_gap"]
        )

    bins = pd.cut(probs, bins=n_bins, labels=False, include_lowest=True)
    frame = pd.DataFrame({"bin": bins, "prob": probs, "label": labels})

    grouped = frame.groupby("bin", dropna=False)
    result = grouped.agg(
        count=("label", "size"),
        pred_mean=("prob", "mean"),
        obs_rate=("label", "mean"),
        bin_lower=("prob", "min"),
        bin_upper=("prob", "max"),
    ).reset_index()

    result["abs_gap"] = (result["pred_mean"] - result["obs_rate"]).abs()
    return result[["bin", "bin_lower", "bin_upper", "count", "pred_mean", "obs_rate", "abs_gap"]]
