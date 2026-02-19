"""Posterior predictive inference utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


def posterior_predictive_summary(predictions: pd.Series) -> pd.DataFrame:
    """Create global summary statistics from posterior-like predictions."""
    numeric = pd.to_numeric(predictions, errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "mean": [float(numeric.mean())],
            "std": [float(numeric.std(ddof=0) if len(numeric) else 0.0)],
            "p05": [float(numeric.quantile(0.05) if len(numeric) else 0.0)],
            "p50": [float(numeric.quantile(0.50) if len(numeric) else 0.0)],
            "p95": [float(numeric.quantile(0.95) if len(numeric) else 0.0)],
        }
    )


def summarize_posterior_draws(
    posterior_draws: pd.DataFrame,
    *,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> pd.DataFrame:
    """Summarize row-wise posterior draws into risk intervals.

    Parameters
    ----------
    posterior_draws:
        Rows are observations and columns are posterior samples.
    """
    draws = posterior_draws.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if draws.empty:
        return pd.DataFrame(columns=["mean", "median", "lower", "upper", "std"])

    return pd.DataFrame(
        {
            "mean": draws.mean(axis=1),
            "median": draws.median(axis=1),
            "lower": draws.quantile(lower_q, axis=1),
            "upper": draws.quantile(upper_q, axis=1),
            "std": draws.std(axis=1, ddof=0),
        },
        index=draws.index,
    )


def exceedance_probability(
    posterior_draws: pd.DataFrame,
    *,
    threshold: float,
) -> pd.Series:
    """Compute P(risk > threshold) per row from posterior draws."""
    draws = posterior_draws.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if draws.empty:
        return pd.Series(dtype="float64")
    exceed = (draws.to_numpy(dtype=float) > threshold).mean(axis=1)
    return pd.Series(exceed, index=draws.index, name="exceedance_probability", dtype="float64")


def posterior_predictive_risk_summary(
    posterior_draws: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return combined posterior risk summary with exceedance probabilities."""
    summary = summarize_posterior_draws(posterior_draws)
    if summary.empty:
        return summary
    summary["prob_exceeds_threshold"] = exceedance_probability(posterior_draws, threshold=threshold)
    summary["risk_flag"] = (summary["prob_exceeds_threshold"] >= 0.5).astype(int)
    summary["uncertainty_width"] = np.maximum(0.0, summary["upper"] - summary["lower"])
    return summary
