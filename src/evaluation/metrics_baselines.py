"""Evaluation metrics for baseline (frequentist) classifiers."""

from __future__ import annotations

import logging
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _as_binary_series(values: Sequence[int] | pd.Series, name: str) -> pd.Series:
    series = pd.Series(values, copy=False).astype(int)
    if series.empty:
        raise ValueError(f"{name} is empty.")
    invalid = ~series.isin([0, 1])
    if invalid.any():
        raise ValueError(f"{name} must contain only binary values 0/1.")
    return series.reset_index(drop=True)


def _as_probability_series(values: Sequence[float] | pd.Series, name: str) -> pd.Series:
    series = pd.Series(values, copy=False).astype(float).reset_index(drop=True)
    if series.empty:
        raise ValueError(f"{name} is empty.")
    if series.isna().all():
        raise ValueError(f"{name} contains only NaN values.")
    clipped = series.clip(0.0, 1.0)
    if not np.allclose(series.fillna(0.0), clipped.fillna(0.0)):
        logger.warning("Probabilities in %s were clipped to [0, 1].", name)
    return clipped


def false_alarm_rate(y_true: Sequence[int] | pd.Series, y_pred: Sequence[int] | pd.Series) -> float:
    """Compute false alarm rate: FP / (FP + TN)."""
    y_true_series = _as_binary_series(y_true, "y_true")
    y_pred_series = _as_binary_series(y_pred, "y_pred")
    if len(y_true_series) != len(y_pred_series):
        raise ValueError("y_true and y_pred must have equal length.")

    negatives = y_true_series == 0
    denominator = int(negatives.sum())
    if denominator == 0:
        logger.warning("False alarm rate undefined because no negative instances were present.")
        return float("nan")
    false_positives = int(((y_pred_series == 1) & negatives).sum())
    return false_positives / denominator


def lead_time_steps(
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
    *,
    max_lookback_steps: int | None = 8,
    temporal_index: Sequence[object] | pd.Series | None = None,
    district: Sequence[object] | pd.Series | None = None,
) -> pd.Series:
    """Estimate per-outbreak lead time in index steps.

    Lead time is defined on outbreak onset points where `y_true` transitions 0 -> 1.
    For each onset, the nearest prior alert (`y_pred == 1`) contributes lead steps.
    """
    if max_lookback_steps is not None and max_lookback_steps < 0:
        raise ValueError("max_lookback_steps must be >= 0 when provided.")

    y_true_series = _as_binary_series(y_true, "y_true")
    y_pred_series = _as_binary_series(y_pred, "y_pred")
    if len(y_true_series) != len(y_pred_series):
        raise ValueError("y_true and y_pred must have equal length.")

    n_rows = len(y_true_series)
    frame = pd.DataFrame({"y_true": y_true_series, "y_pred": y_pred_series})
    frame["_row_order"] = np.arange(n_rows)

    if temporal_index is not None:
        temporal_series = pd.Series(temporal_index, copy=False).reset_index(drop=True)
        if len(temporal_series) != n_rows:
            raise ValueError("temporal_index must have equal length to y_true/y_pred.")
        frame["_time"] = temporal_series
        frame["_sort_time"] = pd.to_datetime(temporal_series, errors="coerce")
    else:
        frame["_time"] = pd.NaT
        frame["_sort_time"] = pd.NaT

    if district is not None:
        district_series = pd.Series(district, copy=False).reset_index(drop=True)
        if len(district_series) != n_rows:
            raise ValueError("district must have equal length to y_true/y_pred.")
        frame["_district"] = district_series.astype("string").fillna("__missing__")
    else:
        frame["_district"] = "__all__"

    frame = frame.sort_values(["_district", "_sort_time", "_row_order"], na_position="last").reset_index(drop=True)

    lead_values: list[int] = []
    for _, group in frame.groupby("_district", sort=False, dropna=False):
        starts = (group["y_true"] == 1) & (group["y_true"].shift(1, fill_value=0) == 0)
        onset_positions = np.flatnonzero(starts.to_numpy())

        y_pred_group = group["y_pred"].to_numpy()
        for onset_pos in onset_positions:
            prior_alert_positions = np.flatnonzero(y_pred_group[:onset_pos] == 1)
            if prior_alert_positions.size == 0:
                lead_values.append(0)
                continue

            lead = int(onset_pos - prior_alert_positions.max())
            if max_lookback_steps is not None and lead > max_lookback_steps:
                lead_values.append(0)
                continue
            lead_values.append(lead)

    return pd.Series(lead_values, name="lead_time_steps", dtype="int64")


def simple_lead_time_utility(
    y_true: Sequence[int] | pd.Series,
    y_pred: Sequence[int] | pd.Series,
    utility_per_step: float = 1.0,
    max_credit_steps: int = 4,
    *,
    max_lookback_steps: int | None = 8,
    temporal_index: Sequence[object] | pd.Series | None = None,
    district: Sequence[object] | pd.Series | None = None,
) -> float:
    """Compute simple lead-time utility from outbreak onset lead steps."""
    if max_credit_steps < 0:
        raise ValueError("max_credit_steps must be >= 0.")

    lead = lead_time_steps(
        y_true=y_true,
        y_pred=y_pred,
        max_lookback_steps=max_lookback_steps,
        temporal_index=temporal_index,
        district=district,
    )
    if lead.empty:
        return 0.0
    credited = lead.clip(lower=0, upper=max_credit_steps)
    return float(credited.mean() * utility_per_step)


def evaluate_baseline_predictions(
    y_true: Sequence[int] | pd.Series,
    y_pred_proba: Sequence[float] | pd.Series,
    threshold: float = 0.5,
    *,
    max_lookback_steps: int | None = 8,
    temporal_index: Sequence[object] | pd.Series | None = None,
    district: Sequence[object] | pd.Series | None = None,
) -> Dict[str, float]:
    """Compute baseline metrics with imbalanced-learning coverage.

    Returns a metrics dictionary containing:
    accuracy, precision, recall, f1, roc_auc, pr_auc, kappa,
    false_alarm_rate, lead_time_mean, lead_time_utility.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be within [0, 1].")

    y_true_series = _as_binary_series(y_true, "y_true")
    y_proba_series = _as_probability_series(y_pred_proba, "y_pred_proba")
    if len(y_true_series) != len(y_proba_series):
        raise ValueError("y_true and y_pred_proba must have equal length.")

    y_pred = (y_proba_series >= threshold).astype(int)

    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true_series, y_pred)),
        "precision": float(precision_score(y_true_series, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_series, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_series, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true_series, y_pred)),
        "false_alarm_rate": false_alarm_rate(y_true_series, y_pred),
    }

    unique_labels = y_true_series.nunique(dropna=True)
    if unique_labels > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true_series, y_proba_series))
        metrics["pr_auc"] = float(average_precision_score(y_true_series, y_proba_series))
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
