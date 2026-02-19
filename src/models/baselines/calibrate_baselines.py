"""Calibration helpers for baseline model probabilities."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def _identity_calibration(probabilities: pd.Series) -> pd.Series:
    return pd.to_numeric(probabilities, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def calibrate_probability_series(
    y_true: pd.Series,
    y_prob: pd.Series,
    *,
    method: str = "isotonic",
) -> pd.Series:
    """Calibrate one probability series with isotonic or Platt scaling.

    Fallback behavior:
    - If sklearn is unavailable, returns identity calibration.
    - If labels are single-class, returns identity calibration.
    """
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0.0)
    probs = _identity_calibration(y_prob)

    if labels.nunique() <= 1:
        LOGGER.info("Skipping calibration due to single-class labels")
        return probs

    try:
        if method.lower() == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrated = calibrator.fit_transform(probs.to_numpy(), labels.to_numpy())
            return pd.Series(calibrated, index=probs.index, dtype="float64").clip(0.0, 1.0)

        if method.lower() in {"platt", "sigmoid"}:
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression(max_iter=1000)
            calibrator.fit(probs.to_numpy().reshape(-1, 1), labels.to_numpy())
            calibrated = calibrator.predict_proba(probs.to_numpy().reshape(-1, 1))[:, 1]
            return pd.Series(calibrated, index=probs.index, dtype="float64").clip(0.0, 1.0)

        LOGGER.warning("Unknown calibration method '%s'; applying identity", method)
        return probs

    except Exception as calibration_error:
        LOGGER.warning("Calibration failed (%s). Applying identity calibration.", calibration_error)
        return probs


def calibrate_predictions(
    predictions: pd.DataFrame,
    y_true: pd.Series | None = None,
    *,
    method: str = "isotonic",
) -> pd.DataFrame:
    """Calibrate per-model baseline probability columns.

    If ``y_true`` is None, this performs safe clipping-only calibration.
    """
    clipped = predictions.apply(_identity_calibration, axis=0)
    if y_true is None:
        return clipped

    calibrated_columns: dict[str, pd.Series] = {}
    for column in clipped.columns:
        calibrated_columns[column] = calibrate_probability_series(
            y_true=y_true,
            y_prob=clipped[column],
            method=method,
        )
    return pd.DataFrame(calibrated_columns, index=clipped.index)


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    """Compute Brier score for binary labels and probabilities."""
    labels = pd.to_numeric(y_true, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    probs = _identity_calibration(y_prob).to_numpy(dtype=float)
    if labels.size == 0:
        return 0.0
    return float(np.mean((probs - labels) ** 2))
