"""Prediction interfaces for baseline models."""

from __future__ import annotations

import logging
from typing import Mapping

import pandas as pd

from src.models.baselines.baseline_models import BaselineModel
from src.models.baselines.threshold_rules import apply_threshold

LOGGER = logging.getLogger(__name__)


def predict_baselines(models: Mapping[str, BaselineModel], X: pd.DataFrame) -> pd.DataFrame:
    """Generate clipped probability predictions from trained baseline models."""
    outputs: dict[str, pd.Series] = {}
    for model_name, model in models.items():
        try:
            outputs[model_name] = model.predict_proba(X).clip(0.0, 1.0)
        except Exception as predict_error:
            LOGGER.warning("Prediction failed for model '%s': %s", model_name, predict_error)
            outputs[model_name] = pd.Series(0.0, index=X.index, dtype="float64")
    return pd.DataFrame(outputs, index=X.index)


def predict_baseline_alerts(
    models: Mapping[str, BaselineModel],
    X: pd.DataFrame,
    *,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate binary alert outputs from baseline model probabilities."""
    probabilities = predict_baselines(models, X)
    return probabilities.apply(lambda column: apply_threshold(column, threshold=threshold), axis=0)
