"""Tests for bayesian scaffold model."""

from __future__ import annotations

import pandas as pd

from src.models.bayesian import hierarchical_model


def test_bayesian_model_fit_predict() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)
        preds = model.predict(X)
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader

    assert len(preds) == 3


def test_bayesian_predict_with_uncertainty_returns_intervals_and_metadata() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = pd.DataFrame(
            {
                "district": ["A", "A", "B"],
                "date": ["2016-01-01", "2016-01-08", "2016-01-15"],
                "temp_anomaly": [0.1, 0.2, -0.1],
            }
        )
        y = pd.Series([0.0, 2.0, 1.0])
        thresholds = pd.Series([1.0, 3.0, 1.0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)
        risk_frame, metadata = model.predict_with_uncertainty(X, outbreak_threshold=thresholds)
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader

    assert {"risk_mean", "risk_q05", "risk_q95", "threshold_cases", "bayesian_risk"}.issubset(risk_frame.columns)
    assert len(risk_frame) == len(X)
    assert bool(metadata.get("degraded_mode", False)) is True
    assert metadata.get("threshold_basis") == "provided_series"
    assert float(risk_frame.loc[0, "risk_mean"]) >= float(risk_frame.loc[1, "risk_mean"])
