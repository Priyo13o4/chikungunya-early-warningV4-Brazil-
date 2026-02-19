"""Tests for feature engineering pipeline."""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.build_feature_matrix import build_feature_matrix
from src.feature_engineering.spatial_features import build_spatial_features
from src.feature_engineering.temporal_features import build_temporal_features


def test_build_feature_matrix_runs() -> None:
    df = pd.DataFrame({"date": ["2025-01-01"], "district": ["A"], "temperature": [30], "rainfall": [10]})
    features = build_feature_matrix(df)
    assert "month" in features.columns
    assert "district_code" in features.columns
    assert "temp_rain_interaction" in features.columns


def test_case_rolling_std_defaults_to_zero_for_insufficient_periods() -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-08"],
            "district": ["A", "A"],
            "cases": [5, 9],
        }
    )
    features = build_temporal_features(df)
    assert features["case_rolling_std"].tolist() == [0.0, 0.0]


def test_case_temporal_predictors_are_lagged() -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
            "district": ["A", "A", "A"],
            "cases": [1, 3, 10],
        }
    )

    features = build_temporal_features(df)

    assert pd.isna(features.loc[0, "case_rolling_mean"])
    assert features.loc[1, "case_rolling_mean"] == 1.0
    assert features.loc[2, "case_rolling_mean"] == 2.0
    assert features.loc[2, "case_velocity"] == 2.0


def test_spatial_neighbor_mean_cases_uses_lagged_neighbor_values() -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-01", "2025-01-08", "2025-01-08"],
            "district": ["A", "B", "A", "B"],
            "lat": [0.0, 0.0, 0.0, 0.0],
            "lon": [0.0, 1.0, 0.0, 1.0],
            "cases": [1, 10, 2, 999],
        }
    )

    features = build_spatial_features(df, k_neighbors=1)

    assert pd.isna(features.loc[0, "neighbor_mean_cases"])
    assert pd.isna(features.loc[1, "neighbor_mean_cases"])
    assert features.loc[2, "neighbor_mean_cases"] == 10.0
    assert features.loc[3, "neighbor_mean_cases"] == 1.0
