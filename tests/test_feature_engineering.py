"""Tests for feature engineering pipeline."""

from __future__ import annotations

import pandas as pd

from src.feature_engineering.build_feature_matrix import build_feature_matrix
from src.feature_engineering.mechanistic_features import build_mechanistic_features
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


def test_profile_without_lat_lon_does_not_create_neighbor_feature() -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-08"],
            "district": ["A", "A"],
            "municipality_id": [1001, 1001],
            "state": ["S", "S"],
            "cases": [1, 2],
            "rt": [0.9, 1.1],
            "p_rt1": [0.4, 0.6],
            "receptivo": [1, 1],
            "outbreak_label": [0, 1],
        }
    )

    features = build_feature_matrix(df)

    assert "district_code" in features.columns
    assert "neighbor_mean_cases" not in features.columns


def test_profile_without_climate_sources_does_not_create_mechanistic_climate_features() -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-08"],
            "district": ["A", "A"],
            "municipality_id": [1001, 1001],
            "state": ["S", "S"],
            "cases": [1, 2],
            "rt": [0.9, 1.1],
            "p_rt1": [0.4, 0.6],
            "receptivo": [1, 1],
            "outbreak_label": [0, 1],
        }
    )

    features = build_feature_matrix(df)

    assert "month" in features.columns
    assert "season" in features.columns
    assert "outbreak_label" in features.columns
    assert "temp_kelvin" not in features.columns
    assert "temp_celsius" not in features.columns
    assert "temp_anomaly" not in features.columns
    assert "degree_days_20" not in features.columns
    assert "temp_optimal" not in features.columns
    assert "rainfall_4wk" not in features.columns
    assert "lai_anomaly" not in features.columns
    assert "temp_rain_interaction" not in features.columns


def test_mechanistic_climate_features_are_lagged_by_one_week() -> None:
    df = pd.DataFrame(
        {
            "date": ["2025-01-01", "2025-01-08", "2025-01-15"],
            "district": ["A", "A", "A"],
            "temperature": [20.0, 21.0, 22.0],
            "rainfall": [1.0, 10.0, 1000.0],
        }
    )

    features = build_mechanistic_features(df)

    assert pd.isna(features.loc[0, "temp_celsius"])
    assert features.loc[1, "temp_celsius"] == 20.0
    assert features.loc[2, "temp_celsius"] == 21.0
    assert pd.isna(features.loc[0, "rainfall_4wk"])
    assert features.loc[1, "rainfall_4wk"] == 1.0
    assert features.loc[2, "rainfall_4wk"] == 11.0
    assert pd.isna(features.loc[0, "temp_rain_interaction"])
    assert features.loc[1, "temp_rain_interaction"] == 20.0
