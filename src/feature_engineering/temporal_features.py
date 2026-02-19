"""Temporal feature engineering for lagged and calendar effects."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _resolve_case_column(df: pd.DataFrame) -> str | None:
    for candidate in ("cases", "case_count", "weekly_cases"):
        if candidate in df.columns:
            return candidate
    return None


def build_temporal_features(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    district_column: str = "district",
    lag_periods: Iterable[int] = (1, 2),
) -> pd.DataFrame:
    """Build temporal features with conservative sparse-data handling.

    Added features when possible:
    - ``case_rolling_mean``: 4-week rolling mean (min_periods=1)
    - ``case_rolling_std``: 4-week rolling std (min_periods=3), ``fillna(0)``
    - ``case_velocity``: first difference of cases by district
    - ``case_acceleration``: first difference of velocity by district
    - ``case_lag_1``, ``case_lag_2`` (configurable)
    """
    LOGGER.info("Building temporal features")
    output = df.copy()

    if date_column in output.columns:
        dates = pd.to_datetime(output[date_column], errors="coerce")
        output[date_column] = dates
        output["year"] = dates.dt.year.astype("Int64")
        output["month"] = dates.dt.month.astype("Int64")
        output["weekofyear"] = dates.dt.isocalendar().week.astype("Int64")
    else:
        LOGGER.warning("Date column '%s' missing; skipping calendar decomposition", date_column)

    case_column = _resolve_case_column(output)
    if case_column is None:
        LOGGER.warning("No case column found; skipping temporal case-derived features")
        return output

    case_values = pd.to_numeric(output[case_column], errors="coerce")
    has_district = district_column in output.columns
    if has_district:
        sort_keys = [district_column]
        if date_column in output.columns:
            sort_keys.append(date_column)
        output = output.sort_values(sort_keys).copy()
        case_values = pd.to_numeric(output[case_column], errors="coerce")
        lagged_case_values = case_values.groupby(output[district_column], dropna=False).shift(1)

        output["case_rolling_mean"] = lagged_case_values.groupby(output[district_column], dropna=False).transform(
            lambda series: series.rolling(window=4, min_periods=1).mean()
        )
        output["case_rolling_std"] = (
            lagged_case_values.groupby(output[district_column], dropna=False)
            .transform(lambda series: series.rolling(window=4, min_periods=3).std())
            .fillna(0.0)
        )
        output["case_velocity"] = lagged_case_values.groupby(output[district_column], dropna=False).diff()
        output["case_acceleration"] = output["case_velocity"].groupby(output[district_column], dropna=False).diff()
        for lag in lag_periods:
            output[f"case_lag_{lag}"] = case_values.groupby(output[district_column], dropna=False).shift(lag)
    else:
        LOGGER.warning("District column '%s' missing; using global temporal features", district_column)
        if date_column in output.columns:
            output = output.sort_values([date_column]).copy()
            case_values = pd.to_numeric(output[case_column], errors="coerce")
        lagged_case_values = case_values.shift(1)
        output["case_rolling_mean"] = lagged_case_values.rolling(window=4, min_periods=1).mean()
        output["case_rolling_std"] = lagged_case_values.rolling(window=4, min_periods=3).std().fillna(0.0)
        output["case_velocity"] = lagged_case_values.diff()
        output["case_acceleration"] = output["case_velocity"].diff()
        for lag in lag_periods:
            output[f"case_lag_{lag}"] = case_values.shift(lag)

    return output


def run(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    district_column: str = "district",
    lag_periods: Iterable[int] = (1, 2),
) -> pd.DataFrame:
    """Entrypoint for temporal feature generation."""
    return build_temporal_features(
        df,
        date_column=date_column,
        district_column=district_column,
        lag_periods=lag_periods,
    )
