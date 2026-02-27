"""Mechanistic feature generation for climate-disease relationships."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

_TEMPERATURE_CANDIDATES: tuple[str, ...] = (
    "temp_kelvin",
    "temperature_kelvin",
    "t2m",
    "temperature_celsius",
    "temp_celsius",
    "temperature",
    "temp",
    "mean_temp",
)
_RAINFALL_CANDIDATES: tuple[str, ...] = ("rainfall", "precipitation", "rain_mm")
_LAI_CANDIDATES: tuple[str, ...] = ("lai", "leaf_area_index")


def _pick_first_existing(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _pick_temperature_column(columns: Sequence[str]) -> str | None:
    direct_match = _pick_first_existing(columns, _TEMPERATURE_CANDIDATES)
    if direct_match is not None:
        return direct_match

    lower_map = {str(column).lower(): str(column) for column in columns}
    keyword_matches = [
        original_name
        for lowered_name, original_name in lower_map.items()
        if ("temp" in lowered_name) or ("t2m" in lowered_name)
    ]
    return keyword_matches[0] if keyword_matches else None


def _infer_temperature_units(series: pd.Series, column_name: str) -> str:
    lowered_name = column_name.lower()
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return "celsius"

    q05 = float(numeric.quantile(0.05))
    q95 = float(numeric.quantile(0.95))
    median = float(numeric.median())

    if "kelvin" in lowered_name or lowered_name.endswith("_k"):
        return "kelvin"
    if "celsius" in lowered_name or lowered_name.endswith("_c"):
        return "celsius"

    if q05 >= 150.0 and q95 <= 360.0:
        return "kelvin"
    if -40.0 <= q05 <= 80.0 and -40.0 <= q95 <= 80.0:
        return "celsius"
    if median > 120.0:
        return "kelvin"
    return "celsius"


def _is_degenerate(series: pd.Series, *, eps: float = 1e-12) -> bool:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return True
    if int(numeric.nunique()) <= 1:
        return True
    return float(numeric.var(ddof=0)) <= eps


def _extract_season(month: pd.Series) -> pd.Series:
    """Map month number to broad India-relevant seasons."""
    month_numeric = pd.to_numeric(month, errors="coerce").astype("Int64")
    season = pd.Series(pd.array([pd.NA] * len(month_numeric), dtype="string"), index=month_numeric.index)
    season = season.mask(month_numeric.isin([12, 1, 2]), "winter")
    season = season.mask(month_numeric.isin([3, 4, 5]), "pre_monsoon")
    season = season.mask(month_numeric.isin([6, 7, 8, 9]), "monsoon")
    season = season.mask(month_numeric.isin([10, 11]), "post_monsoon")
    return season


def _ensure_month_column(df: pd.DataFrame, date_column: str) -> pd.Series:
    if "month" in df.columns:
        return pd.to_numeric(df["month"], errors="coerce").astype("Int64")
    if date_column in df.columns:
        parsed = pd.to_datetime(df[date_column], errors="coerce")
        return parsed.dt.month.astype("Int64")
    return pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)


def build_mechanistic_features(
    df: pd.DataFrame,
    *,
    district_column: str = "district",
    date_column: str = "date",
    enable_climate_features: bool = True,
) -> pd.DataFrame:
    """Build mechanistic, climate-informed features conservatively.

    Features added when inputs are available:
    - ``temp_celsius = temp_kelvin - 273.15`` (or from ``temperature`` as fallback)
    - ``season`` from month
    - ``temp_anomaly`` by district+season
    - ``degree_days_20``
    - ``temp_optimal`` as 25-30°C indicator
    - ``rainfall_4wk`` rolling sum by district ordered by date
    - ``lai_anomaly`` by district+season

    Notes
    -----
    - Sparse or missing inputs result in safe no-op for specific features.
    - Legacy compatibility feature ``temp_rain_interaction`` is kept when possible.
    """
    LOGGER.info("Building mechanistic features")
    output = df.copy()
    output["month"] = _ensure_month_column(output, date_column=date_column)
    output["season"] = _extract_season(output["month"])

    if not enable_climate_features:
        LOGGER.info("Climate-derived mechanistic features disabled for current dataset profile")
        return output

    temp_column = _pick_temperature_column(output.columns)
    rainfall_column = _pick_first_existing(output.columns, _RAINFALL_CANDIDATES)
    lai_column = _pick_first_existing(output.columns, _LAI_CANDIDATES)

    if temp_column is None:
        LOGGER.warning("No temperature column found; skipping temperature-derived mechanistic features")
    else:
        temp_values = pd.to_numeric(output[temp_column], errors="coerce")
        inferred_units = _infer_temperature_units(temp_values, temp_column)
        if inferred_units == "kelvin":
            output["temp_kelvin"] = temp_values
            output["temp_celsius"] = temp_values - 273.15
        else:
            output["temp_celsius"] = temp_values
            output["temp_kelvin"] = temp_values + 273.15

        temp_has_signal = not _is_degenerate(output["temp_celsius"])

        degree_days = (output["temp_celsius"] - 20.0).clip(lower=0.0)
        if temp_has_signal and _is_degenerate(degree_days):
            adaptive_baseline = float(pd.to_numeric(output["temp_celsius"], errors="coerce").median(skipna=True))
            degree_days = (output["temp_celsius"] - adaptive_baseline).clip(lower=0.0)
            LOGGER.warning(
                "degree_days_20 was degenerate with fixed threshold; applied adaptive baseline %.3f to preserve signal",
                adaptive_baseline,
            )
        output["degree_days_20"] = degree_days

        temp_optimal = output["temp_celsius"].between(25.0, 30.0, inclusive="both").astype("Int64")
        if temp_has_signal and _is_degenerate(pd.to_numeric(temp_optimal, errors="coerce")):
            center = float(pd.to_numeric(output["temp_celsius"], errors="coerce").median(skipna=True))
            temp_optimal = output["temp_celsius"].between(center - 2.5, center + 2.5, inclusive="both").astype("Int64")
            LOGGER.warning(
                "temp_optimal was degenerate with fixed 25-30C band; applied adaptive band centered at %.3fC",
                center,
            )
        output["temp_optimal"] = temp_optimal

        if district_column in output.columns:
            group_cols = [district_column, "season"]
            district_season_temp_mean = output.groupby(group_cols, dropna=False)["temp_celsius"].transform("mean")
            temp_anomaly = output["temp_celsius"] - district_season_temp_mean
            if temp_has_signal and _is_degenerate(temp_anomaly):
                district_temp_mean = output.groupby(district_column, dropna=False)["temp_celsius"].transform("mean")
                temp_anomaly = output["temp_celsius"] - district_temp_mean
            if temp_has_signal and _is_degenerate(temp_anomaly):
                season_temp_mean = output.groupby("season", dropna=False)["temp_celsius"].transform("mean")
                temp_anomaly = output["temp_celsius"] - season_temp_mean
            if temp_has_signal and _is_degenerate(temp_anomaly):
                temp_anomaly = output["temp_celsius"] - output["temp_celsius"].mean(skipna=True)
            output["temp_anomaly"] = temp_anomaly
        else:
            LOGGER.warning("Missing district column '%s'; using season-only temp anomaly", district_column)
            season_temp_mean = output.groupby("season", dropna=False)["temp_celsius"].transform("mean")
            output["temp_anomaly"] = output["temp_celsius"] - season_temp_mean

    if rainfall_column is None:
        LOGGER.warning("No rainfall column found; skipping rainfall_4wk")
    elif district_column not in output.columns:
        LOGGER.warning("Missing district column '%s'; skipping rainfall_4wk", district_column)
    else:
        sort_keys = [district_column]
        if date_column in output.columns:
            output[date_column] = pd.to_datetime(output[date_column], errors="coerce")
            sort_keys.append(date_column)
        else:
            LOGGER.warning("Missing date column '%s'; rainfall_4wk will use row order within district", date_column)
        output = output.sort_values(sort_keys).copy()
        rain_numeric = pd.to_numeric(output[rainfall_column], errors="coerce")
        output["rainfall_4wk"] = rain_numeric.groupby(output[district_column], dropna=False).transform(
            lambda series: series.rolling(window=4, min_periods=1).sum()
        )

    if lai_column is None:
        LOGGER.warning("No LAI column found; skipping lai_anomaly")
    else:
        lai_numeric = pd.to_numeric(output[lai_column], errors="coerce")
        if district_column in output.columns:
            lai_mean = lai_numeric.groupby([output[district_column], output["season"]], dropna=False).transform("mean")
            output["lai_anomaly"] = lai_numeric - lai_mean
        else:
            lai_mean = lai_numeric.groupby(output["season"], dropna=False).transform("mean")
            output["lai_anomaly"] = lai_numeric - lai_mean

    if temp_column is not None and rainfall_column is not None:
        output["temp_rain_interaction"] = pd.to_numeric(output["temp_celsius"], errors="coerce") * pd.to_numeric(
            output[rainfall_column],
            errors="coerce",
        )

    return output


def run(
    df: pd.DataFrame,
    *,
    district_column: str = "district",
    date_column: str = "date",
    enable_climate_features: bool = True,
) -> pd.DataFrame:
    """Entrypoint for mechanistic feature generation."""
    return build_mechanistic_features(
        df,
        district_column=district_column,
        date_column=date_column,
        enable_climate_features=enable_climate_features,
    )
