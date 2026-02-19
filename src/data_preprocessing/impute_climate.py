"""Climate variable imputation routines."""

from __future__ import annotations

import logging
from typing import Mapping, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_CLIMATE_COLUMNS: tuple[str, ...] = (
    "rainfall",
    "temperature",
    "temp_kelvin",
    "humidity",
)

DEFAULT_CLIP_BOUNDS: dict[str, tuple[float, float]] = {
    "rainfall": (0.0, 2_000.0),
    "temperature": (-10.0, 60.0),
    "temp_kelvin": (240.0, 340.0),
    "humidity": (0.0, 100.0),
}


def _infer_temperature_units(series: pd.Series, column_name: str) -> str:
    lowered_name = str(column_name).lower()
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


def _normalize_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()

    if "temp_kelvin" in output.columns:
        kelvin = pd.to_numeric(output["temp_kelvin"], errors="coerce")
        celsius_existing = pd.to_numeric(output["temperature"], errors="coerce") if "temperature" in output.columns else pd.Series(index=output.index, dtype=float)
        celsius = celsius_existing.fillna(kelvin - 273.15)
        kelvin = kelvin.fillna(celsius + 273.15)
        output["temp_kelvin"] = kelvin
        output["temperature"] = celsius
        return output

    source_column: str | None = None
    for candidate in ("temperature", "temp", "temperature_kelvin", "t2m"):
        if candidate in output.columns:
            source_column = candidate
            break

    if source_column is None:
        return output

    source_values = pd.to_numeric(output[source_column], errors="coerce")
    inferred_units = _infer_temperature_units(source_values, source_column)
    if inferred_units == "kelvin":
        output["temp_kelvin"] = source_values
        output["temperature"] = source_values - 273.15
    else:
        output["temperature"] = source_values
        output["temp_kelvin"] = source_values + 273.15
    return output


def _ensure_month_column(df: pd.DataFrame, date_column: str = "date") -> pd.Series:
    """Return a month series from existing month column or parsed date."""
    if "month" in df.columns:
        month = pd.to_numeric(df["month"], errors="coerce")
        return month.astype("Int64")
    if date_column in df.columns:
        parsed = pd.to_datetime(df[date_column], errors="coerce")
        return parsed.dt.month.astype("Int64")
    return pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"), index=df.index)


def clip_climate_values(
    df: pd.DataFrame,
    clip_bounds: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Clip climate values to plausible physical ranges."""
    output = _normalize_temperature_columns(df)
    for column, bounds in clip_bounds.items():
        if column in output.columns:
            lower, upper = bounds
            output[column] = pd.to_numeric(output[column], errors="coerce").clip(lower=lower, upper=upper)
    return output


def _impute_single_climate_column(df: pd.DataFrame, climate_column: str) -> pd.Series:
    """Impute one climate column using district+month median then global median."""
    values = pd.to_numeric(df[climate_column], errors="coerce")
    district_month_median = values.groupby([df["district"], df["month"]], dropna=False).transform("median")
    filled = values.fillna(district_month_median)
    global_median = values.median(skipna=True)
    if pd.notna(global_median):
        filled = filled.fillna(global_median)
    return filled


def impute_climate(
    df: pd.DataFrame,
    *,
    climate_columns: Sequence[str] = DEFAULT_CLIMATE_COLUMNS,
    clip_bounds: Mapping[str, tuple[float, float]] = DEFAULT_CLIP_BOUNDS,
    date_column: str = "date",
) -> pd.DataFrame:
    """Impute climate values conservatively for sparse data.

    Strategy
    --------
    1) Clip impossible values.
    2) Impute by district+month median.
    3) Fallback to global median.

    Epidemiological case columns are never imputed here.
    """
    LOGGER.info("Imputing climate variables")
    output = df.copy()
    output = clip_climate_values(output, clip_bounds=clip_bounds)
    output["month"] = _ensure_month_column(output, date_column=date_column)

    if "district" not in output.columns:
        LOGGER.warning("'district' column missing; using global median fallback only")
        for column in climate_columns:
            if column in output.columns:
                numeric = pd.to_numeric(output[column], errors="coerce")
                median = numeric.median(skipna=True)
                output[column] = numeric.fillna(median) if pd.notna(median) else numeric
        return output

    for column in climate_columns:
        if column not in output.columns:
            LOGGER.debug("Skipping absent climate column '%s'", column)
            continue
        missing_before = int(output[column].isna().sum())
        output[column] = _impute_single_climate_column(output, column)
        missing_after = int(output[column].isna().sum())
        LOGGER.info("Imputed climate column '%s': missing %d -> %d", column, missing_before, missing_after)

    if "temperature" in output.columns or "temp_kelvin" in output.columns:
        celsius = pd.to_numeric(output.get("temperature"), errors="coerce") if "temperature" in output.columns else pd.Series(index=output.index, dtype=float)
        kelvin = pd.to_numeric(output.get("temp_kelvin"), errors="coerce") if "temp_kelvin" in output.columns else pd.Series(index=output.index, dtype=float)
        celsius = celsius.fillna(kelvin - 273.15)
        kelvin = kelvin.fillna(celsius + 273.15)
        output["temperature"] = celsius
        output["temp_kelvin"] = kelvin

    return output


def run(df: pd.DataFrame) -> pd.DataFrame:
    """Entrypoint for the climate imputation phase."""
    return impute_climate(df)
