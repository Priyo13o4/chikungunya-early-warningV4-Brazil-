"""Data cleaning routines for epidemiological and climate records."""

from __future__ import annotations

import logging
from typing import Iterable, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_REQUIRED_COLUMNS: tuple[str, ...] = ("district", "date")
DEFAULT_NUMERIC_COLUMNS: tuple[str, ...] = (
    "cases",
    "rainfall",
    "temperature",
    "temp",
    "temp_kelvin",
    "humidity",
    "population",
)

_COLUMN_ALIASES: dict[str, str] = {
    "state_ut": "state",
    "preci": "rainfall",
    "longitute": "longitude",
}


def _normalize_known_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known source aliases to canonical pipeline column names."""
    output = df.copy()
    rename_map = {column: _COLUMN_ALIASES[column] for column in output.columns if column in _COLUMN_ALIASES}
    if rename_map:
        output = output.rename(columns=rename_map)
    return output


def _synthesize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Build a canonical date column from common day/month/year fields when missing."""
    output = df.copy()
    if "date" in output.columns:
        return output
    if {"day", "mon", "year"}.issubset(output.columns):
        output["date"] = pd.to_datetime(
            {
                "year": pd.to_numeric(output["year"], errors="coerce"),
                "month": pd.to_numeric(output["mon"], errors="coerce"),
                "day": pd.to_numeric(output["day"], errors="coerce"),
            },
            errors="coerce",
        )
        return output
    return output


def enforce_required_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    """Raise a clear error if required columns are missing."""
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def filter_years(
    df: pd.DataFrame,
    date_column: str = "date",
    start_year: int = 2009,
    end_year: int = 2019,
) -> pd.DataFrame:
    """Filter rows to a year range using a date column if present."""
    output = df.copy()
    if date_column not in output.columns:
        LOGGER.warning("Date column '%s' not found; skipping year filtering", date_column)
        return output
    parsed_dates = pd.to_datetime(output[date_column], errors="coerce")
    years = parsed_dates.dt.year
    mask = years.between(start_year, end_year, inclusive="both")
    kept = int(mask.fillna(False).sum())
    LOGGER.info("Year filter %s-%s kept %d/%d rows", start_year, end_year, kept, len(output))
    output = output.loc[mask.fillna(False)].copy()
    output[date_column] = parsed_dates.loc[output.index]
    return output


def deduplicate_rows(df: pd.DataFrame, subset: Sequence[str] | None = None) -> pd.DataFrame:
    """Drop duplicates conservatively, optionally using a subset of keys."""
    output = df.copy()
    before = len(output)
    output = output.drop_duplicates(subset=list(subset) if subset else None)
    removed = before - len(output)
    if removed:
        LOGGER.info("Removed %d duplicate rows", removed)
    return output


def coerce_numeric_columns(df: pd.DataFrame, numeric_columns: Iterable[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric with invalid values as missing."""
    output = df.copy()
    for column in numeric_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


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


def normalize_temperature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize temperature semantics across raw aliases.

    Output conventions:
    - ``temperature`` is always Celsius.
    - ``temp_kelvin`` is always Kelvin.
    - Original source columns are preserved.
    """
    output = df.copy()
    source_column: str | None = None
    for candidate in ("temp_kelvin", "temperature_kelvin", "temperature", "temp", "t2m"):
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


def clean_data(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] = DEFAULT_REQUIRED_COLUMNS,
    numeric_columns: Iterable[str] = DEFAULT_NUMERIC_COLUMNS,
    start_year: int = 2009,
    end_year: int = 2019,
    dedupe_subset: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Clean records with schema checks and conservative coercion.

    Notes
    -----
    - This function never fabricates epidemiological values.
    - Numeric coercion only converts invalid values to missing.
    """
    LOGGER.info("Cleaning data with %d rows", len(df))
    cleaned = df.copy()
    cleaned.columns = [str(col).strip().lower() for col in cleaned.columns]
    cleaned = _normalize_known_aliases(cleaned)
    cleaned = _synthesize_date_column(cleaned)
    enforce_required_columns(cleaned, required_columns=required_columns)
    cleaned = filter_years(cleaned, start_year=start_year, end_year=end_year)
    cleaned = deduplicate_rows(cleaned, subset=dedupe_subset)
    cleaned = coerce_numeric_columns(cleaned, numeric_columns=numeric_columns)
    cleaned = normalize_temperature_columns(cleaned)
    LOGGER.info("Finished cleaning with shape=%s", cleaned.shape)
    return cleaned


def run(
    df: pd.DataFrame,
    *,
    start_year: int = 2009,
    end_year: int = 2019,
) -> pd.DataFrame:
    """Entrypoint for the cleaning phase."""
    return clean_data(df, start_year=start_year, end_year=end_year)
