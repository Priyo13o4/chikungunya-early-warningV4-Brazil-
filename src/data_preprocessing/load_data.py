"""Data loading utilities for raw chikungunya and optional census data.

This module is intentionally import-safe: no I/O occurs at import time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable
import re

import pandas as pd

LOGGER = logging.getLogger(__name__)

SUPPORTED_TABULAR_SUFFIXES: tuple[str, ...] = (".csv", ".xlsx", ".xls")
POPULATION_DISCOVERY_KEYWORDS: tuple[str, ...] = ("census", "indiastatedist", "population")

DEFAULT_DATE_COLUMNS: tuple[str, ...] = ("date", "week_start", "month_start")
DEFAULT_DTYPES: dict[str, str] = {
    "district": "string",
    "state": "string",
}


def normalize_column_name(column_name: str) -> str:
    """Normalize a raw column name to a safe snake_case-like representation."""
    normalized = str(column_name).strip().lower()
    normalized = normalized.replace("%", "pct")
    normalized = normalized.replace("/", "_")
    normalized = normalized.replace("-", "_")
    normalized = "_".join(part for part in normalized.split())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    output = df.copy()
    output.columns = [normalize_column_name(col) for col in output.columns]
    return output


def parse_date_columns(df: pd.DataFrame, date_columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Parse known or provided date columns to pandas datetime."""
    output = df.copy()
    candidates = tuple(date_columns) if date_columns is not None else DEFAULT_DATE_COLUMNS
    for column in candidates:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column], errors="coerce")
    return output


def _read_csv(path: Path, dtype_overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Read a CSV with conservative defaults and optional dtype overrides."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    LOGGER.info("Reading CSV from %s", path)
    return pd.read_csv(path, low_memory=False, dtype=dtype_overrides)


def _read_excel(path: Path, dtype_overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Read an Excel workbook with optional dtype overrides."""
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {path}")
    LOGGER.info("Reading Excel from %s", path)
    return pd.read_excel(path, dtype=dtype_overrides)


def _read_table(path: Path, dtype_overrides: dict[str, str] | None = None) -> pd.DataFrame:
    """Read supported tabular files by extension."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _read_csv(path, dtype_overrides=dtype_overrides)
    if suffix in {".xlsx", ".xls"}:
        return _read_excel(path, dtype_overrides=dtype_overrides)
    raise ValueError(
        f"Unsupported tabular file extension '{path.suffix}' for {path}. "
        f"Supported extensions: {', '.join(SUPPORTED_TABULAR_SUFFIXES)}"
    )


def _normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def discover_population_data_path(search_dir: Path = Path("data/raw")) -> Path | None:
    """Discover likely population/census file from data/raw-like locations."""
    if not search_dir.exists():
        LOGGER.info("Population auto-discovery skipped: search directory not found at %s", search_dir)
        return None

    scored_candidates: list[tuple[int, int, str, Path]] = []
    extension_priority = {".csv": 0, ".xlsx": 1, ".xls": 2}
    for file_path in search_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_TABULAR_SUFFIXES:
            continue
        normalized_name = _normalize_for_match(file_path.stem)
        keyword_rank = next(
            (idx for idx, keyword in enumerate(POPULATION_DISCOVERY_KEYWORDS) if keyword in normalized_name),
            None,
        )
        if keyword_rank is None:
            continue
        scored_candidates.append(
            (
                keyword_rank,
                extension_priority.get(file_path.suffix.lower(), 99),
                str(file_path).lower(),
                file_path,
            )
        )

    if not scored_candidates:
        LOGGER.info(
            "Population auto-discovery found no matching files in %s (keywords=%s, extensions=%s)",
            search_dir,
            ",".join(POPULATION_DISCOVERY_KEYWORDS),
            ",".join(SUPPORTED_TABULAR_SUFFIXES),
        )
        return None

    scored_candidates.sort()
    discovered_path = scored_candidates[0][3]
    LOGGER.info("Population auto-discovery selected: %s", discovered_path)
    return discovered_path


def load_epiclim_data(
    path: Path,
    date_columns: Iterable[str] | None = None,
    dtype_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load Epiclim data, normalize column names, and parse date columns."""
    dtypes = dict(DEFAULT_DTYPES)
    if dtype_overrides:
        dtypes.update(dtype_overrides)
    df = _read_table(path, dtype_overrides=dtypes)
    df = normalize_columns(df)
    df = parse_date_columns(df, date_columns=date_columns)
    LOGGER.info("Loaded Epiclim data from %s with shape=%s", path, df.shape)
    return df


def load_census_data(
    path: Path | None,
    date_columns: Iterable[str] | None = None,
    dtype_overrides: dict[str, str] | None = None,
    discovery_dir: Path = Path("data/raw"),
) -> pd.DataFrame | None:
    """Load optional census data; return ``None`` if path is not provided."""
    census_path = path
    if census_path is None:
        census_path = discover_population_data_path(search_dir=discovery_dir)

    if census_path is None:
        LOGGER.info("No census/population dataset provided or discovered; skipping census load")
        return None

    if path is not None:
        LOGGER.info("Using explicit population path from CLI/config: %s", path)

    dtypes = dict(DEFAULT_DTYPES)
    if dtype_overrides:
        dtypes.update(dtype_overrides)

    try:
        df = _read_table(census_path, dtype_overrides=dtypes)
    except FileNotFoundError:
        LOGGER.warning("Population dataset path does not exist: %s. Continuing without census merge.", census_path)
        return None
    except ImportError as import_error:
        LOGGER.warning(
            "Population dataset requires optional Excel dependency that is unavailable (%s). "
            "Continuing without census merge.",
            import_error,
        )
        return None
    except ValueError as read_error:
        LOGGER.warning("Population dataset could not be loaded (%s). Continuing without census merge.", read_error)
        return None

    df = normalize_columns(df)
    df = parse_date_columns(df, date_columns=date_columns)
    LOGGER.info("Loaded census/population data from %s with shape=%s", census_path, df.shape)
    return df


def run(
    epiclim_path: Path,
    census_path: Path | None = None,
    date_columns: Iterable[str] | None = None,
    discovery_dir: Path = Path("data/raw"),
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Entrypoint for the loading phase.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        A tuple of ``(epiclim_df, census_df_or_none)``.
    """
    epiclim_df = load_epiclim_data(epiclim_path, date_columns=date_columns)
    census_df = load_census_data(census_path, date_columns=date_columns, discovery_dir=discovery_dir)
    return epiclim_df, census_df
