"""Temporal cross-validation splitter utilities for baseline models."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterator

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeSeriesCVConfig:
    """Configuration for rolling-origin temporal splits.

    The default setup yields approximately:
    - 2009-2013 -> 2014
    - 2010-2014 -> 2015
    - 2011-2015 -> 2016
    - 2012-2016 -> 2017
    - 2013-2017 -> 2018
    - 2014-2018 -> 2019
    """

    date_column: str = "date"
    target_column: str = "outbreak_label"
    start_train_year: int = 2009
    first_valid_year: int = 2014
    last_valid_year: int = 2019
    train_window_years: int = 5
    thesis_strict: bool = False
    skip_single_class_folds: bool = True
    minimum_evaluated_folds: int = 1


def _resolve_year_series(df: pd.DataFrame, date_column: str) -> pd.Series:
    """Return row-wise year values from date-like columns."""
    if date_column in df.columns:
        parsed = pd.to_datetime(df[date_column], errors="coerce")
        return parsed.dt.year
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce")
    raise ValueError(
        f"Temporal CV requires either '{date_column}' or 'year' column."
    )


def _is_single_class(y: pd.Series) -> bool:
    """Check if a target vector has fewer than two classes."""
    non_na = y.dropna()
    return non_na.nunique() <= 1


def generate_time_splits(
    df: pd.DataFrame,
    config: TimeSeriesCVConfig,
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield leakage-safe rolling-origin train/validation index splits.

    Invalid folds are skipped when they have no train/validation rows or when
    training labels are single-class and ``skip_single_class_folds=True``.
    """
    for fold in build_fold_ledger(df, config):
        if fold["status"] != "yielded":
            LOGGER.info(
                "Skipping fold train=%s-%s valid=%s due to %s",
                fold["train_start_year"],
                fold["train_end_year"],
                fold["valid_year"],
                fold["reason"],
            )
            continue
        yield pd.Index(fold["train_index"]), pd.Index(fold["valid_index"])


def build_fold_ledger(df: pd.DataFrame, config: TimeSeriesCVConfig) -> list[dict[str, Any]]:
    """Build full fold accounting with yield/skip status and reasons."""
    ledger: list[dict[str, Any]] = []
    if df.empty:
        LOGGER.warning("Received empty dataframe; no CV folds generated")
        return ledger

    years = _resolve_year_series(df, config.date_column)
    for valid_year in range(config.first_valid_year, config.last_valid_year + 1):
        train_start = valid_year - config.train_window_years
        train_end = valid_year - 1

        train_mask = years.between(train_start, train_end, inclusive="both")
        valid_mask = years == valid_year
        train_index = df.index[train_mask.fillna(False)]
        valid_index = df.index[valid_mask.fillna(False)]

        status = "yielded"
        reason = "ok"

        if train_index.empty or valid_index.empty:
            status = "skipped"
            reason = "empty_slice"
        else:
            train_year_max = years.loc[train_index].max(skipna=True)
            valid_year_min = years.loc[valid_index].min(skipna=True)
            if pd.notna(train_year_max) and pd.notna(valid_year_min) and train_year_max >= valid_year_min:
                status = "skipped"
                reason = "potential_leakage"
            elif config.skip_single_class_folds and config.target_column in df.columns:
                y_train = pd.to_numeric(df.loc[train_index, config.target_column], errors="coerce")
                if _is_single_class(y_train):
                    status = "skipped"
                    reason = "single_class_train"

        ledger.append(
            {
                "valid_year": int(valid_year),
                "train_start_year": int(train_start),
                "train_end_year": int(train_end),
                "rows_train": int(len(train_index)),
                "rows_valid": int(len(valid_index)),
                "status": status,
                "reason": reason,
                "train_index": train_index.tolist(),
                "valid_index": valid_index.tolist(),
            }
        )

    return ledger
