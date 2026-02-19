from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterator

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeSeriesCVConfig:
    date_column: str = "date"
    target_column: str = "outbreak_label"
    start_train_year: int = 2015
    first_valid_year: int = 2016
    last_valid_year: int = 2020
    train_window_years: int = 5
    thesis_strict: bool = False
    skip_single_class_folds: bool = True
    minimum_evaluated_folds: int = 1


def _cfg_bool(config: Any, key: str, default: bool = False) -> bool:
    value = getattr(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _cfg_int(config: Any, key: str, default: int) -> int:
    raw = getattr(config, key, default)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return int(default)


def _fold_plan(config: Any) -> tuple[str, list[tuple[int, int, int]]]:
    thesis_strict = _cfg_bool(config, "thesis_strict", False)
    if thesis_strict:
        plan = [(2015, valid_year - 1, valid_year) for valid_year in range(2016, 2021)]
        return "thesis_strict", plan

    start_train_year = _cfg_int(config, "start_train_year", 2015)
    first_valid_year = _cfg_int(config, "first_valid_year", 2016)
    last_valid_year = _cfg_int(config, "last_valid_year", 2020)
    plan = [
        (start_train_year, valid_year - 1, valid_year)
        for valid_year in range(first_valid_year, last_valid_year + 1)
    ]
    return "range", plan


def _resolve_year_series(df: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column in df.columns:
        parsed = pd.to_datetime(df[date_column], errors="coerce")
        return parsed.dt.year
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce")
    raise ValueError(f"CV adapter requires '{date_column}' or 'year' column")


def _is_single_class(y: pd.Series) -> bool:
    non_na = y.dropna()
    return non_na.nunique() <= 1


def build_fold_ledger(df: pd.DataFrame, config: TimeSeriesCVConfig) -> list[dict[str, Any]]:
    if df.empty:
        return []

    years = _resolve_year_series(df, config.date_column)
    ledger: list[dict[str, Any]] = []
    fold_mode, plan = _fold_plan(config)

    LOGGER.info(
        "Brazil CV fold mode=%s thesis_strict=%s folds=%d",
        fold_mode,
        _cfg_bool(config, "thesis_strict", False),
        len(plan),
    )

    for train_start, train_end, valid_year in plan:

        train_mask = years.between(train_start, train_end, inclusive="both")
        valid_mask = years == valid_year

        train_index = df.index[train_mask.fillna(False)]
        valid_index = df.index[valid_mask.fillna(False)]

        status = "yielded"
        reason = "ok"

        if train_index.empty or valid_index.empty:
            status = "skipped"
            reason = "empty_slice"
        elif config.skip_single_class_folds and config.target_column in df.columns:
            y_train = pd.to_numeric(df.loc[train_index, config.target_column], errors="coerce")
            if _is_single_class(y_train):
                status = "skipped"
                reason = "single_class_train"

        ledger.append(
            {
                "fold_mode": fold_mode,
                "thesis_strict": _cfg_bool(config, "thesis_strict", False),
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


def generate_time_splits(df: pd.DataFrame, config: TimeSeriesCVConfig) -> Iterator[tuple[pd.Index, pd.Index]]:
    for fold in build_fold_ledger(df, config):
        if fold["status"] != "yielded":
            continue
        yield pd.Index(fold["train_index"]), pd.Index(fold["valid_index"])
