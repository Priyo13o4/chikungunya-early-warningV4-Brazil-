"""Temporal cross-validation splitter utilities for baseline models."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterator

import numpy as np
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
    thesis_strict_mode: str = "expanding"
    skip_single_class_folds: bool = True
    minimum_evaluated_folds: int = 1
    min_outbreak_count_per_fold: int = 1
    min_class_ratio_per_fold: float = 0.0
    max_class_ratio_per_fold: float = 1.0
    min_municipality_count_per_fold: int = 1
    min_train_span_years: int = 1
    fail_on_gate_violation: bool = True


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


def _gate_violations(
    *,
    df: pd.DataFrame,
    train_index: pd.Index,
    valid_index: pd.Index,
    years: pd.Series,
    config: TimeSeriesCVConfig,
) -> tuple[list[str], dict[str, Any]]:
    metrics: dict[str, Any] = {
        "outbreaks_train": 0,
        "outbreaks_valid": 0,
        "positive_ratio_train": float("nan"),
        "positive_ratio_valid": float("nan"),
        "municipalities_train": 0,
        "train_span_years": 0,
    }
    violations: list[str] = []

    if config.target_column in df.columns:
        y_train = pd.to_numeric(df.loc[train_index, config.target_column], errors="coerce").fillna(0.0)
        y_valid = pd.to_numeric(df.loc[valid_index, config.target_column], errors="coerce").fillna(0.0)
        outbreaks_train = int((y_train > 0.5).sum())
        outbreaks_valid = int((y_valid > 0.5).sum())
        ratio_train = float((y_train > 0.5).mean()) if len(y_train) else float("nan")
        ratio_valid = float((y_valid > 0.5).mean()) if len(y_valid) else float("nan")
        metrics.update(
            {
                "outbreaks_train": outbreaks_train,
                "outbreaks_valid": outbreaks_valid,
                "positive_ratio_train": ratio_train,
                "positive_ratio_valid": ratio_valid,
            }
        )

        min_outbreak = max(1, int(config.min_outbreak_count_per_fold))
        if outbreaks_train < min_outbreak or outbreaks_valid < min_outbreak:
            violations.append("min_outbreak_count")

        min_ratio = float(config.min_class_ratio_per_fold)
        max_ratio = float(config.max_class_ratio_per_fold)
        if not np.isfinite(ratio_train) or ratio_train < min_ratio or ratio_train > max_ratio:
            violations.append("train_class_ratio_bounds")
        if not np.isfinite(ratio_valid) or ratio_valid < min_ratio or ratio_valid > max_ratio:
            violations.append("valid_class_ratio_bounds")

    municipality_column = "municipality_id" if "municipality_id" in df.columns else ("district" if "district" in df.columns else None)
    if municipality_column is not None:
        municipality_count = int(df.loc[train_index, municipality_column].astype("string").nunique(dropna=True))
        metrics["municipalities_train"] = municipality_count
        if municipality_count < max(1, int(config.min_municipality_count_per_fold)):
            violations.append("min_municipality_count")

    train_years = pd.to_numeric(years.loc[train_index], errors="coerce").dropna().astype(int)
    if not train_years.empty:
        span_years = int(train_years.max() - train_years.min() + 1)
        metrics["train_span_years"] = span_years
        if span_years < max(1, int(config.min_train_span_years)):
            violations.append("min_train_span_years")

    return sorted(set(violations)), metrics


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
    thesis_mode = str(getattr(config, "thesis_strict_mode", "expanding")).strip().lower()
    for valid_year in range(config.first_valid_year, config.last_valid_year + 1):
        if bool(config.thesis_strict):
            if thesis_mode == "rolling":
                train_start = max(int(config.start_train_year), valid_year - int(config.train_window_years))
            else:
                train_start = int(config.start_train_year)
        else:
            train_start = valid_year - int(config.train_window_years)
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

        gate_metrics: dict[str, Any] | None = None
        if status == "yielded":
            violations, gate_metrics = _gate_violations(
                df=df,
                train_index=train_index,
                valid_index=valid_index,
                years=years,
                config=config,
            )
            if violations:
                if bool(config.fail_on_gate_violation):
                    raise RuntimeError(
                        "CV statistical gate failure "
                        f"valid_year={valid_year} train={train_start}-{train_end} violations={violations} metrics={gate_metrics}"
                    )
                status = "skipped"
                reason = "statistical_gate_failed"

        ledger.append(
            {
                "valid_year": int(valid_year),
                "train_start_year": int(train_start),
                "train_end_year": int(train_end),
                "rows_train": int(len(train_index)),
                "rows_valid": int(len(valid_index)),
                "status": status,
                "reason": reason,
                "gate_metrics": gate_metrics,
                "train_index": train_index.tolist(),
                "valid_index": valid_index.tolist(),
            }
        )

    return ledger
