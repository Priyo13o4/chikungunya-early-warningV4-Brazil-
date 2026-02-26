"""Memory optimization filtering helpers for pipeline runtime."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from src.pipeline_runtime import config_runtime as runtime_config


def stable_district_shard(value: Any, shard_count: int) -> int:
    normalized = str(value).strip().lower()
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(shard_count)


def apply_memory_optimization_filters(
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    config: runtime_config.MemoryOptimizationConfig,
    date_column: str = "date",
    district_column: str = "district",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mode = str(config.mode or "off").lower()
    report: dict[str, Any] = {
        "mode": mode,
        "active": False,
        "reproducibility": {
            "district_hash": "md5_lower_utf8_hex8_mod",
        },
        "inputs": {
            "labeled_rows": int(len(labeled_df)),
            "feature_rows": int(len(features_df)),
            "labeled_districts": int(labeled_df.get(district_column, pd.Series(dtype=object)).nunique(dropna=True)),
        },
        "filters": {
            "year_window": {
                "requested": list(config.train_year_window) if config.train_year_window is not None else None,
                "applied": False,
            },
            "district_shard": {
                "requested_count": config.district_shard_count,
                "requested_index": config.district_shard_index,
                "applied": False,
            },
        },
        "warnings": [],
    }

    if mode == "off":
        report["outputs"] = {
            "labeled_rows": int(len(labeled_df)),
            "feature_rows": int(len(features_df)),
            "labeled_districts": int(labeled_df.get(district_column, pd.Series(dtype=object)).nunique(dropna=True)),
        }
        return labeled_df, features_df, report

    keep_mask = pd.Series(True, index=labeled_df.index, dtype="bool")

    apply_year_filter = mode in {"year_window", "hybrid"}
    if apply_year_filter:
        if config.train_year_window is None:
            report["warnings"].append("year_window_mode_requested_but_train_year_window_missing")
        elif date_column not in labeled_df.columns:
            report["warnings"].append("year_window_mode_requested_but_date_column_missing")
        else:
            start_year, end_year = config.train_year_window
            date_series = pd.to_datetime(labeled_df[date_column], errors="coerce")
            year_mask = date_series.dt.year.between(int(start_year), int(end_year), inclusive="both").fillna(False)
            keep_mask &= year_mask
            report["filters"]["year_window"].update(
                {
                    "applied": True,
                    "effective": [int(start_year), int(end_year)],
                    "kept_rows": int(year_mask.sum()),
                }
            )

    apply_shard_filter = mode in {"district_shard", "hybrid"}
    if apply_shard_filter:
        shard_count = config.district_shard_count
        shard_index = config.district_shard_index
        if shard_count is None:
            report["warnings"].append("district_shard_mode_requested_but_district_shard_count_missing")
        elif district_column not in labeled_df.columns:
            report["warnings"].append("district_shard_mode_requested_but_district_column_missing")
        else:
            safe_index = int(shard_index if shard_index is not None else 0) % int(shard_count)
            district_values = labeled_df[district_column].fillna("__missing_district__").astype(str)
            shard_series = district_values.map(lambda value: stable_district_shard(value, int(shard_count)))
            shard_mask = shard_series.eq(int(safe_index)).fillna(False)
            keep_mask &= shard_mask
            report["filters"]["district_shard"].update(
                {
                    "applied": True,
                    "effective_count": int(shard_count),
                    "effective_index": int(safe_index),
                    "kept_rows": int(shard_mask.sum()),
                }
            )

    filtered_labeled = labeled_df.loc[keep_mask].copy()
    if features_df.index.equals(labeled_df.index):
        filtered_features = features_df.loc[keep_mask].copy()
    else:
        selected_index = pd.Index(filtered_labeled.index)
        filtered_features = features_df.loc[features_df.index.intersection(selected_index)].copy()

    report["active"] = bool(len(filtered_labeled) != len(labeled_df))
    report["outputs"] = {
        "labeled_rows": int(len(filtered_labeled)),
        "feature_rows": int(len(filtered_features)),
        "labeled_districts": int(filtered_labeled.get(district_column, pd.Series(dtype=object)).nunique(dropna=True)),
        "rows_removed": int(len(labeled_df) - len(filtered_labeled)),
    }
    return filtered_labeled, filtered_features, report
