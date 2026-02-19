"""Tests for CV splitter utilities."""

from __future__ import annotations

import pandas as pd

from src.models.baselines.cv_splitter import TimeSeriesCVConfig, build_fold_ledger, generate_time_splits


def test_generate_time_splits_produces_splits() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2009-01-01", periods=12, freq="YS"),
            "outbreak_label": [0, 1] * 6,
            "x": range(12),
        }
    )
    config = TimeSeriesCVConfig(
        first_valid_year=2014,
        last_valid_year=2018,
        train_window_years=4,
        skip_single_class_folds=False,
    )
    splits = list(generate_time_splits(df, config))
    assert len(splits) >= 1

    years = pd.to_datetime(df["date"]).dt.year
    for train_idx, valid_idx in splits:
        assert years.loc[train_idx].max() < years.loc[valid_idx].min()


def test_build_fold_ledger_emits_status_and_reason() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01", "2014-01-01"]),
            "outbreak_label": [0, 0, 0, 0, 1],
        }
    )
    config = TimeSeriesCVConfig(
        first_valid_year=2013,
        last_valid_year=2014,
        train_window_years=2,
        skip_single_class_folds=True,
    )

    ledger = build_fold_ledger(df, config)

    assert len(ledger) == 2
    assert {entry["status"] for entry in ledger} <= {"yielded", "skipped"}
    assert all("reason" in entry for entry in ledger)
    assert any(entry["status"] == "skipped" for entry in ledger)
