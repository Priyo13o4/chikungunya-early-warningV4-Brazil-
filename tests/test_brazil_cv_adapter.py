from __future__ import annotations

import pandas as pd

from projects.brazil_chik.cv_adapter import build_fold_ledger, generate_time_splits
from src.models.baselines.cv_splitter import TimeSeriesCVConfig


def _make_yearly_df(start_year: int = 2015, end_year: int = 2020) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    return pd.DataFrame(
        {
            "date": pd.to_datetime([f"{year}-01-01" for year in years]),
            "outbreak_label": [1] * len(years),
            "municipality_id": ["m1"] * len(years),
            "feature": list(range(len(years))),
        }
    )


def test_brazil_cv_adapter_thesis_strict_emits_exact_folds() -> None:
    df = _make_yearly_df(2015, 2020)
    config = TimeSeriesCVConfig(
        start_train_year=2009,
        first_valid_year=2014,
        last_valid_year=2025,
        train_window_years=5,
        thesis_strict=True,
        skip_single_class_folds=False,
    )

    ledger = build_fold_ledger(df, config)
    yielded = [row for row in ledger if row["status"] == "yielded"]

    expected = [
        (2015, 2015, 2016),
        (2015, 2016, 2017),
        (2015, 2017, 2018),
        (2015, 2018, 2019),
        (2015, 2019, 2020),
    ]
    observed = [(row["train_start_year"], row["train_end_year"], row["valid_year"]) for row in yielded]

    assert observed == expected
    assert all(row["fold_mode"] == "thesis_strict" for row in ledger)
    assert all(row["thesis_strict"] is True for row in ledger)
    assert all(row["reason"] == "ok" for row in yielded)
    assert all((row["gate_metrics"] or {}).get("outbreaks_valid", 0) >= 1 for row in yielded)
    assert all((row["gate_metrics"] or {}).get("municipalities_train", 0) >= 1 for row in yielded)
    assert config.fail_on_gate_violation is True


def test_brazil_cv_adapter_range_mode_respects_configured_bounds() -> None:
    df = _make_yearly_df(2016, 2021)
    config = TimeSeriesCVConfig(
        start_train_year=2016,
        first_valid_year=2019,
        last_valid_year=2021,
        thesis_strict=False,
        skip_single_class_folds=False,
    )

    ledger = build_fold_ledger(df, config)
    yielded = [row for row in ledger if row["status"] == "yielded"]

    observed = [(row["train_start_year"], row["train_end_year"], row["valid_year"]) for row in yielded]
    assert observed == [(2016, 2018, 2019), (2016, 2019, 2020), (2016, 2020, 2021)]
    assert all(row["fold_mode"] == "range" for row in ledger)
    assert all(row["thesis_strict"] is False for row in ledger)
    assert all((row["gate_metrics"] or {}).get("outbreaks_valid", 0) >= 1 for row in yielded)
    assert all((row["gate_metrics"] or {}).get("municipalities_train", 0) >= 1 for row in yielded)
    assert config.fail_on_gate_violation is True


def test_brazil_cv_adapter_splits_match_ledger_deterministically() -> None:
    df = _make_yearly_df(2015, 2020)
    config = TimeSeriesCVConfig(
        start_train_year=2015,
        first_valid_year=2016,
        last_valid_year=2020,
        thesis_strict=False,
        skip_single_class_folds=False,
    )

    ledger = build_fold_ledger(df, config)
    expected_pairs = [
        (tuple(row["train_index"]), tuple(row["valid_index"]))
        for row in ledger
        if row["status"] == "yielded"
    ]

    observed_pairs = [(tuple(train_idx.tolist()), tuple(valid_idx.tolist())) for train_idx, valid_idx in generate_time_splits(df, config)]

    assert observed_pairs == expected_pairs
    assert observed_pairs == [
        ((0,), (1,)),
        ((0, 1), (2,)),
        ((0, 1, 2), (3,)),
        ((0, 1, 2, 3), (4,)),
        ((0, 1, 2, 3, 4), (5,)),
    ]
    assert config.fail_on_gate_violation is True
