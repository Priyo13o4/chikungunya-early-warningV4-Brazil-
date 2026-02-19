"""Tests for outbreak labeling utilities."""

from __future__ import annotations

import pandas as pd

from src.data_preprocessing.label_outbreaks import label_outbreaks


def test_label_outbreaks_percentile_labels_by_district() -> None:
    df = pd.DataFrame(
        {
            "district": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "cases": [1, 2, 3, 4, 5, 10, 11, 12, 13, 14],
        }
    )

    labeled = label_outbreaks(df, selected_percentile=75, use_percentile_labels=True)

    assert "threshold_p70" in labeled.columns
    assert "threshold_p75" in labeled.columns
    assert "threshold_p80" in labeled.columns
    assert "outbreak_label_p75" in labeled.columns
    assert labeled["outbreak_label"].tolist() == labeled["outbreak_label_p75"].tolist()

    a_rows = labeled[labeled["district"] == "A"].reset_index(drop=True)
    b_rows = labeled[labeled["district"] == "B"].reset_index(drop=True)
    assert a_rows["threshold_p75"].iloc[0] == 4.0
    assert b_rows["threshold_p75"].iloc[0] == 13.0
    assert a_rows["outbreak_label_p75"].tolist() == [0, 0, 0, 1, 1]
    assert b_rows["outbreak_label_p75"].tolist() == [0, 0, 0, 1, 1]
