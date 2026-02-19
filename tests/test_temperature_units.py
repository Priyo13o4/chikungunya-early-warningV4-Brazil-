from __future__ import annotations

import pandas as pd

from src.data_preprocessing.clean_data import clean_data
from src.data_preprocessing.impute_climate import impute_climate


def test_clean_data_normalizes_kelvin_temp_alias() -> None:
    raw = pd.DataFrame(
        {
            "district": ["A", "A"],
            "day": [1, 8],
            "mon": [1, 1],
            "year": [2015, 2015],
            "temp": [300.15, 301.15],
            "rainfall": [10.0, 20.0],
            "humidity": [80.0, 81.0],
        }
    )

    cleaned = clean_data(raw)

    assert "temp_kelvin" in cleaned.columns
    assert "temperature" in cleaned.columns
    assert cleaned["temp_kelvin"].round(2).tolist() == [300.15, 301.15]
    assert cleaned["temperature"].round(2).tolist() == [27.00, 28.00]


def test_impute_climate_does_not_clip_kelvin_with_celsius_bounds() -> None:
    df = pd.DataFrame(
        {
            "district": ["A", "A", "B"],
            "date": ["2015-01-01", "2015-01-08", "2015-01-01"],
            "temperature": [300.15, None, 305.15],
            "rainfall": [10.0, 12.0, 8.0],
            "humidity": [70.0, None, 75.0],
        }
    )

    imputed = impute_climate(df)

    assert "temp_kelvin" in imputed.columns
    assert float(imputed["temp_kelvin"].min()) > 250.0
    assert float(imputed["temp_kelvin"].max()) < 330.0
    assert float(imputed["temperature"].max()) < 60.0
    assert imputed["temperature"].nunique(dropna=True) > 1
