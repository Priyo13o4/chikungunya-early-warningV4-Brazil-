"""Outbreak labeling logic for supervised learning targets."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_PERCENTILES: tuple[int, ...] = (70, 75, 80)


def _quantile_to_fraction(percentile: int) -> float:
    """Convert integer percentile to quantile fraction."""
    if not 0 <= percentile <= 100:
        raise ValueError(f"Percentile must be between 0 and 100, got {percentile}")
    return percentile / 100.0


def compute_district_thresholds(
    df: pd.DataFrame,
    *,
    case_column: str = "cases",
    district_column: str = "district",
    percentiles: Iterable[int] = DEFAULT_PERCENTILES,
) -> pd.DataFrame:
    """Compute district-specific case thresholds at requested percentiles."""
    if case_column not in df.columns:
        raise ValueError(f"Missing case column: {case_column}")
    if district_column not in df.columns:
        raise ValueError(f"Missing district column: {district_column}")

    thresholds = pd.DataFrame(index=df.index)
    case_values = pd.to_numeric(df[case_column], errors="coerce")
    for percentile in percentiles:
        fraction = _quantile_to_fraction(percentile)
        threshold_column = f"threshold_p{percentile}"
        thresholds[threshold_column] = case_values.groupby(df[district_column], dropna=False).transform(
            lambda series: series.quantile(fraction)
        )
    return thresholds


def add_outbreak_labels(
    df: pd.DataFrame,
    *,
    case_column: str = "cases",
    district_column: str = "district",
    percentiles: Iterable[int] = DEFAULT_PERCENTILES,
) -> pd.DataFrame:
    """Add binary outbreak labels for district-specific percentile thresholds."""
    output = df.copy()
    thresholds = compute_district_thresholds(
        output,
        case_column=case_column,
        district_column=district_column,
        percentiles=percentiles,
    )
    output = pd.concat([output, thresholds], axis=1)
    cases = pd.to_numeric(output[case_column], errors="coerce")

    for percentile in percentiles:
        threshold_column = f"threshold_p{percentile}"
        label_column = f"outbreak_label_p{percentile}"
        output[label_column] = (
            (cases >= output[threshold_column])
            & cases.notna()
            & output[threshold_column].notna()
        ).astype(int)

    return output


def choose_label_column(
    df: pd.DataFrame,
    *,
    preferred_percentile: int = 75,
    alias: str = "outbreak_label",
) -> pd.DataFrame:
    """Select one generated label column as canonical model target."""
    selected_column = f"outbreak_label_p{preferred_percentile}"
    if selected_column not in df.columns:
        available = sorted(column for column in df.columns if column.startswith("outbreak_label_p"))
        raise ValueError(
            f"Requested label column '{selected_column}' not found. Available: {available}"
        )
    output = df.copy()
    output[alias] = output[selected_column].astype("Int64")
    return output


def label_outbreaks(
    df: pd.DataFrame,
    case_column: str = "cases",
    threshold: float = 10.0,
    *,
    district_column: str = "district",
    percentiles: Iterable[int] = DEFAULT_PERCENTILES,
    selected_percentile: int = 75,
    use_percentile_labels: bool = True,
) -> pd.DataFrame:
    """Create outbreak labels.

    Default behavior computes district-specific percentile labels (70/75/80) and
    exposes ``outbreak_label`` from the selected percentile. A compatibility
    fallback can be used to create a fixed-threshold label.
    """
    output = df.copy()
    if case_column not in output.columns:
        LOGGER.warning("Case column '%s' missing; filling labels with 0", case_column)
        output["outbreak_label"] = 0
        return output

    if use_percentile_labels and district_column in output.columns:
        LOGGER.info("Labeling outbreaks with district percentile thresholds")
        output = add_outbreak_labels(
            output,
            case_column=case_column,
            district_column=district_column,
            percentiles=percentiles,
        )
        output = choose_label_column(
            output,
            preferred_percentile=selected_percentile,
            alias="outbreak_label",
        )
        return output

    LOGGER.info("Labeling outbreaks using fixed threshold=%s", threshold)
    output["outbreak_label"] = (pd.to_numeric(output[case_column], errors="coerce") >= threshold).astype(int)
    return output


def run(
    df: pd.DataFrame,
    *,
    case_column: str = "cases",
    selected_percentile: int = 75,
) -> pd.DataFrame:
    """Entrypoint for outbreak labeling phase."""
    return label_outbreaks(
        df,
        case_column=case_column,
        selected_percentile=selected_percentile,
        use_percentile_labels=True,
    )
