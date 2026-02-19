"""Utilities to identify and filter promising districts for focused visualizations."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _first_present_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def identify_promising_districts(
    frame: pd.DataFrame,
    *,
    district_col: str = "district",
    top_k: int = 20,
    risk_candidates: tuple[str, ...] = (
        "risk_score",
        "bayesian_risk",
        "risk",
        "probability",
        "outbreak_probability",
    ),
    fallback_case_candidates: tuple[str, ...] = ("cases", "case_count"),
) -> list[str]:
    """Select top-K districts by risk score with fallback to observed case volume."""
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")
    if district_col not in frame.columns:
        LOGGER.warning("District column '%s' missing; cannot identify promising districts.", district_col)
        return []

    district_series = frame[district_col].astype(str)
    metric_column = _first_present_column(frame, risk_candidates)
    metric_name = "risk"
    if metric_column is None:
        metric_column = _first_present_column(frame, fallback_case_candidates)
        metric_name = "cases"

    if metric_column is None:
        LOGGER.warning("No risk/probability/cases column found; using district frequency fallback.")
        district_counts = district_series.value_counts().head(top_k)
        return district_counts.index.astype(str).tolist()

    metric_values = pd.to_numeric(frame[metric_column], errors="coerce").fillna(0.0)
    summary = (
        pd.DataFrame({district_col: district_series, "metric": metric_values})
        .groupby(district_col, as_index=True)["metric"]
        .mean()
        .sort_values(ascending=False)
        .head(top_k)
    )
    LOGGER.info(
        "Selected %d promising districts using %s column '%s'.",
        len(summary),
        metric_name,
        metric_column,
    )
    return summary.index.astype(str).tolist()


def filter_to_promising_districts(
    frame: pd.DataFrame,
    *,
    district_col: str = "district",
    top_k: int = 20,
    risk_candidates: tuple[str, ...] = (
        "risk_score",
        "bayesian_risk",
        "risk",
        "probability",
        "outbreak_probability",
    ),
    fallback_case_candidates: tuple[str, ...] = ("cases", "case_count"),
) -> pd.DataFrame:
    """Return frame restricted to top-K promising districts."""
    districts = identify_promising_districts(
        frame,
        district_col=district_col,
        top_k=top_k,
        risk_candidates=risk_candidates,
        fallback_case_candidates=fallback_case_candidates,
    )
    if not districts:
        return frame.copy()
    return frame[frame[district_col].astype(str).isin(districts)].copy()
