from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

_THRESHOLD_PERCENTILE_PATTERN = re.compile(r"^threshold_p(?P<percentile>\d+(?:\.\d+)?)$")


def parse_threshold_percentile(column_name: str) -> float | None:
    """Parse numeric percentile from threshold column names like threshold_p75."""
    match = _THRESHOLD_PERCENTILE_PATTERN.match(str(column_name))
    if match is None:
        return None
    try:
        return float(match.group("percentile"))
    except (TypeError, ValueError):
        return None


def _percentile_candidates(frame: pd.DataFrame) -> list[tuple[float, str]]:
    candidates: list[tuple[float, str]] = []
    for column in frame.columns:
        if not isinstance(column, str):
            continue
        percentile = parse_threshold_percentile(column)
        if percentile is None:
            continue
        candidates.append((float(percentile), column))
    # Numeric percentile sort first, then name for deterministic tie-breaking.
    return sorted(candidates, key=lambda item: (item[0], item[1]))


def resolve_threshold_series(
    *,
    frame: pd.DataFrame,
    default_threshold: float,
    outbreak_threshold: pd.Series | None = None,
    preferred_threshold_column: str | None = None,
    preferred_percentile: float | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Resolve threshold cases from authoritative sources with deterministic fallbacks."""
    default_value = float(default_threshold)

    if outbreak_threshold is not None:
        threshold_series = pd.to_numeric(pd.Series(outbreak_threshold, index=frame.index), errors="coerce").fillna(default_value)
        return threshold_series, {
            "threshold_basis": "provided_series",
            "threshold_resolution_reason": "explicit_outbreak_threshold_argument",
            "threshold_default": default_value,
            "threshold_column": None,
        }

    if "outbreak_threshold" in frame.columns:
        threshold_series = pd.to_numeric(frame["outbreak_threshold"], errors="coerce").fillna(default_value)
        return threshold_series, {
            "threshold_basis": "column:outbreak_threshold",
            "threshold_resolution_reason": "explicit_outbreak_threshold_column",
            "threshold_default": default_value,
            "threshold_column": "outbreak_threshold",
        }

    candidates = _percentile_candidates(frame)
    candidate_columns = [column for _, column in candidates]

    if preferred_threshold_column and preferred_threshold_column in candidate_columns:
        threshold_series = pd.to_numeric(frame[preferred_threshold_column], errors="coerce").fillna(default_value)
        return threshold_series, {
            "threshold_basis": f"column:{preferred_threshold_column}",
            "threshold_resolution_reason": "preferred_percentile_threshold_column",
            "threshold_default": default_value,
            "threshold_column": preferred_threshold_column,
            "threshold_columns_detected": candidate_columns,
        }

    if preferred_percentile is not None and len(candidates) > 0:
        preferred_matches = [
            (percentile, column)
            for percentile, column in candidates
            if np.isclose(percentile, float(preferred_percentile), atol=1e-9)
        ]
        if preferred_matches:
            selected_column = preferred_matches[-1][1]
            threshold_series = pd.to_numeric(frame[selected_column], errors="coerce").fillna(default_value)
            return threshold_series, {
                "threshold_basis": f"column:{selected_column}",
                "threshold_resolution_reason": "preferred_percentile_threshold_column",
                "threshold_default": default_value,
                "threshold_column": selected_column,
                "threshold_columns_detected": candidate_columns,
            }

    if len(candidates) == 1:
        selected_column = candidates[0][1]
        threshold_series = pd.to_numeric(frame[selected_column], errors="coerce").fillna(default_value)
        return threshold_series, {
            "threshold_basis": f"column:{selected_column}",
            "threshold_resolution_reason": "single_percentile_threshold_column",
            "threshold_default": default_value,
            "threshold_column": selected_column,
            "threshold_columns_detected": candidate_columns,
        }

    if len(candidates) > 1:
        selected_column = candidates[-1][1]
        threshold_series = pd.to_numeric(frame[selected_column], errors="coerce").fillna(default_value)
        return threshold_series, {
            "threshold_basis": f"column:{selected_column}",
            "threshold_resolution_reason": "highest_percentile_threshold_column",
            "threshold_default": default_value,
            "threshold_column": selected_column,
            "threshold_columns_detected": candidate_columns,
        }

    return pd.Series(default_value, index=frame.index, dtype="float64"), {
        "threshold_basis": "default",
        "threshold_resolution_reason": "no_threshold_columns_found",
        "threshold_default": default_value,
        "threshold_column": None,
    }
