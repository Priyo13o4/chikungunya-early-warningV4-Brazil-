"""Feature validation checks."""

from __future__ import annotations

import logging
from typing import Any, Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_MECHANISTIC_FEATURES: tuple[str, ...] = (
    "temp_anomaly",
    "degree_days_20",
    "rainfall_4wk",
    "lai_anomaly",
    "temp_optimal",
    "temp_celsius",
    "temp_rain_interaction",
)


def validate_feature_matrix(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] | None = None,
    missingness_threshold: float = 0.40,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate expected feature matrix quality and schema.

    Checks include:
    - required feature availability
    - per-column missingness against threshold
    - presence of infinities in numeric columns

    Returns
    -------
    dict[str, Any]
        Structured report with pass/fail and diagnostics.
    """
    LOGGER.info("Validating feature matrix")
    if not 0.0 <= missingness_threshold <= 1.0:
        raise ValueError("missingness_threshold must be in [0, 1]")

    required = list(required_columns) if required_columns is not None else []
    missing_required = [column for column in required if column not in df.columns]

    numeric = df.select_dtypes(include=["number"])
    inf_counts = numeric.isin([float("inf"), float("-inf")]).sum()
    columns_with_infinities = [column for column, count in inf_counts.items() if int(count) > 0]
    inf_count_total = int(inf_counts.sum()) if not inf_counts.empty else 0

    missingness = df.isna().mean().sort_values(ascending=False)
    high_missingness = {
        column: float(rate)
        for column, rate in missingness.items()
        if float(rate) > missingness_threshold
    }

    passed = not missing_required and not columns_with_infinities and not high_missingness
    report: dict[str, Any] = {
        "passed": bool(passed),
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "missing_required_features": missing_required,
        "columns_with_infinities": columns_with_infinities,
        "infinite_value_count": inf_count_total,
        "missingness_threshold": float(missingness_threshold),
        "high_missingness_columns": high_missingness,
        "required_features_checked": required,
    }

    if missing_required:
        LOGGER.warning("Missing required features: %s", missing_required)
    if columns_with_infinities:
        LOGGER.warning("Columns with infinities: %s", columns_with_infinities)
    if high_missingness:
        LOGGER.warning("Columns above missingness threshold %.2f: %s", missingness_threshold, high_missingness)

    if strict and not passed:
        raise ValueError(f"Feature validation failed: {report}")
    return report


def gate_mechanistic_features(
    df: pd.DataFrame,
    *,
    mechanistic_features: Sequence[str] = DEFAULT_MECHANISTIC_FEATURES,
    variance_threshold: float = 1e-10,
    uniqueness_threshold: float = 0.01,
    min_non_missing_samples: int = 24,
    strict: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Gate low-quality mechanistic features and optionally drop them.

    Default behavior is non-fatal: flagged features are removed from the returned frame.
    When ``strict=True``, any flagged feature raises ``ValueError``.
    """
    if variance_threshold < 0.0:
        raise ValueError("variance_threshold must be >= 0")
    if not 0.0 <= uniqueness_threshold <= 1.0:
        raise ValueError("uniqueness_threshold must be in [0, 1]")
    if min_non_missing_samples < 1:
        raise ValueError("min_non_missing_samples must be >= 1")

    present = [feature for feature in mechanistic_features if feature in df.columns]
    details: dict[str, Any] = {}
    flagged: list[str] = []

    for feature in present:
        numeric = pd.to_numeric(df[feature], errors="coerce")
        non_missing = numeric.dropna()
        variance = float(non_missing.var(ddof=0)) if len(non_missing) > 1 else 0.0
        unique_count = int(non_missing.nunique(dropna=True))
        uniqueness_rate = float(unique_count / max(len(non_missing), 1))

        reasons: list[str] = []
        if len(non_missing) >= min_non_missing_samples:
            if variance <= variance_threshold:
                reasons.append("near_zero_variance")
            if uniqueness_rate <= uniqueness_threshold:
                reasons.append("near_zero_uniqueness")

        feature_report = {
            "variance": variance,
            "uniqueness_rate": uniqueness_rate,
            "non_missing_count": int(len(non_missing)),
            "min_non_missing_samples": int(min_non_missing_samples),
            "flagged": bool(reasons),
            "reasons": reasons,
        }
        details[feature] = feature_report
        if reasons:
            flagged.append(feature)

    report: dict[str, Any] = {
        "passed": len(flagged) == 0,
        "strict_mode": bool(strict),
        "variance_threshold": float(variance_threshold),
        "uniqueness_threshold": float(uniqueness_threshold),
        "min_non_missing_samples": int(min_non_missing_samples),
        "mechanistic_features_checked": present,
        "flagged_features": flagged,
        "dropped_features": [],
        "feature_details": details,
    }

    if flagged and strict:
        raise ValueError(f"Mechanistic feature quality gate failed: {report}")

    if flagged:
        LOGGER.warning("Quality gate flagged mechanistic features: %s", flagged)
        filtered = df.drop(columns=flagged, errors="ignore").copy()
        report["dropped_features"] = list(flagged)
        return filtered, report

    return df.copy(), report


def run(
    df: pd.DataFrame,
    *,
    required_columns: Sequence[str] | None = None,
    missingness_threshold: float = 0.40,
    strict: bool = False,
) -> dict[str, Any]:
    """Entrypoint for feature validation."""
    return validate_feature_matrix(
        df,
        required_columns=required_columns,
        missingness_threshold=missingness_threshold,
        strict=strict,
    )
