from __future__ import annotations

import logging
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

_LEAKAGE_COLUMN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^outbreak_label($|_)", flags=re.IGNORECASE),
    re.compile(r"^target($|_)", flags=re.IGNORECASE),
    re.compile(r"(^|_)future($|_)", flags=re.IGNORECASE),
    re.compile(r"(^|_)next($|_)", flags=re.IGNORECASE),
    re.compile(r"(^|_)lead(_|$)", flags=re.IGNORECASE),
    re.compile(r"t\+\d+", flags=re.IGNORECASE),
)

_DISTRIBUTION_FIT_TOKENS: tuple[str, ...] = (
    "zscore",
    "standardized",
    "standardised",
    "minmax",
    "quantile",
    "boxcox",
    "yeojohnson",
)


DEFAULT_REQUIRED_FEATURES: tuple[str, ...] = (
    "case_lag_1",
    "case_lag_2",
    "case_lag_3",
    "case_lag_4",
    "case_roll4_mean",
    "case_roll4_var",
    "case_roll4_cv",
    "case_roll4_max",
    "case_trend_1w",
    "case_trend_4w",
    "case_cum12",
)


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    lower_to_source = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        source = lower_to_source.get(candidate.lower())
        if source is not None:
            return source
    return None


def _find_leakage_columns(columns: Sequence[str]) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        if any(pattern.search(column) for pattern in _LEAKAGE_COLUMN_PATTERNS):
            flagged.append(column)
    return sorted(set(flagged))


def _find_distribution_fit_columns(columns: Sequence[str]) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if "case" not in lowered and "rt" not in lowered:
            continue
        if any(token in lowered for token in _DISTRIBUTION_FIT_TOKENS):
            flagged.append(column)
    return sorted(set(flagged))


def _detect_input_date_issues(
    df: pd.DataFrame,
    *,
    date_column: str,
    district_column: str,
) -> list[str]:
    issues: list[str] = []
    if date_column not in df.columns:
        return [f"missing date column '{date_column}'"]

    working = df.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    if district_column not in working.columns:
        working[district_column] = "UNKNOWN"

    nat_count = int(working[date_column].isna().sum())
    if nat_count > 0:
        issues.append(f"found {nat_count} rows with invalid/missing dates")

    duplicate_mask = working.duplicated(subset=[district_column, date_column], keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count > 0:
        issues.append(f"found {duplicate_count} duplicate district/date rows")

    order_issues = 0
    for _, group in working.groupby(district_column, dropna=False, sort=False):
        dates = group[date_column]
        decreases = (dates.diff().dt.total_seconds() < 0).fillna(False)
        order_issues += int(decreases.sum())
    if order_issues > 0:
        issues.append(f"found {order_issues} out-of-order timestamps within district series")

    return issues


def _expected_causal_features(output: pd.DataFrame, *, district_column: str) -> dict[str, pd.Series]:
    grouped = output.groupby(district_column, dropna=False, sort=False)
    expected: dict[str, pd.Series] = {}
    raw_lags: dict[int, pd.Series] = {}

    for lag in (1, 2, 3, 4):
        raw_lag = grouped["cases"].shift(lag)
        raw_lags[lag] = raw_lag
        expected[f"case_lag_{lag}"] = raw_lag.fillna(0.0)

    prior = grouped["cases"].shift(1)
    expected["case_roll4_mean"] = (
        prior.groupby(output[district_column], dropna=False)
        .transform(lambda series: series.rolling(4, min_periods=1).mean())
        .fillna(0.0)
    )
    expected["case_roll4_var"] = (
        prior.groupby(output[district_column], dropna=False)
        .transform(lambda series: series.rolling(4, min_periods=1).var())
        .fillna(0.0)
    )
    expected["case_roll4_max"] = (
        prior.groupby(output[district_column], dropna=False)
        .transform(lambda series: series.rolling(4, min_periods=1).max())
        .fillna(0.0)
    )
    safe_mean = expected["case_roll4_mean"].replace(0.0, np.nan)
    expected["case_roll4_cv"] = (
        np.sqrt(expected["case_roll4_var"].clip(lower=0.0)) / safe_mean
    ).fillna(0.0)
    expected["case_trend_1w"] = (raw_lags[1] - raw_lags[2]).fillna(0.0)
    expected["case_trend_4w"] = (raw_lags[1] - raw_lags[4]).fillna(0.0)
    expected["case_cum12"] = (
        prior.groupby(output[district_column], dropna=False)
        .transform(lambda series: series.rolling(12, min_periods=1).sum())
        .fillna(0.0)
    )

    if "rt" in output.columns:
        expected["rt_lag_1"] = grouped["rt"].shift(1).fillna(0.0)
    if "p_rt1" in output.columns:
        expected["p_rt1_lag_1"] = grouped["p_rt1"].shift(1).fillna(0.0)
    if "receptivo" in output.columns:
        expected["receptivo_lag_1"] = grouped["receptivo"].shift(1).fillna(0.0)
    return expected


def _validate_features(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame,
    *,
    date_column: str,
    district_column: str,
    required_features: Sequence[str],
    strict: bool,
) -> dict[str, Any]:
    issues: list[str] = []
    input_date_issues = _detect_input_date_issues(
        input_df,
        date_column=date_column,
        district_column=district_column,
    )
    if input_date_issues:
        issues.extend(input_date_issues)

    leakage_columns = _find_leakage_columns([str(column) for column in input_df.columns])
    if leakage_columns:
        issues.append(f"found potential leakage columns in input: {leakage_columns}")

    distribution_fit_columns = _find_distribution_fit_columns([str(column) for column in output_df.columns])
    if distribution_fit_columns:
        issues.append(
            "found case/rt distribution-fit transform columns in feature matrix: "
            f"{distribution_fit_columns}"
        )

    missing_required = [feature for feature in required_features if feature not in output_df.columns]
    if missing_required:
        issues.append(f"missing required features: {missing_required}")

    if date_column in output_df.columns and district_column in output_df.columns:
        for district, group in output_df.groupby(district_column, dropna=False, sort=False):
            dates = pd.to_datetime(group[date_column], errors="coerce")
            if not dates.is_monotonic_increasing:
                issues.append(f"output date ordering is not monotonic for district={district}")
                break

    expected = _expected_causal_features(output_df, district_column=district_column)
    for feature_name, expected_series in expected.items():
        if feature_name not in output_df.columns:
            continue
        actual = pd.to_numeric(output_df[feature_name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        expected_values = pd.to_numeric(expected_series, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        if not np.allclose(actual, expected_values, rtol=1e-6, atol=1e-8, equal_nan=True):
            issues.append(
                "causality assertion failed for feature "
                f"'{feature_name}' (values differ from <=t-1 expected construction)"
            )

    report: dict[str, Any] = {
        "passed": not issues,
        "strict_mode": bool(strict),
        "issues": issues,
        "input_date_issues": input_date_issues,
        "leakage_columns": leakage_columns,
        "distribution_fit_columns": distribution_fit_columns,
        "missing_required_features": missing_required,
    }
    if issues:
        error_message = "Brazil feature adapter validation failed: " + " | ".join(issues)
        if strict:
            raise ValueError(error_message)
        LOGGER.warning(error_message)
    return report


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    district_column: str = "district",
    validate: bool = True,
    strict_validation: bool = False,
    enable_quality_gate: bool = False,
    quality_gate_variance_threshold: float = 1e-10,
    quality_gate_uniqueness_threshold: float = 0.01,
    quality_gate_min_non_missing_samples: int = 24,
    quality_gate_report_path: Path | None = None,
    required_features: Sequence[str] = DEFAULT_REQUIRED_FEATURES,
    write_output: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    _ = (
        enable_quality_gate,
        quality_gate_variance_threshold,
        quality_gate_uniqueness_threshold,
        quality_gate_min_non_missing_samples,
        quality_gate_report_path,
    )

    output = df.copy()
    if date_column in output.columns:
        output[date_column] = pd.to_datetime(output[date_column], errors="coerce")
    else:
        output[date_column] = pd.NaT

    if district_column not in output.columns:
        output[district_column] = "UNKNOWN"

    case_column = _resolve_column(output, ("cases", "casos", "case_count", "weekly_cases"))
    if case_column is None:
        output["cases"] = 0.0
        case_column = "cases"

    output[case_column] = pd.to_numeric(output[case_column], errors="coerce").fillna(0.0).clip(lower=0.0)

    if case_column != "cases":
        output["cases"] = output[case_column]

    output = output.sort_values([district_column, date_column]).reset_index(drop=True)
    grouped = output.groupby(district_column, dropna=False)

    for lag in (1, 2, 3, 4):
        output[f"case_lag_{lag}"] = grouped["cases"].shift(lag)

    prior = grouped["cases"].shift(1)
    output["case_roll4_mean"] = prior.groupby(output[district_column], dropna=False).transform(
        lambda series: series.rolling(4, min_periods=1).mean()
    )
    output["case_roll4_var"] = prior.groupby(output[district_column], dropna=False).transform(
        lambda series: series.rolling(4, min_periods=1).var()
    )
    output["case_roll4_max"] = prior.groupby(output[district_column], dropna=False).transform(
        lambda series: series.rolling(4, min_periods=1).max()
    )

    safe_mean = output["case_roll4_mean"].replace(0.0, np.nan)
    output["case_roll4_cv"] = np.sqrt(output["case_roll4_var"].clip(lower=0.0)) / safe_mean

    output["case_trend_1w"] = output["case_lag_1"] - output["case_lag_2"]
    output["case_trend_4w"] = output["case_lag_1"] - output["case_lag_4"]
    output["case_cum12"] = prior.groupby(output[district_column], dropna=False).transform(
        lambda series: series.rolling(12, min_periods=1).sum()
    )

    rt_column = _resolve_column(output, ("rt",))
    if rt_column is not None:
        output["rt_lag_1"] = grouped[rt_column].shift(1)

    p_rt1_column = _resolve_column(output, ("p_rt1", "p.rt1"))
    if p_rt1_column is not None:
        output["p_rt1_lag_1"] = grouped[p_rt1_column].shift(1)

    receptivo_column = _resolve_column(output, ("receptivo",))
    if receptivo_column is not None:
        output["receptivo_lag_1"] = grouped[receptivo_column].shift(1)

    output["year"] = output[date_column].dt.year.astype("Int64")
    output["month"] = output[date_column].dt.month.astype("Int64")
    output["weekofyear"] = output[date_column].dt.isocalendar().week.astype("Int64")

    # Neutral-value imputation for early history:
    # rows without sufficient lagged history are deterministically filled with 0.0
    # so the adapter remains causal and model-ready from the first timestep.
    neutral_fill_columns = [
        "case_lag_1",
        "case_lag_2",
        "case_lag_3",
        "case_lag_4",
        "case_roll4_mean",
        "case_roll4_var",
        "case_roll4_cv",
        "case_roll4_max",
        "case_trend_1w",
        "case_trend_4w",
        "case_cum12",
        "rt_lag_1",
        "p_rt1_lag_1",
        "receptivo_lag_1",
    ]
    for column in neutral_fill_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce").fillna(0.0)

    if validate:
        _validate_features(
            df,
            output,
            date_column=date_column,
            district_column=district_column,
            required_features=required_features,
            strict=strict_validation,
        )

    if write_output:
        final_output = output_path or Path("data/features/feature_matrix.csv")
        final_output.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(final_output, index=False)

    return output


def run(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    district_column: str = "district",
    validate: bool = True,
    strict_validation: bool = False,
    enable_quality_gate: bool = False,
    quality_gate_variance_threshold: float = 1e-10,
    quality_gate_uniqueness_threshold: float = 0.01,
    quality_gate_min_non_missing_samples: int = 24,
    quality_gate_report_path: Path | None = None,
    required_features: Sequence[str] = DEFAULT_REQUIRED_FEATURES,
    write_output: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    return build_feature_matrix(
        df,
        date_column=date_column,
        district_column=district_column,
        validate=validate,
        strict_validation=strict_validation,
        enable_quality_gate=enable_quality_gate,
        quality_gate_variance_threshold=quality_gate_variance_threshold,
        quality_gate_uniqueness_threshold=quality_gate_uniqueness_threshold,
        quality_gate_min_non_missing_samples=quality_gate_min_non_missing_samples,
        quality_gate_report_path=quality_gate_report_path,
        required_features=required_features,
        write_output=write_output,
        output_path=output_path,
    )
