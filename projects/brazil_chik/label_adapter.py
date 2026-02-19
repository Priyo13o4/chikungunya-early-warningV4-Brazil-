from __future__ import annotations

from typing import Iterable

import pandas as pd

DEFAULT_PERCENTILES: tuple[int, ...] = (70, 75, 80)


def _resolve_case_column(df: pd.DataFrame, case_column: str) -> str:
    if case_column in df.columns:
        return case_column
    for candidate in ("cases", "casos", "case_count", "weekly_cases"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No case column found for outbreak labeling")


def _resolve_year_series(df: pd.DataFrame, date_column: str) -> pd.Series:
    if date_column in df.columns:
        parsed = pd.to_datetime(df[date_column], errors="coerce")
        return parsed.dt.year
    if "year" in df.columns:
        return pd.to_numeric(df["year"], errors="coerce")
    return pd.Series(pd.NA, index=df.index)


def _past_only_thresholds(
    df: pd.DataFrame,
    *,
    district_column: str,
    case_column: str,
    date_column: str,
    quantile: float,
) -> pd.Series:
    ordered = df.copy()
    ordered["_label_original_order"] = pd.RangeIndex(len(ordered))
    if date_column in ordered.columns:
        ordered["_label_date"] = pd.to_datetime(ordered[date_column], errors="coerce")
    else:
        ordered["_label_date"] = pd.NaT

    ordered = ordered.sort_values(
        by=[district_column, "_label_date", "_label_original_order"],
        ascending=[True, True, True],
        na_position="last",
    )
    expanded = (
        ordered.groupby(district_column, dropna=False)[case_column]
        .expanding(min_periods=1)
        .quantile(quantile)
        .reset_index(level=0, drop=True)
    )
    ordered["_label_threshold"] = expanded.astype(float)
    restored = ordered.sort_values("_label_original_order")["_label_threshold"]
    restored.index = df.index
    return restored


def _assign_fold_train_thresholds(
    df: pd.DataFrame,
    *,
    district_column: str,
    case_column: str,
    date_column: str,
    quantile: float,
    first_valid_year: int,
    last_valid_year: int,
    start_train_year: int | None,
    train_window_years: int | None,
) -> pd.Series:
    years = _resolve_year_series(df, date_column)
    thresholds = pd.Series(float("nan"), index=df.index, dtype="float64")

    for valid_year in range(int(first_valid_year), int(last_valid_year) + 1):
        train_end_year = valid_year - 1
        if start_train_year is not None:
            train_start_year = int(start_train_year)
        elif train_window_years is not None and int(train_window_years) > 0:
            train_start_year = int(valid_year - int(train_window_years))
        else:
            train_start_year = int(years.min(skipna=True)) if years.notna().any() else valid_year

        train_mask = years.between(train_start_year, train_end_year, inclusive="both")
        valid_mask = years == valid_year
        if not bool(train_mask.fillna(False).any()) or not bool(valid_mask.fillna(False).any()):
            continue

        train_df = df.loc[train_mask.fillna(False), [district_column, case_column]].copy()
        train_thresholds = train_df.groupby(district_column, dropna=False)[case_column].quantile(quantile)

        valid_index = df.index[valid_mask.fillna(False)]
        valid_districts = df.loc[valid_index, district_column]
        assigned = valid_districts.map(train_thresholds).astype("float64")

        if train_df[district_column].isna().any():
            nan_threshold = float(train_df.loc[train_df[district_column].isna(), case_column].quantile(quantile))
            assigned = assigned.where(valid_districts.notna(), nan_threshold)

        thresholds.loc[valid_index] = assigned

    return thresholds


def label_outbreaks(
    df: pd.DataFrame,
    case_column: str = "cases",
    threshold: float = 10.0,
    *,
    district_column: str = "district",
    percentiles: Iterable[int] = DEFAULT_PERCENTILES,
    selected_percentile: int = 75,
    use_percentile_labels: bool = True,
    date_column: str = "date",
    first_valid_year: int | None = None,
    last_valid_year: int | None = None,
    start_train_year: int | None = None,
    train_window_years: int | None = None,
    threshold_scope: str = "train_fold",
) -> pd.DataFrame:
    threshold
    if district_column not in df.columns:
        raise ValueError(f"Missing district column '{district_column}'")

    if threshold_scope not in {"train_fold", "past_only", "global"}:
        raise ValueError("threshold_scope must be one of {'train_fold', 'past_only', 'global'}")

    output = df.copy()
    resolved_case_column = _resolve_case_column(output, case_column)
    output[resolved_case_column] = pd.to_numeric(output[resolved_case_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    if resolved_case_column != "cases":
        output["cases"] = output[resolved_case_column]

    if not use_percentile_labels:
        future_cases = output.groupby(district_column, dropna=False)["cases"].shift(-1)
        output["outbreak_label"] = (future_cases > float(threshold)).fillna(False).astype(int)
        return output

    future_cases = output.groupby(district_column, dropna=False)["cases"].shift(-1)
    for percentile in percentiles:
        q = float(percentile) / 100.0
        threshold_column = f"threshold_p{int(percentile)}"
        label_column = f"outbreak_label_p{int(percentile)}"

        threshold_series: pd.Series
        if threshold_scope == "global":
            threshold_series = output.groupby(district_column, dropna=False)["cases"].transform(
                lambda series: series.quantile(q)
            )
        elif (
            threshold_scope == "train_fold"
            and first_valid_year is not None
            and last_valid_year is not None
        ):
            threshold_series = _assign_fold_train_thresholds(
                output,
                district_column=district_column,
                case_column="cases",
                date_column=date_column,
                quantile=q,
                first_valid_year=int(first_valid_year),
                last_valid_year=int(last_valid_year),
                start_train_year=start_train_year,
                train_window_years=train_window_years,
            )
            missing_mask = threshold_series.isna()
            if missing_mask.any():
                threshold_series = threshold_series.copy()
                threshold_series.loc[missing_mask] = _past_only_thresholds(
                    output.loc[missing_mask],
                    district_column=district_column,
                    case_column="cases",
                    date_column=date_column,
                    quantile=q,
                )
        else:
            threshold_series = _past_only_thresholds(
                output,
                district_column=district_column,
                case_column="cases",
                date_column=date_column,
                quantile=q,
            )

        output[threshold_column] = pd.to_numeric(threshold_series, errors="coerce")
        output[label_column] = (
            (future_cases > pd.to_numeric(output[threshold_column], errors="coerce"))
            & future_cases.notna()
            & output[threshold_column].notna()
        ).astype(int)

    selected_column = f"outbreak_label_p{int(selected_percentile)}"
    if selected_column not in output.columns:
        raise ValueError(f"Selected percentile label not available: {selected_column}")
    output["outbreak_label"] = output[selected_column].astype(int)

    return output


def run(
    df: pd.DataFrame,
    *,
    case_column: str = "cases",
    selected_percentile: int = 75,
) -> pd.DataFrame:
    return label_outbreaks(
        df,
        case_column=case_column,
        selected_percentile=selected_percentile,
        use_percentile_labels=True,
    )
