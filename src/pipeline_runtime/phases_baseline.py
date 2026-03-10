from __future__ import annotations

from dataclasses import asdict
import inspect
import logging
from pathlib import Path
import re
import shutil
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.baselines.cv_splitter import TimeSeriesCVConfig
from src.pipeline_runtime.io_artifacts import build_suppressed_metric_payload
from src.pipeline_runtime.phase_context import BaselinePhaseResult, SharedPhaseState

LOGGER = logging.getLogger(__name__)

_FORBIDDEN_COLUMN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^outbreak_label$", flags=re.IGNORECASE),
    re.compile(r"^outbreak_label_.*$", flags=re.IGNORECASE),
    re.compile(r"^outbreak_label_p.*$", flags=re.IGNORECASE),
    re.compile(r"^threshold_.*$", flags=re.IGNORECASE),
    re.compile(r"^threshold_p.*$", flags=re.IGNORECASE),
)
_RAW_CASE_TARGET_ALIASES: set[str] = {"cases", "case_count", "weekly_cases", "outbreak_target", "target"}
_DISTRIBUTION_FIT_TOKENS: tuple[str, ...] = (
    "zscore",
    "standardized",
    "standardised",
    "minmax",
    "quantile",
    "boxcox",
    "yeojohnson",
)
_CLIMATE_COLUMN_TOKENS: tuple[str, ...] = ("rain", "temp", "humid", "precip", "climate", "lai", "month", "week", "year")
_BASELINE_CLIMATE_REQUIRED_CONTEXT_COLUMNS: tuple[str, ...] = ("date", "year", "month", "weekofyear", "district")
_REQUESTED_COVARIATE_ALIAS_TOKENS: dict[str, tuple[str, ...]] = {
    "rainfall": ("rain", "precip"),
    "precipitation": ("rain", "precip"),
    "temperature": ("temp",),
    "humidity": ("humid", "umid"),
    "month_sin": ("month",),
    "month_cos": ("month",),
}


def build_model_input_df(
    feature_df: pd.DataFrame,
    *,
    target_column: str = "outbreak_label",
    climate_only: bool = False,
    requested_covariates: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dropped: list[str] = []
    for column in feature_df.columns:
        column_lower = column.lower()
        pattern_forbidden = any(pattern.match(column) for pattern in _FORBIDDEN_COLUMN_PATTERNS)
        raw_case_alias = column_lower in _RAW_CASE_TARGET_ALIASES
        direct_target_alias = column_lower == target_column.lower()
        if pattern_forbidden or raw_case_alias or direct_target_alias:
            dropped.append(column)

    output = feature_df.drop(columns=sorted(set(dropped)), errors="ignore").copy()

    requested_covariates = [str(value).strip() for value in (requested_covariates or []) if str(value).strip()]
    dropped_non_climate_columns: list[str] = []
    selected_climate_columns: list[str] = []
    if climate_only:
        requested_lookup = {name.lower() for name in requested_covariates}
        selected: list[str] = []
        for column in output.columns:
            lowered = str(column).lower()
            if requested_lookup:
                if lowered in requested_lookup:
                    selected.append(str(column))
                    continue
                requested_alias_hit = False
                for requested_name in requested_lookup:
                    alias_tokens = _REQUESTED_COVARIATE_ALIAS_TOKENS.get(requested_name, ())
                    if alias_tokens and any(token in lowered for token in alias_tokens):
                        requested_alias_hit = True
                        break
                if requested_alias_hit:
                    selected.append(str(column))
                continue
            if any(token in lowered for token in _CLIMATE_COLUMN_TOKENS):
                selected.append(str(column))
        if requested_lookup and not selected:
            for column in output.columns:
                lowered = str(column).lower()
                if any(token in lowered for token in _CLIMATE_COLUMN_TOKENS):
                    selected.append(str(column))
        for context_column in _BASELINE_CLIMATE_REQUIRED_CONTEXT_COLUMNS:
            if context_column in output.columns:
                selected.append(context_column)
        selected_climate_columns = sorted(set(selected))
        dropped_non_climate_columns = sorted([str(column) for column in output.columns if str(column) not in set(selected_climate_columns)])
        output = output.loc[:, selected_climate_columns].copy()

    audit = {
        "input_feature_count": int(feature_df.shape[1]),
        "output_feature_count": int(output.shape[1]),
        "dropped_forbidden_columns": sorted(set(dropped)),
        "climate_only_filter_enabled": bool(climate_only),
        "climate_only_requested_covariates": requested_covariates,
        "climate_only_selected_columns": selected_climate_columns,
        "dropped_non_climate_columns": dropped_non_climate_columns,
        "forbidden_patterns": [pattern.pattern for pattern in _FORBIDDEN_COLUMN_PATTERNS],
        "raw_case_aliases": sorted(_RAW_CASE_TARGET_ALIASES),
    }
    return output, audit


def find_distribution_fit_columns(columns: list[str]) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if "case" not in lowered and "rt" not in lowered:
            continue
        if any(token in lowered for token in _DISTRIBUTION_FIT_TOKENS):
            flagged.append(str(column))
    return sorted(set(flagged))


def audit_train_fold_threshold_scope(
    df: pd.DataFrame,
    *,
    selected_percentile: int,
    cv_config: TimeSeriesCVConfig,
    case_column: str = "cases",
    district_column: str = "district",
    date_column: str = "date",
) -> dict[str, Any]:
    threshold_column = f"threshold_p{int(selected_percentile)}"
    required_columns = {case_column, district_column, date_column, threshold_column}
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        return {
            "checked": False,
            "violation_count": 0,
            "sample_violations": [],
            "missing_columns": missing,
            "reason": "missing_required_columns",
        }

    working = df[[case_column, district_column, date_column, threshold_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working["_year"] = working[date_column].dt.year
    working[case_column] = pd.to_numeric(working[case_column], errors="coerce")
    working[threshold_column] = pd.to_numeric(working[threshold_column], errors="coerce")

    quantile = float(selected_percentile) / 100.0
    total_violations = 0
    sample_violations: list[dict[str, Any]] = []

    for valid_year in range(int(cv_config.first_valid_year), int(cv_config.last_valid_year) + 1):
        train_end_year = valid_year - 1
        train_start_year = int(getattr(cv_config, "start_train_year", 0) or 0)
        if train_start_year <= 0:
            train_start_year = int(valid_year - max(1, int(cv_config.train_window_years)))

        train_mask = working["_year"].between(train_start_year, train_end_year, inclusive="both")
        valid_mask = working["_year"] == valid_year
        if not bool(train_mask.fillna(False).any()) or not bool(valid_mask.fillna(False).any()):
            continue

        train_df = working.loc[train_mask.fillna(False), [district_column, case_column]].dropna(subset=[case_column])
        if train_df.empty:
            continue
        train_quantiles = train_df.groupby(district_column, dropna=False)[case_column].quantile(quantile)

        valid_df = working.loc[valid_mask.fillna(False), [district_column, threshold_column]].copy()
        valid_df["_expected"] = valid_df[district_column].map(train_quantiles)
        comparable = valid_df[threshold_column].notna() & valid_df["_expected"].notna()
        if not comparable.any():
            continue

        delta = (valid_df.loc[comparable, threshold_column] - valid_df.loc[comparable, "_expected"]).abs()
        violation_mask = delta > 1e-9
        year_violations = int(violation_mask.sum())
        if year_violations <= 0:
            continue

        total_violations += year_violations
        if len(sample_violations) < 5:
            violating_rows = valid_df.loc[comparable].loc[violation_mask].head(5 - len(sample_violations))
            for _, row in violating_rows.iterrows():
                sample_violations.append(
                    {
                        "valid_year": int(valid_year),
                        "district": None if pd.isna(row[district_column]) else str(row[district_column]),
                        "observed_threshold": float(row[threshold_column]),
                        "expected_train_quantile": float(row["_expected"]),
                    }
                )

    return {
        "checked": True,
        "violation_count": int(total_violations),
        "sample_violations": sample_violations,
        "missing_columns": [],
        "reason": "ok",
    }


def collect_baseline_oof_fold_ids(output_root: Path) -> list[int]:
    fold_ids: list[int] = []
    for fold_dir in sorted(output_root.glob("fold_*")):
        predictions_path = fold_dir / "predictions.csv"
        if not predictions_path.exists():
            continue
        try:
            frame = pd.read_csv(predictions_path)
        except Exception:
            continue
        if frame.empty:
            continue
        match = re.search(r"fold_(\d+)$", fold_dir.name)
        if match:
            fold_ids.append(int(match.group(1)))
    return fold_ids


def clear_stale_baseline_artifacts(output_root: Path) -> dict[str, int]:
    removed_fold_dirs = 0
    removed_files = 0
    if not output_root.exists():
        return {"removed_fold_dirs": 0, "removed_files": 0}

    for fold_dir in sorted(output_root.glob("fold_*")):
        if fold_dir.is_dir():
            shutil.rmtree(fold_dir, ignore_errors=True)
            removed_fold_dirs += 1

    for stale_name in ("cv_metrics.csv", "cv_metrics_aggregate.csv", "fold_ledger.json"):
        stale_path = output_root / stale_name
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()
            removed_files += 1

    return {"removed_fold_dirs": int(removed_fold_dirs), "removed_files": int(removed_files)}


def collect_baseline_oof_scores(
    *,
    output_root: Path,
    expected_index: pd.Index,
) -> pd.Series:
    oof = pd.Series(np.nan, index=expected_index, dtype="float64")
    for fold_dir in sorted(output_root.glob("fold_*")):
        predictions_path = fold_dir / "predictions.csv"
        if not predictions_path.exists():
            continue
        fold_predictions = pd.read_csv(predictions_path, index_col=0)
        if fold_predictions.empty:
            continue
        fold_scores = fold_predictions.mean(axis=1).astype(float)
        fold_scores.index = pd.Index(fold_scores.index)
        valid_index = expected_index.intersection(fold_scores.index)
        if valid_index.empty:
            continue
        oof.loc[valid_index] = fold_scores.loc[valid_index].astype(float).to_numpy()
    return oof


def collect_baseline_oof_predictions(
    *,
    output_root: Path,
    expected_index: pd.Index,
) -> pd.DataFrame:
    collected_frames: list[pd.DataFrame] = []
    for fold_dir in sorted(output_root.glob("fold_*")):
        predictions_path = fold_dir / "predictions.csv"
        if not predictions_path.exists():
            continue
        try:
            fold_predictions = pd.read_csv(predictions_path, index_col=0)
        except Exception:
            continue
        if fold_predictions.empty:
            continue
        fold_predictions.index = pd.Index(fold_predictions.index)
        collected_frames.append(fold_predictions)

    if not collected_frames:
        return pd.DataFrame(index=expected_index)

    combined = pd.concat(collected_frames, axis=0)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.reindex(expected_index)
    for column in combined.columns:
        combined[column] = pd.to_numeric(combined[column], errors="coerce")
    return combined


def extract_feature_importances(
    models: dict[str, Any],
    feature_names: list[str],
) -> pd.Series | None:
    if not models or not feature_names:
        return None

    collected: list[pd.Series] = []
    for model_name, model in models.items():
        values: np.ndarray | None = None
        if hasattr(model, "feature_importances_"):
            values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        elif hasattr(model, "coef_"):
            coef_values = np.asarray(getattr(model, "coef_"), dtype=float)
            values = np.abs(coef_values).mean(axis=0) if coef_values.ndim > 1 else np.abs(coef_values)

        if values is None or values.size != len(feature_names):
            continue

        importance = pd.Series(values, index=feature_names, dtype=float)
        importance.name = str(model_name)
        collected.append(importance)

    if not collected:
        return None
    return pd.concat(collected, axis=1).mean(axis=1).sort_values(ascending=False)


def run_baseline_phase(
    *,
    state: SharedPhaseState,
    paths: Any,
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    selected_percentile: int,
    effective_cv_config: TimeSeriesCVConfig,
    effective_seed: int,
    strict_feature_gate: bool,
    baseline_compute_backend: str,
    skip_baselines: bool,
    export_detailed_csv: bool,
    model_names: list[str] | None,
    baseline_climate_only: bool = False,
    baseline_requested_covariates: list[str] | None = None,
    lead_time_max_lookback_steps: int,
    threshold_scope_audit: dict[str, Any],
    cv_ledger_callable: Callable[..., Any],
    cv_split_callable: Callable[..., Any],
    train_baselines_fn: Callable[..., dict[str, Any]],
    baseline_training_config_cls: Any,
    predict_baselines_fn: Callable[..., pd.DataFrame],
    evaluate_baseline_predictions_fn: Callable[..., dict[str, float]],
    collect_baseline_oof_scores_fn: Callable[..., pd.Series],
    collect_baseline_oof_predictions_fn: Callable[..., pd.DataFrame],
    collect_baseline_oof_fold_ids_fn: Callable[..., list[int]],
    safe_write_json_fn: Callable[[dict[str, Any], Path], None],
) -> BaselinePhaseResult:
    LOGGER.info("Phase: baselines")
    target = labeled_df.get("outbreak_label", pd.Series(0, index=labeled_df.index))
    target = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
    bayesian_count_target = pd.to_numeric(labeled_df.get("cases", pd.Series(0.0, index=labeled_df.index)), errors="coerce")
    bayesian_count_target = bayesian_count_target.fillna(0.0).clip(lower=0.0)
    threshold_column_name = f"threshold_p{selected_percentile}"
    bayesian_threshold_series = pd.to_numeric(
        labeled_df.get(threshold_column_name, pd.Series(np.nan, index=labeled_df.index)),
        errors="coerce",
    )
    temporal_index = labeled_df.get("date")
    district_index = labeled_df.get("district")

    model_input_df, leakage_audit = build_model_input_df(
        features_df,
        target_column="outbreak_label",
        climate_only=bool(baseline_climate_only),
        requested_covariates=baseline_requested_covariates or [],
    )
    if baseline_climate_only and model_input_df.shape[1] == 0:
        message = "Baseline climate-only mode selected no usable covariates; cannot train baselines."
        if strict_feature_gate:
            raise RuntimeError(message)
        LOGGER.warning(message)
        state.degraded_reasons.append({"code": "baseline_climate_only_empty_covariates"})

    def _align_to_model_input(series: pd.Series | None) -> pd.Series | None:
        if series is None:
            return None
        if len(series) == len(model_input_df):
            return pd.Series(series.to_numpy(), index=model_input_df.index)
        return series.reindex(model_input_df.index)

    target = _align_to_model_input(target)
    bayesian_count_target = _align_to_model_input(bayesian_count_target)
    bayesian_threshold_series = _align_to_model_input(bayesian_threshold_series)
    temporal_index = _align_to_model_input(temporal_index)
    district_index = _align_to_model_input(district_index)

    distribution_fit_columns = find_distribution_fit_columns(model_input_df.columns.tolist())
    if distribution_fit_columns:
        message = (
            "Found distribution-fit transform columns in model input; expected train-fold-only transforms "
            f"or causal features. flagged_columns={distribution_fit_columns}"
        )
        if strict_feature_gate:
            raise RuntimeError(message)
        LOGGER.warning(message)
        state.degraded_reasons.append(
            {
                "code": "distribution_fit_transform_columns_detected",
                "columns": distribution_fit_columns,
            }
        )

    leakage_audit["train_fold_threshold_scope_audit"] = threshold_scope_audit
    leakage_audit["distribution_fit_columns"] = distribution_fit_columns
    leakage_audit_path = paths.outputs_reports / "model_input_leakage_audit.json"
    safe_write_json_fn(leakage_audit, leakage_audit_path)
    state.artifacts["model_input_leakage_audit"] = leakage_audit_path

    fold_ledger_frame = model_input_df.copy()
    fold_ledger_frame[effective_cv_config.target_column] = target
    fold_ledger = cv_ledger_callable(fold_ledger_frame, effective_cv_config)
    fold_ledger_path = paths.outputs_reports / "fold_ledger.json"
    safe_write_json_fn(
        {
            "run_id": state.run_id,
            "cv_config": asdict(effective_cv_config),
            "total_folds": int(len(fold_ledger)),
            "yielded_folds": int(sum(1 for fold in fold_ledger if fold.get("status") == "yielded")),
            "minimum_evaluated_folds": int(effective_cv_config.minimum_evaluated_folds),
            "folds": fold_ledger,
        },
        fold_ledger_path,
    )
    state.artifacts["fold_ledger"] = fold_ledger_path

    baseline_score: pd.Series | None = None
    baseline_oof_score: pd.Series | None = None
    baseline_oof_predictions: pd.DataFrame | None = None
    baseline_metrics: dict[str, float] | None = None
    baseline_metrics_fullfit: dict[str, float] | None = None
    baseline_model_metrics: pd.DataFrame | None = None
    baseline_models: dict[str, Any] = {}
    baseline_headline_eligible = False
    baseline_evaluated_fold_count = 0
    if not skip_baselines:
        LOGGER.info("Baseline compute backend: %s", baseline_compute_backend)
        baseline_output_root = paths.outputs_models / "baselines"
        baseline_output_root.mkdir(parents=True, exist_ok=True)
        cleanup_summary = clear_stale_baseline_artifacts(baseline_output_root)
        if cleanup_summary["removed_fold_dirs"] or cleanup_summary["removed_files"]:
            LOGGER.info(
                "Cleared stale baseline artifacts before run (fold_dirs=%d, files=%d)",
                int(cleanup_summary["removed_fold_dirs"]),
                int(cleanup_summary["removed_files"]),
            )

        baseline_train_kwargs: dict[str, Any] = {
            "model_names": model_names,
            "config": baseline_training_config_cls(
                random_state=effective_seed,
                compute_backend=str(baseline_compute_backend),
            ),
            "cv_config": effective_cv_config,
        }
        train_signature = inspect.signature(train_baselines_fn)
        if "build_fold_ledger_fn" in train_signature.parameters:
            baseline_train_kwargs["build_fold_ledger_fn"] = cv_ledger_callable
        if "generate_time_splits_fn" in train_signature.parameters:
            baseline_train_kwargs["generate_time_splits_fn"] = cv_split_callable
        if "case_series" in train_signature.parameters:
            baseline_train_kwargs["case_series"] = bayesian_count_target
        if "district_series" in train_signature.parameters:
            baseline_train_kwargs["district_series"] = district_index
        if "temporal_series" in train_signature.parameters:
            baseline_train_kwargs["temporal_series"] = temporal_index
        if "selected_percentile" in train_signature.parameters:
            baseline_train_kwargs["selected_percentile"] = int(selected_percentile)
        if "fold_local_labeling" in train_signature.parameters:
            baseline_train_kwargs["fold_local_labeling"] = True
        if "fold_local_climate_imputation" in train_signature.parameters:
            baseline_train_kwargs["fold_local_climate_imputation"] = True
        baseline_models = train_baselines_fn(
            model_input_df,
            target,
            **baseline_train_kwargs,
        )
        baseline_predictions = predict_baselines_fn(baseline_models, model_input_df)
        if export_detailed_csv:
            baseline_predictions_path = paths.outputs_models / "detailed" / "baseline_predictions.csv"
            baseline_predictions_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_predictions.to_csv(baseline_predictions_path, index=False)
            state.artifacts["baseline_predictions"] = baseline_predictions_path

        if not baseline_predictions.empty:
            baseline_score = baseline_predictions.mean(axis=1)
            baseline_metrics_fullfit = evaluate_baseline_predictions_fn(
                target,
                baseline_score,
                max_lookback_steps=lead_time_max_lookback_steps,
                temporal_index=temporal_index,
                district=district_index,
            )
            safe_write_json_fn(baseline_metrics_fullfit, paths.outputs_metrics / "baseline_metrics_fullfit.json")
            state.artifacts["baseline_metrics_fullfit"] = paths.outputs_metrics / "baseline_metrics_fullfit.json"

            baseline_oof_score = collect_baseline_oof_scores_fn(
                output_root=paths.outputs_models / "baselines",
                expected_index=model_input_df.index,
            )
            baseline_oof_predictions = collect_baseline_oof_predictions_fn(
                output_root=paths.outputs_models / "baselines",
                expected_index=model_input_df.index,
            )
            baseline_evaluated_fold_count = len(collect_baseline_oof_fold_ids_fn(paths.outputs_models / "baselines"))
            valid_oof_mask = baseline_oof_score.notna()
            if valid_oof_mask.any() and baseline_evaluated_fold_count >= int(effective_cv_config.minimum_evaluated_folds):
                baseline_metrics = evaluate_baseline_predictions_fn(
                    target.loc[valid_oof_mask],
                    baseline_oof_score.loc[valid_oof_mask],
                    max_lookback_steps=lead_time_max_lookback_steps,
                    temporal_index=temporal_index.loc[valid_oof_mask] if temporal_index is not None else None,
                    district=district_index.loc[valid_oof_mask] if district_index is not None else None,
                )
                safe_write_json_fn(baseline_metrics, paths.outputs_metrics / "baseline_metrics.json")
                state.artifacts["baseline_metrics"] = paths.outputs_metrics / "baseline_metrics.json"
                baseline_headline_eligible = True

                if baseline_oof_predictions is not None and not baseline_oof_predictions.empty:
                    configured_model_names = (
                        [str(name) for name in model_names]
                        if model_names
                        else [str(name) for name in baseline_models.keys()]
                    )
                    if not configured_model_names:
                        configured_model_names = [str(name) for name in baseline_oof_predictions.columns]
                    allowed_model_set = set(configured_model_names)
                    filtered_model_columns = [
                        column for column in baseline_oof_predictions.columns if str(column) in allowed_model_set
                    ]
                    baseline_oof_predictions = baseline_oof_predictions.loc[:, filtered_model_columns]

                    model_metric_rows: list[dict[str, Any]] = []
                    for model_name in baseline_oof_predictions.columns:
                        model_score = pd.to_numeric(baseline_oof_predictions[model_name], errors="coerce")
                        model_mask = model_score.notna()
                        if not model_mask.any():
                            continue
                        metrics_row = evaluate_baseline_predictions_fn(
                            target.loc[model_mask],
                            model_score.loc[model_mask],
                            max_lookback_steps=lead_time_max_lookback_steps,
                            temporal_index=temporal_index.loc[model_mask] if temporal_index is not None else None,
                            district=district_index.loc[model_mask] if district_index is not None else None,
                        )
                        metrics_row["model"] = str(model_name)
                        model_metric_rows.append(metrics_row)

                    if model_metric_rows:
                        baseline_model_metrics = pd.DataFrame(model_metric_rows)
                        baseline_model_metrics = baseline_model_metrics[
                            ["model", *[c for c in baseline_model_metrics.columns if c != "model"]]
                        ]
                        tracka_metric_path = paths.outputs_metrics / "tracka_model_scores.csv"
                        baseline_model_metrics.to_csv(tracka_metric_path, index=False)
                        state.artifacts["tracka_model_scores"] = tracka_metric_path
            elif valid_oof_mask.any():
                state.degraded_reasons.append(
                    {
                        "code": "baseline_insufficient_folds",
                        "minimum_required": int(effective_cv_config.minimum_evaluated_folds),
                        "evaluated_folds": int(baseline_evaluated_fold_count),
                    }
                )
                LOGGER.warning(
                    "Baseline headline metrics suppressed: evaluated folds=%d below minimum=%d",
                    int(baseline_evaluated_fold_count),
                    int(effective_cv_config.minimum_evaluated_folds),
                )
            else:
                LOGGER.warning("No baseline OOF predictions available; headline baseline metrics not produced.")
                state.degraded_reasons.append(
                    {
                        "code": "baseline_no_oof_predictions",
                        "minimum_required": int(effective_cv_config.minimum_evaluated_folds),
                        "evaluated_folds": int(baseline_evaluated_fold_count),
                    }
                )
    else:
        LOGGER.info("Baseline phase skipped by flag")

    baseline_metrics_fullfit_path = paths.outputs_metrics / "baseline_metrics_fullfit.json"
    if baseline_metrics_fullfit is None:
        baseline_metrics_fullfit = build_suppressed_metric_payload(
            run_id=state.run_id,
            track="baseline_fullfit",
            reason="baseline_not_available",
        )
        safe_write_json_fn(baseline_metrics_fullfit, baseline_metrics_fullfit_path)
    state.artifacts["baseline_metrics_fullfit"] = baseline_metrics_fullfit_path

    baseline_metrics_path = paths.outputs_metrics / "baseline_metrics.json"
    if baseline_metrics is None:
        baseline_metrics = build_suppressed_metric_payload(
            run_id=state.run_id,
            track="baseline_oof",
            reason="baseline_headline_not_available",
        )
        safe_write_json_fn(baseline_metrics, baseline_metrics_path)
    state.artifacts["baseline_metrics"] = baseline_metrics_path

    baseline_backend_metadata_path = paths.outputs_metrics / "baseline_backend_metadata.json"
    baseline_training_config_path = paths.outputs_models / "baselines" / "training_config.json"
    safe_write_json_fn(
        {
            "run_id": state.run_id,
            "compute_backend_effective": str(baseline_compute_backend),
            "skip_baselines": bool(skip_baselines),
            "model_names": [str(name) for name in model_names] if model_names else None,
            "training_config_path": str(baseline_training_config_path) if baseline_training_config_path.exists() else None,
        },
        baseline_backend_metadata_path,
    )
    state.artifacts["baseline_backend_metadata"] = baseline_backend_metadata_path

    LOGGER.info(
        "Baseline phase summary | skip=%s headline_eligible=%s evaluated_folds=%d models_trained=%d",
        bool(skip_baselines),
        bool(baseline_headline_eligible),
        int(baseline_evaluated_fold_count),
        int(len(baseline_models)),
    )

    return BaselinePhaseResult(
        model_input_df=model_input_df,
        target=target,
        bayesian_count_target=bayesian_count_target,
        bayesian_threshold_series=bayesian_threshold_series,
        temporal_index=temporal_index,
        district_index=district_index,
        baseline_score=baseline_score,
        baseline_oof_score=baseline_oof_score,
        baseline_oof_predictions=baseline_oof_predictions,
        baseline_metrics=baseline_metrics,
        baseline_metrics_fullfit=baseline_metrics_fullfit,
        baseline_model_metrics=baseline_model_metrics,
        baseline_models=baseline_models,
        baseline_headline_eligible=baseline_headline_eligible,
        baseline_evaluated_fold_count=baseline_evaluated_fold_count,
    )
