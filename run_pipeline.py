"""End-to-end orchestration for the chikungunya early warning pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import logging
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import pandas as pd

from config.paths import ensure_directories
from src.data_preprocessing.clean_data import clean_data
from src.data_preprocessing.impute_climate import impute_climate
from src.data_preprocessing.label_outbreaks import label_outbreaks
from src.data_preprocessing.load_data import run as load_phase
from src.data_preprocessing.merge_population import merge_population
from src.decision_layer.cost_loss import AlertThresholds, assign_alert_levels, optimize_decision_threshold
from src.evaluation.compare_tracks import build_comparison_table, export_track_comparison
from src.evaluation.metrics_baselines import evaluate_baseline_predictions, lead_time_steps
from src.evaluation.metrics_bayesian import evaluate_bayesian_predictions
from src.feature_engineering.build_feature_matrix import build_feature_matrix
from src.models.baselines.predict_baselines import predict_baselines
from src.models.baselines.train_baselines import BaselineTrainingConfig, train_baselines
from src.models.baselines.model_registry import list_default_model_names
from src.models.baselines.cv_splitter import TimeSeriesCVConfig, build_fold_ledger, generate_time_splits
from src.models.bayesian.diagnostics import check_convergence, extract_rhat_ess
from src.pipeline_runtime import config_runtime as runtime_config
from src.pipeline_runtime import compute_backend as runtime_backend
from src.pipeline_runtime import io_artifacts as runtime_artifacts
from src.pipeline_runtime.phase_context import SharedPhaseState
from src.pipeline_runtime import phases_baseline as runtime_baseline
from src.pipeline_runtime import phases_bayesian as runtime_bayesian
from src.pipeline_runtime import phases_eval_decision as runtime_eval_decision
from src.visualization.diagnostic_plots import (
    plot_convergence_comparison,
    plot_posterior_predictive_check,
    plot_residuals,
    plot_trace,
)
from src.visualization.exploratory import (
    plot_case_distribution,
    plot_missingness_summary,
    plot_temporal_coverage_heatmap,
)
from src.visualization.feature_plots import plot_correlation_heatmap, plot_feature_importance
from src.visualization.performance_plots import (
    plot_calibration_curve,
    plot_track_delta_heatmap,
    plot_track_comparison_shared_metrics_bar,
    plot_tracka_model_score_comparison,
    plot_confusion_matrix_grid,
    plot_brier_lead_time_summary,
    plot_lead_time_boxplot,
    plot_pr_curve,
    plot_roc_curve,
)
from src.visualization.risk_maps import plot_decision_alert_trend, plot_risk_trajectory, plot_top_risk_districts

LOGGER = logging.getLogger(__name__)

_FORBIDDEN_COLUMN_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^outbreak_label$", flags=re.IGNORECASE),
    re.compile(r"^outbreak_label_.*$", flags=re.IGNORECASE),
    re.compile(r"^outbreak_label_p.*$", flags=re.IGNORECASE),
    re.compile(r"^threshold_.*$", flags=re.IGNORECASE),
    re.compile(r"^threshold_p.*$", flags=re.IGNORECASE),
)
_RAW_CASE_TARGET_ALIASES: set[str] = {"cases", "case_count", "weekly_cases", "outbreak_target", "target"}
_ADAPTER_CALLABLE_KEYS: tuple[str, ...] = (
    "load_data",
    "label_outbreaks",
    "build_feature_matrix",
    "build_fold_ledger",
    "generate_time_splits",
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

_CONTRACT_REPORT_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "run_metadata.json",
    "fold_ledger.json",
    "degraded_run.json",
    "feature_quality_gate_report.json",
    "model_input_leakage_audit.json",
)
_CONTRACT_METRIC_FILES: tuple[str, ...] = (
    "baseline_backend_metadata.json",
    "baseline_metrics.json",
    "baseline_metrics_fullfit.json",
    "bayesian_metrics.json",
    "bayesian_metrics_fullfit.json",
    "bayesian_risk_metadata.json",
    "track_comparison.csv",
    "track_comparison.md",
    "decision_alerts.csv",
    "bayesian_convergence_summary.csv",
    "bayesian_convergence_summary.md",
)


def _build_model_input_df(
    feature_df: pd.DataFrame,
    *,
    target_column: str = "outbreak_label",
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
    audit = {
        "input_feature_count": int(feature_df.shape[1]),
        "output_feature_count": int(output.shape[1]),
        "dropped_forbidden_columns": sorted(set(dropped)),
        "forbidden_patterns": [pattern.pattern for pattern in _FORBIDDEN_COLUMN_PATTERNS],
        "raw_case_aliases": sorted(_RAW_CASE_TARGET_ALIASES),
    }
    return output, audit


def _build_bayesian_config(strict_dependencies: bool, bayesian_settings: dict[str, Any]) -> Any:
    from src.models.bayesian.hierarchical_model import BayesianModelConfig

    return BayesianModelConfig(
        strict_dependencies=strict_dependencies,
        draws=int(bayesian_settings.get("draws", BayesianModelConfig.draws)),
        tune=int(bayesian_settings.get("tune", BayesianModelConfig.tune)),
        chains=int(bayesian_settings.get("chains", BayesianModelConfig.chains)),
        bayesian_progress=bool(bayesian_settings.get("bayesian_progress", BayesianModelConfig.bayesian_progress)),
        target_accept=float(bayesian_settings.get("target_accept", BayesianModelConfig.target_accept)),
        max_treedepth=int(bayesian_settings.get("max_treedepth", BayesianModelConfig.max_treedepth)),
        bayesian_simplified_mode=bool(
            bayesian_settings.get("bayesian_simplified_mode", BayesianModelConfig.bayesian_simplified_mode)
        ),
        max_convergence_retries=int(
            bayesian_settings.get("max_convergence_retries", BayesianModelConfig.max_convergence_retries)
        ),
        divergence_warn_threshold=int(
            bayesian_settings.get("divergence_warn_threshold", BayesianModelConfig.divergence_warn_threshold)
        ),
        rhat_warn_threshold=float(
            bayesian_settings.get("rhat_warn_threshold", BayesianModelConfig.rhat_warn_threshold)
        ),
        ess_warn_threshold=float(
            bayesian_settings.get("ess_warn_threshold", BayesianModelConfig.ess_warn_threshold)
        ),
        random_seed=int(bayesian_settings.get("random_seed", BayesianModelConfig.random_seed)),
        force_full_bayesian=bool(
            bayesian_settings.get("force_full_bayesian", BayesianModelConfig.force_full_bayesian)
        ),
        outbreak_threshold_default_cases=float(
            bayesian_settings.get(
                "outbreak_threshold_default_cases",
                BayesianModelConfig.outbreak_threshold_default_cases,
            )
        ),
        posterior_sample_cap=int(
            bayesian_settings.get("posterior_sample_cap", BayesianModelConfig.posterior_sample_cap)
        ),
    )


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        LOGGER.warning("Model config not found at %s; using defaults", config_path)
        return {}
    try:
        import yaml
    except Exception as yaml_error:
        LOGGER.warning("PyYAML not available (%s); unable to read %s", yaml_error, config_path)
        return {}

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as load_error:
        LOGGER.warning("Unable to load YAML config from %s: %s", config_path, load_error)
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _deep_merge_dict(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(existing, value)
        else:
            merged[key] = value
    return merged


def _import_callable(path_spec: str) -> Callable[..., Any]:
    spec = str(path_spec).strip()
    if ":" in spec:
        module_name, attribute_name = spec.split(":", 1)
    else:
        module_name, _, attribute_name = spec.rpartition(".")
    if not module_name or not attribute_name:
        raise ValueError(f"Invalid callable import path: '{path_spec}'")
    module = importlib.import_module(module_name)
    loaded = getattr(module, attribute_name, None)
    if not callable(loaded):
        raise TypeError(f"Imported object is not callable: '{path_spec}'")
    return loaded


def _resolve_adapter_callable(
    adapter_config: dict[str, Any],
    *,
    key: str,
    default: Callable[..., Any],
) -> Callable[..., Any]:
    configured = adapter_config.get(key)
    if not configured:
        return default
    try:
        loaded = _import_callable(str(configured))
    except Exception as import_error:
        LOGGER.warning(
            "Unable to load adapter callable '%s' from key '%s' (%s); using default %s",
            configured,
            key,
            import_error,
            default.__name__,
        )
        return default
    LOGGER.info("Adapter callable enabled for '%s': %s", key, configured)
    return loaded


def _is_brazil_adapter_config_active(adapter_config: dict[str, Any]) -> bool:
    if not isinstance(adapter_config, dict) or not adapter_config:
        return False

    callable_specs: list[str] = []
    for key in _ADAPTER_CALLABLE_KEYS:
        value = adapter_config.get(key)
        if value is None:
            continue
        callable_specs.append(str(value).strip())

    if not callable_specs:
        return False
    return all(spec.startswith("projects.brazil_chik.") for spec in callable_specs)


def _find_distribution_fit_columns(columns: list[str]) -> list[str]:
    flagged: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if "case" not in lowered and "rt" not in lowered:
            continue
        if any(token in lowered for token in _DISTRIBUTION_FIT_TOKENS):
            flagged.append(str(column))
    return sorted(set(flagged))


def _audit_train_fold_threshold_scope(
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


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def _call_load_data_compat(
    load_data_callable: Callable[..., Any],
    raw_data_path: Path,
    population_data_path: Path | None,
    *,
    start_year: int,
    end_year: int,
    discovery_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    signature = inspect.signature(load_data_callable)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    kwargs: dict[str, Any] = {"discovery_dir": discovery_dir}
    if "start_year" in parameters or accepts_var_kwargs:
        kwargs["start_year"] = int(start_year)
    if "end_year" in parameters or accepts_var_kwargs:
        kwargs["end_year"] = int(end_year)

    try:
        return load_data_callable(raw_data_path, population_data_path, **kwargs)
    except TypeError as call_error:
        fallback_kwargs = {"discovery_dir": discovery_dir}
        unsupported_kw_error = "unexpected keyword" in str(call_error).lower() and (
            "start_year" in str(call_error) or "end_year" in str(call_error)
        )
        if kwargs != fallback_kwargs and unsupported_kw_error:
            LOGGER.info(
                "Loader callable rejected year bounds (%s); falling back to legacy loader call shape",
                call_error,
            )
            return load_data_callable(raw_data_path, population_data_path, **fallback_kwargs)
        raise


def _call_label_outbreaks_compat(
    label_callable: Callable[..., Any],
    df: pd.DataFrame,
    *,
    selected_percentile: int,
    use_percentile_labels: bool,
    cv_config: TimeSeriesCVConfig,
    strict_mode: bool,
) -> pd.DataFrame:
    signature = inspect.signature(label_callable)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    kwargs: dict[str, Any] = {
        "selected_percentile": selected_percentile,
        "use_percentile_labels": use_percentile_labels,
        "date_column": cv_config.date_column,
        "first_valid_year": int(cv_config.first_valid_year),
        "last_valid_year": int(cv_config.last_valid_year),
        "start_train_year": int(cv_config.start_train_year),
        "train_window_years": int(cv_config.train_window_years),
        "threshold_scope": "train_fold",
    }

    if not accepts_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}

    try:
        return label_callable(df, **kwargs)
    except TypeError as call_error:
        unsupported_kw_error = "unexpected keyword" in str(call_error).lower()
        if kwargs and unsupported_kw_error:
            if strict_mode:
                raise RuntimeError(
                    "Label callable does not support train-fold leakage-control args under strict mode "
                    f"(error: {call_error})"
                ) from call_error
            LOGGER.info(
                "Label callable rejected advanced args (%s); falling back to legacy label call shape",
                call_error,
            )
            return label_callable(
                df,
                selected_percentile=selected_percentile,
                use_percentile_labels=use_percentile_labels,
            )
        raise


def _resolve_cv_config(
    *,
    cv_config_path: Path,
    date_column: str,
    target_column: str,
    end_year: int,
) -> tuple[TimeSeriesCVConfig, dict[str, Any]]:
    raw_cv_config = _load_yaml_config(cv_config_path)

    first_valid_year = int(raw_cv_config.get("first_valid_year", 0) or 0)
    last_valid_year = int(raw_cv_config.get("last_valid_year", 0) or 0)
    n_splits = int(raw_cv_config.get("n_splits", 0) or 0)

    if first_valid_year <= 0:
        first_valid_year = max(2009, (end_year - max(1, n_splits) + 1) if n_splits > 0 else 2014)
    if last_valid_year <= 0:
        last_valid_year = end_year

    cv_cfg = TimeSeriesCVConfig(
        date_column=str(raw_cv_config.get("date_column", date_column)),
        target_column=str(raw_cv_config.get("target_column", target_column)),
        start_train_year=int(raw_cv_config.get("start_train_year", 2009)),
        first_valid_year=int(first_valid_year),
        last_valid_year=int(last_valid_year),
        train_window_years=int(raw_cv_config.get("train_window_years", 5)),
        thesis_strict=bool(raw_cv_config.get("thesis_strict", False)),
        skip_single_class_folds=bool(raw_cv_config.get("skip_single_class_folds", True)),
        minimum_evaluated_folds=int(raw_cv_config.get("minimum_evaluated_folds", max(1, min(3, n_splits or 5)))),
    )
    effective = {
        **raw_cv_config,
        **asdict(cv_cfg),
    }
    return cv_cfg, effective


def _clear_headline_artifacts(metrics_dir: Path) -> None:
    for filename in (
        "baseline_metrics.json",
        "bayesian_metrics.json",
        "track_comparison.csv",
        "track_comparison.md",
        "bayesian_convergence_summary.csv",
        "bayesian_convergence_summary.md",
    ):
        path = metrics_dir / filename
        if path.exists():
            path.unlink()


def _clear_contract_artifacts(*, reports_dir: Path, metrics_dir: Path) -> None:
    for filename in _CONTRACT_REPORT_FILES:
        path = reports_dir / filename
        if path.exists():
            path.unlink()
    for filename in _CONTRACT_METRIC_FILES:
        path = metrics_dir / filename
        if path.exists():
            path.unlink()


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_sha(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit if commit else None


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(v) for v in value]
    return value


def _stable_district_shard(value: Any, shard_count: int) -> int:
    normalized = str(value).strip().lower()
    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(shard_count)


def _apply_memory_optimization_filters(
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    config: runtime_config.MemoryOptimizationConfig,
    date_column: str = "date",
    district_column: str = "district",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    mode = str(config.mode or "off").lower()
    report: dict[str, Any] = {
        "mode": mode,
        "active": False,
        "reproducibility": {
            "district_hash": "md5_lower_utf8_hex8_mod",
        },
        "inputs": {
            "labeled_rows": int(len(labeled_df)),
            "feature_rows": int(len(features_df)),
            "labeled_districts": int(labeled_df.get(district_column, pd.Series(dtype=object)).nunique(dropna=True)),
        },
        "filters": {
            "year_window": {
                "requested": list(config.train_year_window) if config.train_year_window is not None else None,
                "applied": False,
            },
            "district_shard": {
                "requested_count": config.district_shard_count,
                "requested_index": config.district_shard_index,
                "applied": False,
            },
        },
        "warnings": [],
    }

    if mode == "off":
        report["outputs"] = {
            "labeled_rows": int(len(labeled_df)),
            "feature_rows": int(len(features_df)),
            "labeled_districts": int(labeled_df.get(district_column, pd.Series(dtype=object)).nunique(dropna=True)),
        }
        return labeled_df, features_df, report

    keep_mask = pd.Series(True, index=labeled_df.index, dtype="bool")

    apply_year_filter = mode in {"year_window", "hybrid"}
    if apply_year_filter:
        if config.train_year_window is None:
            report["warnings"].append("year_window_mode_requested_but_train_year_window_missing")
        elif date_column not in labeled_df.columns:
            report["warnings"].append("year_window_mode_requested_but_date_column_missing")
        else:
            start_year, end_year = config.train_year_window
            date_series = pd.to_datetime(labeled_df[date_column], errors="coerce")
            year_mask = date_series.dt.year.between(int(start_year), int(end_year), inclusive="both").fillna(False)
            keep_mask &= year_mask
            report["filters"]["year_window"].update(
                {
                    "applied": True,
                    "effective": [int(start_year), int(end_year)],
                    "kept_rows": int(year_mask.sum()),
                }
            )

    apply_shard_filter = mode in {"district_shard", "hybrid"}
    if apply_shard_filter:
        shard_count = config.district_shard_count
        shard_index = config.district_shard_index
        if shard_count is None:
            report["warnings"].append("district_shard_mode_requested_but_district_shard_count_missing")
        elif district_column not in labeled_df.columns:
            report["warnings"].append("district_shard_mode_requested_but_district_column_missing")
        else:
            safe_index = int(shard_index if shard_index is not None else 0) % int(shard_count)
            district_values = labeled_df[district_column].fillna("__missing_district__").astype(str)
            shard_series = district_values.map(lambda value: _stable_district_shard(value, int(shard_count)))
            shard_mask = shard_series.eq(int(safe_index)).fillna(False)
            keep_mask &= shard_mask
            report["filters"]["district_shard"].update(
                {
                    "applied": True,
                    "effective_count": int(shard_count),
                    "effective_index": int(safe_index),
                    "kept_rows": int(shard_mask.sum()),
                }
            )

    filtered_labeled = labeled_df.loc[keep_mask].copy()
    if features_df.index.equals(labeled_df.index):
        filtered_features = features_df.loc[keep_mask].copy()
    else:
        selected_index = pd.Index(filtered_labeled.index)
        filtered_features = features_df.loc[features_df.index.intersection(selected_index)].copy()

    report["active"] = bool(len(filtered_labeled) != len(labeled_df))
    report["outputs"] = {
        "labeled_rows": int(len(filtered_labeled)),
        "feature_rows": int(len(filtered_features)),
        "labeled_districts": int(filtered_labeled.get(district_column, pd.Series(dtype=object)).nunique(dropna=True)),
        "rows_removed": int(len(labeled_df) - len(filtered_labeled)),
    }
    return filtered_labeled, filtered_features, report


def _build_suppressed_metric_payload(*, run_id: str, track: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "track": track,
        "suppressed": True,
        "reason": reason,
    }


def _build_contract_track_comparison_placeholder(*, run_id: str, reason: str) -> tuple[pd.DataFrame, str]:
    frame = pd.DataFrame(
        [
            {
                "metric": "headline_comparison",
                "baseline": None,
                "bayesian": None,
                "delta": None,
                "status": "suppressed",
                "reason": reason,
                "run_id": run_id,
            }
        ]
    )
    markdown = (
        "# Track Comparison\n\n"
        "| metric | baseline | bayesian | delta | status | reason | run_id |\n"
        "|---|---:|---:|---:|---|---|---|\n"
        f"| headline_comparison |  |  |  | suppressed | {reason} | {run_id} |\n"
    )
    return frame, markdown


def _write_bayesian_convergence_summary(
    *,
    metrics_dir: Path,
    run_id: str,
    diagnostics: dict[str, Any],
    convergence_artifact_path: Path | None = None,
    convergence: dict[str, Any] | None,
) -> tuple[Path, Path]:
    if convergence_artifact_path is not None and convergence_artifact_path.exists():
        convergence_payload = json.loads(convergence_artifact_path.read_text(encoding="utf-8"))
    else:
        convergence_payload = convergence or {}
    summary_row = {
        "run_id": run_id,
        "mode_used": str(diagnostics.get("mode_used", "not_run")),
        "degraded_mode": bool(diagnostics.get("degraded_mode", False)),
        "fallback_used": bool(diagnostics.get("fallback_used", False)),
        "converged": bool(convergence_payload.get("converged", False)),
        "divergences": float(convergence_payload.get("divergences", 0.0) or 0.0),
        "divergence_threshold": float(convergence_payload.get("divergence_threshold", float("nan"))),
        "max_tree_depth": float(convergence_payload.get("max_tree_depth", 0.0) or 0.0),
        "max_tree_depth_threshold": float(convergence_payload.get("max_tree_depth_threshold", float("nan"))),
        "r_hat_max": float(convergence_payload.get("r_hat_max", 0.0) or 0.0),
        "rhat_threshold": float(convergence_payload.get("rhat_threshold", float("nan"))),
        "ess_min": float(convergence_payload.get("ess_min", 0.0) or 0.0),
        "ess_threshold": float(convergence_payload.get("ess_threshold", float("nan"))),
    }
    summary_frame = pd.DataFrame([summary_row])
    csv_path = metrics_dir / "bayesian_convergence_summary.csv"
    md_path = metrics_dir / "bayesian_convergence_summary.md"
    summary_frame.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Bayesian Convergence Summary\n\n"
        + summary_frame.to_markdown(index=False),
        encoding="utf-8",
    )
    return csv_path, md_path


def _validate_manifest_contract(
    *,
    run_manifest_path: Path,
    artifacts: dict[str, Path],
    required_artifact_keys: set[str],
    expected_run_id: str,
) -> None:
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_artifacts = manifest.get("artifacts", {})

    missing_in_manifest = sorted(required_artifact_keys.difference(manifest_artifacts.keys()))
    missing_in_artifacts = sorted(required_artifact_keys.difference(artifacts.keys()))
    if missing_in_manifest or missing_in_artifacts:
        raise RuntimeError(
            "Contract artifact key mismatch "
            f"(manifest_missing={missing_in_manifest}, artifacts_missing={missing_in_artifacts})"
        )

    mismatched_paths: list[str] = []
    missing_files: list[str] = []
    for key in sorted(required_artifact_keys):
        artifact_path = artifacts[key]
        manifest_path = Path(str(manifest_artifacts[key]))
        if str(artifact_path) != str(manifest_path):
            mismatched_paths.append(key)
        if not artifact_path.exists():
            missing_files.append(str(artifact_path))
    if mismatched_paths or missing_files:
        raise RuntimeError(
            "Contract artifact parity failed "
            f"(path_mismatch={mismatched_paths}, missing_files={missing_files})"
        )

    metadata = json.loads(Path(str(manifest_artifacts["run_metadata"])).read_text(encoding="utf-8"))
    fold_ledger = json.loads(Path(str(manifest_artifacts["fold_ledger"])).read_text(encoding="utf-8"))
    degraded_run = json.loads(Path(str(manifest_artifacts["degraded_run"])).read_text(encoding="utf-8"))

    run_id_values = {
        "manifest": str(manifest.get("run_id")),
        "run_metadata": str(metadata.get("run_id")),
        "fold_ledger": str(fold_ledger.get("run_id")),
        "degraded_run": str(degraded_run.get("run_id")),
    }
    if any(value != expected_run_id for value in run_id_values.values()):
        raise RuntimeError(f"run_id parity mismatch: {run_id_values} expected={expected_run_id}")

    suppressed_manifest = bool(manifest.get("headline_claims", {}).get("suppressed", False))
    suppressed_degraded = bool(degraded_run.get("suppress_headline_comparison", False))
    if suppressed_manifest != suppressed_degraded:
        raise RuntimeError(
            "degraded_run parity mismatch "
            f"(manifest_suppressed={suppressed_manifest}, degraded_suppressed={suppressed_degraded})"
        )


def _collect_baseline_oof_fold_ids(output_root: Path) -> list[int]:
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


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for pipeline execution."""
    return runtime_config.parse_args()


def _safe_write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _run_bayesian_track(
    features_df: pd.DataFrame,
    count_target: pd.Series,
    *,
    outbreak_threshold: pd.Series | None,
    strict_dependencies: bool,
    bayesian_settings: dict[str, Any],
    compute_backend_requested: str = "cpu",
    compute_backend_effective: str = "cpu",
) -> tuple[pd.DataFrame | None, Any | None, dict[str, Any]]:
    backend_requested = str(compute_backend_requested or "cpu")
    backend_effective = str(compute_backend_effective or "cpu")
    bayesian_runtime_backend = "cpu"
    backend_fallback_reason: str | None = None
    if backend_effective != "cpu":
        backend_fallback_reason = (
            f"Bayesian implementation does not currently support '{backend_effective}' execution; using CPU runtime path"
        )
        LOGGER.warning("%s", backend_fallback_reason)

    try:
        from src.models.bayesian.hierarchical_model import BayesianModelConfig, HierarchicalBayesianModel

        config = _build_bayesian_config(strict_dependencies, bayesian_settings)

        if config.force_full_bayesian:
            config = BayesianModelConfig(
                **{
                    **config.__dict__,
                    "bayesian_simplified_mode": False,
                    "max_convergence_retries": 0,
                }
            )
            LOGGER.info("Bayesian mode request: full latent AR only (simplified fallback disabled).")

        bayesian_model = HierarchicalBayesianModel(config=config).fit(features_df, count_target)
        risk_frame, predictive_metadata = bayesian_model.predict_with_uncertainty(
            features_df,
            outbreak_threshold=outbreak_threshold,
        )
        fallback_used = bool(float(bayesian_model.diagnostics_summary_.get("fallback", 0.0)) > 0.0)
        mode_used = "fallback" if fallback_used else ("simplified" if bayesian_model.simplified_used_ else "full_latent_ar")
        diagnostics = {
            "simplified_mode": bool(bayesian_model.simplified_used_),
            "force_full_bayesian": bool(config.force_full_bayesian),
            "mode_used": mode_used,
            "fallback_used": fallback_used,
            "degraded_mode": bool(predictive_metadata.get("degraded_mode", False) or fallback_used),
            "compute_backend_requested": backend_requested,
            "compute_backend_effective": backend_effective,
            "compute_backend_runtime": bayesian_runtime_backend,
            "compute_backend_fallback_used": bool(backend_fallback_reason is not None),
            "compute_backend_fallback_reason": backend_fallback_reason,
            "target_semantics": "count_likelihood",
            "target_column_used": "cases",
            "risk_summary_columns": ["risk_mean", "risk_q05", "risk_q95", "threshold_cases"],
            **predictive_metadata,
            **bayesian_model.diagnostics_summary_,
        }
        return risk_frame, bayesian_model.idata_, diagnostics
    except ImportError as import_error:
        LOGGER.warning("Skipping Bayesian phase due to missing optional dependencies: %s", import_error)
        if strict_dependencies:
            raise
        return None, None, {
            "degraded_mode": True,
            "fallback_used": True,
            "mode_used": "missing_dependencies",
            "degraded_reason": "missing_optional_dependencies",
            "error": str(import_error),
            "compute_backend_requested": backend_requested,
            "compute_backend_effective": backend_effective,
            "compute_backend_runtime": bayesian_runtime_backend,
            "compute_backend_fallback_used": bool(backend_fallback_reason is not None),
            "compute_backend_fallback_reason": backend_fallback_reason,
        }


def _collect_baseline_oof_scores(
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


def _collect_baseline_oof_predictions(
    *,
    output_root: Path,
    expected_index: pd.Index,
) -> pd.DataFrame:
    """Collect per-model OOF predictions across temporal CV folds."""
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


def _collect_bayesian_oof_scores(
    *,
    features_df: pd.DataFrame,
    outbreak_target: pd.Series,
    count_target: pd.Series,
    strict_dependencies: bool,
    bayesian_settings: dict[str, Any],
    cv_config: TimeSeriesCVConfig,
    threshold_series: pd.Series | None = None,
    fail_on_error: bool = False,
    date_column: str = "date",
    target_column: str = "outbreak_label",
    generate_time_splits_fn: Callable[[pd.DataFrame, TimeSeriesCVConfig], Any] = generate_time_splits,
) -> pd.Series:
    from src.models.bayesian.hierarchical_model import HierarchicalBayesianModel

    oof = pd.Series(np.nan, index=features_df.index, dtype="float64")

    cv_frame = features_df.copy()
    cv_frame[target_column] = pd.to_numeric(outbreak_target, errors="coerce").fillna(0).astype(int)
    cv_config = TimeSeriesCVConfig(
        date_column=cv_config.date_column or date_column,
        target_column=cv_config.target_column or target_column,
        start_train_year=cv_config.start_train_year,
        first_valid_year=cv_config.first_valid_year,
        last_valid_year=cv_config.last_valid_year,
        train_window_years=cv_config.train_window_years,
        thesis_strict=getattr(cv_config, "thesis_strict", False),
        skip_single_class_folds=cv_config.skip_single_class_folds,
        minimum_evaluated_folds=cv_config.minimum_evaluated_folds,
    )

    for train_idx, valid_idx in generate_time_splits_fn(cv_frame, cv_config):
        y_train_binary = pd.to_numeric(outbreak_target.loc[train_idx], errors="coerce").fillna(0).astype(int)
        y_train_counts = pd.to_numeric(count_target.loc[train_idx], errors="coerce").fillna(0.0)
        if y_train_binary.nunique(dropna=True) <= 1:
            continue
        try:
            model = HierarchicalBayesianModel(config=_build_bayesian_config(strict_dependencies, bayesian_settings))
            model.fit(features_df.loc[train_idx], y_train_counts)
            fold_threshold = threshold_series.loc[valid_idx] if threshold_series is not None else None
            fold_pred = model.predict_with_uncertainty(
                features_df.loc[valid_idx],
                outbreak_threshold=fold_threshold,
            )[0]["risk_mean"].clip(0.0, 1.0)
            oof.loc[valid_idx] = fold_pred.astype(float)
        except Exception as fold_error:
            if fail_on_error:
                raise RuntimeError(f"Bayesian OOF fold failed under strict/full mode: {fold_error}") from fold_error
            LOGGER.warning("Bayesian OOF fold skipped due to error: %s", fold_error)

    return oof


def _cleanup_nonessential_metric_csvs(metrics_dir: Path) -> None:
    """Remove legacy non-essential CSV artifacts from outputs/metrics."""
    for filename in ("baseline_predictions.csv", "bayesian_risk.csv", "track_comparison_wide.csv"):
        path = metrics_dir / filename
        if path.exists():
            path.unlink()
            LOGGER.info("Removed legacy non-essential metrics CSV: %s", path)


def _cleanup_legacy_figure_placeholders(figures_dir: Path) -> None:
    """Remove legacy placeholders and superseded figure artifacts."""
    legacy_placeholders = {
        "diagnostic_plots.txt",
        "exploratory_summary.txt",
        "feature_plots.txt",
        "performance_plots.txt",
        "risk_maps.txt",
        "diagnostic_trace_plot_skipped.txt",
        "features_shap_summary_skipped.txt",
        "risk_alerts_map_skipped.txt",
    }
    for path in figures_dir.glob("*.txt"):
        if path.name in legacy_placeholders:
            path.unlink(missing_ok=True)
            LOGGER.info("Removed legacy figure placeholder: %s", path)

    superseded_figures = {
        "performance_track_metric_comparison_bar.png",
        "track_comparison_shared_metrics.png",
    }
    for filename in superseded_figures:
        legacy_path = figures_dir / filename
        if legacy_path.exists():
            legacy_path.unlink(missing_ok=True)
            LOGGER.info("Removed superseded figure artifact: %s", legacy_path)


def _extract_feature_importances(
    models: dict[str, Any],
    feature_names: list[str],
) -> pd.Series | None:
    """Aggregate model-specific importances into a single ranked series."""
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


def run(
    *,
    model_config_path: Path = Path("config/model_config.yaml"),
    adapter_config_path: Path | None = None,
    cv_config_path: Path = Path("config/cv_config.yaml"),
    raw_data_path: Path,
    population_data_path: Path | None = None,
    start_year: int = 2009,
    end_year: int = 2019,
    selected_percentile: int = 75,
    skip_baselines: bool = False,
    skip_bayesian: bool = False,
    skip_visualizations: bool = False,
    decision_cost: float = 0.2,
    decision_loss: float = 1.0,
    lead_time_max_lookback_steps: int = 8,
    strict_bayesian_deps: bool = False,
    strict_feature_gate: bool = False,
    force_full_bayesian: bool = False,
    bayesian_overrides: dict[str, Any] | None = None,
    export_detailed_csv: bool = False,
    model_names: list[str] | None = None,
    seed: int | None = None,
    cli_args_snapshot: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Execute pipeline stages in the required end-to-end order."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    run_started_at = datetime.now(timezone.utc).isoformat()

    paths = ensure_directories()
    _cleanup_nonessential_metric_csvs(paths.outputs_metrics)
    runtime_artifacts.clear_headline_artifacts(paths.outputs_metrics)
    runtime_artifacts.clear_contract_artifacts(reports_dir=paths.outputs_reports, metrics_dir=paths.outputs_metrics)

    raw_model_config = runtime_config.load_yaml_config(model_config_path)
    adapter_overlay = runtime_config.load_yaml_config(adapter_config_path) if adapter_config_path is not None else {}
    if adapter_overlay:
        raw_model_config = runtime_config.deep_merge_dict(raw_model_config, adapter_overlay)

    adapter_config = raw_model_config.get("adapter", {})
    if not isinstance(adapter_config, dict):
        adapter_config = {}

    load_data_callable = runtime_config.resolve_adapter_callable(adapter_config, key="load_data", default=load_phase)
    label_callable = runtime_config.resolve_adapter_callable(adapter_config, key="label_outbreaks", default=label_outbreaks)
    feature_callable = runtime_config.resolve_adapter_callable(adapter_config, key="build_feature_matrix", default=build_feature_matrix)
    cv_ledger_callable = runtime_config.resolve_adapter_callable(adapter_config, key="build_fold_ledger", default=build_fold_ledger)
    cv_split_callable = runtime_config.resolve_adapter_callable(adapter_config, key="generate_time_splits", default=generate_time_splits)

    effective_seed = int(seed if seed is not None else raw_model_config.get("random_seed", 42))
    runtime_config.set_global_seed(effective_seed)

    if model_names is None:
        configured_models = raw_model_config.get("baseline_models")
        if isinstance(configured_models, list):
            model_names = [str(name) for name in configured_models]

    effective_cv_config, effective_cv_config_dict = runtime_config.resolve_cv_config(
        cv_config_path=cv_config_path,
        date_column="date",
        target_column="outbreak_label",
        end_year=end_year,
    )

    bayesian_settings = raw_model_config.get("bayesian_model", {}).copy()
    if not isinstance(bayesian_settings, dict):
        bayesian_settings = {}
    memory_optimization_config = runtime_config.parse_memory_optimization_config(raw_model_config)
    compute_backend_config = runtime_config.parse_compute_backend_config(raw_model_config)
    backend_resolution = runtime_backend.resolve_backends(compute_backend_config)
    baseline_backend_effective = str(backend_resolution.get("baseline_backend", "cpu"))
    bayesian_backend_effective = str(backend_resolution.get("bayesian_backend", "cpu"))
    decision_settings = raw_model_config.get("decision", {})
    if not isinstance(decision_settings, dict):
        decision_settings = {}

    decision_policy_version = str(decision_settings.get("policy_version", "v1"))
    decision_optimize_threshold = bool(decision_settings.get("optimize_threshold", True))
    decision_optimization_risk_basis = str(decision_settings.get("optimization_risk_basis", "bayesian_oof_risk_mean"))
    decision_optimization_min_samples = int(decision_settings.get("optimization_min_samples", 100))
    decision_optimization_grid_size = int(decision_settings.get("optimization_grid_size", 101))
    decision_optimization_grid_min = float(decision_settings.get("optimization_grid_min", 0.0))
    decision_optimization_grid_max = float(decision_settings.get("optimization_grid_max", 1.0))
    decision_risk_score_basis = str(decision_settings.get("risk_score_basis", "bayesian_risk_q95"))
    alert_thresholds_raw = decision_settings.get("alert_thresholds", {})
    if not isinstance(alert_thresholds_raw, dict):
        alert_thresholds_raw = {}
    alert_thresholds = AlertThresholds(
        yellow=float(alert_thresholds_raw.get("yellow", 0.3)),
        orange=float(alert_thresholds_raw.get("orange", 0.5)),
        red=float(alert_thresholds_raw.get("red", 0.7)),
    )
    alert_thresholds.validate()

    bayesian_settings["random_seed"] = effective_seed
    if bayesian_overrides:
        bayesian_settings.update({key: value for key, value in bayesian_overrides.items() if value is not None})

    brazil_adapter_active = runtime_config.is_brazil_adapter_config_active(adapter_config)
    requested_force_full = bool(
        force_full_bayesian
        or bayesian_settings.get("force_full_bayesian", False)
        or adapter_config.get("force_full_bayesian_project_only", False)
    )
    force_full_effective = bool(requested_force_full and brazil_adapter_active)
    if requested_force_full and not brazil_adapter_active:
        LOGGER.warning(
            "Ignoring force_full_bayesian request because Brazil adapter config is not active; "
            "falling back to configured non-forced Bayesian mode."
        )
    bayesian_settings["force_full_bayesian"] = bool(force_full_effective)
    if force_full_effective:
        bayesian_settings["bayesian_simplified_mode"] = False
        bayesian_settings["max_convergence_retries"] = 0
        if not skip_baselines:
            LOGGER.info("Force-full Bayesian mode active; skipping baseline track by design")
        skip_baselines = True

    previous_convergence_path = paths.outputs_models / "bayesian" / "diagnostics" / "convergence.json"
    previous_bayesian_convergence: dict[str, Any] | None = None
    if previous_convergence_path.exists():
        try:
            previous_bayesian_convergence = json.loads(previous_convergence_path.read_text(encoding="utf-8"))
        except Exception as read_error:
            LOGGER.warning("Unable to load previous Bayesian convergence diagnostics: %s", read_error)

    LOGGER.info("Phase: load")
    raw_df, population_df = runtime_config.call_load_data_compat(
        load_data_callable,
        raw_data_path,
        population_data_path,
        start_year=start_year,
        end_year=end_year,
        discovery_dir=raw_data_path.parent,
    )

    LOGGER.info("Phase: clean")
    cleaned_df = clean_data(raw_df, start_year=start_year, end_year=end_year)

    LOGGER.info("Phase: impute")
    imputed_df = impute_climate(cleaned_df)

    LOGGER.info("Phase: merge")
    merged_df = imputed_df
    if population_df is not None:
        merged_df = merge_population(imputed_df, population_df)
    else:
        LOGGER.info("No population data provided; merge phase completed with passthrough frame")

    LOGGER.info("Phase: labels")
    labeled_df = runtime_config.call_label_outbreaks_compat(
        label_callable,
        merged_df,
        selected_percentile=selected_percentile,
        use_percentile_labels=True,
        cv_config=effective_cv_config,
        strict_mode=bool(strict_feature_gate),
    )
    threshold_scope_audit = runtime_baseline.audit_train_fold_threshold_scope(
        labeled_df,
        selected_percentile=selected_percentile,
        cv_config=effective_cv_config,
    )

    state = SharedPhaseState(run_id=run_id)
    if not bool(threshold_scope_audit.get("checked", False)):
        message = (
            "Train-fold threshold scope audit could not be completed "
            f"(reason={threshold_scope_audit.get('reason')}, "
            f"missing_columns={threshold_scope_audit.get('missing_columns', [])})"
        )
        if strict_feature_gate:
            raise RuntimeError(message)
        LOGGER.warning(message)
        state.degraded_reasons.append(
            {
                "code": "train_fold_threshold_scope_audit_incomplete",
                "reason": str(threshold_scope_audit.get("reason", "unknown")),
                "missing_columns": threshold_scope_audit.get("missing_columns", []),
            }
        )
    elif int(threshold_scope_audit.get("violation_count", 0)) > 0:
        message = (
            "Detected threshold scope violations against train-fold-only quantiles "
            f"(violation_count={int(threshold_scope_audit.get('violation_count', 0))}, "
            f"sample={threshold_scope_audit.get('sample_violations', [])})"
        )
        if strict_feature_gate:
            raise RuntimeError(message)
        LOGGER.warning(message)
        state.degraded_reasons.append(
            {
                "code": "train_fold_threshold_scope_violation",
                "violation_count": int(threshold_scope_audit.get("violation_count", 0)),
                "sample_violations": threshold_scope_audit.get("sample_violations", []),
            }
        )

    labeled_output = paths.data_processed / "epiclim_labeled.csv"
    labeled_df.to_csv(labeled_output, index=False)
    LOGGER.info("Phase: feature matrix")
    feature_output = paths.data_features / "feature_matrix.csv"
    features_df = feature_callable(
        labeled_df,
        strict_validation=strict_feature_gate,
        write_output=True,
        output_path=feature_output,
    )
    state.artifacts["labeled_data"] = labeled_output
    state.artifacts["feature_matrix"] = feature_output

    labeled_df, features_df, memory_optimization_report = _apply_memory_optimization_filters(
        labeled_df,
        features_df,
        config=memory_optimization_config,
        date_column=effective_cv_config.date_column,
        district_column="district",
    )
    if bool(memory_optimization_report.get("active", False)):
        labeled_df.to_csv(labeled_output, index=False)
        features_df.to_csv(feature_output, index=False)
    memory_report_path = paths.outputs_reports / "memory_optimization_report.json"
    runtime_artifacts.safe_write_json(
        {
            "run_id": run_id,
            **memory_optimization_report,
            "bayesian_subset": asdict(memory_optimization_config.bayesian_subset),
        },
        memory_report_path,
    )
    state.artifacts["memory_optimization_report"] = memory_report_path

    reproducibility_pack = {
        "effective_cli_args": runtime_artifacts.json_compatible(
            cli_args_snapshot
            if cli_args_snapshot is not None
            else {
                "model_config": model_config_path,
                "adapter_config": adapter_config_path,
                "cv_config": cv_config_path,
                "raw_data": raw_data_path,
                "population_data": population_data_path,
                "start_year": start_year,
                "end_year": end_year,
                "selected_percentile": selected_percentile,
                "skip_baselines": skip_baselines,
                "skip_bayesian": skip_bayesian,
                "skip_visualizations": skip_visualizations,
                "decision_cost": decision_cost,
                "decision_loss": decision_loss,
                "lead_time_max_lookback_steps": lead_time_max_lookback_steps,
                "strict_bayesian_deps": strict_bayesian_deps,
                "strict_feature_gate": strict_feature_gate,
                "force_full_bayesian": force_full_bayesian,
                "bayesian_overrides": bayesian_overrides,
                "export_detailed_csv": export_detailed_csv,
                "model_names": model_names,
                "seed": seed,
            }
        ),
        "git_commit_sha": runtime_artifacts.git_commit_sha(paths.root),
        "python_version": sys.version,
        "input_hashes_sha256": {
            "raw_data": runtime_artifacts.sha256_file(raw_data_path),
            "model_config": runtime_artifacts.sha256_file(model_config_path),
            "adapter_config": runtime_artifacts.sha256_file(adapter_config_path) if adapter_config_path is not None else None,
            "cv_config": runtime_artifacts.sha256_file(cv_config_path),
        },
    }

    run_metadata_path = paths.outputs_reports / "run_metadata.json"
    runtime_artifacts.safe_write_json(
        {
            "run_id": run_id,
            "started_at_utc": run_started_at,
            "effective_seed": int(effective_seed),
            "model_config_path": str(model_config_path),
            "adapter_config_path": str(adapter_config_path) if adapter_config_path is not None else None,
            "cv_config_path": str(cv_config_path),
            "effective_cv_config": effective_cv_config_dict,
            "adapter": {
                "enabled": bool(adapter_config),
                "callables": {
                    "load_data": str(adapter_config.get("load_data", "default")),
                    "label_outbreaks": str(adapter_config.get("label_outbreaks", "default")),
                    "build_feature_matrix": str(adapter_config.get("build_feature_matrix", "default")),
                    "build_fold_ledger": str(adapter_config.get("build_fold_ledger", "default")),
                    "generate_time_splits": str(adapter_config.get("generate_time_splits", "default")),
                },
                "brazil_adapter_active": bool(brazil_adapter_active),
            },
            "bayesian_mode_control": {
                "force_full_requested": bool(requested_force_full),
                "force_full_effective": bool(force_full_effective),
                "brazil_adapter_active": bool(brazil_adapter_active),
                "bayesian_simplified_mode": bool(bayesian_settings.get("bayesian_simplified_mode", False)),
                "max_convergence_retries": int(bayesian_settings.get("max_convergence_retries", 0)),
            },
            "compute_backend": backend_resolution,
            "memory_optimization": {
                **asdict(memory_optimization_config),
                "active": bool(memory_optimization_report.get("active", False)),
                "rows_before": int(memory_optimization_report.get("inputs", {}).get("labeled_rows", len(labeled_df))),
                "rows_after": int(memory_optimization_report.get("outputs", {}).get("labeled_rows", len(labeled_df))),
            },
            "reproducibility": reproducibility_pack,
        },
        run_metadata_path,
    )
    state.artifacts["run_metadata"] = run_metadata_path

    feature_quality_gate_path = paths.outputs_reports / "feature_quality_gate_report.json"
    if not feature_quality_gate_path.exists():
        runtime_artifacts.safe_write_json(
            {
                "run_id": run_id,
                "passed": False,
                "strict_mode": bool(strict_feature_gate),
                "flagged_features": [],
                "dropped_features": [],
                "note": "feature quality gate report was not emitted by feature pipeline",
            },
            feature_quality_gate_path,
        )
    state.artifacts["feature_quality_gate_report"] = feature_quality_gate_path

    baseline_result = runtime_baseline.run_baseline_phase(
        state=state,
        paths=paths,
        labeled_df=labeled_df,
        features_df=features_df,
        selected_percentile=selected_percentile,
        effective_cv_config=effective_cv_config,
        effective_seed=effective_seed,
        strict_feature_gate=strict_feature_gate,
        baseline_compute_backend=baseline_backend_effective,
        skip_baselines=skip_baselines,
        export_detailed_csv=export_detailed_csv,
        model_names=model_names,
        lead_time_max_lookback_steps=lead_time_max_lookback_steps,
        threshold_scope_audit=threshold_scope_audit,
        cv_ledger_callable=cv_ledger_callable,
        train_baselines_fn=train_baselines,
        baseline_training_config_cls=BaselineTrainingConfig,
        predict_baselines_fn=predict_baselines,
        evaluate_baseline_predictions_fn=evaluate_baseline_predictions,
        collect_baseline_oof_scores_fn=_collect_baseline_oof_scores,
        collect_baseline_oof_predictions_fn=_collect_baseline_oof_predictions,
        collect_baseline_oof_fold_ids_fn=_collect_baseline_oof_fold_ids,
        safe_write_json_fn=_safe_write_json,
    )

    strict_or_full_bayesian_mode = bool(strict_bayesian_deps or bayesian_settings.get("force_full_bayesian", False))
    bayesian_result = runtime_bayesian.run_bayesian_phase(
        state=state,
        paths=paths,
        model_input_df=baseline_result.model_input_df,
        target=baseline_result.target,
        bayesian_count_target=baseline_result.bayesian_count_target,
        bayesian_threshold_series=baseline_result.bayesian_threshold_series,
        temporal_index=baseline_result.temporal_index,
        district_index=baseline_result.district_index,
        skip_bayesian=skip_bayesian,
        strict_bayesian_deps=strict_bayesian_deps,
        bayesian_settings=bayesian_settings,
        bayesian_compute_backend_requested=str(compute_backend_config.mode),
        bayesian_compute_backend_effective=bayesian_backend_effective,
        effective_cv_config=effective_cv_config,
        lead_time_max_lookback_steps=lead_time_max_lookback_steps,
        export_detailed_csv=export_detailed_csv,
        strict_or_full_bayesian_mode=strict_or_full_bayesian_mode,
        bayesian_subset_config=asdict(memory_optimization_config.bayesian_subset),
        bayesian_subset_seed=int(effective_seed),
        cv_split_callable=cv_split_callable,
        run_bayesian_track_fn=_run_bayesian_track,
        collect_bayesian_oof_scores_fn=_collect_bayesian_oof_scores,
        evaluate_bayesian_predictions_fn=evaluate_bayesian_predictions,
        check_convergence_fn=check_convergence,
        extract_rhat_ess_fn=extract_rhat_ess,
        safe_write_json_fn=_safe_write_json,
    )

    eval_decision_result = runtime_eval_decision.run_evaluation_and_decision_phase(
        state=state,
        paths=paths,
        labeled_df=labeled_df,
        target=baseline_result.target,
        baseline_headline_eligible=baseline_result.baseline_headline_eligible,
        bayesian_headline_eligible=bayesian_result.bayesian_headline_eligible,
        baseline_metrics=baseline_result.baseline_metrics or {},
        bayesian_metrics=bayesian_result.bayesian_metrics or {},
        bayesian_score=bayesian_result.bayesian_score,
        bayesian_oof_score=bayesian_result.bayesian_oof_score,
        baseline_oof_score=baseline_result.baseline_oof_score,
        bayesian_sampling_diagnostics=bayesian_result.bayesian_sampling_diagnostics,
        bayesian_convergence_payload=bayesian_result.bayesian_convergence_payload,
        bayesian_converged=bayesian_result.bayesian_converged,
        temporal_index=baseline_result.temporal_index,
        district_index=baseline_result.district_index,
        run_id=run_id,
        decision_cost=decision_cost,
        decision_loss=decision_loss,
        decision_optimize_threshold=decision_optimize_threshold,
        decision_optimization_risk_basis=decision_optimization_risk_basis,
        decision_optimization_min_samples=decision_optimization_min_samples,
        decision_optimization_grid_size=decision_optimization_grid_size,
        decision_optimization_grid_min=decision_optimization_grid_min,
        decision_optimization_grid_max=decision_optimization_grid_max,
        decision_risk_score_basis=decision_risk_score_basis,
        decision_policy_version=decision_policy_version,
        alert_thresholds=alert_thresholds,
        export_detailed_csv=export_detailed_csv,
        optimize_decision_threshold_fn=optimize_decision_threshold,
        assign_alert_levels_fn=assign_alert_levels,
        export_track_comparison_fn=export_track_comparison,
        build_comparison_table_fn=build_comparison_table,
        safe_write_json_fn=_safe_write_json,
    )

    LOGGER.info("Phase: visualizations")
    if not skip_visualizations:
        _cleanup_legacy_figure_placeholders(paths.outputs_figures)
        legacy_lead_time_plot = paths.outputs_figures / "performance_lead_time_boxplot.png"
        if legacy_lead_time_plot.exists():
            legacy_lead_time_plot.unlink()

        viz_frame = labeled_df.copy()
        viz_frame["risk_score"] = eval_decision_result.decision_frame["risk_score"]

        if {"date", "district"}.issubset(viz_frame.columns):
            try:
                plot_temporal_coverage_heatmap(
                    viz_frame,
                    date_col="date",
                    district_col="district",
                    top_k_districts=20,
                    output_dir=paths.outputs_figures,
                )
            except ValueError as exploratory_error:
                LOGGER.warning("Temporal exploratory plot skipped: %s", exploratory_error)

        if "cases" in viz_frame.columns:
            try:
                plot_case_distribution(viz_frame, case_col="cases", output_dir=paths.outputs_figures)
            except ValueError as exploratory_error:
                LOGGER.warning("Case distribution plot skipped: %s", exploratory_error)

        plot_missingness_summary(viz_frame, output_dir=paths.outputs_figures)

        numeric_feature_frame = features_df.select_dtypes(include=[np.number]).copy()
        if not numeric_feature_frame.empty:
            try:
                plot_correlation_heatmap(numeric_feature_frame, output_dir=paths.outputs_figures)
            except ValueError as feature_error:
                LOGGER.warning("Feature correlation plot skipped: %s", feature_error)

        feature_importances = _extract_feature_importances(baseline_result.baseline_models, feature_names=features_df.columns.tolist())
        if feature_importances is not None:
            try:
                plot_feature_importance(feature_importances, top_k=20, output_dir=paths.outputs_figures)
            except ValueError as feature_error:
                LOGGER.warning("Feature importance plot skipped: %s", feature_error)

        track_b_score = bayesian_result.bayesian_score if bayesian_result.bayesian_score is not None else eval_decision_result.decision_frame["risk_score"]
        plot_residuals(
            baseline_result.target,
            track_b_score,
            filename="trackb_residuals.png",
            output_dir=paths.outputs_figures,
        )

        track_b_array = pd.Series(track_b_score, copy=False).to_numpy(dtype=float)
        if track_b_array.size == 0:
            posterior_predictive_samples = np.zeros((1, 1), dtype=float)
        else:
            max_points = 10000
            if track_b_array.size > max_points:
                rng = np.random.default_rng(effective_seed)
                sampled_idx = np.sort(rng.choice(track_b_array.size, size=max_points, replace=False))
                track_b_array = track_b_array[sampled_idx]
            sample_count = 25
            posterior_predictive_samples = np.broadcast_to(track_b_array, (sample_count, track_b_array.size))
        plot_posterior_predictive_check(
            baseline_result.target,
            posterior_predictive_samples,
            filename="trackb_posterior_predictive_check.png",
            output_dir=paths.outputs_figures,
        )

        if bayesian_result.bayesian_idata is not None:
            plot_trace(
                bayesian_result.bayesian_idata,
                filename="trackb_trace_plot.png",
                output_dir=paths.outputs_figures,
            )

        y_true = baseline_result.target.reset_index(drop=True)
        y_score = pd.Series(track_b_score, copy=False).reset_index(drop=True)
        if y_true.nunique(dropna=True) > 1 and len(y_true) > 1:
            try:
                plot_roc_curve(y_true, y_score, output_dir=paths.outputs_figures)
                plot_pr_curve(y_true, y_score, output_dir=paths.outputs_figures)
                plot_calibration_curve(
                    y_true,
                    y_score,
                    filename="trackb_calibration_curve.png",
                    output_dir=paths.outputs_figures,
                )
            except ValueError as metric_error:
                LOGGER.warning("Performance plots skipped: %s", metric_error)

            y_pred = (y_score >= 0.5).astype(int)
            plot_confusion_matrix_grid(
                y_true,
                predictions={"decision": y_pred},
                output_dir=paths.outputs_figures,
            )

            lead_times = lead_time_steps(
                y_true,
                y_pred,
                max_lookback_steps=lead_time_max_lookback_steps,
                temporal_index=baseline_result.temporal_index,
                district=baseline_result.district_index,
            )
            trackb_lead_rows = pd.DataFrame(
                {
                    "track": ["Track B"] * max(len(lead_times), 1),
                    "lead_time": lead_times.tolist() if not lead_times.empty else [0.0],
                }
            )
            plot_lead_time_boxplot(
                trackb_lead_rows,
                filename="trackb_lead_time_boxplot.png",
                output_dir=paths.outputs_figures,
            )

            brier_value = float(np.mean((y_true.astype(float).to_numpy() - y_score.astype(float).to_numpy()) ** 2))
            lead_time_mean = float(lead_times.mean()) if not lead_times.empty else 0.0
            plot_brier_lead_time_summary(
                brier_score=brier_value,
                lead_time_mean=lead_time_mean,
                output_dir=paths.outputs_figures,
            )

        if eval_decision_result.comparison_table is not None:
            try:
                plot_track_delta_heatmap(eval_decision_result.comparison_table, output_dir=paths.outputs_figures)
                plot_track_comparison_shared_metrics_bar(eval_decision_result.comparison_table, output_dir=paths.outputs_figures)
            except ValueError as comparison_error:
                LOGGER.warning("Track comparison plots skipped: %s", comparison_error)

        if {"date", "alert_level"}.issubset(eval_decision_result.decision_frame.columns):
            try:
                plot_decision_alert_trend(
                    eval_decision_result.decision_frame,
                    date_col="date",
                    alert_col="alert_level",
                    output_dir=paths.outputs_figures,
                )
            except ValueError as alert_trend_error:
                LOGGER.warning("Decision alert trend plot skipped: %s", alert_trend_error)

        if baseline_result.baseline_model_metrics is not None and not baseline_result.baseline_model_metrics.empty:
            try:
                plot_tracka_model_score_comparison(
                    baseline_result.baseline_model_metrics,
                    filename="tracka_models_all_scores.png",
                    output_dir=paths.outputs_figures,
                )
            except ValueError as tracka_error:
                LOGGER.warning("Track A model score comparison plot skipped: %s", tracka_error)

        if {"date", "district"}.issubset(labeled_df.columns):
            risk_plot_frame = pd.DataFrame(
                {
                    "date": pd.to_datetime(labeled_df["date"], errors="coerce"),
                    "district": labeled_df["district"],
                    "risk_score": eval_decision_result.decision_frame["risk_score"],
                    "cases": labeled_df.get("cases", pd.Series(0.0, index=labeled_df.index)),
                }
            )
            risk_plot_frame = risk_plot_frame.dropna(subset=["date", "district", "risk_score"])
            if not risk_plot_frame.empty:
                plot_risk_trajectory(
                    risk_plot_frame,
                    date_col="date",
                    risk_col="risk_score",
                    district_col="district",
                    top_n=10,
                    filename="trackb_risk_trajectory.png",
                    output_dir=paths.outputs_figures,
                )
                plot_top_risk_districts(
                    risk_plot_frame,
                    district_col="district",
                    risk_col="risk_score",
                    top_n=20,
                    output_dir=paths.outputs_figures,
                )

        bayesian_convergence_path = state.artifacts.get("bayesian_convergence")
        if bayesian_convergence_path is not None and Path(bayesian_convergence_path).exists():
            try:
                current_convergence = json.loads(Path(bayesian_convergence_path).read_text(encoding="utf-8"))
                plot_convergence_comparison(
                    current_diagnostics=current_convergence,
                    previous_diagnostics=previous_bayesian_convergence,
                    filename="bayesian_convergence_comparison.png",
                    output_dir=paths.outputs_figures,
                )
                state.artifacts["bayesian_convergence_comparison"] = paths.outputs_figures / "bayesian_convergence_comparison.png"
            except Exception as comparison_error:
                LOGGER.warning("Unable to generate convergence comparison figure: %s", comparison_error)

        _cleanup_legacy_figure_placeholders(paths.outputs_figures)
    else:
        LOGGER.info("Visualization phase skipped by flag")

    run_completed_at = datetime.now(timezone.utc).isoformat()
    run_manifest_path = paths.outputs_reports / "run_manifest.json"
    state.artifacts["run_manifest"] = run_manifest_path

    contract_required_keys = {
        "run_manifest",
        "run_metadata",
        "fold_ledger",
        "degraded_run",
        "baseline_backend_metadata",
        "baseline_metrics",
        "baseline_metrics_fullfit",
        "bayesian_metrics",
        "bayesian_metrics_fullfit",
        "bayesian_risk_metadata",
        "track_comparison_csv",
        "track_comparison_md",
        "decision_alerts",
        "bayesian_convergence_summary_csv",
        "bayesian_convergence_summary_md",
        "feature_quality_gate_report",
        "model_input_leakage_audit",
    }

    runtime_artifacts.safe_write_json(
        {
            "run_id": run_id,
            "started_at_utc": run_started_at,
            "completed_at_utc": run_completed_at,
            "effective_seed": int(effective_seed),
            "effective_cv_config": asdict(effective_cv_config),
            "reproducibility": reproducibility_pack,
            "fold_accounting": {
                "minimum_evaluated_folds": int(effective_cv_config.minimum_evaluated_folds),
                "evaluated_folds": int(baseline_result.baseline_evaluated_fold_count),
                "fold_ledger_path": str(state.artifacts["fold_ledger"]),
            },
            "degraded_run": eval_decision_result.degraded_run_payload,
            "memory_optimization": {
                **memory_optimization_report,
                "bayesian_subset": asdict(memory_optimization_config.bayesian_subset),
            },
            "artifacts": {name: str(path) for name, path in state.artifacts.items()},
            "contract_required_artifacts": sorted(contract_required_keys),
            "headline_claims": {
                "baseline_metrics": bool(baseline_result.baseline_headline_eligible),
                "bayesian_metrics": bool(bayesian_result.bayesian_headline_eligible),
                "track_comparison": bool((eval_decision_result.comparison_table is not None) and (not eval_decision_result.suppress_headline_comparison)),
                "suppressed": bool(eval_decision_result.suppress_headline_comparison),
            },
        },
        run_manifest_path,
    )

    runtime_artifacts.validate_manifest_contract(
        run_manifest_path=run_manifest_path,
        artifacts=state.artifacts,
        required_artifact_keys=contract_required_keys,
        expected_run_id=run_id,
    )

    LOGGER.info("Pipeline completed successfully")
    return state.artifacts


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    run(
        model_config_path=args.model_config,
        adapter_config_path=args.adapter_config,
        cv_config_path=args.cv_config,
        raw_data_path=args.raw_data,
        population_data_path=args.population_data,
        start_year=args.start_year,
        end_year=args.end_year,
        selected_percentile=args.selected_percentile,
        skip_baselines=args.skip_baselines,
        skip_bayesian=args.skip_bayesian,
        skip_visualizations=args.skip_visualizations,
        decision_cost=args.decision_cost,
        decision_loss=args.decision_loss,
        lead_time_max_lookback_steps=args.lead_time_max_lookback_steps,
        strict_bayesian_deps=args.strict_bayesian_deps,
        strict_feature_gate=args.strict_feature_gate,
        force_full_bayesian=args.force_full_bayesian,
        bayesian_overrides={
            "draws": args.bayesian_draws,
            "tune": args.bayesian_tune,
            "chains": args.bayesian_chains,
            "bayesian_progress": args.bayesian_progress,
            "target_accept": args.bayesian_target_accept,
            "max_treedepth": args.bayesian_max_treedepth,
            "bayesian_simplified_mode": args.bayesian_simplified_mode,
            "force_full_bayesian": args.force_full_bayesian,
        },
        export_detailed_csv=args.export_detailed_csv,
        model_names=args.model_names,
        seed=args.seed,
        cli_args_snapshot=vars(args),
    )


if __name__ == "__main__":
    main()
