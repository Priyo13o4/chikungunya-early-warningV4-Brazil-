"""End-to-end orchestration for the chikungunya early warning pipeline."""

from __future__ import annotations

import os
import site
import sys

# --- JAX GPU Path Fix ---
# Dynamically inject pip-installed NVIDIA library paths into LD_LIBRARY_PATH
# This must happen before any JAX or PyMC imports so that XLA can find cuSPARSE, cuDNN, etc.
try:
    site_packages = site.getsitepackages()[0]
    nvidia_dir = os.path.join(site_packages, 'nvidia')
    if os.path.exists(nvidia_dir):
        nvidia_libs = [
            os.path.join(nvidia_dir, d, 'lib')
            for d in os.listdir(nvidia_dir)
            if os.path.isdir(os.path.join(nvidia_dir, d, 'lib'))
        ]
        if nvidia_libs:
            new_ld_path = ':'.join(nvidia_libs)
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{current_ld_path}" if current_ld_path else new_ld_path
            
    # Force JAX to only look for CUDA and CPU, suppressing the TPU warning
    os.environ['JAX_PLATFORMS'] = 'cuda,cpu'
except Exception as e:
    print(f"Warning: Failed to inject NVIDIA library paths: {e}")
# ------------------------

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import json
import logging
from pathlib import Path
import re
from time import perf_counter
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
from src.evaluation.metrics_baselines import evaluate_baseline_predictions
from src.evaluation.metrics_bayesian import evaluate_bayesian_predictions
from src.feature_engineering.build_feature_matrix import build_feature_matrix
from src.models.baselines.predict_baselines import predict_baselines
from src.models.baselines.train_baselines import BaselineTrainingConfig, train_baselines
from src.models.baselines.model_registry import list_default_model_names
from src.models.baselines.cv_splitter import TimeSeriesCVConfig, build_fold_ledger, generate_time_splits
from src.models.bayesian.diagnostics import check_convergence, extract_rhat_ess
from src.pipeline_runtime import config_runtime as runtime_config
from src.pipeline_runtime import compute_backend as runtime_backend
from src.pipeline_runtime import curated_contract as runtime_curated_contract
from src.pipeline_runtime import io_artifacts as runtime_artifacts
from src.pipeline_runtime import memory_filters as runtime_memory_filters
from src.pipeline_runtime.phase_context import SharedPhaseState
from src.pipeline_runtime import phases_baseline as runtime_baseline
from src.pipeline_runtime import phases_bayesian as runtime_bayesian
from src.pipeline_runtime import phases_eval_decision as runtime_eval_decision
from src.pipeline_runtime import phases_visualization as runtime_visualization

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
_DEFAULT_CURATED_MUNICIPALITIES_PATH = Path("resources/curated_municipalities_v1.json")
_BAYESIAN_OOF_HARD_FAIL_MARKERS: tuple[str, ...] = (
    "cv statistical gate failure",
    "statistical_gate_failed",
    "missing required climate covariates",
    "pipeline must provide the configured bayesian covariate set explicitly",
)


def _find_forbidden_feature_columns(
    feature_df: pd.DataFrame,
    *,
    target_column: str = "outbreak_label",
) -> list[str]:
    forbidden: list[str] = []
    for column in feature_df.columns:
        lowered = str(column).lower()
        pattern_forbidden = any(pattern.match(str(column)) for pattern in _FORBIDDEN_COLUMN_PATTERNS)
        raw_case_alias = lowered in _RAW_CASE_TARGET_ALIASES
        direct_target_alias = lowered == target_column.lower()
        if pattern_forbidden or raw_case_alias or direct_target_alias:
            forbidden.append(str(column))
    return sorted(set(forbidden))


def _assert_no_forbidden_feature_columns(
    feature_df: pd.DataFrame,
    *,
    target_column: str = "outbreak_label",
    strict: bool = True,
) -> None:
    forbidden = _find_forbidden_feature_columns(feature_df, target_column=target_column)
    if forbidden:
        message = (
            "Feature matrix contains forbidden/leaky columns. "
            f"forbidden_columns={forbidden}"
        )
        if strict:
            raise RuntimeError(f"{message} Cannot proceed in strict mode.")
        LOGGER.warning("%s Proceeding because strict_feature_gate is disabled.", message)


def _load_curated_municipality_contract(contract_path: Path) -> dict[str, Any]:
    return runtime_curated_contract.load_curated_municipality_contract(contract_path)


def _apply_curated_municipality_filter(
    df: pd.DataFrame,
    *,
    curated_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return runtime_curated_contract.apply_curated_municipality_filter(df, curated_ids=curated_ids)


def _build_bayesian_config(strict_dependencies: bool, bayesian_settings: dict[str, Any]) -> Any:
    from src.models.bayesian.hierarchical_model import BayesianModelConfig

    raw_convergence_failure_mode = bayesian_settings.get("convergence_failure_mode", BayesianModelConfig.convergence_failure_mode)
    convergence_failure_mode = str(raw_convergence_failure_mode).strip().lower()
    if convergence_failure_mode not in {"strict", "warn"}:
        LOGGER.warning(
            "Invalid bayesian_model.convergence_failure_mode '%s'; using '%s'",
            raw_convergence_failure_mode,
            BayesianModelConfig.convergence_failure_mode,
        )
        convergence_failure_mode = BayesianModelConfig.convergence_failure_mode

    sampling_backend = runtime_config.normalize_sampling_backend(
        bayesian_settings.get("sampling_backend", BayesianModelConfig.sampling_backend)
    )
    raw_covariates = bayesian_settings.get("climate_covariates", BayesianModelConfig.climate_covariates)
    if isinstance(raw_covariates, (list, tuple)):
        climate_covariates = tuple(str(value).strip() for value in raw_covariates if str(value).strip())
    else:
        climate_covariates = BayesianModelConfig.climate_covariates
    if not climate_covariates:
        LOGGER.warning(
            "Invalid bayesian_model.climate_covariates '%s'; using defaults",
            raw_covariates,
        )
        climate_covariates = BayesianModelConfig.climate_covariates

    return BayesianModelConfig(
        climate_covariates=climate_covariates,
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
        convergence_failure_mode=convergence_failure_mode,
        sampling_backend=sampling_backend,
        outbreak_threshold_default_cases=float(
            bayesian_settings.get(
                "outbreak_threshold_default_cases",
                BayesianModelConfig.outbreak_threshold_default_cases,
            )
        ),
        posterior_sample_cap=int(
            bayesian_settings.get("posterior_sample_cap", BayesianModelConfig.posterior_sample_cap)
        ),
        predictive_chunk_rows=int(
            bayesian_settings.get("predictive_chunk_rows", BayesianModelConfig.predictive_chunk_rows)
        ),
    )


def _call_build_feature_matrix_compat(
    feature_callable: Callable[..., Any],
    df: pd.DataFrame,
    *,
    strict_validation: bool,
    write_output: bool,
    output_path: Path,
    enable_quality_gate: bool,
    quality_gate_report_path: Path,
) -> pd.DataFrame:
    signature = inspect.signature(feature_callable)
    parameters = signature.parameters
    accepts_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())

    kwargs: dict[str, Any] = {
        "strict_validation": strict_validation,
        "write_output": write_output,
        "output_path": output_path,
        "enable_quality_gate": enable_quality_gate,
        "quality_gate_report_path": quality_gate_report_path,
    }
    if not accepts_var_kwargs:
        kwargs = {key: value for key, value in kwargs.items() if key in parameters}

    try:
        result = feature_callable(df, **kwargs)
    except TypeError as call_error:
        unsupported_kw_error = "unexpected keyword" in str(call_error).lower()
        if kwargs and unsupported_kw_error:
            fallback_kwargs = {
                "strict_validation": strict_validation,
                "write_output": write_output,
                "output_path": output_path,
            }
            if not accepts_var_kwargs:
                fallback_kwargs = {key: value for key, value in fallback_kwargs.items() if key in parameters}
            LOGGER.info(
                "Feature callable rejected quality-gate args (%s); falling back to legacy feature call shape",
                call_error,
            )
            result = feature_callable(df, **fallback_kwargs)
        else:
            raise

    if not isinstance(result, pd.DataFrame):
        raise RuntimeError("Feature callable returned non-DataFrame output")
    return result


def _apply_memory_optimization_filters(
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    *,
    config: runtime_config.MemoryOptimizationConfig,
    date_column: str = "date",
    district_column: str = "district",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    return runtime_memory_filters.apply_memory_optimization_filters(
        labeled_df,
        features_df,
        config=config,
        date_column=date_column,
        district_column=district_column,
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
    payload = runtime_artifacts.json_compatible(data)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


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
    sampling_backend_requested = runtime_config.normalize_sampling_backend(
        bayesian_settings.get("sampling_backend", "auto")
    )
    convergence_failure_mode = str(bayesian_settings.get("convergence_failure_mode", "strict")).strip().lower()
    LOGGER.info(
        "Bayesian phase start: backend requested=%s, resolved=%s, sampling_backend=%s, convergence_failure_mode=%s",
        backend_requested,
        backend_effective,
        sampling_backend_requested,
        convergence_failure_mode,
    )
    configured_covariates = list(
        runtime_config.resolve_bayesian_climate_covariates(bayesian_settings=bayesian_settings)
    )

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

        bayesian_model = HierarchicalBayesianModel(config=config).fit(
            features_df,
            count_target,
            compute_backend_effective=backend_effective,
        )
        risk_frame, predictive_metadata = bayesian_model.predict_with_uncertainty(
            features_df,
            outbreak_threshold=outbreak_threshold,
        )
        sampling_diag = dict(getattr(bayesian_model, "sampling_diagnostics_", {}) or {})
        bayesian_runtime_backend = str(sampling_diag.get("actual_runtime_backend", "cpu"))
        sampling_backend_effective = str(sampling_diag.get("sampling_backend_effective", "pymc"))
        sampling_backend_fallback_reason = sampling_diag.get("sampling_backend_fallback_reason")
        backend_implemented = bool(sampling_diag.get("backend_implemented", True))
        backend_fallback_reason = (
            sampling_backend_fallback_reason
            if sampling_backend_fallback_reason is not None
            else (
                f"Bayesian implementation does not currently support '{backend_effective}' execution; using CPU runtime path"
                if not backend_implemented
                else None
            )
        )
        fallback_used = bool(float(bayesian_model.diagnostics_summary_.get("fallback", 0.0)) > 0.0)
        mode_used = "fallback" if fallback_used else ("simplified" if bayesian_model.simplified_used_ else "full_latent_ar")
        diagnostics = {
            "climate_covariates": configured_covariates,
            "simplified_mode": bool(bayesian_model.simplified_used_),
            "force_full_bayesian": bool(config.force_full_bayesian),
            "mode_used": mode_used,
            "fallback_used": fallback_used,
            "degraded_mode": bool(predictive_metadata.get("degraded_mode", False) or fallback_used),
            "requested_backend": backend_requested,
            "resolved_backend": backend_effective,
            "actual_runtime_backend": bayesian_runtime_backend,
            "backend_implemented": backend_implemented,
            "fallback_reason": backend_fallback_reason,
            "sampling_backend_requested": sampling_backend_requested,
            "sampling_backend_effective": sampling_backend_effective,
            "sampling_backend_fallback_reason": sampling_backend_fallback_reason,
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
        bayesian_runtime_backend = "cpu"
        sampling_backend_effective = "pymc"
        sampling_backend_fallback_reason = f"missing optional dependencies ({import_error})"
        backend_implemented = bool(backend_effective == "cpu")
        backend_fallback_reason = (
            sampling_backend_fallback_reason
            if backend_effective != "cpu"
            else sampling_backend_fallback_reason
        )
        return None, None, {
            "climate_covariates": configured_covariates,
            "degraded_mode": True,
            "fallback_used": True,
            "mode_used": "missing_dependencies",
            "degraded_reason": "missing_optional_dependencies",
            "error": str(import_error),
            "requested_backend": backend_requested,
            "resolved_backend": backend_effective,
            "actual_runtime_backend": bayesian_runtime_backend,
            "backend_implemented": backend_implemented,
            "fallback_reason": backend_fallback_reason,
            "sampling_backend_requested": sampling_backend_requested,
            "sampling_backend_effective": sampling_backend_effective,
            "sampling_backend_fallback_reason": sampling_backend_fallback_reason,
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
    compute_backend_effective: str = "cpu",
) -> pd.Series:
    from src.models.bayesian.hierarchical_model import HierarchicalBayesianModel

    oof = pd.Series(np.nan, index=features_df.index, dtype="float64")

    cv_effective = cv_config
    if not str(cv_effective.date_column).strip() or not str(cv_effective.target_column).strip():
        cv_payload = asdict(cv_effective)
        if not str(cv_payload.get("date_column", "")).strip():
            cv_payload["date_column"] = date_column
        if not str(cv_payload.get("target_column", "")).strip():
            cv_payload["target_column"] = target_column
        cv_effective = TimeSeriesCVConfig(**cv_payload)

    cv_frame = features_df.copy()
    cv_frame[cv_effective.target_column] = pd.to_numeric(outbreak_target, errors="coerce").fillna(0).astype(int)
    requested_covariates = list(runtime_config.resolve_bayesian_climate_covariates(bayesian_settings=bayesian_settings))

    for train_idx, valid_idx in generate_time_splits_fn(cv_frame, cv_effective):
        y_train_binary = pd.to_numeric(outbreak_target.loc[train_idx], errors="coerce").fillna(0).astype(int)
        y_train_counts = pd.to_numeric(count_target.loc[train_idx], errors="coerce").fillna(0.0)
        if y_train_binary.nunique(dropna=True) <= 1:
            continue
        try:
            fold_train_features = features_df.loc[train_idx]
            fold_valid_features = features_df.loc[valid_idx]

            train_selection = runtime_config.select_bayesian_covariates_by_availability(
                frame=fold_train_features,
                requested_covariates=requested_covariates,
            )
            train_selected_covariates = list(train_selection.get("selected_covariates", []))
            if not train_selected_covariates:
                LOGGER.warning(
                    "Bayesian OOF fold skipped: no viable covariates in train split after availability alignment. requested=%s excluded=%s",
                    train_selection.get("requested_covariates", requested_covariates),
                    train_selection.get("excluded_covariates", []),
                )
                continue

            valid_selection = runtime_config.select_bayesian_covariates_by_availability(
                frame=fold_valid_features,
                requested_covariates=train_selected_covariates,
            )
            fold_selected_covariates = list(valid_selection.get("selected_covariates", []))
            if not fold_selected_covariates:
                LOGGER.warning(
                    "Bayesian OOF fold skipped: no viable covariates in validation split after availability alignment. train_selected=%s excluded=%s",
                    train_selected_covariates,
                    valid_selection.get("excluded_covariates", []),
                )
                continue

            fold_settings = dict(bayesian_settings)
            fold_settings["climate_covariates"] = fold_selected_covariates

            model = HierarchicalBayesianModel(config=_build_bayesian_config(strict_dependencies, fold_settings))
            model.fit(fold_train_features, y_train_counts, compute_backend_effective=compute_backend_effective)
            fold_threshold = threshold_series.loc[valid_idx] if threshold_series is not None else None
            fold_pred = model.predict_with_uncertainty(
                fold_valid_features,
                outbreak_threshold=fold_threshold,
            )[0]["risk_mean"].clip(0.0, 1.0)
            oof.loc[valid_idx] = fold_pred.astype(float)
        except Exception as fold_error:
            fold_error_message = str(fold_error).strip().lower()
            hard_fail_violation = any(marker in fold_error_message for marker in _BAYESIAN_OOF_HARD_FAIL_MARKERS)
            if hard_fail_violation:
                raise RuntimeError(
                    f"Bayesian OOF fold failed due to strict contract/gate violation: {fold_error}"
                ) from fold_error
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
    strict_feature_gate: bool = True,
    force_full_bayesian: bool = False,
    bayesian_overrides: dict[str, Any] | None = None,
    bayesian_profile_mode: str | None = None,
    export_detailed_csv: bool = False,
    model_names: list[str] | None = None,
    seed: int | None = None,
    cli_args_snapshot: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Execute pipeline stages in the required end-to-end order."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{uuid4().hex[:8]}"
    run_started_at = datetime.now(timezone.utc).isoformat()
    run_started_perf = perf_counter()

    phase_starts: dict[str, float] = {}

    def _phase_context_suffix(**context: Any) -> str:
        if not context:
            return ""
        rendered = ", ".join(f"{key}={value}" for key, value in context.items())
        return f" | {rendered}"

    def _phase_start(name: str, **context: Any) -> None:
        phase_starts[name] = perf_counter()
        LOGGER.info("Phase[%s] START%s", name, _phase_context_suffix(**context))

    def _phase_end(name: str, **context: Any) -> None:
        started_at = phase_starts.pop(name, None)
        elapsed = (perf_counter() - started_at) if started_at is not None else float("nan")
        elapsed_fragment = f"elapsed={elapsed:.2f}s"
        LOGGER.info("Phase[%s] END | %s%s", name, elapsed_fragment, _phase_context_suffix(**context))

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
    bayesian_profiles = raw_model_config.get("bayesian_model_profiles", {})
    if not isinstance(bayesian_profiles, dict):
        bayesian_profiles = {}
    runtime_settings = raw_model_config.get("runtime", {})
    if not isinstance(runtime_settings, dict):
        runtime_settings = {}
    sampling_backend_runtime_value = runtime_settings.get("sampling_backend")
    sampling_backend_model_value = bayesian_settings.get("sampling_backend")
    sampling_backend_requested_source = "bayesian_model"
    sampling_backend_requested_raw = sampling_backend_model_value
    if sampling_backend_requested_raw is None and sampling_backend_runtime_value is not None:
        sampling_backend_requested_source = "runtime"
        sampling_backend_requested_raw = sampling_backend_runtime_value
    if sampling_backend_requested_raw is None:
        sampling_backend_requested_source = "default"
        sampling_backend_requested_raw = "auto"
    bayesian_settings["sampling_backend"] = runtime_config.normalize_sampling_backend(sampling_backend_requested_raw)

    # Resolve two concrete Bayesian configs:
    # - fullfit: final model pass
    # - cv: OOF/CV pass (typically lighter for iterative testing)
    bayesian_settings_fullfit, bayesian_settings_cv, bayesian_profile_usage = runtime_config.resolve_bayesian_profile_settings(
        bayesian_settings=bayesian_settings,
        bayesian_profiles=bayesian_profiles,
        profile_mode=bayesian_profile_mode,
    )

    # CLI overrides are applied to both paths so tests and fullfit stay in sync when explicitly overridden.
    if bayesian_overrides:
        non_null_overrides = {key: value for key, value in bayesian_overrides.items() if value is not None}
        if non_null_overrides:
            bayesian_settings_fullfit.update(non_null_overrides)
            bayesian_settings_cv.update(non_null_overrides)

    bayesian_settings_fullfit["random_seed"] = effective_seed
    bayesian_settings_cv["random_seed"] = effective_seed
    bayesian_settings_fullfit["sampling_backend"] = runtime_config.normalize_sampling_backend(
        bayesian_settings_fullfit.get("sampling_backend", "auto")
    )
    bayesian_settings_cv["sampling_backend"] = runtime_config.normalize_sampling_backend(
        bayesian_settings_cv.get("sampling_backend", "auto")
    )

    cv_subset_mode_active = False
    memory_optimization_config = runtime_config.parse_memory_optimization_config(raw_model_config)
    compute_backend_config = runtime_config.parse_compute_backend_config(raw_model_config)
    backend_resolution = runtime_backend.resolve_backends(compute_backend_config)
    baseline_backend_effective = str(backend_resolution.get("baseline_backend", "cpu"))
    bayesian_backend_effective = str(backend_resolution.get("bayesian_backend", "cpu"))
    bayesian_requested_backend = str(compute_backend_config.mode)
    bayesian_resolved_backend = str(bayesian_backend_effective)
    bayesian_actual_runtime_backend = "cpu"
    bayesian_backend_implemented = bool(bayesian_actual_runtime_backend == bayesian_resolved_backend)
    bayesian_backend_fallback_reason: str | None = None
    if not bayesian_backend_implemented:
        bayesian_backend_fallback_reason = (
            f"Bayesian implementation does not currently support '{bayesian_resolved_backend}' execution; using CPU runtime path"
        )
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

    brazil_adapter_active = runtime_config.is_brazil_adapter_config_active(adapter_config)
    requested_force_full = bool(
        force_full_bayesian
        or bayesian_settings_fullfit.get("force_full_bayesian", False)
        or adapter_config.get("force_full_bayesian_project_only", False)
    )
    force_full_effective = bool(requested_force_full and brazil_adapter_active)
    if requested_force_full and not brazil_adapter_active:
        LOGGER.warning(
            "Ignoring force_full_bayesian request because Brazil adapter config is not active; "
            "falling back to configured non-forced Bayesian mode."
        )
    bayesian_settings_fullfit["force_full_bayesian"] = bool(force_full_effective)
    bayesian_settings_cv["force_full_bayesian"] = bool(force_full_effective)
    if force_full_effective:
        bayesian_settings_fullfit["bayesian_simplified_mode"] = False
        bayesian_settings_fullfit["max_convergence_retries"] = 0
        bayesian_settings_cv["bayesian_simplified_mode"] = False
        bayesian_settings_cv["max_convergence_retries"] = 0
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

    _phase_start("load", start_year=int(start_year), end_year=int(end_year))
    raw_df, population_df = runtime_config.call_load_data_compat(
        load_data_callable,
        raw_data_path,
        population_data_path,
        start_year=start_year,
        end_year=end_year,
        discovery_dir=raw_data_path.parent,
    )
    _phase_end(
        "load",
        raw_rows=int(len(raw_df)),
        population_rows=int(len(population_df)) if population_df is not None else 0,
    )

    curated_contract_path = Path(
        str(raw_model_config.get("curated_municipalities_path", _DEFAULT_CURATED_MUNICIPALITIES_PATH))
    )
    _phase_start("clean")
    cleaned_df = clean_data(raw_df, start_year=start_year, end_year=end_year)
    _phase_end("clean", rows=int(len(cleaned_df)))

    _phase_start("impute")
    imputed_df = impute_climate(cleaned_df)
    _phase_end("impute", rows=int(len(imputed_df)))

    _phase_start("merge", population_attached=bool(population_df is not None))
    merged_df = imputed_df
    if population_df is not None:
        merged_df = merge_population(imputed_df, population_df)
    else:
        LOGGER.info("No population data provided; merge phase completed with passthrough frame")

    if "date" in merged_df.columns:
        merged_df = merged_df.copy()
        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
    if "district" in merged_df.columns and "date" in merged_df.columns:
        before_rows = int(len(merged_df))
        merged_df = (
            merged_df.sort_values(["district", "date"], ascending=[True, True], na_position="last")
            .drop_duplicates(subset=["district", "date"], keep="last")
            .reset_index(drop=True)
        )
        dropped_rows = int(before_rows - len(merged_df))
        if dropped_rows > 0:
            LOGGER.info(
                "Pre-label canonicalization removed %s duplicate district/date rows",
                dropped_rows,
            )
    _phase_end("merge", rows=int(len(merged_df)))

    curated_contract = _load_curated_municipality_contract(curated_contract_path)
    curated_municipality_ids = set(curated_contract["municipality_ids"])

    label_input_df, curated_filter_report = _apply_curated_municipality_filter(
        merged_df,
        curated_ids=curated_municipality_ids,
    )
    if label_input_df.empty:
        raise RuntimeError(
            "Curated municipality filtering removed all rows before labeling. "
            f"contract_path={curated_contract_path}"
        )

    _phase_start("labels", selected_percentile=int(selected_percentile))
    labeled_df = runtime_config.call_label_outbreaks_compat(
        label_callable,
        label_input_df,
        selected_percentile=selected_percentile,
        use_percentile_labels=True,
        cv_config=effective_cv_config,
        strict_mode=bool(strict_feature_gate),
    )
    _phase_end("labels", rows=int(len(labeled_df)))

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
    _phase_start("feature_matrix")
    feature_quality_gate_path = paths.outputs_reports / "feature_quality_gate_report.json"
    feature_output = paths.data_features / "feature_matrix.csv"
    features_df = _call_build_feature_matrix_compat(
        feature_callable,
        labeled_df,
        strict_validation=strict_feature_gate,
        write_output=True,
        output_path=feature_output,
        enable_quality_gate=True,
        quality_gate_report_path=feature_quality_gate_path,
    )
    _assert_no_forbidden_feature_columns(
        features_df,
        target_column="outbreak_label",
        strict=bool(strict_feature_gate),
    )
    _phase_end("feature_matrix", rows=int(len(features_df)), columns=int(features_df.shape[1]))
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
                "bayesian_profile_mode": bayesian_profile_mode,
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
            "curated_municipalities_contract": curated_contract.get("sha256"),
        },
    }

    run_metadata_path = paths.outputs_reports / "run_metadata.json"
    run_metadata_payload: dict[str, Any] = {
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
                "bayesian_simplified_mode": bool(bayesian_settings_fullfit.get("bayesian_simplified_mode", False)),
                "max_convergence_retries": int(bayesian_settings_fullfit.get("max_convergence_retries", 0)),
                "convergence_failure_mode": str(bayesian_settings_fullfit.get("convergence_failure_mode", "warn")),
            },
            "bayesian_backend": {
                "requested_backend": bayesian_requested_backend,
                "resolved_backend": bayesian_resolved_backend,
                "actual_runtime_backend": bayesian_actual_runtime_backend,
                "backend_implemented": bool(bayesian_backend_implemented),
                "fallback_reason": bayesian_backend_fallback_reason,
                "sampling_backend_requested": str(bayesian_settings_fullfit.get("sampling_backend", "auto")),
                "sampling_backend_effective": "pymc",
                "sampling_backend_fallback_reason": bayesian_backend_fallback_reason,
                "sampling_backend_requested_source": sampling_backend_requested_source,
            },
            "sampling_backend_requested": str(bayesian_settings_fullfit.get("sampling_backend", "auto")),
            "sampling_backend_effective": "pymc",
            "sampling_backend_fallback_reason": bayesian_backend_fallback_reason,
            "bayesian_profile_usage": {
                **bayesian_profile_usage,
                "cv_subset_mode_active": bool(cv_subset_mode_active),
            },
            "compute_backend": backend_resolution,
            "memory_optimization": {
                **asdict(memory_optimization_config),
                "active": bool(memory_optimization_report.get("active", False)),
                "rows_before": int(memory_optimization_report.get("inputs", {}).get("labeled_rows", len(labeled_df))),
                "rows_after": int(memory_optimization_report.get("outputs", {}).get("labeled_rows", len(labeled_df))),
            },
            "curated_municipalities": {
                "path": str(curated_contract_path),
                "version": str(curated_contract.get("version", "v1")),
                "source": str(curated_contract.get("source", "unknown")),
                "count": int(curated_contract.get("count", 0)),
                "sha256": curated_contract.get("sha256"),
                "filter_report": curated_filter_report,
            },
            "reproducibility": reproducibility_pack,
    }
    runtime_artifacts.safe_write_json(
        run_metadata_payload,
        run_metadata_path,
    )
    state.artifacts["run_metadata"] = run_metadata_path

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

    _phase_start("baseline", skip=bool(skip_baselines))
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
        cv_split_callable=cv_split_callable,
        train_baselines_fn=train_baselines,
        baseline_training_config_cls=BaselineTrainingConfig,
        predict_baselines_fn=predict_baselines,
        evaluate_baseline_predictions_fn=evaluate_baseline_predictions,
        collect_baseline_oof_scores_fn=_collect_baseline_oof_scores,
        collect_baseline_oof_predictions_fn=_collect_baseline_oof_predictions,
        collect_baseline_oof_fold_ids_fn=_collect_baseline_oof_fold_ids,
        safe_write_json_fn=_safe_write_json,
    )
    _phase_end(
        "baseline",
        headline_eligible=bool(baseline_result.baseline_headline_eligible),
        evaluated_folds=int(baseline_result.baseline_evaluated_fold_count),
    )

    strict_or_full_bayesian_mode = bool(strict_bayesian_deps or bayesian_settings_fullfit.get("force_full_bayesian", False))
    _phase_start("bayesian", skip=bool(skip_bayesian))
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
        bayesian_settings_fullfit=bayesian_settings_fullfit,
        bayesian_settings_cv=bayesian_settings_cv,
        bayesian_profile_usage=bayesian_profile_usage,
        bayesian_compute_backend_requested=str(compute_backend_config.mode),
        bayesian_compute_backend_effective=bayesian_backend_effective,
        effective_cv_config=effective_cv_config,
        lead_time_max_lookback_steps=lead_time_max_lookback_steps,
        export_detailed_csv=export_detailed_csv,
        strict_or_full_bayesian_mode=strict_or_full_bayesian_mode,
        bayesian_subset_config=asdict(memory_optimization_config.bayesian_subset),
        cv_subset_mode_active=bool(cv_subset_mode_active),
        bayesian_subset_seed=int(effective_seed),
        cv_split_callable=cv_split_callable,
        run_bayesian_track_fn=_run_bayesian_track,
        collect_bayesian_oof_scores_fn=_collect_bayesian_oof_scores,
        evaluate_bayesian_predictions_fn=evaluate_bayesian_predictions,
        check_convergence_fn=check_convergence,
        extract_rhat_ess_fn=extract_rhat_ess,
        safe_write_json_fn=_safe_write_json,
    )
    _phase_end(
        "bayesian",
        headline_eligible=bool(bayesian_result.bayesian_headline_eligible),
        converged=bool(bayesian_result.bayesian_converged),
    )

    bayesian_sampling_diagnostics = bayesian_result.bayesian_sampling_diagnostics or {}
    run_metadata_payload["sampling_backend_requested"] = str(
        bayesian_sampling_diagnostics.get("sampling_backend_requested", bayesian_settings.get("sampling_backend", "auto"))
    )
    run_metadata_payload["sampling_backend_effective"] = str(
        bayesian_sampling_diagnostics.get("sampling_backend_effective", "pymc")
    )
    run_metadata_payload["sampling_backend_fallback_reason"] = bayesian_sampling_diagnostics.get(
        "sampling_backend_fallback_reason"
    )
    covariates_effective_raw = bayesian_sampling_diagnostics.get("climate_covariates")
    if covariates_effective_raw is None:
        covariates_effective = []
    else:
        covariates_effective = list(covariates_effective_raw)

    if (
        "climate_covariates_requested" in bayesian_sampling_diagnostics
        and bayesian_sampling_diagnostics.get("climate_covariates_requested") is not None
    ):
        covariates_requested = list(bayesian_sampling_diagnostics.get("climate_covariates_requested", []))
    else:
        covariates_requested = list(covariates_effective)

    run_metadata_payload["bayesian_covariates_effective"] = covariates_effective
    run_metadata_payload["bayesian_covariates_requested"] = covariates_requested
    run_metadata_payload["bayesian_covariate_selection"] = bayesian_sampling_diagnostics.get(
        "covariate_selection",
        {},
    )
    run_metadata_payload["bayesian_backend"] = {
        **run_metadata_payload.get("bayesian_backend", {}),
        "actual_runtime_backend": str(bayesian_sampling_diagnostics.get("actual_runtime_backend", "cpu")),
        "backend_implemented": bool(bayesian_sampling_diagnostics.get("backend_implemented", True)),
        "fallback_reason": bayesian_sampling_diagnostics.get("fallback_reason"),
        "sampling_backend_requested": run_metadata_payload["sampling_backend_requested"],
        "sampling_backend_effective": run_metadata_payload["sampling_backend_effective"],
        "sampling_backend_fallback_reason": run_metadata_payload["sampling_backend_fallback_reason"],
    }
    run_metadata_payload["bayesian_profile_usage"] = {
        **run_metadata_payload.get("bayesian_profile_usage", {}),
        **(bayesian_sampling_diagnostics.get("bayesian_profile_usage") or {}),
        "cv_subset_mode_active": bool(
            bayesian_sampling_diagnostics.get("cv_subset_mode_active", cv_subset_mode_active)
        ),
    }
    runtime_artifacts.safe_write_json(run_metadata_payload, run_metadata_path)

    fold_ledger_path = state.artifacts.get("fold_ledger")
    if fold_ledger_path is not None and Path(fold_ledger_path).exists():
        try:
            fold_ledger_payload = json.loads(Path(fold_ledger_path).read_text(encoding="utf-8"))
        except Exception as fold_ledger_error:
            LOGGER.warning("Unable to load fold_ledger artifact for covariate parity update: %s", fold_ledger_error)
        else:
            if isinstance(fold_ledger_payload, dict):
                fold_ledger_payload["bayesian_covariates_requested"] = list(covariates_requested)
                fold_ledger_payload["bayesian_covariates_effective"] = list(covariates_effective)
                fold_ledger_payload["bayesian_covariate_selection"] = run_metadata_payload.get(
                    "bayesian_covariate_selection",
                    {},
                )
                runtime_artifacts.safe_write_json(fold_ledger_payload, Path(fold_ledger_path))

    _phase_start("evaluation_decision")
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
        bayesian_profile_usage=run_metadata_payload.get("bayesian_profile_usage", {}),
        cv_subset_mode_active=bool(
            run_metadata_payload.get("bayesian_profile_usage", {}).get("cv_subset_mode_active", cv_subset_mode_active)
        ),
        bayesian_convergence_payload=bayesian_result.bayesian_convergence_payload,
        bayesian_converged=bayesian_result.bayesian_converged,
        temporal_index=baseline_result.temporal_index,
        district_index=baseline_result.district_index,
        run_id=run_id,
        bayesian_covariates_requested=list(covariates_requested),
        bayesian_covariates_effective=list(covariates_effective),
        bayesian_covariate_selection=run_metadata_payload.get("bayesian_covariate_selection", {}),
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
    _phase_end(
        "evaluation_decision",
        decision_rows=int(len(eval_decision_result.decision_frame)),
        suppress_headline=bool(eval_decision_result.suppress_headline_comparison),
    )

    _phase_start("visualization_payload")
    bayesian_convergence_path = state.artifacts.get("bayesian_convergence")
    payload_csv_path, payload_metadata_path = runtime_visualization.build_visualization_payload(
        paths=paths,
        run_id=run_id,
        labeled_df=labeled_df,
        features_df=features_df,
        baseline_result=baseline_result,
        bayesian_result=bayesian_result,
        eval_decision_result=eval_decision_result,
        bayesian_convergence_path=bayesian_convergence_path,
        safe_write_json_fn=_safe_write_json,
    )
    state.artifacts["visualization_payload_csv"] = payload_csv_path
    state.artifacts["visualization_payload_metadata"] = payload_metadata_path
    _phase_end("visualization_payload", payload_rows=int(len(labeled_df)))

    _phase_start("visualizations", skip=bool(skip_visualizations))
    if not skip_visualizations:
        visualization_artifacts = runtime_visualization.run_visualization_phase(
            paths=paths,
            run_id=run_id,
            labeled_df=labeled_df,
            features_df=features_df,
            baseline_result=baseline_result,
            bayesian_result=bayesian_result,
            eval_decision_result=eval_decision_result,
            previous_bayesian_convergence=previous_bayesian_convergence,
            lead_time_max_lookback_steps=lead_time_max_lookback_steps,
            effective_seed=effective_seed,
            bayesian_convergence_path=bayesian_convergence_path,
        )
        state.artifacts.update(visualization_artifacts)
    else:
        LOGGER.info(
            "Visualization phase skipped by flag; payload available for standalone run at %s",
            payload_csv_path,
        )
    _phase_end("visualizations")

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
            "curated_municipalities": {
                "path": str(curated_contract_path),
                "version": str(curated_contract.get("version", "v1")),
                "source": str(curated_contract.get("source", "unknown")),
                "count": int(curated_contract.get("count", 0)),
                "sha256": curated_contract.get("sha256"),
                "filter_report": curated_filter_report,
            },
            "artifacts": {name: str(path) for name, path in state.artifacts.items()},
            "contract_required_artifacts": sorted(contract_required_keys),
            "bayesian_covariates": {
                "requested": list(run_metadata_payload.get("bayesian_covariates_requested", [])),
                "effective": list(run_metadata_payload.get("bayesian_covariates_effective", [])),
                "selection": run_metadata_payload.get("bayesian_covariate_selection", {}),
            },
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

    LOGGER.info("Pipeline completed successfully | total_elapsed=%.2fs", (perf_counter() - run_started_perf))
    return state.artifacts


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
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
        bayesian_profile_mode=args.bayesian_profile_mode,
        export_detailed_csv=args.export_detailed_csv,
        model_names=args.model_names,
        seed=args.seed,
        cli_args_snapshot=vars(args),
    )


if __name__ == "__main__":
    main()
