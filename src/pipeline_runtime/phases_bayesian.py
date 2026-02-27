from __future__ import annotations

from dataclasses import asdict
import inspect
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.baselines.cv_splitter import TimeSeriesCVConfig, generate_time_splits
from src.pipeline_runtime.config_runtime import (
    build_bayesian_config,
    resolve_bayesian_climate_covariates,
    select_bayesian_covariates_by_availability,
)
from src.pipeline_runtime.io_artifacts import build_suppressed_metric_payload
from src.pipeline_runtime.phase_context import BayesianPhaseResult, SharedPhaseState

LOGGER = logging.getLogger(__name__)

_BAYESIAN_OOF_HARD_FAIL_MARKERS: tuple[str, ...] = (
    "cv statistical gate failure",
    "statistical_gate_failed",
    "missing required climate covariates",
    "pipeline must provide the configured bayesian covariate set explicitly",
)


def _build_bayesian_backend_metadata(*, requested_backend: str, resolved_backend: str) -> dict[str, Any]:
    requested = str(requested_backend or "cpu")
    resolved = str(resolved_backend or "cpu")
    actual_runtime_backend = "cpu"
    backend_implemented = bool(actual_runtime_backend == resolved)
    fallback_reason: str | None = None
    if not backend_implemented:
        fallback_reason = (
            f"Bayesian implementation does not currently support '{resolved}' execution; using CPU runtime path"
        )
    return {
        "requested_backend": requested,
        "resolved_backend": resolved,
        "actual_runtime_backend": actual_runtime_backend,
        "backend_implemented": backend_implemented,
        "fallback_reason": fallback_reason,
    }


def _resolve_convergence_failure_mode(
    *,
    bayesian_settings: dict[str, Any],
    strict_or_full_bayesian_mode: bool,
) -> tuple[str, bool, str | None]:
    configured_mode = bayesian_settings.get("convergence_failure_mode")
    configured_explicitly = configured_mode is not None

    if configured_explicitly:
        normalized = str(configured_mode).strip().lower()
        if normalized not in {"strict", "warn"}:
            LOGGER.warning(
                "Invalid bayesian_model.convergence_failure_mode '%s'; using strict",
                configured_mode,
            )
            normalized = "strict"
        return normalized, True, None

    default_mode = "strict" if bool(strict_or_full_bayesian_mode) else "warn"
    draws = int(bayesian_settings.get("draws", 0) or 0)
    chains = int(bayesian_settings.get("chains", 0) or 0)
    if chains < 2 or draws < 50:
        reason = (
            "auto-downgraded to warn: low-sample smoke settings "
            f"(chains={chains}, draws={draws}) with no explicit convergence_failure_mode"
        )
        return "warn", False, reason
    return default_mode, False, None


def _select_bayesian_subset_index(
    *,
    model_input_df: pd.DataFrame,
    target: pd.Series,
    bayesian_count_target: pd.Series,
    config: dict[str, Any],
    seed: int,
    date_column: str = "date",
) -> tuple[pd.Index, dict[str, Any]]:
    normalized: dict[str, Any] = config if isinstance(config, dict) else {}
    enabled = bool(normalized.get("enabled", False))
    max_rows = int(normalized.get("max_rows", 0) or 0)
    strategy = str(normalized.get("strategy", "recent_years")).strip().lower()
    if strategy not in {"recent_years", "top_cases", "random_stratified"}:
        strategy = "recent_years"

    full_index = pd.Index(model_input_df.index)
    metadata = {
        "enabled": enabled,
        "strategy": strategy,
        "max_rows": int(max_rows),
        "source_rows": int(len(full_index)),
        "selected_rows": int(len(full_index)),
        "selection_seed": int(seed),
        "selection_reproducibility_key": f"strategy={strategy}|max_rows={max_rows}|seed={seed}",
        "applied": False,
    }

    if not enabled or max_rows <= 0 or len(full_index) <= max_rows:
        return full_index, metadata

    if strategy == "top_cases":
        scores = pd.to_numeric(bayesian_count_target, errors="coerce").fillna(0.0)
        selected_index = scores.sort_values(ascending=False).head(max_rows).index
    elif strategy == "random_stratified":
        rng = np.random.default_rng(int(seed))
        y = pd.to_numeric(target, errors="coerce").fillna(0).astype(int)
        y = y.reindex(full_index)
        candidates = pd.Series(index=full_index, data=False, dtype="bool")
        class_values = sorted(int(value) for value in y.dropna().unique())
        allocated = 0
        for class_value in class_values:
            class_idx = y.index[y == class_value]
            if len(class_idx) == 0:
                continue
            share = max(1, int(round(max_rows * (len(class_idx) / max(len(full_index), 1)))))
            share = min(share, len(class_idx))
            sampled = rng.choice(np.asarray(class_idx), size=int(share), replace=False)
            candidates.loc[pd.Index(sampled)] = True
            allocated += int(share)
        if allocated < max_rows:
            remaining_pool = candidates.index[~candidates]
            if len(remaining_pool) > 0:
                top_up = min(max_rows - allocated, len(remaining_pool))
                sampled = rng.choice(np.asarray(remaining_pool), size=int(top_up), replace=False)
                candidates.loc[pd.Index(sampled)] = True
        selected_index = candidates[candidates].index[:max_rows]
    else:
        if date_column in model_input_df.columns:
            ordering = pd.to_datetime(model_input_df[date_column], errors="coerce")
        else:
            ordering = pd.Series(np.arange(len(full_index)), index=full_index, dtype="float64")
        selected_index = ordering.sort_values(ascending=False).head(max_rows).index

    selected_index = pd.Index(selected_index)
    metadata.update(
        {
            "applied": True,
            "selected_rows": int(len(selected_index)),
            "coverage": float(len(selected_index) / max(len(full_index), 1)),
        }
    )
    return selected_index, metadata


def run_bayesian_track(
    features_df: pd.DataFrame,
    count_target: pd.Series,
    *,
    outbreak_threshold: pd.Series | None,
    strict_dependencies: bool,
    bayesian_settings: dict[str, Any],
    compute_backend_requested: str = "cpu",
    compute_backend_effective: str = "cpu",
) -> tuple[pd.DataFrame | None, Any | None, dict[str, Any]]:
    backend_metadata = _build_bayesian_backend_metadata(
        requested_backend=str(compute_backend_requested or "cpu"),
        resolved_backend=str(compute_backend_effective or "cpu"),
    )
    sampling_backend_requested = str(bayesian_settings.get("sampling_backend", "auto")).strip().lower()
    convergence_failure_mode = str(bayesian_settings.get("convergence_failure_mode", "strict")).strip().lower()
    LOGGER.info(
        "Bayesian phase start: backend requested=%s, resolved=%s, sampling_backend=%s, convergence_failure_mode=%s",
        backend_metadata["requested_backend"],
        backend_metadata["resolved_backend"],
        sampling_backend_requested,
        convergence_failure_mode,
    )
    if backend_metadata["fallback_reason"]:
        LOGGER.warning("%s", backend_metadata["fallback_reason"])

    configured_covariates: list[str] = list(
        resolve_bayesian_climate_covariates(bayesian_settings=bayesian_settings)
    )
    try:
        from src.models.bayesian.hierarchical_model import BayesianModelConfig, HierarchicalBayesianModel

        config = build_bayesian_config(strict_dependencies, bayesian_settings)
        configured_covariates = list(config.climate_covariates)

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
            compute_backend_effective=str(backend_metadata["resolved_backend"]),
        )
        risk_frame, predictive_metadata = bayesian_model.predict_with_uncertainty(
            features_df,
            outbreak_threshold=outbreak_threshold,
        )
        sampling_diag = dict(getattr(bayesian_model, "sampling_diagnostics_", {}) or {})
        actual_runtime_backend = str(sampling_diag.get("actual_runtime_backend", backend_metadata["actual_runtime_backend"]))
        sampling_backend_effective = str(sampling_diag.get("sampling_backend_effective", "pymc"))
        sampling_backend_fallback_reason = sampling_diag.get("sampling_backend_fallback_reason")
        backend_implemented = bool(sampling_diag.get("backend_implemented", backend_metadata["backend_implemented"]))
        fallback_reason = sampling_backend_fallback_reason or backend_metadata["fallback_reason"]
        if fallback_reason is not None:
            actual_runtime_backend = "cpu"
        fallback_used = bool(float(bayesian_model.diagnostics_summary_.get("fallback", 0.0)) > 0.0)
        mode_used = "fallback" if fallback_used else ("simplified" if bayesian_model.simplified_used_ else "full_latent_ar")
        diagnostics = {
            "climate_covariates": configured_covariates,
            "simplified_mode": bool(bayesian_model.simplified_used_),
            "force_full_bayesian": bool(config.force_full_bayesian),
            "mode_used": mode_used,
            "fallback_used": fallback_used,
            "degraded_mode": bool(predictive_metadata.get("degraded_mode", False) or fallback_used),
            "requested_backend": backend_metadata["requested_backend"],
            "resolved_backend": backend_metadata["resolved_backend"],
            "actual_runtime_backend": actual_runtime_backend,
            "backend_implemented": backend_implemented,
            "fallback_reason": fallback_reason,
            "sampling_backend_requested": sampling_backend_requested,
            "sampling_backend_effective": sampling_backend_effective,
            "sampling_backend_fallback_reason": sampling_backend_fallback_reason,
            "compute_backend_requested": backend_metadata["requested_backend"],
            "compute_backend_effective": backend_metadata["resolved_backend"],
            "compute_backend_runtime": actual_runtime_backend,
            "compute_backend_fallback_used": bool(fallback_reason is not None),
            "compute_backend_fallback_reason": fallback_reason,
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
        fallback_reason = f"missing optional dependencies ({import_error})"
        return None, None, {
            "climate_covariates": configured_covariates,
            "degraded_mode": True,
            "fallback_used": True,
            "mode_used": "missing_dependencies",
            "degraded_reason": "missing_optional_dependencies",
            "error": str(import_error),
            "requested_backend": backend_metadata["requested_backend"],
            "resolved_backend": backend_metadata["resolved_backend"],
            "actual_runtime_backend": backend_metadata["actual_runtime_backend"],
            "backend_implemented": bool(backend_metadata["backend_implemented"]),
            "fallback_reason": fallback_reason,
            "sampling_backend_requested": sampling_backend_requested,
            "sampling_backend_effective": "pymc",
            "sampling_backend_fallback_reason": fallback_reason,
            "compute_backend_requested": backend_metadata["requested_backend"],
            "compute_backend_effective": backend_metadata["resolved_backend"],
            "compute_backend_runtime": backend_metadata["actual_runtime_backend"],
            "compute_backend_fallback_used": True,
            "compute_backend_fallback_reason": fallback_reason,
        }


def collect_bayesian_oof_scores(
    *,
    features_df: pd.DataFrame,
    outbreak_target: pd.Series,
    count_target: pd.Series,
    strict_dependencies: bool,
    bayesian_settings: dict[str, Any],
    cv_config: TimeSeriesCVConfig,
    threshold_series: pd.Series | None = None,
    fail_on_error: bool = False,
    compute_backend_effective: str = "cpu",
    date_column: str = "date",
    target_column: str = "outbreak_label",
    generate_time_splits_fn: Callable[[pd.DataFrame, TimeSeriesCVConfig], Any] = generate_time_splits,
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
    requested_covariates = list(resolve_bayesian_climate_covariates(bayesian_settings=bayesian_settings))

    for train_idx, valid_idx in generate_time_splits_fn(cv_frame, cv_effective):
        y_train_binary = pd.to_numeric(outbreak_target.loc[train_idx], errors="coerce").fillna(0).astype(int)
        y_train_counts = pd.to_numeric(count_target.loc[train_idx], errors="coerce").fillna(0.0)
        if y_train_binary.nunique(dropna=True) <= 1:
            continue
        try:
            fold_train_features = features_df.loc[train_idx]
            fold_valid_features = features_df.loc[valid_idx]

            train_selection = select_bayesian_covariates_by_availability(
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

            valid_selection = select_bayesian_covariates_by_availability(
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

            model = HierarchicalBayesianModel(config=build_bayesian_config(strict_dependencies, fold_settings))
            model.fit(
                fold_train_features,
                y_train_counts,
                compute_backend_effective=compute_backend_effective,
            )
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


def run_bayesian_phase(
    *,
    state: SharedPhaseState,
    paths: Any,
    model_input_df: pd.DataFrame,
    target: pd.Series,
    bayesian_count_target: pd.Series,
    bayesian_threshold_series: pd.Series,
    temporal_index: pd.Series | None,
    district_index: pd.Series | None,
    skip_bayesian: bool,
    strict_bayesian_deps: bool,
    bayesian_settings_fullfit: dict[str, Any],
    bayesian_settings_cv: dict[str, Any],
    bayesian_profile_usage: dict[str, Any],
    bayesian_compute_backend_requested: str,
    bayesian_compute_backend_effective: str,
    effective_cv_config: TimeSeriesCVConfig,
    lead_time_max_lookback_steps: int,
    export_detailed_csv: bool,
    strict_or_full_bayesian_mode: bool,
    bayesian_subset_config: dict[str, Any],
    cv_subset_mode_active: bool,
    bayesian_subset_seed: int,
    cv_split_callable: Callable[..., Any],
    run_bayesian_track_fn: Callable[..., tuple[pd.DataFrame | None, Any | None, dict[str, Any]]],
    collect_bayesian_oof_scores_fn: Callable[..., pd.Series],
    evaluate_bayesian_predictions_fn: Callable[..., dict[str, float]],
    check_convergence_fn: Callable[..., dict[str, Any]],
    extract_rhat_ess_fn: Callable[..., pd.DataFrame],
    safe_write_json_fn: Callable[[dict[str, Any], Path], None],
) -> BayesianPhaseResult:
    LOGGER.info("Phase: bayesian")
    bayesian_score: pd.Series | None = None
    bayesian_risk_frame: pd.DataFrame | None = None
    bayesian_oof_score: pd.Series | None = None
    bayesian_idata: Any | None = None
    configured_covariates = list(
        resolve_bayesian_climate_covariates(bayesian_settings=bayesian_settings_fullfit)
    )
    default_covariate_selection: dict[str, Any] = {
        "requested_covariates": list(configured_covariates),
        "selected_covariates": list(configured_covariates),
        "excluded_covariates": [],
        "missing_covariates": [],
        "required_covariates": ["month", "year", "weekofyear"],
        "required_covariates_present": True,
        "viable_count": int(len(configured_covariates)),
        "requested_count": int(len(configured_covariates)),
    }
    bayesian_sampling_diagnostics: dict[str, Any] = {
        "climate_covariates": configured_covariates,
        "climate_covariates_requested": list(configured_covariates),
        "covariate_selection": default_covariate_selection,
        "mode_used": "not_run" if skip_bayesian else "pending",
        "degraded_mode": False,
        "fallback_used": False,
    }
    bayesian_metrics: dict[str, float] | None = None
    bayesian_metrics_fullfit: dict[str, float] | None = None
    bayesian_headline_eligible = False
    bayesian_convergence_payload: dict[str, Any] | None = None
    bayesian_converged: bool | None = None
    suppressed_fullfit_reason = "bayesian_not_available"
    suppressed_oof_reason = "bayesian_headline_not_available"
    convergence_failure_mode, convergence_mode_explicit, convergence_mode_auto_reason = _resolve_convergence_failure_mode(
        bayesian_settings=bayesian_settings_fullfit,
        strict_or_full_bayesian_mode=bool(strict_or_full_bayesian_mode),
    )
    bayesian_settings_fullfit["convergence_failure_mode"] = convergence_failure_mode
    bayesian_settings_cv["convergence_failure_mode"] = convergence_failure_mode
    if convergence_mode_auto_reason is not None:
        LOGGER.warning("Bayesian convergence failure policy %s", convergence_mode_auto_reason)

    if not skip_bayesian:
        subset_index, subset_metadata = _select_bayesian_subset_index(
            model_input_df=model_input_df,
            target=target,
            bayesian_count_target=bayesian_count_target,
            config=bayesian_subset_config,
            seed=int(bayesian_subset_seed),
            date_column=effective_cv_config.date_column or "date",
        )
        subset_model_input_df = model_input_df.loc[subset_index]
        subset_target = target.loc[subset_index]
        subset_count_target = bayesian_count_target.loc[subset_index]
        subset_threshold_series = bayesian_threshold_series.loc[subset_index]

        covariate_selection = select_bayesian_covariates_by_availability(
            frame=subset_model_input_df,
            requested_covariates=configured_covariates,
        )
        selected_covariates = list(covariate_selection.get("selected_covariates", []))
        bayesian_sampling_diagnostics["climate_covariates_requested"] = list(
            covariate_selection.get("requested_covariates", configured_covariates)
        )
        bayesian_sampling_diagnostics["covariate_selection"] = covariate_selection
        bayesian_sampling_diagnostics["climate_covariates"] = selected_covariates

        if not selected_covariates:
            LOGGER.warning(
                "Bayesian track suppressed: no viable climate covariates after availability-first alignment. requested=%s excluded=%s",
                covariate_selection.get("requested_covariates", []),
                covariate_selection.get("excluded_covariates", []),
            )
            state.degraded_reasons.append(
                {
                    "code": "bayesian_no_viable_covariates",
                    "reason": "no_viable_climate_covariates_after_alignment",
                    "requested_covariates": list(covariate_selection.get("requested_covariates", [])),
                    "excluded_covariates": list(covariate_selection.get("excluded_covariates", [])),
                }
            )
            suppressed_fullfit_reason = "bayesian_no_viable_covariates"
            suppressed_oof_reason = "bayesian_no_viable_covariates"
            bayesian_sampling_diagnostics.update(
                {
                    "degraded_mode": True,
                    "fallback_used": True,
                    "mode_used": "suppressed_no_viable_covariates",
                    "degraded_reason": "bayesian_no_viable_covariates",
                    "error": "No Bayesian climate covariates passed availability-first alignment.",
                }
            )
        else:
            bayesian_settings_fullfit["climate_covariates"] = list(selected_covariates)
            bayesian_settings_cv["climate_covariates"] = list(selected_covariates)

            bayes_track_kwargs: dict[str, Any] = {
                "outbreak_threshold": subset_threshold_series,
                "strict_dependencies": strict_bayesian_deps,
                "bayesian_settings": bayesian_settings_fullfit,
            }
            bayes_track_signature = inspect.signature(run_bayesian_track_fn)
            if "compute_backend_requested" in bayes_track_signature.parameters:
                bayes_track_kwargs["compute_backend_requested"] = bayesian_compute_backend_requested
            if "compute_backend_effective" in bayes_track_signature.parameters:
                bayes_track_kwargs["compute_backend_effective"] = bayesian_compute_backend_effective

            bayesian_risk_frame, bayesian_idata, bayesian_sampling_diagnostics = run_bayesian_track_fn(
                subset_model_input_df,
                subset_count_target,
                **bayes_track_kwargs,
            )
            bayesian_sampling_diagnostics["covariate_selection"] = covariate_selection
            bayesian_sampling_diagnostics["climate_covariates_requested"] = list(
                covariate_selection.get("requested_covariates", configured_covariates)
            )
            bayesian_sampling_diagnostics["climate_covariates"] = list(
                bayesian_sampling_diagnostics.get("climate_covariates", selected_covariates)
            )
        bayesian_sampling_diagnostics["bayesian_subset"] = subset_metadata
        if "climate_covariates" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["climate_covariates"] = configured_covariates
        if "compute_backend_requested" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["compute_backend_requested"] = str(bayesian_compute_backend_requested)
        if "compute_backend_effective" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["compute_backend_effective"] = str(bayesian_compute_backend_effective)
        if "compute_backend_runtime" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["compute_backend_runtime"] = "cpu"
        if "requested_backend" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["requested_backend"] = str(
                bayesian_sampling_diagnostics.get("compute_backend_requested", bayesian_compute_backend_requested)
            )
        if "resolved_backend" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["resolved_backend"] = str(
                bayesian_sampling_diagnostics.get("compute_backend_effective", bayesian_compute_backend_effective)
            )
        if "actual_runtime_backend" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["actual_runtime_backend"] = str(
                bayesian_sampling_diagnostics.get("compute_backend_runtime", "cpu")
            )
        if "backend_implemented" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["backend_implemented"] = bool(
                str(bayesian_sampling_diagnostics.get("actual_runtime_backend", "cpu"))
                == str(bayesian_sampling_diagnostics.get("resolved_backend", "cpu"))
            )
        if "fallback_reason" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["fallback_reason"] = (
                f"Bayesian implementation does not currently support '{bayesian_sampling_diagnostics.get('resolved_backend', 'cpu')}' execution; using CPU runtime path"
                if not bool(bayesian_sampling_diagnostics.get("backend_implemented", True))
                else None
            )
        if "compute_backend_fallback_used" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["compute_backend_fallback_used"] = bool(
                bayesian_sampling_diagnostics.get("fallback_reason") is not None
            )
        if "compute_backend_fallback_reason" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["compute_backend_fallback_reason"] = (
                bayesian_sampling_diagnostics.get("fallback_reason")
            )
        if bayesian_sampling_diagnostics.get("compute_backend_fallback_reason") is not None:
            bayesian_sampling_diagnostics["actual_runtime_backend"] = "cpu"
            bayesian_sampling_diagnostics["compute_backend_runtime"] = "cpu"
        if "sampling_backend_requested" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["sampling_backend_requested"] = str(
                bayesian_settings_fullfit.get("sampling_backend", "auto")
            )
        if "sampling_backend_effective" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["sampling_backend_effective"] = "pymc"
        if "sampling_backend_fallback_reason" not in bayesian_sampling_diagnostics:
            bayesian_sampling_diagnostics["sampling_backend_fallback_reason"] = bayesian_sampling_diagnostics.get(
                "fallback_reason"
            )
        bayesian_sampling_diagnostics["convergence_failure_mode"] = convergence_failure_mode
        bayesian_sampling_diagnostics["convergence_failure_mode_explicit"] = bool(convergence_mode_explicit)
        bayesian_sampling_diagnostics["convergence_failure_mode_auto_reason"] = convergence_mode_auto_reason
        bayesian_sampling_diagnostics["bayesian_profile_usage"] = dict(bayesian_profile_usage)
        bayesian_sampling_diagnostics["cv_subset_mode_active"] = bool(cv_subset_mode_active)
        if bool(bayesian_sampling_diagnostics.get("degraded_mode", False)):
            state.degraded_reasons.append(
                {
                    "code": "bayesian_degraded_mode",
                    "mode_used": str(bayesian_sampling_diagnostics.get("mode_used", "unknown")),
                    "reason": str(bayesian_sampling_diagnostics.get("degraded_reason", "degraded_mode")),
                }
            )
        if bayesian_risk_frame is not None and not bayesian_risk_frame.empty:
            bayesian_subset_score = pd.to_numeric(bayesian_risk_frame["risk_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
            bayesian_score = pd.Series(0.0, index=model_input_df.index, dtype="float64")
            assign_count = min(len(subset_index), len(bayesian_subset_score))
            if assign_count < len(subset_index):
                LOGGER.warning(
                    "Bayesian subset prediction count (%d) differs from subset rows (%d); truncating assignment",
                    len(bayesian_subset_score),
                    len(subset_index),
                )
            bayesian_score.loc[subset_index[:assign_count]] = bayesian_subset_score.iloc[:assign_count].to_numpy(dtype=float)

            risk_intervals_path = paths.outputs_metrics / "bayesian_risk_intervals.csv"
            bayesian_risk_frame.to_csv(risk_intervals_path, index=False)
            state.artifacts["bayesian_risk_intervals"] = risk_intervals_path

            if export_detailed_csv:
                bayesian_path = paths.outputs_models / "detailed" / "bayesian_risk.csv"
                bayesian_path.parent.mkdir(parents=True, exist_ok=True)
                bayesian_risk_frame.to_csv(bayesian_path, index=False)
                state.artifacts["bayesian_risk"] = bayesian_path

            bayesian_metrics_fullfit = evaluate_bayesian_predictions_fn(
                subset_target,
                bayesian_subset_score,
                max_lookback_steps=lead_time_max_lookback_steps,
                temporal_index=temporal_index.loc[subset_index] if temporal_index is not None else None,
                district=district_index.loc[subset_index] if district_index is not None else None,
            )
            safe_write_json_fn(bayesian_metrics_fullfit, paths.outputs_metrics / "bayesian_metrics_fullfit.json")
            state.artifacts["bayesian_metrics_fullfit"] = paths.outputs_metrics / "bayesian_metrics_fullfit.json"

            if not bool(bayesian_sampling_diagnostics.get("degraded_mode", False)):
                bayes_oof_kwargs: dict[str, Any] = {
                    "features_df": subset_model_input_df,
                    "outbreak_target": subset_target,
                    "count_target": subset_count_target,
                    "strict_dependencies": strict_bayesian_deps,
                    "bayesian_settings": bayesian_settings_cv,
                    "cv_config": effective_cv_config,
                    "threshold_series": subset_threshold_series,
                    "fail_on_error": True,
                }
                if "generate_time_splits_fn" in inspect.signature(collect_bayesian_oof_scores_fn).parameters:
                    bayes_oof_kwargs["generate_time_splits_fn"] = cv_split_callable
                if "compute_backend_effective" in inspect.signature(collect_bayesian_oof_scores_fn).parameters:
                    bayes_oof_kwargs["compute_backend_effective"] = bayesian_compute_backend_effective
                bayesian_oof_subset = collect_bayesian_oof_scores_fn(**bayes_oof_kwargs)
                bayesian_oof_score = pd.Series(np.nan, index=model_input_df.index, dtype="float64")
                bayesian_oof_score.loc[bayesian_oof_subset.index] = bayesian_oof_subset.astype(float).to_numpy()
                valid_bayes_oof_mask = bayesian_oof_score.notna()
                if valid_bayes_oof_mask.any():
                    bayesian_metrics = evaluate_bayesian_predictions_fn(
                        target.loc[valid_bayes_oof_mask],
                        bayesian_oof_score.loc[valid_bayes_oof_mask],
                        max_lookback_steps=lead_time_max_lookback_steps,
                        temporal_index=temporal_index.loc[valid_bayes_oof_mask] if temporal_index is not None else None,
                        district=district_index.loc[valid_bayes_oof_mask] if district_index is not None else None,
                    )
                    safe_write_json_fn(bayesian_metrics, paths.outputs_metrics / "bayesian_metrics.json")
                    state.artifacts["bayesian_metrics"] = paths.outputs_metrics / "bayesian_metrics.json"
                    bayesian_headline_eligible = True
                else:
                    LOGGER.warning("No Bayesian OOF predictions available; headline Bayesian metrics not produced.")
                    state.degraded_reasons.append({"code": "bayesian_no_oof_predictions"})
            else:
                LOGGER.warning("Bayesian track is in degraded/fallback mode; headline Bayesian OOF metrics suppressed.")

            if bayesian_idata is not None:
                bayesian_diag_dir = paths.outputs_models / "bayesian" / "diagnostics"
                bayesian_diag_dir.mkdir(parents=True, exist_ok=True)

                convergence = check_convergence_fn(
                    bayesian_idata,
                    divergence_threshold=float(bayesian_settings_fullfit.get("divergence_warn_threshold", 0.0)),
                    rhat_threshold=float(bayesian_settings_fullfit.get("rhat_warn_threshold", 1.05)),
                    ess_threshold=float(bayesian_settings_fullfit.get("ess_warn_threshold", 200.0)),
                    max_tree_depth_threshold=float(bayesian_settings_fullfit.get("max_treedepth", 12)),
                )
                convergence.update(
                    {
                        "simplified_mode": bool(bayesian_sampling_diagnostics.get("simplified_mode", False)),
                        "mode_used": str(bayesian_sampling_diagnostics.get("mode_used", "unknown")),
                        "force_full_bayesian": bool(bayesian_sampling_diagnostics.get("force_full_bayesian", False)),
                        "fallback_used": bool(bayesian_sampling_diagnostics.get("fallback_used", False)),
                        "degraded_mode": bool(bayesian_sampling_diagnostics.get("degraded_mode", False)),
                        "threshold_basis": str(bayesian_sampling_diagnostics.get("threshold_basis", "default")),
                        "threshold_column": bayesian_sampling_diagnostics.get("threshold_column"),
                        "threshold_default": float(bayesian_sampling_diagnostics.get("threshold_default", 1.0)),
                        "configured_draws": int(bayesian_settings_fullfit.get("draws", 0)),
                        "configured_tune": int(bayesian_settings_fullfit.get("tune", 0)),
                        "configured_chains": int(bayesian_settings_fullfit.get("chains", 0)),
                        "configured_bayesian_progress": bool(bayesian_settings_fullfit.get("bayesian_progress", True)),
                        "configured_target_accept": float(bayesian_settings_fullfit.get("target_accept", 0.0)),
                        "configured_max_treedepth": int(bayesian_settings_fullfit.get("max_treedepth", 0)),
                        "convergence_failure_mode": convergence_failure_mode,
                        "convergence_failure_mode_explicit": bool(convergence_mode_explicit),
                        "convergence_failure_mode_auto_reason": convergence_mode_auto_reason,
                        "requested_backend": str(
                            bayesian_sampling_diagnostics.get("requested_backend", "cpu")
                        ),
                        "resolved_backend": str(
                            bayesian_sampling_diagnostics.get("resolved_backend", "cpu")
                        ),
                        "actual_runtime_backend": str(
                            bayesian_sampling_diagnostics.get("actual_runtime_backend", "cpu")
                        ),
                        "backend_implemented": bool(
                            bayesian_sampling_diagnostics.get("backend_implemented", True)
                        ),
                        "fallback_reason": bayesian_sampling_diagnostics.get("fallback_reason"),
                        "sampling_backend_requested": str(
                            bayesian_sampling_diagnostics.get("sampling_backend_requested", "auto")
                        ),
                        "sampling_backend_effective": str(
                            bayesian_sampling_diagnostics.get("sampling_backend_effective", "pymc")
                        ),
                        "sampling_backend_fallback_reason": bayesian_sampling_diagnostics.get(
                            "sampling_backend_fallback_reason"
                        ),
                        "compute_backend_requested": str(
                            bayesian_sampling_diagnostics.get("compute_backend_requested", "cpu")
                        ),
                        "compute_backend_effective": str(
                            bayesian_sampling_diagnostics.get("compute_backend_effective", "cpu")
                        ),
                        "compute_backend_runtime": str(
                            bayesian_sampling_diagnostics.get("compute_backend_runtime", "cpu")
                        ),
                        "compute_backend_fallback_used": bool(
                            bayesian_sampling_diagnostics.get("compute_backend_fallback_used", False)
                        ),
                        "compute_backend_fallback_reason": bayesian_sampling_diagnostics.get(
                            "compute_backend_fallback_reason"
                        ),
                    }
                )
                safe_write_json_fn(convergence, bayesian_diag_dir / "convergence.json")
                bayesian_convergence_payload = convergence
                state.artifacts["bayesian_convergence"] = bayesian_diag_dir / "convergence.json"

                mode_artifact = {
                    "requested_mode": "full_latent_ar"
                    if bool(bayesian_settings_fullfit.get("force_full_bayesian", False))
                    else ("simplified" if bool(bayesian_settings_fullfit.get("bayesian_simplified_mode", False)) else "auto"),
                    "used_mode": str(convergence.get("mode_used", "unknown")),
                    "force_full_bayesian": bool(convergence.get("force_full_bayesian", False)),
                    "simplified_mode": bool(convergence.get("simplified_mode", False)),
                    "fallback_used": bool(convergence.get("fallback_used", False)),
                    "degraded_mode": bool(convergence.get("degraded_mode", False)),
                    "threshold_basis": str(convergence.get("threshold_basis", "default")),
                    "threshold_column": convergence.get("threshold_column"),
                    "threshold_default": float(convergence.get("threshold_default", 1.0)),
                }
                safe_write_json_fn(mode_artifact, bayesian_diag_dir / "mode.json")
                state.artifacts["bayesian_mode"] = bayesian_diag_dir / "mode.json"

                if not convergence.get("converged", False):
                    bayesian_converged = False
                    state.degraded_reasons.append(
                        {
                            "code": "bayesian_convergence_failed",
                            "mode_used": str(convergence.get("mode_used", "unknown")),
                            "reason": "convergence_not_met",
                        }
                    )
                    if convergence_failure_mode == "strict":
                        raise RuntimeError("Bayesian convergence check failed under strict/full mode")
                    bayesian_headline_eligible = False
                    bayesian_metrics = build_suppressed_metric_payload(
                        run_id=state.run_id,
                        track="bayesian_oof",
                        reason="bayesian_convergence_not_met",
                    )
                    safe_write_json_fn(bayesian_metrics, paths.outputs_metrics / "bayesian_metrics.json")
                    state.artifacts["bayesian_metrics"] = paths.outputs_metrics / "bayesian_metrics.json"
                    LOGGER.warning(
                        "Bayesian convergence failure handled in '%s' mode: divergences=%.0f, max_tree_depth=%.0f, r_hat_max=%.4f, ess_min=%.1f",
                        convergence_failure_mode,
                        convergence["divergences"],
                        convergence["max_tree_depth"],
                        convergence["r_hat_max"],
                        convergence["ess_min"],
                    )
                else:
                    bayesian_converged = True

                try:
                    diagnostics_frame = extract_rhat_ess_fn(bayesian_idata)
                    diagnostics_csv_path = bayesian_diag_dir / "rhat_ess.csv"
                    diagnostics_frame.to_csv(diagnostics_csv_path, index=False)
                    state.artifacts["bayesian_rhat_ess"] = diagnostics_csv_path
                except ImportError as diagnostics_error:
                    LOGGER.warning("Unable to export detailed Bayesian diagnostics: %s", diagnostics_error)
    else:
        LOGGER.info("Bayesian phase skipped by flag")

    bayesian_metrics_fullfit_path = paths.outputs_metrics / "bayesian_metrics_fullfit.json"
    if bayesian_metrics_fullfit is None:
        bayesian_metrics_fullfit = build_suppressed_metric_payload(
            run_id=state.run_id,
            track="bayesian_fullfit",
            reason=suppressed_fullfit_reason,
        )
        safe_write_json_fn(bayesian_metrics_fullfit, bayesian_metrics_fullfit_path)
    state.artifacts["bayesian_metrics_fullfit"] = bayesian_metrics_fullfit_path

    bayesian_metrics_path = paths.outputs_metrics / "bayesian_metrics.json"
    if bayesian_metrics is None:
        bayesian_metrics = build_suppressed_metric_payload(
            run_id=state.run_id,
            track="bayesian_oof",
            reason=suppressed_oof_reason,
        )
        safe_write_json_fn(bayesian_metrics, bayesian_metrics_path)
    state.artifacts["bayesian_metrics"] = bayesian_metrics_path

    return BayesianPhaseResult(
        bayesian_score=bayesian_score,
        bayesian_risk_frame=bayesian_risk_frame,
        bayesian_oof_score=bayesian_oof_score,
        bayesian_idata=bayesian_idata,
        bayesian_sampling_diagnostics=bayesian_sampling_diagnostics,
        bayesian_metrics=bayesian_metrics,
        bayesian_metrics_fullfit=bayesian_metrics_fullfit,
        bayesian_headline_eligible=bayesian_headline_eligible,
        bayesian_convergence_payload=bayesian_convergence_payload,
        bayesian_converged=bayesian_converged,
    )
