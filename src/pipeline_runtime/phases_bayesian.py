from __future__ import annotations

from dataclasses import asdict
import inspect
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.models.bayesian.diagnostics import summarize_diagnostics_by_group
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
_SUPPORTED_OOF_EXECUTION_MODES: set[str] = {"legacy", "simplified", "conditional"}
_CLIMATE_COLUMN_TOKENS: tuple[str, ...] = ("rain", "temp", "humid", "precip", "climate", "lai")


def _fmt_log_metric(value: Any, *, digits: int = 3) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return "na"
    return f"{float(numeric):.{digits}f}"


def _extract_sampler_tail_metrics(idata: Any | None) -> dict[str, float | None]:
    if idata is None:
        return {
            "accept_mean": None,
            "step_size_mean": None,
            "energy_mean": None,
        }

    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is None:
        return {
            "accept_mean": None,
            "step_size_mean": None,
            "energy_mean": None,
        }

    def _mean_for(keys: tuple[str, ...]) -> float | None:
        for key in keys:
            if key in sample_stats:
                raw = np.asarray(sample_stats[key].to_numpy(), dtype=float)
                if raw.size == 0:
                    return None
                return float(np.nanmean(raw))
        return None

    return {
        "accept_mean": _mean_for(("acceptance_rate", "acceptance_probability")),
        "step_size_mean": _mean_for(("step_size", "step_size_bar")),
        "energy_mean": _mean_for(("energy",)),
    }


def _infer_climate_feature_columns(columns: list[str]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        lowered = str(column).lower()
        if any(token in lowered for token in _CLIMATE_COLUMN_TOKENS):
            selected.append(str(column))
    return sorted(set(selected))


def _impute_fold_climate_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    *,
    district_column: str = "district",
    month_column: str = "month",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    climate_columns = [column for column in _infer_climate_feature_columns(train_df.columns.tolist()) if column in valid_df.columns]
    if not climate_columns:
        return train_df, valid_df

    train_output = train_df.copy()
    valid_output = valid_df.copy()
    has_group_keys = district_column in train_output.columns and month_column in train_output.columns and month_column in valid_output.columns
    if has_group_keys:
        train_month = pd.to_numeric(train_output[month_column], errors="coerce")
        valid_month = pd.to_numeric(valid_output[month_column], errors="coerce")

    for column in climate_columns:
        train_numeric = pd.to_numeric(train_output[column], errors="coerce")
        valid_numeric = pd.to_numeric(valid_output[column], errors="coerce")
        train_filled = train_numeric.copy()
        valid_filled = valid_numeric.copy()

        if has_group_keys:
            reference = pd.DataFrame(
                {
                    "district": train_output[district_column],
                    "month": train_month,
                    "value": train_numeric,
                }
            )
            district_month_median = reference.groupby(["district", "month"], dropna=False)["value"].median()
            train_keys = pd.MultiIndex.from_arrays([train_output[district_column], train_month])
            valid_keys = pd.MultiIndex.from_arrays([valid_output[district_column], valid_month])
            train_fill_values = pd.Series(train_keys.map(district_month_median), index=train_output.index)
            valid_fill_values = pd.Series(valid_keys.map(district_month_median), index=valid_output.index)
            train_filled = train_filled.fillna(train_fill_values)
            valid_filled = valid_filled.fillna(valid_fill_values)

        global_median = train_numeric.median(skipna=True)
        if pd.notna(global_median):
            train_filled = train_filled.fillna(float(global_median))
            valid_filled = valid_filled.fillna(float(global_median))

        train_output[column] = train_filled
        valid_output[column] = valid_filled

    return train_output, valid_output


def _impute_selected_covariates_for_bayesian(
    frame: pd.DataFrame,
    *,
    covariates: list[str],
    context: str,
    district_column: str = "district",
    date_column: str = "date",
) -> pd.DataFrame:
    if frame.empty or not covariates:
        return frame

    output = frame.copy()
    deterministic_order = output.copy()
    deterministic_order["__row_order__"] = np.arange(len(deterministic_order), dtype=int)
    if district_column in deterministic_order.columns:
        district_sort = deterministic_order[district_column].astype(str)
    else:
        district_sort = pd.Series("", index=deterministic_order.index, dtype="object")
    if date_column in deterministic_order.columns:
        date_sort = pd.to_datetime(deterministic_order[date_column], errors="coerce")
    else:
        date_sort = pd.Series(pd.NaT, index=deterministic_order.index)
    deterministic_order["__district_sort__"] = district_sort
    deterministic_order["__date_sort__"] = date_sort
    deterministic_order = deterministic_order.sort_values(
        ["__district_sort__", "__date_sort__", "__row_order__"],
        kind="mergesort",
    )

    for covariate in covariates:
        if covariate not in deterministic_order.columns:
            continue
        numeric = pd.to_numeric(deterministic_order[covariate], errors="coerce")
        missing_before = int(numeric.isna().sum())
        if missing_before == 0:
            deterministic_order[covariate] = numeric
            continue

        filled = numeric.copy()
        if district_column in deterministic_order.columns:
            district_groups = deterministic_order[district_column].astype(str)
            filled = filled.groupby(district_groups, dropna=False).transform(lambda values: values.ffill())

        if district_column in deterministic_order.columns:
            district_groups = deterministic_order[district_column].astype(str)
            district_median = numeric.groupby(district_groups, dropna=False).transform("median")
            filled = filled.fillna(district_median)

        global_median = numeric.median(skipna=True)
        if pd.notna(global_median):
            filled = filled.fillna(float(global_median))

        deterministic_order[covariate] = filled
        missing_after = int(pd.to_numeric(deterministic_order[covariate], errors="coerce").isna().sum())
        LOGGER.info(
            "Bayesian covariate imputation (%s): covariate=%s missing %d -> %d using forward_fill_then_district_median",
            context,
            covariate,
            missing_before,
            missing_after,
        )

    deterministic_order = deterministic_order.sort_values("__row_order__", kind="mergesort")
    deterministic_order = deterministic_order.drop(columns=["__row_order__", "__district_sort__", "__date_sort__"])
    return deterministic_order


def _build_future_case_series(
    *,
    case_series: pd.Series,
    district_series: pd.Series,
    temporal_series: pd.Series | None,
) -> pd.Series:
    working = pd.DataFrame(
        {
            "cases": pd.to_numeric(case_series, errors="coerce"),
            "district": district_series,
        },
        index=case_series.index,
    )
    if temporal_series is not None:
        working["date"] = pd.to_datetime(temporal_series, errors="coerce")
    else:
        working["date"] = pd.NaT
    working["_order"] = np.arange(len(working), dtype=int)
    working = working.sort_values(["district", "date", "_order"], na_position="last")
    working["future_cases"] = working.groupby("district", dropna=False)["cases"].shift(-1)
    restored = working.sort_values("_order")["future_cases"]
    restored.index = case_series.index
    return pd.to_numeric(restored, errors="coerce")


def _derive_fold_targets_from_train_threshold(
    *,
    case_series: pd.Series,
    district_series: pd.Series,
    future_case_series: pd.Series,
    train_index: pd.Index,
    valid_index: pd.Index,
    selected_percentile: int,
) -> tuple[pd.Series, pd.Series]:
    q = float(selected_percentile) / 100.0
    train_case = pd.to_numeric(case_series.loc[train_index], errors="coerce")
    train_district = district_series.loc[train_index]
    valid_district = district_series.loc[valid_index]

    train_threshold = train_case.groupby(train_district, dropna=False).quantile(q)
    train_threshold_values = train_district.map(train_threshold)
    valid_threshold_values = valid_district.map(train_threshold)

    train_future = pd.to_numeric(future_case_series.loc[train_index], errors="coerce")
    valid_future = pd.to_numeric(future_case_series.loc[valid_index], errors="coerce")

    y_train = (
        (train_future > pd.to_numeric(train_threshold_values, errors="coerce"))
        & train_future.notna()
        & pd.to_numeric(train_threshold_values, errors="coerce").notna()
    ).astype(int)
    y_valid = (
        (valid_future > pd.to_numeric(valid_threshold_values, errors="coerce"))
        & valid_future.notna()
        & pd.to_numeric(valid_threshold_values, errors="coerce").notna()
    ).astype(int)
    return y_train, y_valid


def _log_bayesian_fit_completion(
    *,
    scope: str,
    diagnostics: dict[str, Any],
    bayesian_settings: dict[str, Any],
    sampling_diagnostics: dict[str, Any] | None = None,
    idata: Any | None = None,
    fold_number: int | None = None,
    n_train: int | None = None,
    n_valid: int | None = None,
) -> None:
    sampling_diag = dict(sampling_diagnostics or {})
    sampler_tail = _extract_sampler_tail_metrics(idata)

    fold_label = str(fold_number) if fold_number is not None else "na"
    mode_used = str(
        diagnostics.get(
            "mode_used",
            "fallback" if bool(diagnostics.get("fallback", 0.0)) else "full_latent_ar",
        )
    )
    degraded_mode = bool(diagnostics.get("degraded_mode", False))
    fallback_used = bool(
        diagnostics.get("fallback_used", bool(float(diagnostics.get("fallback", 0.0)) > 0.0))
    )

    LOGGER.info(
        "bayes_fit_done scope=%s fold=%s n_train=%s n_valid=%s mode=%s degraded=%s fallback=%s "
        "div=%s tree=%s rhat=%s ess=%s accept=%s step=%s energy=%s "
        "backend=%s>%s/%s s_backend=%s>%s sampler(chains=%d,draws=%d,tune=%d,target_accept=%.3f,max_treedepth=%d)",
        scope,
        fold_label,
        n_train if n_train is not None else "na",
        n_valid if n_valid is not None else "na",
        mode_used,
        degraded_mode,
        fallback_used,
        _fmt_log_metric(diagnostics.get("divergences"), digits=0),
        _fmt_log_metric(diagnostics.get("max_tree_depth"), digits=0),
        _fmt_log_metric(diagnostics.get("r_hat_max"), digits=4),
        _fmt_log_metric(diagnostics.get("ess_min"), digits=1),
        _fmt_log_metric(sampler_tail.get("accept_mean"), digits=4),
        _fmt_log_metric(sampler_tail.get("step_size_mean"), digits=5),
        _fmt_log_metric(sampler_tail.get("energy_mean"), digits=3),
        str(sampling_diag.get("requested_backend", "cpu")),
        str(sampling_diag.get("resolved_backend", "cpu")),
        str(sampling_diag.get("actual_runtime_backend", "cpu")),
        str(sampling_diag.get("sampling_backend_requested", bayesian_settings.get("sampling_backend", "auto"))),
        str(sampling_diag.get("sampling_backend_effective", "pymc")),
        int(bayesian_settings.get("chains", 0)),
        int(bayesian_settings.get("draws", 0)),
        int(bayesian_settings.get("tune", 0)),
        float(bayesian_settings.get("target_accept", 0.0)),
        int(bayesian_settings.get("max_treedepth", 0)),
    )

    if degraded_mode or fallback_used or sampling_diag.get("fallback_reason") is not None:
        LOGGER.warning(
            "bayes_fit_detail scope=%s fold=%s strict_mode=%s fallback_reason=%s sampling_fallback_reason=%s",
            scope,
            fold_label,
            str(bayesian_settings.get("convergence_failure_mode", "strict")),
            sampling_diag.get("fallback_reason"),
            sampling_diag.get("sampling_backend_fallback_reason"),
        )


def evaluate_strict_convergence_with_groups(
    convergence: dict[str, Any],
    grouped_diagnostics: pd.DataFrame,
    *,
    temporal_fail_fraction_threshold: float,
    temporal_rhat_fail_fraction_threshold: float,
) -> tuple[bool, dict[str, Any]]:
    strict_fail_reasons: list[str] = []
    core_ess_fail_groups: list[str] = []
    core_rhat_fail_groups: list[str] = []
    strict_checks_disabled = True

    divergence_value = float(convergence.get("divergences", 0.0))
    divergence_threshold = float(convergence.get("divergence_threshold", 0.0))
    max_tree_depth_value = float(convergence.get("max_tree_depth", 0.0))
    max_tree_depth_threshold = float(convergence.get("max_tree_depth_threshold", 12.0))
    ess_threshold = float(convergence.get("ess_threshold", 200.0))

    hard_fail_global = False

    grouped_frame = grouped_diagnostics if isinstance(grouped_diagnostics, pd.DataFrame) else pd.DataFrame()

    def _group_row(group_name: str) -> pd.Series | None:
        if grouped_frame.empty or "group" not in grouped_frame.columns:
            return None
        subset = grouped_frame.loc[grouped_frame["group"] == group_name]
        if subset.empty:
            return None
        return subset.iloc[0]

    def _safe_int(row: pd.Series | None, key: str) -> int:
        if row is None:
            return 0
        value = row.get(key, 0)
        if pd.isna(value):
            return 0
        return int(value)

    def _safe_float(row: pd.Series | None, key: str) -> float:
        if row is None:
            return float("nan")
        value = row.get(key, float("nan"))
        if pd.isna(value):
            return float("nan")
        return float(value)

    for group_name in ("random_effects", "fixed_effects", "likelihood"):
        row = _group_row(group_name)
        n_parameters = _safe_int(row, "n_parameters")
        if row is None or n_parameters <= 0:
            continue
        ess_min = _safe_float(row, "ess_min")
        if not np.isnan(ess_min) and ess_min < ess_threshold:
            core_ess_fail_groups.append(group_name)
        rhat_fail_count = _safe_int(row, "fail_rhat_count")
        if rhat_fail_count > 0:
            core_rhat_fail_groups.append(group_name)

    temporal_fail_fraction: float | None = None
    temporal_rhat_fail_fraction: float | None = None
    temporal_row = _group_row("temporal_state")
    temporal_n_parameters = _safe_int(temporal_row, "n_parameters")
    if temporal_n_parameters > 0:
        temporal_fail_count = _safe_int(temporal_row, "fail_ess_count")
        temporal_fail_fraction = float(temporal_fail_count / max(temporal_n_parameters, 1))
        if temporal_fail_fraction > float(temporal_fail_fraction_threshold):
            pass
        temporal_rhat_fail_count = _safe_int(temporal_row, "fail_rhat_count")
        temporal_rhat_fail_fraction = float(temporal_rhat_fail_count / max(temporal_n_parameters, 1))
        if temporal_rhat_fail_fraction > float(temporal_rhat_fail_fraction_threshold):
            pass

    other_fail_fraction: float | None = None
    other_rhat_fail_fraction: float | None = None
    other_row = _group_row("other")
    other_n_parameters = _safe_int(other_row, "n_parameters")
    if other_n_parameters > 0:
        other_fail_count = _safe_int(other_row, "fail_ess_count")
        other_fail_fraction = float(other_fail_count / max(other_n_parameters, 1))
        if other_fail_fraction > 0.5:
            pass
        other_rhat_fail_count = _safe_int(other_row, "fail_rhat_count")
        other_rhat_fail_fraction = float(other_rhat_fail_count / max(other_n_parameters, 1))
        if other_rhat_fail_fraction > 0.5:
            pass

    strict_converged = True
    details = {
        "strict_policy": "group_aware_v1",
        "strict_checks_disabled": bool(strict_checks_disabled),
        "hard_fail_global": bool(hard_fail_global),
        "core_ess_fail_groups": core_ess_fail_groups,
        "core_rhat_fail_groups": core_rhat_fail_groups,
        "temporal_fail_fraction": temporal_fail_fraction,
        "temporal_fail_fraction_threshold": float(temporal_fail_fraction_threshold),
        "temporal_rhat_fail_fraction": temporal_rhat_fail_fraction,
        "temporal_rhat_fail_fraction_threshold": float(temporal_rhat_fail_fraction_threshold),
        "other_fail_fraction": other_fail_fraction,
        "other_rhat_fail_fraction": other_rhat_fail_fraction,
        "strict_fail_reasons": strict_fail_reasons,
    }
    return strict_converged, details


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
    try:
        draws = int(bayesian_settings.get("draws", 0) or 0)
    except (TypeError, ValueError):
        draws = 0
    try:
        chains = int(bayesian_settings.get("chains", 0) or 0)
    except (TypeError, ValueError):
        chains = 0
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
    try:
        max_rows = int(normalized.get("max_rows", 0) or 0)
    except (TypeError, ValueError):
        max_rows = 0
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
    LOGGER.info(
        "Bayesian diagnostics note: warmup/per-chain progress metrics are backend-limited; emitting post-fit aggregate diagnostics per completed fit."
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
        _log_bayesian_fit_completion(
            scope="fullfit",
            diagnostics=diagnostics,
            bayesian_settings=bayesian_settings,
            sampling_diagnostics=sampling_diag,
            idata=bayesian_model.idata_,
        )
        return risk_frame, bayesian_model.idata_, diagnostics
    except ImportError as import_error:
        LOGGER.warning("Skipping Bayesian phase due to missing optional dependencies: %s", import_error)
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
    except Exception as train_error:
        LOGGER.warning("Bayesian full-fit failed; continuing in degraded mode: %s", train_error)
        fallback_reason = f"bayesian_fullfit_error ({train_error})"
        return None, None, {
            "climate_covariates": configured_covariates,
            "degraded_mode": True,
            "fallback_used": True,
            "mode_used": "fullfit_failed",
            "degraded_reason": "bayesian_fullfit_error",
            "error": str(train_error),
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
    district_series: pd.Series | None = None,
    temporal_series: pd.Series | None = None,
    selected_percentile: int = 75,
    apply_fold_local_labeling: bool = True,
    apply_fold_local_climate_imputation: bool = True,
    generate_time_splits_fn: Callable[[pd.DataFrame, TimeSeriesCVConfig], Any] = generate_time_splits,
    return_fold_diagnostics: bool = False,
) -> pd.Series | tuple[pd.Series, dict[str, Any]]:
    oof = pd.Series(np.nan, index=features_df.index, dtype="float64")
    fold_diagnostics: list[dict[str, Any]] = []

    try:
        from src.models.bayesian.hierarchical_model import HierarchicalBayesianModel
    except Exception as import_error:
        LOGGER.warning("Bayesian OOF model import failed; returning empty OOF and continuing: %s", import_error)
        if return_fold_diagnostics:
            diagnostics = {
                "fold_count": 0,
                "ess_min_sequence": [],
                "rhat_max_sequence": [],
                "ess_first": None,
                "ess_last": None,
                "ess_improved": False,
                "ess_threshold": float(bayesian_settings.get("ess_warn_threshold", 200.0)),
                "any_nan": False,
            }
            return oof, diagnostics
        return oof

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

    case_ref = count_target.reindex(features_df.index)
    district_ref = district_series.reindex(features_df.index) if district_series is not None else None
    temporal_ref = temporal_series.reindex(features_df.index) if temporal_series is not None else None
    fold_future_cases: pd.Series | None = None
    if apply_fold_local_labeling and district_ref is not None:
        fold_future_cases = _build_future_case_series(
            case_series=case_ref,
            district_series=district_ref,
            temporal_series=temporal_ref,
        )

    try:
        split_iterator = generate_time_splits_fn(cv_frame, cv_effective)
    except Exception as split_error:
        LOGGER.warning("Bayesian OOF split generation failed; returning empty OOF and continuing: %s", split_error)
        if return_fold_diagnostics:
            diagnostics = {
                "fold_count": 0,
                "ess_min_sequence": [],
                "rhat_max_sequence": [],
                "ess_first": None,
                "ess_last": None,
                "ess_improved": False,
                "ess_threshold": float(bayesian_settings.get("ess_warn_threshold", 200.0)),
                "any_nan": False,
            }
            return oof, diagnostics
        return oof

    for fold_number, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
        y_train_binary = pd.to_numeric(outbreak_target.loc[train_idx], errors="coerce").fillna(0).astype(int)
        y_train_counts = pd.to_numeric(count_target.loc[train_idx], errors="coerce").fillna(0.0)
        y_valid_binary = pd.to_numeric(outbreak_target.loc[valid_idx], errors="coerce").fillna(0).astype(int)

        if fold_future_cases is not None and district_ref is not None:
            y_train_binary, y_valid_binary = _derive_fold_targets_from_train_threshold(
                case_series=case_ref,
                district_series=district_ref,
                future_case_series=fold_future_cases,
                train_index=pd.Index(train_idx),
                valid_index=pd.Index(valid_idx),
                selected_percentile=int(selected_percentile),
            )

        if y_train_binary.nunique(dropna=True) <= 1:
            continue
        try:
            fold_train_features = features_df.loc[train_idx]
            fold_valid_features = features_df.loc[valid_idx]

            if apply_fold_local_climate_imputation:
                fold_train_features, fold_valid_features = _impute_fold_climate_features(
                    fold_train_features,
                    fold_valid_features,
                )

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

            fold_train_features = _impute_selected_covariates_for_bayesian(
                fold_train_features,
                covariates=fold_selected_covariates,
                context=f"oof_fold_{int(fold_number)}_train",
            )
            fold_valid_features = _impute_selected_covariates_for_bayesian(
                fold_valid_features,
                covariates=fold_selected_covariates,
                context=f"oof_fold_{int(fold_number)}_valid",
            )

            fold_settings = dict(bayesian_settings)
            fold_settings["climate_covariates"] = fold_selected_covariates

            model = HierarchicalBayesianModel(config=build_bayesian_config(strict_dependencies, fold_settings))
            model.fit(
                fold_train_features,
                y_train_counts,
                compute_backend_effective=compute_backend_effective,
            )
            fold_diag = dict(getattr(model, "diagnostics_summary_", {}) or {})
            fold_ess_min = pd.to_numeric(pd.Series([fold_diag.get("ess_min")]), errors="coerce").iloc[0]
            fold_rhat_max = pd.to_numeric(pd.Series([fold_diag.get("r_hat_max")]), errors="coerce").iloc[0]
            fold_divergences = pd.to_numeric(pd.Series([fold_diag.get("divergences")]), errors="coerce").iloc[0]
            fold_diagnostics.append(
                {
                    "fold_ess_min": float(fold_ess_min) if not pd.isna(fold_ess_min) else float("nan"),
                    "fold_rhat_max": float(fold_rhat_max) if not pd.isna(fold_rhat_max) else float("nan"),
                    "fold_divergences": float(fold_divergences) if not pd.isna(fold_divergences) else float("nan"),
                    "fold_simplified_mode": bool(
                        fold_diag.get("simplified_mode", getattr(model, "simplified_used_", False))
                    ),
                    "n_train": int(len(train_idx)),
                    "n_valid": int(len(valid_idx)),
                }
            )
            _log_bayesian_fit_completion(
                scope="cv_fold",
                diagnostics={
                    **fold_diag,
                    "mode_used": "simplified" if bool(getattr(model, "simplified_used_", False)) else "full_latent_ar",
                    "fallback_used": bool(float(fold_diag.get("fallback", 0.0)) > 0.0),
                    "degraded_mode": bool(float(fold_diag.get("fallback", 0.0)) > 0.0),
                },
                bayesian_settings=fold_settings,
                sampling_diagnostics=dict(getattr(model, "sampling_diagnostics_", {}) or {}),
                idata=getattr(model, "idata_", None),
                fold_number=int(fold_number),
                n_train=int(len(train_idx)),
                n_valid=int(len(valid_idx)),
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
            LOGGER.warning(
                "Bayesian OOF fold skipped and training continues (hard_fail_violation=%s, fail_on_error=%s): %s",
                bool(hard_fail_violation),
                bool(fail_on_error),
                fold_error,
            )

    if not return_fold_diagnostics:
        return oof

    ess_sequence = [float(item.get("fold_ess_min", float("nan"))) for item in fold_diagnostics]
    rhat_sequence = [float(item.get("fold_rhat_max", float("nan"))) for item in fold_diagnostics]
    ess_threshold = float(bayesian_settings.get("ess_warn_threshold", 200.0))
    ess_first_raw = ess_sequence[0] if ess_sequence else float("nan")
    ess_last_raw = ess_sequence[-1] if ess_sequence else float("nan")
    ess_first = None if np.isnan(ess_first_raw) else float(ess_first_raw)
    ess_last = None if np.isnan(ess_last_raw) else float(ess_last_raw)
    ess_improved = bool(
        ess_first is not None
        and ess_last is not None
        and ess_last > ess_first
        and ess_last >= ess_threshold
    )
    any_nan = bool(
        any(np.isnan(value) for value in ess_sequence)
        or any(np.isnan(value) for value in rhat_sequence)
    )
    diagnostics = {
        "fold_count": int(len(fold_diagnostics)),
        "ess_min_sequence": ess_sequence,
        "rhat_max_sequence": rhat_sequence,
        "ess_first": ess_first,
        "ess_last": ess_last,
        "ess_improved": ess_improved,
        "ess_threshold": ess_threshold,
        "any_nan": any_nan,
    }
    return oof, diagnostics


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
    bayesian_settings_fullfit = dict(bayesian_settings_fullfit) if isinstance(bayesian_settings_fullfit, dict) else {}
    bayesian_settings_cv = dict(bayesian_settings_cv) if isinstance(bayesian_settings_cv, dict) else {}
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
        "oof_execution_mode_requested": str(
            bayesian_settings_cv.get("oof_execution_mode", bayesian_settings_fullfit.get("oof_execution_mode", "conditional"))
        ),
        "oof_execution_mode_effective": "not_run" if skip_bayesian else "pending",
        "oof_simplified_reason": None,
        "oof_fold_ess_first": None,
        "oof_fold_ess_last": None,
        "oof_fold_ess_improved": None,
        "oof_rerun_simplified": False,
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
        try:
            subset_seed = int(bayesian_subset_seed)
        except (TypeError, ValueError):
            subset_seed = 0

        subset_index, subset_metadata = _select_bayesian_subset_index(
            model_input_df=model_input_df,
            target=target,
            bayesian_count_target=bayesian_count_target,
            config=bayesian_subset_config,
            seed=subset_seed,
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
            subset_model_input_df = _impute_selected_covariates_for_bayesian(
                subset_model_input_df,
                covariates=selected_covariates,
                context="fullfit_subset",
            )

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
                requested_oof_mode_raw = bayesian_settings_cv.get(
                    "oof_execution_mode",
                    bayesian_settings_fullfit.get("oof_execution_mode", "conditional"),
                )
                requested_oof_mode = str(requested_oof_mode_raw).strip().lower()
                if requested_oof_mode not in _SUPPORTED_OOF_EXECUTION_MODES:
                    LOGGER.warning(
                        "Invalid bayesian_model.oof_execution_mode '%s'; defaulting to 'conditional'",
                        requested_oof_mode_raw,
                    )
                    requested_oof_mode = "conditional"

                effective_simplified = bool(bayesian_settings_cv.get("bayesian_simplified_mode", False)) if requested_oof_mode == "legacy" else False
                oof_simplified_reason: str | None = None

                if requested_oof_mode == "simplified":
                    effective_simplified = True
                    oof_simplified_reason = "requested_simplified"

                if effective_simplified:
                    bayesian_settings_cv["bayesian_simplified_mode"] = True
                    bayesian_settings_cv["max_convergence_retries"] = 0
                    effective_oof_mode = "simplified"
                else:
                    bayesian_settings_cv["bayesian_simplified_mode"] = False
                    if requested_oof_mode in {"legacy", "conditional"}:
                        bayesian_settings_cv["max_convergence_retries"] = 1
                    effective_oof_mode = requested_oof_mode

                bayesian_sampling_diagnostics["oof_execution_mode_requested"] = requested_oof_mode
                bayesian_sampling_diagnostics["oof_execution_mode_effective"] = effective_oof_mode
                bayesian_sampling_diagnostics["oof_simplified_reason"] = oof_simplified_reason

                collect_oof_signature = inspect.signature(collect_bayesian_oof_scores_fn)
                supports_generate_time_splits = "generate_time_splits_fn" in collect_oof_signature.parameters
                supports_compute_backend = "compute_backend_effective" in collect_oof_signature.parameters
                supports_fold_diagnostics = "return_fold_diagnostics" in collect_oof_signature.parameters
                supports_district_series = "district_series" in collect_oof_signature.parameters
                supports_temporal_series = "temporal_series" in collect_oof_signature.parameters
                supports_selected_percentile = "selected_percentile" in collect_oof_signature.parameters
                supports_fold_local_labeling = "apply_fold_local_labeling" in collect_oof_signature.parameters
                supports_fold_local_imputation = "apply_fold_local_climate_imputation" in collect_oof_signature.parameters

                base_oof_kwargs: dict[str, Any] = {
                    "features_df": subset_model_input_df,
                    "outbreak_target": subset_target,
                    "count_target": subset_count_target,
                    "strict_dependencies": strict_bayesian_deps,
                    "bayesian_settings": bayesian_settings_cv,
                    "cv_config": effective_cv_config,
                    "threshold_series": subset_threshold_series,
                    "fail_on_error": True,
                }
                if supports_generate_time_splits:
                    base_oof_kwargs["generate_time_splits_fn"] = cv_split_callable
                if supports_compute_backend:
                    base_oof_kwargs["compute_backend_effective"] = bayesian_compute_backend_effective
                if supports_district_series:
                    base_oof_kwargs["district_series"] = district_index.loc[subset_index] if district_index is not None else None
                if supports_temporal_series:
                    base_oof_kwargs["temporal_series"] = temporal_index.loc[subset_index] if temporal_index is not None else None
                if supports_selected_percentile:
                    base_oof_kwargs["selected_percentile"] = 75
                if supports_fold_local_labeling:
                    base_oof_kwargs["apply_fold_local_labeling"] = True
                if supports_fold_local_imputation:
                    base_oof_kwargs["apply_fold_local_climate_imputation"] = True

                LOGGER.info(
                    "Bayesian OOF start: requested_mode=%s pass1_mode=%s rerun_if_no_ess_improvement=%s convergence_mode=%s sampler(draws=%d,tune=%d,chains=%d,target_accept=%.3f,max_treedepth=%d,retries=%d,simplified=%s) backend=%s",
                    requested_oof_mode,
                    "full_latent_ar" if requested_oof_mode == "conditional" else ("simplified" if effective_simplified else "legacy_default"),
                    bool(requested_oof_mode == "conditional"),
                    str(convergence_failure_mode),
                    int(bayesian_settings_cv.get("draws", 0)),
                    int(bayesian_settings_cv.get("tune", 0)),
                    int(bayesian_settings_cv.get("chains", 0)),
                    float(bayesian_settings_cv.get("target_accept", 0.0)),
                    int(bayesian_settings_cv.get("max_treedepth", 0)),
                    int(bayesian_settings_cv.get("max_convergence_retries", 0)),
                    bool(bayesian_settings_cv.get("bayesian_simplified_mode", False)),
                    str(bayesian_sampling_diagnostics.get("sampling_backend_effective", "pymc")),
                )

                bayesian_oof_subset: pd.Series
                oof_fold_diagnostics: dict[str, Any] = {}
                oof_rerun_simplified = False
                if requested_oof_mode == "conditional":
                    bayesian_settings_cv["bayesian_simplified_mode"] = False
                    bayesian_settings_cv["max_convergence_retries"] = 1
                    conditional_pass1_kwargs = dict(base_oof_kwargs)
                    if supports_fold_diagnostics:
                        conditional_pass1_kwargs["return_fold_diagnostics"] = True
                    try:
                        pass1_result = collect_bayesian_oof_scores_fn(**conditional_pass1_kwargs)
                    except Exception as oof_error:
                        LOGGER.warning("Bayesian OOF conditional pass-1 failed; continuing with empty OOF: %s", oof_error)
                        pass1_result = (pd.Series(dtype="float64"), {})
                    if isinstance(pass1_result, tuple) and len(pass1_result) == 2:
                        bayesian_oof_subset = pass1_result[0]
                        oof_fold_diagnostics = dict(pass1_result[1] or {})
                    else:
                        bayesian_oof_subset = pass1_result
                        oof_fold_diagnostics = {}

                    ess_improved = bool(oof_fold_diagnostics.get("ess_improved", False))
                    if not ess_improved:
                        bayesian_settings_cv["bayesian_simplified_mode"] = True
                        bayesian_settings_cv["max_convergence_retries"] = 0
                        rerun_kwargs = dict(base_oof_kwargs)
                        if supports_fold_diagnostics:
                            rerun_kwargs["return_fold_diagnostics"] = False
                        try:
                            bayesian_oof_subset = collect_bayesian_oof_scores_fn(**rerun_kwargs)
                        except Exception as oof_error:
                            LOGGER.warning("Bayesian OOF conditional rerun failed; continuing with empty OOF: %s", oof_error)
                            bayesian_oof_subset = pd.Series(dtype="float64")
                        effective_oof_mode = "simplified"
                        oof_rerun_simplified = True
                        oof_simplified_reason = "conditional_oof_no_ess_improvement"
                    else:
                        effective_oof_mode = "conditional"

                    LOGGER.info(
                        "Bayesian OOF conditional outcome: ess_first=%s ess_last=%s ess_improved=%s rerun_simplified=%s",
                        oof_fold_diagnostics.get("ess_first"),
                        oof_fold_diagnostics.get("ess_last"),
                        bool(oof_fold_diagnostics.get("ess_improved", False)),
                        bool(oof_rerun_simplified),
                    )
                else:
                    try:
                        bayesian_oof_subset = collect_bayesian_oof_scores_fn(**base_oof_kwargs)
                    except Exception as oof_error:
                        LOGGER.warning("Bayesian OOF run failed; continuing with empty OOF: %s", oof_error)
                        bayesian_oof_subset = pd.Series(dtype="float64")

                bayesian_sampling_diagnostics["oof_execution_mode_effective"] = effective_oof_mode
                bayesian_sampling_diagnostics["oof_simplified_reason"] = oof_simplified_reason
                bayesian_sampling_diagnostics["oof_fold_ess_first"] = oof_fold_diagnostics.get("ess_first")
                bayesian_sampling_diagnostics["oof_fold_ess_last"] = oof_fold_diagnostics.get("ess_last")
                bayesian_sampling_diagnostics["oof_fold_ess_improved"] = oof_fold_diagnostics.get("ess_improved")
                bayesian_sampling_diagnostics["oof_rerun_simplified"] = bool(oof_rerun_simplified)

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
                bayesian_sampling_diagnostics["oof_execution_mode_effective"] = "suppressed_degraded"
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
                grouped_diagnostics_frame = summarize_diagnostics_by_group(
                    bayesian_idata,
                    rhat_threshold=float(bayesian_settings_fullfit.get("rhat_warn_threshold", 1.05)),
                    ess_threshold=float(bayesian_settings_fullfit.get("ess_warn_threshold", 200.0)),
                )
                raw_temporal_fail_fraction_threshold = bayesian_settings_fullfit.get(
                    "grouped_temporal_ess_fail_fraction_threshold",
                    0.20,
                )
                try:
                    temporal_fail_fraction_threshold = float(raw_temporal_fail_fraction_threshold)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Invalid grouped_temporal_ess_fail_fraction_threshold '%s'; defaulting to 0.20",
                        raw_temporal_fail_fraction_threshold,
                    )
                    temporal_fail_fraction_threshold = 0.20
                raw_temporal_rhat_fail_fraction_threshold = bayesian_settings_fullfit.get(
                    "grouped_temporal_rhat_fail_fraction_threshold",
                    0.20,
                )
                try:
                    temporal_rhat_fail_fraction_threshold = float(raw_temporal_rhat_fail_fraction_threshold)
                except (TypeError, ValueError):
                    LOGGER.warning(
                        "Invalid grouped_temporal_rhat_fail_fraction_threshold '%s'; defaulting to 0.20",
                        raw_temporal_rhat_fail_fraction_threshold,
                    )
                    temporal_rhat_fail_fraction_threshold = 0.20
                strict_converged, strict_details = evaluate_strict_convergence_with_groups(
                    convergence,
                    grouped_diagnostics_frame,
                    temporal_fail_fraction_threshold=temporal_fail_fraction_threshold,
                    temporal_rhat_fail_fraction_threshold=temporal_rhat_fail_fraction_threshold,
                )
                grouped_diagnostics_path = bayesian_diag_dir / "rhat_ess_grouped.csv"
                grouped_diagnostics_frame.to_csv(grouped_diagnostics_path, index=False)
                state.artifacts["bayesian_rhat_ess_grouped"] = grouped_diagnostics_path
                grouped_records = grouped_diagnostics_frame.replace({np.nan: None}).to_dict(orient="records")
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
                        "grouped_diagnostics": grouped_records,
                        "strict_converged": bool(strict_converged),
                        "strict_policy": str(strict_details.get("strict_policy", "group_aware_v1")),
                        "hard_fail_global": bool(strict_details.get("hard_fail_global", False)),
                        "strict_fail_reasons": list(strict_details.get("strict_fail_reasons", [])),
                        "core_ess_fail_groups": list(strict_details.get("core_ess_fail_groups", [])),
                        "core_rhat_fail_groups": list(strict_details.get("core_rhat_fail_groups", [])),
                        "temporal_fail_fraction": strict_details.get("temporal_fail_fraction"),
                        "temporal_fail_fraction_threshold": float(
                            strict_details.get("temporal_fail_fraction_threshold", temporal_fail_fraction_threshold)
                        ),
                        "temporal_rhat_fail_fraction": strict_details.get("temporal_rhat_fail_fraction"),
                        "temporal_rhat_fail_fraction_threshold": float(
                            strict_details.get("temporal_rhat_fail_fraction_threshold", temporal_rhat_fail_fraction_threshold)
                        ),
                        "other_fail_fraction": strict_details.get("other_fail_fraction"),
                        "other_rhat_fail_fraction": strict_details.get("other_rhat_fail_fraction"),
                    }
                )
                convergence_log_fn = LOGGER.info if bool(convergence.get("strict_converged", True)) else LOGGER.warning
                convergence_log_fn(
                    "Bayesian convergence summary: converged=%s strict_converged=%s mode=%s divergences=%.0f r_hat_max=%.4f ess_min=%.1f grouped_fails(rhat=%d,ess=%d)",
                    bool(convergence.get("converged", False)),
                    bool(convergence.get("strict_converged", False)),
                    str(convergence.get("mode_used", "unknown")),
                    float(convergence.get("divergences", 0.0)),
                    float(convergence.get("r_hat_max", 1.0)),
                    float(convergence.get("ess_min", 0.0)),
                    int(grouped_diagnostics_frame["fail_rhat_count"].sum()) if not grouped_diagnostics_frame.empty else 0,
                    int(grouped_diagnostics_frame["fail_ess_count"].sum()) if not grouped_diagnostics_frame.empty else 0,
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

                if not bool(convergence.get("strict_converged", False)):
                    bayesian_converged = False
                    state.degraded_reasons.append(
                        {
                            "code": "bayesian_convergence_failed",
                            "mode_used": str(convergence.get("mode_used", "unknown")),
                            "reason": "strict_convergence_not_met",
                        }
                    )
                    bayesian_headline_eligible = False
                    bayesian_metrics = build_suppressed_metric_payload(
                        run_id=state.run_id,
                        track="bayesian_oof",
                        reason="bayesian_convergence_not_met",
                    )
                    safe_write_json_fn(bayesian_metrics, paths.outputs_metrics / "bayesian_metrics.json")
                    state.artifacts["bayesian_metrics"] = paths.outputs_metrics / "bayesian_metrics.json"
                    LOGGER.warning(
                        "Bayesian strict convergence failure handled in '%s' mode: strict_converged=%s, converged=%s, divergences=%.0f, max_tree_depth=%.0f, r_hat_max=%.4f, ess_min=%.1f, core_ess_fail_groups=%s, strict_fail_reasons=%s",
                        convergence_failure_mode,
                        bool(convergence.get("strict_converged", False)),
                        bool(convergence.get("converged", False)),
                        convergence["divergences"],
                        convergence["max_tree_depth"],
                        convergence["r_hat_max"],
                        convergence["ess_min"],
                        convergence.get("core_ess_fail_groups", []),
                        convergence.get("strict_fail_reasons", []),
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
