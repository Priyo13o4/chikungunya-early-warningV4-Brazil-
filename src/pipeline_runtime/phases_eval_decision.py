from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score

from src.evaluation.metrics_baselines import evaluate_baseline_predictions
from src.pipeline_runtime.io_artifacts import build_contract_track_comparison_placeholder, write_bayesian_convergence_summary
from src.pipeline_runtime.phase_context import EvalDecisionPhaseResult, SharedPhaseState
from src.threshold_resolution import parse_threshold_percentile, resolve_threshold_series

LOGGER = logging.getLogger(__name__)
_THRESHOLD_GRID_MIN_FLOOR = 0.05


def _build_threshold_grid(*, grid_size: int, grid_min: float, grid_max: float) -> list[float]:
    clipped_min = float(min(max(grid_min, _THRESHOLD_GRID_MIN_FLOOR), 1.0))
    clipped_max = float(min(max(grid_max, _THRESHOLD_GRID_MIN_FLOOR), 1.0))
    if clipped_max < clipped_min:
        clipped_min, clipped_max = clipped_max, clipped_min
    safe_grid_size = max(int(grid_size), 2)
    return [float(value) for value in np.linspace(clipped_min, clipped_max, num=safe_grid_size)]


def _optimize_balanced_bayesian_threshold(
    *,
    y_true: pd.Series,
    probabilities: pd.Series,
    fallback_threshold: float,
    min_samples: int,
    grid_size: int,
    grid_min: float,
    grid_max: float,
) -> dict[str, Any]:
    y = pd.to_numeric(y_true, errors="coerce")
    probs = pd.to_numeric(probabilities, errors="coerce")
    valid = y.notna() & probs.notna()
    y_valid = y.loc[valid].astype(int)
    probs_valid = probs.loc[valid].astype(float).clip(0.0, 1.0)
    sample_size = int(len(y_valid))
    configured_min = float(grid_min)
    effective_min = float(min(max(configured_min, _THRESHOLD_GRID_MIN_FLOOR), 1.0))
    clamp_applied = bool(not np.isclose(configured_min, effective_min))

    payload: dict[str, Any] = {
        "threshold": float(min(max(fallback_threshold, 0.0), 1.0)),
        "f1": float("nan"),
        "accuracy": float("nan"),
        "sample_size": sample_size,
        "optimized": False,
        "method": "fallback_balanced_threshold",
        "objective_primary": "f1",
        "objective_tiebreaker": "accuracy",
        "configured_min": configured_min,
        "effective_min": effective_min,
        "clamp_applied": clamp_applied,
        "clamp_policy": "threshold_grid_min_floor",
    }
    if sample_size < int(min_samples) or y_valid.nunique(dropna=True) < 2:
        return payload

    candidates = _build_threshold_grid(
        grid_size=int(grid_size),
        grid_min=float(grid_min),
        grid_max=float(grid_max),
    )
    best_threshold = float(payload["threshold"])
    best_f1 = float("-inf")
    best_accuracy = float("-inf")
    for threshold in candidates:
        pred = (probs_valid >= float(threshold)).astype(int)
        trial_f1 = float(f1_score(y_valid, pred, zero_division=0))
        trial_accuracy = float(accuracy_score(y_valid, pred))
        if (
            trial_f1 > best_f1
            or (np.isclose(trial_f1, best_f1) and trial_accuracy > best_accuracy)
            or (
                np.isclose(trial_f1, best_f1)
                and np.isclose(trial_accuracy, best_accuracy)
                and abs(float(threshold) - float(fallback_threshold)) < abs(best_threshold - float(fallback_threshold))
            )
        ):
            best_f1 = float(trial_f1)
            best_accuracy = float(trial_accuracy)
            best_threshold = float(threshold)

    return {
        **payload,
        "threshold": float(best_threshold),
        "f1": float(best_f1),
        "accuracy": float(best_accuracy),
        "sample_size": sample_size,
        "optimized": True,
        "method": "empirical_balanced_oof",
    }


def run_evaluation_and_decision_phase(
    *,
    state: SharedPhaseState,
    paths: Any,
    labeled_df: pd.DataFrame,
    target: pd.Series,
    baseline_headline_eligible: bool,
    bayesian_headline_eligible: bool,
    baseline_metrics: dict[str, float],
    bayesian_metrics: dict[str, float],
    bayesian_score: pd.Series | None,
    bayesian_oof_score: pd.Series | None,
    baseline_oof_score: pd.Series | None,
    bayesian_sampling_diagnostics: dict[str, Any],
    bayesian_profile_usage: dict[str, Any],
    cv_subset_mode_active: bool,
    bayesian_convergence_payload: dict[str, Any] | None,
    bayesian_converged: bool | None,
    temporal_index: pd.Series | None,
    district_index: pd.Series | None,
    run_id: str,
    decision_cost: float,
    decision_loss: float,
    decision_optimize_threshold: bool,
    decision_optimization_risk_basis: str,
    decision_optimization_min_samples: int,
    decision_optimization_grid_size: int,
    decision_optimization_grid_min: float,
    decision_optimization_grid_max: float,
    decision_risk_score_basis: str,
    decision_policy_version: str,
    alert_thresholds: Any,
    export_detailed_csv: bool,
    optimize_decision_threshold_fn: Callable[..., dict[str, Any]],
    assign_alert_levels_fn: Callable[..., pd.Series],
    export_track_comparison_fn: Callable[..., dict[str, Path]],
    build_comparison_table_fn: Callable[..., pd.DataFrame],
    safe_write_json_fn: Callable[[dict[str, Any], Path], None],
    bayesian_covariates_requested: list[str] | None = None,
    bayesian_covariates_effective: list[str] | None = None,
    bayesian_covariate_selection: dict[str, Any] | None = None,
) -> EvalDecisionPhaseResult:
    suppress_headline_comparison = False
    covariates_requested_for_payload = list(
        bayesian_covariates_requested
        if bayesian_covariates_requested is not None
        else (
            bayesian_sampling_diagnostics.get("climate_covariates_requested")
            or bayesian_sampling_diagnostics.get("climate_covariates")
            or []
        )
    )
    covariates_effective_for_payload = list(
        bayesian_covariates_effective
        if bayesian_covariates_effective is not None
        else (bayesian_sampling_diagnostics.get("climate_covariates") or [])
    )
    covariate_selection_for_payload = bayesian_covariate_selection
    if not isinstance(covariate_selection_for_payload, dict):
        covariate_selection_for_payload = bayesian_sampling_diagnostics.get("covariate_selection")
    if not isinstance(covariate_selection_for_payload, dict):
        covariate_selection_for_payload = {
            "requested_covariates": list(covariates_requested_for_payload),
            "selected_covariates": list(covariates_effective_for_payload),
            "excluded_covariates": [],
            "missing_covariates": [],
            "required_covariates": ["month", "year", "weekofyear"],
            "required_covariates_present": True,
            "viable_count": int(len(covariates_effective_for_payload)),
            "requested_count": int(len(covariates_requested_for_payload)),
        }
    degraded_run_payload: dict[str, Any] = {}

    LOGGER.info("Phase: decision layer")
    fallback_decision_threshold = min(
        max(float(decision_cost) / max(float(decision_cost) + float(decision_loss), 1e-12), 0.0),
        1.0,
    )
    threshold_opt_payload = {
        "threshold": float(fallback_decision_threshold),
        "objective_cost": float("nan"),
        "sample_size": 0,
        "optimized": False,
        "method": "fallback_cost_loss_ratio",
    }
    oof_risk_for_optimization: pd.Series | None = None
    if decision_optimization_risk_basis == "bayesian_oof_risk_mean":
        oof_risk_for_optimization = bayesian_oof_score
    elif decision_optimization_risk_basis == "baseline_oof_mean":
        oof_risk_for_optimization = baseline_oof_score
    elif decision_optimization_risk_basis == "auto":
        oof_risk_for_optimization = bayesian_oof_score if bayesian_oof_score is not None else baseline_oof_score

    if decision_optimize_threshold and oof_risk_for_optimization is not None:
        threshold_opt_payload = optimize_decision_threshold_fn(
            y_true=target,
            probabilities=oof_risk_for_optimization,
            cost=float(decision_cost),
            loss=float(decision_loss),
            min_samples=int(decision_optimization_min_samples),
            grid_size=int(decision_optimization_grid_size),
            grid_min=float(decision_optimization_grid_min),
            grid_max=float(decision_optimization_grid_max),
        )

    decision_threshold_used = float(threshold_opt_payload.get("threshold", fallback_decision_threshold))
    balanced_threshold_payload: dict[str, Any] = {
        "threshold": float(decision_threshold_used),
        "f1": float("nan"),
        "accuracy": float("nan"),
        "sample_size": 0,
        "optimized": False,
        "method": "bayesian_oof_not_available",
        "objective_primary": "f1",
        "objective_tiebreaker": "accuracy",
    }
    if bayesian_oof_score is not None:
        balanced_threshold_payload = _optimize_balanced_bayesian_threshold(
            y_true=target,
            probabilities=bayesian_oof_score,
            fallback_threshold=float(decision_threshold_used),
            min_samples=int(decision_optimization_min_samples),
            grid_size=int(decision_optimization_grid_size),
            grid_min=float(decision_optimization_grid_min),
            grid_max=float(decision_optimization_grid_max),
        )
    balanced_threshold_used = float(balanced_threshold_payload.get("threshold", decision_threshold_used))

    decision_score: pd.Series | None = None
    decision_score_basis = "target_passthrough"
    normalized_risk_basis = decision_risk_score_basis.strip().lower()
    if normalized_risk_basis in {"bayesian_risk_q95", "bayesian_risk_mean", "bayesian_oof_risk_mean"}:
        decision_score = bayesian_oof_score
        decision_score_basis = "bayesian_oof_risk_mean"
    elif normalized_risk_basis == "baseline_oof_mean":
        decision_score = baseline_oof_score
        decision_score_basis = "baseline_oof_mean"
    elif normalized_risk_basis == "auto":
        if bayesian_oof_score is not None and bayesian_oof_score.notna().any():
            decision_score = bayesian_oof_score
            decision_score_basis = "bayesian_oof_risk_mean"
        else:
            decision_score = baseline_oof_score
            decision_score_basis = "baseline_oof_mean"

    if decision_score is not None:
        decision_score = pd.to_numeric(decision_score, errors="coerce").clip(0.0, 1.0)

    if decision_score is None or not decision_score.notna().any():
        state.degraded_reasons.append(
            {
                "code": "decision_no_oof_risk_available",
                "risk_score_basis_requested": decision_risk_score_basis,
            }
        )
        decision_score_basis = "target_passthrough"
        decision_score = target.astype(float)

    if bayesian_oof_score is not None and bayesian_headline_eligible:
        valid_bayesian_mask = pd.to_numeric(bayesian_oof_score, errors="coerce").notna() & pd.to_numeric(
            target,
            errors="coerce",
        ).notna()
        if bool(valid_bayesian_mask.any()):
            bayesian_probs = pd.to_numeric(bayesian_oof_score.loc[valid_bayesian_mask], errors="coerce").clip(0.0, 1.0)
            bayesian_true = pd.to_numeric(target.loc[valid_bayesian_mask], errors="coerce").fillna(0).astype(int)
            bayesian_pred = (bayesian_probs >= float(decision_threshold_used)).astype(int)
            bayesian_metrics = dict(bayesian_metrics)
            bayesian_metrics["precision"] = float(precision_score(bayesian_true, bayesian_pred, zero_division=0))
            bayesian_metrics["recall"] = float(recall_score(bayesian_true, bayesian_pred, zero_division=0))
            bayesian_metrics["f1"] = float(f1_score(bayesian_true, bayesian_pred, zero_division=0))
            bayesian_metrics["kappa"] = float(cohen_kappa_score(bayesian_true, bayesian_pred))
            bayesian_metrics["threshold_used"] = float(decision_threshold_used)

            bayesian_pred_balanced = (bayesian_probs >= float(balanced_threshold_used)).astype(int)
            bayesian_metrics["balanced_threshold_used"] = float(balanced_threshold_used)
            bayesian_metrics["balanced_threshold_optimized"] = bool(
                balanced_threshold_payload.get("optimized", False)
            )
            bayesian_metrics["balanced_threshold_sample_size"] = int(
                balanced_threshold_payload.get("sample_size", 0) or 0
            )
            bayesian_metrics["balanced_threshold_accuracy"] = float(accuracy_score(bayesian_true, bayesian_pred_balanced))
            bayesian_metrics["balanced_threshold_precision"] = float(
                precision_score(bayesian_true, bayesian_pred_balanced, zero_division=0)
            )
            bayesian_metrics["balanced_threshold_recall"] = float(
                recall_score(bayesian_true, bayesian_pred_balanced, zero_division=0)
            )
            bayesian_metrics["balanced_threshold_f1"] = float(
                f1_score(bayesian_true, bayesian_pred_balanced, zero_division=0)
            )
            bayesian_metrics["balanced_threshold_kappa"] = float(cohen_kappa_score(bayesian_true, bayesian_pred_balanced))
            safe_write_json_fn(bayesian_metrics, paths.outputs_metrics / "bayesian_metrics.json")
            state.artifacts["bayesian_metrics"] = paths.outputs_metrics / "bayesian_metrics.json"

    if baseline_oof_score is not None and baseline_headline_eligible:
        valid_baseline_mask = pd.to_numeric(baseline_oof_score, errors="coerce").notna() & pd.to_numeric(
            target,
            errors="coerce",
        ).notna()
        if bool(valid_baseline_mask.any()):
            baseline_recomputed = evaluate_baseline_predictions(
                y_true=pd.to_numeric(target.loc[valid_baseline_mask], errors="coerce").fillna(0).astype(int),
                y_pred_proba=pd.to_numeric(baseline_oof_score.loc[valid_baseline_mask], errors="coerce").clip(0.0, 1.0),
                threshold=float(decision_threshold_used),
                temporal_index=temporal_index.loc[valid_baseline_mask] if temporal_index is not None else None,
                district=district_index.loc[valid_baseline_mask] if district_index is not None else None,
            )
            baseline_metrics = dict(baseline_metrics)
            if baseline_metrics:
                for key in list(baseline_metrics.keys()):
                    if key in baseline_recomputed:
                        baseline_metrics[key] = float(baseline_recomputed[key])
            else:
                baseline_metrics = {key: float(value) for key, value in baseline_recomputed.items()}
            safe_write_json_fn(baseline_metrics, paths.outputs_metrics / "baseline_metrics.json")
            state.artifacts["baseline_metrics"] = paths.outputs_metrics / "baseline_metrics.json"

    LOGGER.info("Phase: evaluation")
    suppress_headline_comparison = bool(state.degraded_reasons)
    comparison_table = None
    if baseline_headline_eligible and bayesian_headline_eligible and not suppress_headline_comparison:
        comparison_outputs = export_track_comparison_fn(
            baseline_metrics=baseline_metrics,
            bayesian_metrics=bayesian_metrics,
            output_dir=paths.outputs_metrics,
            filename_prefix="track_comparison",
            include_wide_csv=export_detailed_csv,
        )
        state.artifacts["track_comparison_long_csv"] = comparison_outputs["long_csv"]
        state.artifacts["track_comparison_markdown"] = comparison_outputs["markdown"]
        state.artifacts["track_comparison_csv"] = comparison_outputs["long_csv"]
        state.artifacts["track_comparison_md"] = comparison_outputs["markdown"]
        if "wide_csv" in comparison_outputs:
            state.artifacts["track_comparison_wide_csv"] = comparison_outputs["wide_csv"]

        comparison_table = build_comparison_table_fn(
            baseline_metrics=baseline_metrics,
            bayesian_metrics=bayesian_metrics,
        )
    else:
        LOGGER.info("Track comparison skipped because one or both tracks are unavailable or run is degraded")

    degraded_run_payload = {
        "run_id": run_id,
        "degraded": bool(state.degraded_reasons),
        "suppress_headline_comparison": bool(suppress_headline_comparison),
        "reasons": state.degraded_reasons,
        "baseline_headline_eligible": bool(baseline_headline_eligible),
        "bayesian_headline_eligible": bool(bayesian_headline_eligible),
        "bayesian_covariates_requested": list(covariates_requested_for_payload),
        "bayesian_covariates_effective": list(covariates_effective_for_payload),
        "bayesian_covariate_selection": covariate_selection_for_payload,
    }
    degraded_run_path = paths.outputs_reports / "degraded_run.json"
    safe_write_json_fn(degraded_run_payload, degraded_run_path)
    state.artifacts["degraded_run"] = degraded_run_path

    if "track_comparison_csv" not in state.artifacts or "track_comparison_md" not in state.artifacts:
        track_reason = "degraded_run_or_missing_track_metrics"
        placeholder_frame, placeholder_markdown = build_contract_track_comparison_placeholder(
            run_id=run_id,
            reason=track_reason,
        )
        track_csv_path = paths.outputs_metrics / "track_comparison.csv"
        track_md_path = paths.outputs_metrics / "track_comparison.md"
        placeholder_frame.to_csv(track_csv_path, index=False)
        track_md_path.write_text(placeholder_markdown, encoding="utf-8")
        state.artifacts["track_comparison_csv"] = track_csv_path
        state.artifacts["track_comparison_md"] = track_md_path

    preferred_threshold_column = bayesian_sampling_diagnostics.get("threshold_column")
    if not isinstance(preferred_threshold_column, str) or not preferred_threshold_column:
        preferred_threshold_column = None

    preferred_percentile = None
    if preferred_threshold_column is not None:
        preferred_percentile = parse_threshold_percentile(preferred_threshold_column)
    if preferred_percentile is None:
        threshold_basis = str(bayesian_sampling_diagnostics.get("threshold_basis", ""))
        if threshold_basis.startswith("column:"):
            preferred_percentile = parse_threshold_percentile(threshold_basis.split(":", 1)[1])

    threshold_default = pd.to_numeric(
        pd.Series([bayesian_sampling_diagnostics.get("threshold_default", np.nan)]),
        errors="coerce",
    ).iloc[0]
    threshold_default_value = float(threshold_default) if pd.notna(threshold_default) else float("nan")
    threshold_cases_series, _ = resolve_threshold_series(
        frame=labeled_df,
        default_threshold=threshold_default_value,
        preferred_threshold_column=preferred_threshold_column,
        preferred_percentile=preferred_percentile,
    )
    threshold_cases_series = pd.to_numeric(threshold_cases_series, errors="coerce")
    threshold_cases_clamp_mask = threshold_cases_series.notna() & (threshold_cases_series <= 0.0)
    threshold_cases_clamp_count = int(threshold_cases_clamp_mask.sum())
    threshold_cases_clamp_ratio = float(threshold_cases_clamp_count / len(threshold_cases_series)) if len(threshold_cases_series) else 0.0
    if threshold_cases_clamp_count > 0:
        threshold_cases_series.loc[threshold_cases_clamp_mask] = 1.0
        LOGGER.warning(
            "Applied sentinel clamp for non-positive threshold_cases in decision/evaluation layer: clamped_rows=%d",
            threshold_cases_clamp_count,
        )

    decision_frame = pd.DataFrame(index=labeled_df.index)
    decision_frame["run_id"] = run_id
    decision_frame["date"] = labeled_df.get("date", pd.Series(pd.NaT, index=labeled_df.index))
    decision_frame["district"] = labeled_df.get("district", pd.Series(pd.NA, index=labeled_df.index))
    decision_frame["risk_score"] = pd.to_numeric(decision_score, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    decision_frame["risk_mean"] = decision_frame["risk_score"]
    decision_frame["risk_q05"] = decision_frame["risk_score"]
    decision_frame["risk_q95"] = decision_frame["risk_score"]
    decision_frame["threshold_cases"] = threshold_cases_series
    decision_frame["risk_score_basis"] = decision_score_basis
    decision_frame["decision_threshold_used"] = float(decision_threshold_used)
    decision_frame["decision_cost"] = float(decision_cost)
    decision_frame["decision_loss"] = float(decision_loss)
    decision_frame["alert_threshold_yellow"] = float(alert_thresholds.yellow)
    decision_frame["alert_threshold_orange"] = float(alert_thresholds.orange)
    decision_frame["alert_threshold_red"] = float(alert_thresholds.red)
    decision_frame["decision_policy_version"] = decision_policy_version
    decision_frame["alert_level"] = assign_alert_levels_fn(decision_frame["risk_score"], thresholds=alert_thresholds)
    decision_frame["recommended_action"] = (
        pd.to_numeric(decision_frame["risk_score"], errors="coerce").fillna(0.0) >= float(decision_threshold_used)
    ).astype(int)
    decision_frame = decision_frame[
        [
            "run_id",
            "date",
            "district",
            "risk_score",
            "risk_mean",
            "risk_q05",
            "risk_q95",
            "threshold_cases",
            "risk_score_basis",
            "decision_threshold_used",
            "decision_cost",
            "decision_loss",
            "alert_threshold_yellow",
            "alert_threshold_orange",
            "alert_threshold_red",
            "decision_policy_version",
            "alert_level",
            "recommended_action",
        ]
    ]
    decision_path = paths.outputs_metrics / "decision_alerts.csv"
    decision_frame.to_csv(decision_path, index=False)
    state.artifacts["decision_alerts"] = decision_path

    bayesian_headline_effective = bool(bayesian_headline_eligible) and not bool(state.degraded_reasons)
    if bayesian_converged is False:
        bayesian_headline_effective = False
    covariates_requested = list(
        bayesian_sampling_diagnostics.get("climate_covariates_requested")
        or bayesian_sampling_diagnostics.get("climate_covariates")
        or []
    )
    covariates_effective = list(bayesian_sampling_diagnostics.get("climate_covariates") or [])
    covariate_selection_payload = bayesian_sampling_diagnostics.get("covariate_selection")
    if not isinstance(covariate_selection_payload, dict):
        covariate_selection_payload = {
            "requested_covariates": list(covariates_requested),
            "selected_covariates": list(covariates_effective),
            "excluded_covariates": [],
            "missing_covariates": [],
            "required_covariates": ["month", "year", "weekofyear"],
            "required_covariates_present": True,
            "viable_count": int(len(covariates_effective)),
            "requested_count": int(len(covariates_requested)),
        }
    bayesian_risk_metadata_path = paths.outputs_metrics / "bayesian_risk_metadata.json"
    safe_write_json_fn(
        {
            "run_id": run_id,
            "headline_eligible": bool(bayesian_headline_effective),
            "degraded": bool(state.degraded_reasons),
            "converged": None if bayesian_converged is None else bool(bayesian_converged),
            "global_converged": (
                bool(bayesian_convergence_payload.get("global_converged"))
                if bayesian_convergence_payload is not None and bayesian_convergence_payload.get("global_converged") is not None
                else None
            ),
            "strict_converged": (
                bool(bayesian_convergence_payload.get("strict_converged"))
                if bayesian_convergence_payload is not None and bayesian_convergence_payload.get("strict_converged") is not None
                else None
            ),
            "effective_converged": (
                bool(bayesian_convergence_payload.get("effective_converged"))
                if bayesian_convergence_payload is not None and bayesian_convergence_payload.get("effective_converged") is not None
                else None
            ),
            "convergence_mode_used": (
                bayesian_convergence_payload.get("convergence_mode_used")
                if bayesian_convergence_payload is not None
                else None
            ),
            "convergence_checked": bool(bayesian_convergence_payload is not None),
            "mode_used": str(bayesian_sampling_diagnostics.get("mode_used", "not_run")),
            "fallback_used": bool(bayesian_sampling_diagnostics.get("fallback_used", False)),
            "degraded_mode": bool(bayesian_sampling_diagnostics.get("degraded_mode", False)),
            "convergence_failure_mode": str(bayesian_sampling_diagnostics.get("convergence_failure_mode", "warn")),
            "convergence_failure_mode_explicit": bool(
                bayesian_sampling_diagnostics.get("convergence_failure_mode_explicit", False)
            ),
            "convergence_failure_mode_auto_reason": bayesian_sampling_diagnostics.get(
                "convergence_failure_mode_auto_reason"
            ),
            "requested_backend": str(bayesian_sampling_diagnostics.get("requested_backend", "cpu")),
            "resolved_backend": str(bayesian_sampling_diagnostics.get("resolved_backend", "cpu")),
            "actual_runtime_backend": str(bayesian_sampling_diagnostics.get("actual_runtime_backend", "cpu")),
            "backend_implemented": bool(bayesian_sampling_diagnostics.get("backend_implemented", True)),
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
            "climate_covariates_requested": list(covariates_requested),
            "climate_covariates": list(covariates_effective),
            "covariates_requested": list(covariates_requested),
            "covariates_effective": list(covariates_effective),
            "trained_covariate_order": list(
                bayesian_sampling_diagnostics.get("trained_covariate_order", covariates_effective)
            ),
            "trained_covariate_order_locked": bool(
                bayesian_sampling_diagnostics.get("trained_covariate_order_locked", False)
            ),
            "covariate_selection": covariate_selection_payload,
            "compute_backend_requested": str(bayesian_sampling_diagnostics.get("compute_backend_requested", "cpu")),
            "compute_backend_effective": str(bayesian_sampling_diagnostics.get("compute_backend_effective", "cpu")),
            "compute_backend_runtime": str(bayesian_sampling_diagnostics.get("compute_backend_runtime", "cpu")),
            "compute_backend_fallback_used": bool(
                bayesian_sampling_diagnostics.get("compute_backend_fallback_used", False)
            ),
            "compute_backend_fallback_reason": bayesian_sampling_diagnostics.get("compute_backend_fallback_reason"),
            "threshold_basis": str(bayesian_sampling_diagnostics.get("threshold_basis", "default")),
            "label_uses_future_shift": bool(bayesian_sampling_diagnostics.get("label_uses_future_shift", True)),
            "full_fit_retrospective_only": bool(
                bayesian_sampling_diagnostics.get("full_fit_retrospective_only", True)
            ),
            "prospective_claim_allowed": bool(
                bayesian_sampling_diagnostics.get("prospective_claim_allowed", False)
            ),
            "training_mode": str(bayesian_sampling_diagnostics.get("training_mode", "unknown")),
            "decision_threshold_optimization": {
                "enabled": bool(decision_optimize_threshold),
                "risk_basis": decision_optimization_risk_basis,
                "sample_size": int(threshold_opt_payload.get("sample_size", 0) or 0),
                "optimized": bool(threshold_opt_payload.get("optimized", False)),
                "method": str(threshold_opt_payload.get("method", "fallback_cost_loss_ratio")),
                "threshold": float(decision_threshold_used),
            },
            "bayesian_balanced_threshold_sweep": {
                "enabled": bool(bayesian_oof_score is not None),
                "sample_size": int(balanced_threshold_payload.get("sample_size", 0) or 0),
                "optimized": bool(balanced_threshold_payload.get("optimized", False)),
                "method": str(balanced_threshold_payload.get("method", "fallback_balanced_threshold")),
                "threshold": float(balanced_threshold_used),
                "f1": float(balanced_threshold_payload.get("f1", float("nan"))),
                "accuracy": float(balanced_threshold_payload.get("accuracy", float("nan"))),
                "objective_primary": str(balanced_threshold_payload.get("objective_primary", "f1")),
                "objective_tiebreaker": str(balanced_threshold_payload.get("objective_tiebreaker", "accuracy")),
                "grid_size": int(max(int(decision_optimization_grid_size), 2)),
                "grid_min": float(min(max(float(decision_optimization_grid_min), _THRESHOLD_GRID_MIN_FLOOR), 1.0)),
                "grid_max": float(min(max(float(decision_optimization_grid_max), _THRESHOLD_GRID_MIN_FLOOR), 1.0)),
                "configured_min": float(balanced_threshold_payload.get("configured_min", decision_optimization_grid_min)),
                "effective_min": float(balanced_threshold_payload.get("effective_min", decision_optimization_grid_min)),
                "clamp_applied": bool(balanced_threshold_payload.get("clamp_applied", False)),
                "clamp_policy": str(balanced_threshold_payload.get("clamp_policy", "threshold_grid_min_floor")),
            },
            "threshold_cases_clamp_count": int(threshold_cases_clamp_count),
            "threshold_cases_clamp_applied": bool(threshold_cases_clamp_count > 0),
            "threshold_cases_clamp_ratio": float(threshold_cases_clamp_ratio),
            "threshold_cases_clamp_reason": (
                "non_positive_threshold_cases_corrected_to_one"
                if threshold_cases_clamp_count > 0
                else None
            ),
            "bayesian_profile_usage": bayesian_sampling_diagnostics.get("bayesian_profile_usage", bayesian_profile_usage),
            "cv_subset_mode_active": bool(
                bayesian_sampling_diagnostics.get("cv_subset_mode_active", cv_subset_mode_active)
            ),
            "bayesian_subset": bayesian_sampling_diagnostics.get("bayesian_subset", {}),
        },
        bayesian_risk_metadata_path,
    )
    state.artifacts["bayesian_risk_metadata"] = bayesian_risk_metadata_path

    bayes_summary_csv, bayes_summary_md = write_bayesian_convergence_summary(
        metrics_dir=paths.outputs_metrics,
        run_id=run_id,
        diagnostics=bayesian_sampling_diagnostics,
        convergence_artifact_path=state.artifacts.get("bayesian_convergence"),
        convergence=bayesian_convergence_payload,
    )
    state.artifacts["bayesian_convergence_summary_csv"] = bayes_summary_csv
    state.artifacts["bayesian_convergence_summary_md"] = bayes_summary_md

    return EvalDecisionPhaseResult(
        comparison_table=comparison_table,
        decision_frame=decision_frame,
        suppress_headline_comparison=suppress_headline_comparison,
        degraded_run_payload=degraded_run_payload,
        bayesian_headline_effective=bayesian_headline_effective,
        decision_threshold_used=float(decision_threshold_used),
        threshold_opt_payload=threshold_opt_payload,
    )
