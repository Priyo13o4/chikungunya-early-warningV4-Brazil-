from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.evaluation.metrics_baselines import lead_time_steps
from src.pipeline_runtime.phase_context import BaselinePhaseResult, BayesianPhaseResult, EvalDecisionPhaseResult
from src.visualization.diagnostic_plots import (
    plot_convergence_comparison,
    plot_posterior_predictive_check,
    plot_residuals,
    plot_single_district_cases_with_ci,
    plot_trace,
)
from src.visualization.exploratory import (
    plot_case_distribution,
    plot_missingness_summary,
    plot_temporal_coverage_heatmap,
)
from src.visualization.feature_plots import (
    plot_correlation_heatmap,
    plot_covariate_forest_hdi,
    plot_covariate_forest_hdi_from_summary,
    plot_feature_importance,
)
from src.visualization.performance_plots import (
    plot_brier_lead_time_summary,
    plot_calibration_curve_comparison,
    plot_calibration_curve,
    plot_confusion_matrix_grid,
    plot_lead_time_boxplot,
    plot_pr_curve,
    plot_roc_curve,
    plot_threshold_costloss_curve,
    plot_track_comparison_shared_metrics_bar,
    plot_track_delta_heatmap,
    plot_tracka_model_score_comparison,
)
from src.visualization.risk_maps import (
    plot_decision_alert_trend,
    plot_geofaceted_risk_vs_baseline,
    plot_risk_trajectory,
    plot_top_risk_districts,
)

LOGGER = logging.getLogger(__name__)

_FEATURE_PREFIX = "feature__"


def _cleanup_legacy_figure_placeholders(figures_dir: Path) -> None:
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


def _extract_feature_importances(models: dict[str, Any], feature_names: list[str]) -> pd.Series | None:
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


def _coerce_numeric(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(np.nan, index=index, dtype="float64")
    if len(series) != len(index):
        aligned = pd.Series(series, copy=False)
        aligned = aligned.reindex(index)
    else:
        aligned = pd.Series(series, copy=False, index=index)
    return pd.to_numeric(aligned, errors="coerce")


def _align_frame_column(frame: pd.DataFrame | None, column: str, index: pd.Index) -> pd.Series:
    if frame is None or frame.empty or column not in frame.columns:
        return pd.Series(np.nan, index=index, dtype="float64")
    series = pd.to_numeric(pd.Series(frame[column], copy=False), errors="coerce")
    series.index = frame.index
    return pd.to_numeric(series.reindex(index), errors="coerce")


def _extract_covariate_hdi_records_from_idata(
    inference_data: Any | None,
    *,
    covariates: tuple[str, ...] = ("temperature", "rainfall", "humidity"),
) -> list[dict[str, float | str]]:
    if inference_data is None:
        return []
    posterior = getattr(inference_data, "posterior", None)
    if posterior is None or "beta" not in posterior:
        return []

    beta = posterior["beta"]
    coord_name = None
    for candidate in ("covariate", "beta_dim_0"):
        if candidate in getattr(beta, "coords", {}):
            coord_name = candidate
            break
    if coord_name is None:
        return []

    available_covariates = [str(value) for value in beta.coords[coord_name].to_numpy().tolist()]
    target_covariates = [cov for cov in covariates if cov in available_covariates]
    rows: list[dict[str, float | str]] = []
    for covariate in target_covariates:
        values = np.asarray(beta.sel({coord_name: covariate}).to_numpy(), dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        rows.append(
            {
                "covariate": str(covariate),
                "mean": float(np.mean(values)),
                "hdi_low": float(np.quantile(values, 0.025)),
                "hdi_high": float(np.quantile(values, 0.975)),
            }
        )
    return rows


def _collect_payload_inputs(
    *,
    run_id: str,
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    baseline_result: BaselinePhaseResult,
    bayesian_result: BayesianPhaseResult,
    eval_decision_result: EvalDecisionPhaseResult,
    bayesian_convergence_path: Path | None,
    bayesian_idata: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    payload = pd.DataFrame(index=labeled_df.index)
    payload["run_id"] = run_id
    payload["row_index"] = labeled_df.index.to_numpy()
    payload["date"] = labeled_df.get("date", pd.Series(pd.NaT, index=labeled_df.index))
    payload["district"] = labeled_df.get("district", pd.Series(pd.NA, index=labeled_df.index))
    payload["state"] = labeled_df.get("state", pd.Series(pd.NA, index=labeled_df.index))
    payload["cases"] = pd.to_numeric(labeled_df.get("cases", pd.Series(np.nan, index=labeled_df.index)), errors="coerce")
    payload["target_outbreak_label"] = _coerce_numeric(baseline_result.target, labeled_df.index)
    payload["risk_score"] = pd.to_numeric(eval_decision_result.decision_frame.get("risk_score"), errors="coerce")
    payload["trackb_score"] = _coerce_numeric(bayesian_result.bayesian_score, labeled_df.index)
    payload["baseline_oof_score"] = _coerce_numeric(baseline_result.baseline_oof_score, labeled_df.index)
    payload["bayesian_oof_score"] = _coerce_numeric(bayesian_result.bayesian_oof_score, labeled_df.index)
    payload["baseline_alarm"] = (payload["baseline_oof_score"].fillna(0.0) >= 0.5).astype(float)
    payload["bayesian_risk"] = _align_frame_column(bayesian_result.bayesian_risk_frame, "risk_mean", labeled_df.index)
    payload["bayesian_risk"] = payload["bayesian_risk"].fillna(payload["risk_score"])
    payload["risk_q05"] = _align_frame_column(bayesian_result.bayesian_risk_frame, "risk_q05", labeled_df.index)
    payload["risk_q95"] = _align_frame_column(bayesian_result.bayesian_risk_frame, "risk_q95", labeled_df.index)
    payload["cases_mean"] = _align_frame_column(bayesian_result.bayesian_risk_frame, "cases_mean", labeled_df.index)
    payload["cases_q05"] = _align_frame_column(bayesian_result.bayesian_risk_frame, "cases_q05", labeled_df.index)
    payload["cases_q95"] = _align_frame_column(bayesian_result.bayesian_risk_frame, "cases_q95", labeled_df.index)
    payload["alert_level"] = eval_decision_result.decision_frame.get("alert_level", pd.Series(pd.NA, index=labeled_df.index))
    payload["recommended_action"] = pd.to_numeric(
        eval_decision_result.decision_frame.get("recommended_action", pd.Series(np.nan, index=labeled_df.index)),
        errors="coerce",
    )

    feature_column_map: dict[str, str] = {}
    numeric_features = features_df.select_dtypes(include=[np.number]).copy()
    for original_column in numeric_features.columns:
        payload_column = f"{_FEATURE_PREFIX}{original_column}"
        payload[payload_column] = pd.to_numeric(numeric_features[original_column], errors="coerce")
        feature_column_map[payload_column] = str(original_column)

    if payload["trackb_score"].notna().sum() == 0:
        payload["trackb_score"] = payload["risk_score"]

    feature_importances = _extract_feature_importances(
        baseline_result.baseline_models,
        feature_names=features_df.columns.astype(str).tolist(),
    )

    baseline_model_metrics_records: list[dict[str, Any]] = []
    if baseline_result.baseline_model_metrics is not None and not baseline_result.baseline_model_metrics.empty:
        baseline_model_metrics_records = baseline_result.baseline_model_metrics.to_dict(orient="records")

    comparison_table_records: list[dict[str, Any]] = []
    if eval_decision_result.comparison_table is not None and not eval_decision_result.comparison_table.empty:
        comparison_table_records = eval_decision_result.comparison_table.to_dict(orient="records")

    bayesian_convergence_payload: dict[str, Any] | None = None
    if bayesian_convergence_path is not None and Path(bayesian_convergence_path).exists():
        try:
            bayesian_convergence_payload = json.loads(Path(bayesian_convergence_path).read_text(encoding="utf-8"))
        except Exception as read_error:
            LOGGER.warning("Unable to read Bayesian convergence payload for visualization metadata: %s", read_error)

    payload_metadata = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(payload)),
        "feature_column_map": feature_column_map,
        "feature_importances": (
            {str(k): float(v) for k, v in feature_importances.items()} if feature_importances is not None else {}
        ),
        "baseline_model_metrics_records": baseline_model_metrics_records,
        "comparison_table_records": comparison_table_records,
        "bayesian_convergence_path": str(bayesian_convergence_path) if bayesian_convergence_path is not None else None,
        "bayesian_convergence": bayesian_convergence_payload,
        "covariate_hdi_records": _extract_covariate_hdi_records_from_idata(bayesian_idata),
    }
    return payload, payload_metadata


def build_visualization_payload(
    *,
    paths: Any,
    run_id: str,
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    baseline_result: BaselinePhaseResult,
    bayesian_result: BayesianPhaseResult,
    eval_decision_result: EvalDecisionPhaseResult,
    bayesian_convergence_path: Path | None = None,
    safe_write_json_fn: Callable[[dict[str, Any], Path], None],
) -> tuple[Path, Path]:
    payload_frame, payload_metadata = _collect_payload_inputs(
        run_id=run_id,
        labeled_df=labeled_df,
        features_df=features_df,
        baseline_result=baseline_result,
        bayesian_result=bayesian_result,
        eval_decision_result=eval_decision_result,
        bayesian_convergence_path=bayesian_convergence_path,
        bayesian_idata=bayesian_result.bayesian_idata,
    )

    payload_csv_path = paths.outputs_reports / "visualization_payload.csv"
    payload_metadata_path = paths.outputs_reports / "visualization_payload_metadata.json"
    payload_frame.to_csv(payload_csv_path, index=False)

    payload_metadata = {
        **payload_metadata,
        "payload_csv_path": str(payload_csv_path),
    }
    safe_write_json_fn(payload_metadata, payload_metadata_path)
    return payload_csv_path, payload_metadata_path


def _safe_plot(plot_label: str, plot_callable: Callable[..., Any], generated: dict[str, Path], artifact_key: str, **kwargs: Any) -> None:
    try:
        plot_result = plot_callable(**kwargs)
    except ValueError as plot_error:
        LOGGER.warning("%s skipped: %s", plot_label, plot_error)
        return
    except Exception as plot_error:
        LOGGER.warning("%s failed: %s", plot_label, plot_error)
        return

    if isinstance(plot_result, Path):
        generated[artifact_key] = plot_result


def _build_metric_vectors(payload_frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    y_true = pd.to_numeric(payload_frame.get("target_outbreak_label"), errors="coerce")
    y_score = pd.to_numeric(payload_frame.get("trackb_score"), errors="coerce")
    if y_score.notna().sum() == 0:
        y_score = pd.to_numeric(payload_frame.get("risk_score"), errors="coerce")

    valid_mask = y_true.notna() & y_score.notna() & np.isfinite(y_true) & np.isfinite(y_score)
    y_true_valid = y_true.loc[valid_mask].astype(int)
    y_score_valid = y_score.loc[valid_mask].clip(0.0, 1.0)
    y_pred_valid = (y_score_valid >= 0.5).astype(int)
    return y_true_valid, y_score_valid, y_pred_valid


def _run_visualizations_from_payload(
    *,
    payload_frame: pd.DataFrame,
    payload_metadata: dict[str, Any],
    output_dir: Path,
    lead_time_max_lookback_steps: int,
    previous_bayesian_convergence: dict[str, Any] | None,
    effective_seed: int,
) -> dict[str, Path]:
    generated: dict[str, Path] = {}
    _cleanup_legacy_figure_placeholders(output_dir)
    legacy_lead_time_plot = output_dir / "performance_lead_time_boxplot.png"
    if legacy_lead_time_plot.exists():
        legacy_lead_time_plot.unlink(missing_ok=True)

    if {"date", "district"}.issubset(payload_frame.columns):
        _safe_plot(
            "Temporal exploratory plot",
            plot_temporal_coverage_heatmap,
            generated,
            "exploratory_temporal_coverage_heatmap",
            frame=payload_frame,
            date_col="date",
            district_col="district",
            top_k_districts=20,
            output_dir=output_dir,
        )

    if "cases" in payload_frame.columns:
        _safe_plot(
            "Case distribution plot",
            plot_case_distribution,
            generated,
            "exploratory_case_distribution",
            frame=payload_frame,
            case_col="cases",
            output_dir=output_dir,
        )

    _safe_plot(
        "Missingness summary plot",
        plot_missingness_summary,
        generated,
        "exploratory_missingness_summary",
        frame=payload_frame,
        output_dir=output_dir,
    )

    feature_columns = [column for column in payload_frame.columns if str(column).startswith(_FEATURE_PREFIX)]
    if feature_columns:
        numeric_feature_frame = payload_frame.loc[:, feature_columns].copy()
        renamed = {
            column: str(payload_metadata.get("feature_column_map", {}).get(column, column.removeprefix(_FEATURE_PREFIX)))
            for column in feature_columns
        }
        numeric_feature_frame = numeric_feature_frame.rename(columns=renamed)
        _safe_plot(
            "Feature correlation plot",
            plot_correlation_heatmap,
            generated,
            "features_correlation_heatmap",
            feature_frame=numeric_feature_frame,
            output_dir=output_dir,
        )

    feature_importances = payload_metadata.get("feature_importances", {})
    if isinstance(feature_importances, dict) and feature_importances:
        _safe_plot(
            "Feature importance plot",
            plot_feature_importance,
            generated,
            "features_importance",
            importances=feature_importances,
            top_k=20,
            output_dir=output_dir,
        )

    y_true_valid, y_score_valid, y_pred_valid = _build_metric_vectors(payload_frame)
    if len(y_true_valid) > 1:
        _safe_plot(
            "Residual plot",
            plot_residuals,
            generated,
            "trackb_residuals",
            y_true=y_true_valid,
            y_pred=y_score_valid,
            filename="trackb_residuals.png",
            output_dir=output_dir,
        )

        track_b_array = y_score_valid.to_numpy(dtype=float)
        if track_b_array.size == 0:
            posterior_predictive_samples = np.zeros((1, 1), dtype=float)
        else:
            max_points = 10000
            if track_b_array.size > max_points:
                rng = np.random.default_rng(int(effective_seed))
                sampled_idx = np.sort(rng.choice(track_b_array.size, size=max_points, replace=False))
                track_b_array = track_b_array[sampled_idx]
            sample_count = 25
            posterior_predictive_samples = np.broadcast_to(track_b_array, (sample_count, track_b_array.size))

        _safe_plot(
            "Posterior predictive check plot",
            plot_posterior_predictive_check,
            generated,
            "trackb_posterior_predictive_check",
            y_true=y_true_valid,
            posterior_predictive_samples=posterior_predictive_samples,
            filename="trackb_posterior_predictive_check.png",
            output_dir=output_dir,
        )

        if y_true_valid.nunique(dropna=True) > 1:
            _safe_plot(
                "ROC plot",
                plot_roc_curve,
                generated,
                "performance_roc_curve",
                y_true=y_true_valid,
                y_score=y_score_valid,
                output_dir=output_dir,
            )
            _safe_plot(
                "PR plot",
                plot_pr_curve,
                generated,
                "performance_pr_curve",
                y_true=y_true_valid,
                y_score=y_score_valid,
                output_dir=output_dir,
            )
            _safe_plot(
                "Calibration plot",
                plot_calibration_curve,
                generated,
                "trackb_calibration_curve",
                y_true=y_true_valid,
                y_score=y_score_valid,
                filename="trackb_calibration_curve.png",
                output_dir=output_dir,
            )
            baseline_score_valid = pd.to_numeric(payload_frame.get("baseline_oof_score"), errors="coerce").loc[y_true_valid.index]
            if baseline_score_valid.notna().sum() > 10:
                _safe_plot(
                    "Calibration comparison plot",
                    plot_calibration_curve_comparison,
                    generated,
                    "thesis_calibration_comparison",
                    y_true=y_true_valid,
                    bayesian_score=y_score_valid,
                    baseline_score=baseline_score_valid,
                    filename="thesis_calibration_comparison.png",
                    output_dir=output_dir,
                )
            _safe_plot(
                "Confusion matrix plot",
                plot_confusion_matrix_grid,
                generated,
                "performance_confusion_matrix_grid",
                y_true=y_true_valid,
                predictions={"decision": y_pred_valid},
                output_dir=output_dir,
            )
            _safe_plot(
                "Threshold tradeoff curve",
                plot_threshold_costloss_curve,
                generated,
                "thesis_threshold_costloss_curve",
                y_true=y_true_valid,
                y_score=y_score_valid,
                filename="thesis_threshold_costloss_curve.png",
                output_dir=output_dir,
            )

            if {"date", "district"}.issubset(payload_frame.columns):
                temporal_vector = pd.to_datetime(payload_frame.get("date"), errors="coerce")
                district_vector = payload_frame.get("district")
                vector_mask = y_true_valid.index
                lead_times = lead_time_steps(
                    y_true_valid,
                    y_pred_valid,
                    max_lookback_steps=int(lead_time_max_lookback_steps),
                    temporal_index=temporal_vector.loc[vector_mask],
                    district=district_vector.loc[vector_mask] if district_vector is not None else None,
                )
            else:
                lead_times = pd.Series(dtype=float)

            trackb_lead_rows = pd.DataFrame(
                {
                    "track": ["Track B"] * max(len(lead_times), 1),
                    "lead_time": lead_times.tolist() if not lead_times.empty else [0.0],
                }
            )
            _safe_plot(
                "Lead-time boxplot",
                plot_lead_time_boxplot,
                generated,
                "trackb_lead_time_boxplot",
                lead_time_data=trackb_lead_rows,
                filename="trackb_lead_time_boxplot.png",
                output_dir=output_dir,
            )

            brier_value = float(np.mean((y_true_valid.astype(float).to_numpy() - y_score_valid.astype(float).to_numpy()) ** 2))
            lead_time_mean = float(lead_times.mean()) if not lead_times.empty else 0.0
            _safe_plot(
                "Brier/lead-time summary",
                plot_brier_lead_time_summary,
                generated,
                "trackb_brier_leadtime_summary",
                brier_score=brier_value,
                lead_time_mean=lead_time_mean,
                output_dir=output_dir,
            )

    comparison_records = payload_metadata.get("comparison_table_records", [])
    if isinstance(comparison_records, list) and comparison_records:
        comparison_table = pd.DataFrame(comparison_records)
        if not comparison_table.empty:
            _safe_plot(
                "Track delta heatmap",
                plot_track_delta_heatmap,
                generated,
                "performance_track_delta_heatmap",
                comparison_table=comparison_table,
                output_dir=output_dir,
            )
            _safe_plot(
                "Track comparison bar",
                plot_track_comparison_shared_metrics_bar,
                generated,
                "track_comparison_shared_metrics_bar",
                comparison_table=comparison_table,
                output_dir=output_dir,
            )

    if {"date", "alert_level"}.issubset(payload_frame.columns):
        _safe_plot(
            "Decision alert trend plot",
            plot_decision_alert_trend,
            generated,
            "decision_alert_levels_over_time",
            decision_frame=payload_frame,
            date_col="date",
            alert_col="alert_level",
            output_dir=output_dir,
        )

    model_metric_records = payload_metadata.get("baseline_model_metrics_records", [])
    if isinstance(model_metric_records, list) and model_metric_records:
        model_scores = pd.DataFrame(model_metric_records)
        if not model_scores.empty:
            _safe_plot(
                "Track A model score plot",
                plot_tracka_model_score_comparison,
                generated,
                "tracka_models_all_scores",
                model_scores=model_scores,
                filename="tracka_models_all_scores.png",
                output_dir=output_dir,
            )

    if {"date", "district", "risk_score"}.issubset(payload_frame.columns):
        risk_plot_frame = pd.DataFrame(
            {
                "date": pd.to_datetime(payload_frame["date"], errors="coerce"),
                "district": payload_frame["district"],
                "risk_score": pd.to_numeric(payload_frame["risk_score"], errors="coerce"),
                "cases": pd.to_numeric(payload_frame.get("cases", pd.Series(0.0, index=payload_frame.index)), errors="coerce"),
            }
        )
        risk_plot_frame = risk_plot_frame.dropna(subset=["date", "district", "risk_score"])
        if not risk_plot_frame.empty:
            _safe_plot(
                "Risk trajectory plot",
                plot_risk_trajectory,
                generated,
                "trackb_risk_trajectory",
                frame=risk_plot_frame,
                date_col="date",
                risk_col="risk_score",
                district_col="district",
                top_n=10,
                filename="trackb_risk_trajectory.png",
                output_dir=output_dir,
            )
            _safe_plot(
                "Top risk districts plot",
                plot_top_risk_districts,
                generated,
                "risk_top_districts",
                frame=risk_plot_frame,
                district_col="district",
                risk_col="risk_score",
                top_n=20,
                output_dir=output_dir,
            )

    if {"district", "state", "bayesian_risk", "baseline_alarm"}.issubset(payload_frame.columns):
        _safe_plot(
            "Geofaceted spatial risk plot",
            plot_geofaceted_risk_vs_baseline,
            generated,
            "thesis_spatial_risk_map",
            frame=payload_frame,
            district_col="district",
            state_col="state",
            bayesian_risk_col="bayesian_risk",
            baseline_alarm_col="baseline_alarm",
            filename="thesis_spatial_risk_map.png",
            output_dir=output_dir,
        )

    if {"date", "district", "cases", "cases_mean", "cases_q05", "cases_q95"}.issubset(payload_frame.columns):
        _safe_plot(
            "Single district CI time-series",
            plot_single_district_cases_with_ci,
            generated,
            "thesis_single_district_timeseries",
            frame=payload_frame,
            date_col="date",
            district_col="district",
            state_col="state",
            cases_col="cases",
            cases_mean_col="cases_mean",
            cases_q05_col="cases_q05",
            cases_q95_col="cases_q95",
            start_year=2015,
            end_year=2020,
            filename="thesis_single_district_timeseries.png",
            output_dir=output_dir,
        )

    current_convergence = payload_metadata.get("bayesian_convergence")
    if not isinstance(current_convergence, dict):
        bayesian_convergence_path = payload_metadata.get("bayesian_convergence_path")
        if isinstance(bayesian_convergence_path, str) and bayesian_convergence_path.strip():
            convergence_path = Path(bayesian_convergence_path)
            if convergence_path.exists():
                try:
                    current_convergence = json.loads(convergence_path.read_text(encoding="utf-8"))
                except Exception as read_error:
                    LOGGER.warning("Unable to load convergence payload from metadata path: %s", read_error)

    if isinstance(current_convergence, dict) and current_convergence:
        _safe_plot(
            "Bayesian convergence comparison plot",
            plot_convergence_comparison,
            generated,
            "bayesian_convergence_comparison",
            current_diagnostics=current_convergence,
            previous_diagnostics=previous_bayesian_convergence,
            filename="bayesian_convergence_comparison.png",
            output_dir=output_dir,
        )

    bayesian_idata = payload_metadata.get("bayesian_idata")
    covariate_hdi_records = payload_metadata.get("covariate_hdi_records", [])
    if bayesian_idata is not None:
        _safe_plot(
            "Covariate forest HDI plot",
            plot_covariate_forest_hdi,
            generated,
            "thesis_covariate_forest_hdi",
            inference_data=bayesian_idata,
            covariates=("temperature", "rainfall", "humidity"),
            filename="thesis_covariate_forest_hdi.png",
            output_dir=output_dir,
        )
    elif isinstance(covariate_hdi_records, list) and covariate_hdi_records:
        _safe_plot(
            "Covariate forest HDI plot",
            plot_covariate_forest_hdi_from_summary,
            generated,
            "thesis_covariate_forest_hdi",
            summary_records=covariate_hdi_records,
            filename="thesis_covariate_forest_hdi.png",
            output_dir=output_dir,
        )

    if bayesian_idata is not None:
        _safe_plot(
            "Bayesian trace plot",
            plot_trace,
            generated,
            "trackb_trace_plot",
            inference_data=bayesian_idata,
            filename="trackb_trace_plot.png",
            output_dir=output_dir,
        )
    _cleanup_legacy_figure_placeholders(output_dir)
    LOGGER.info("Visualization summary | generated_artifacts=%d", int(len(generated)))
    return generated


def run_visualization_phase(
    *,
    paths: Any,
    run_id: str,
    labeled_df: pd.DataFrame,
    features_df: pd.DataFrame,
    baseline_result: BaselinePhaseResult,
    bayesian_result: BayesianPhaseResult,
    eval_decision_result: EvalDecisionPhaseResult,
    previous_bayesian_convergence: dict[str, Any] | None,
    lead_time_max_lookback_steps: int,
    effective_seed: int,
    bayesian_convergence_path: Path | None = None,
) -> dict[str, Path]:
    payload_frame, payload_metadata = _collect_payload_inputs(
        run_id=run_id,
        labeled_df=labeled_df,
        features_df=features_df,
        baseline_result=baseline_result,
        bayesian_result=bayesian_result,
        eval_decision_result=eval_decision_result,
        bayesian_convergence_path=bayesian_convergence_path,
        bayesian_idata=bayesian_result.bayesian_idata,
    )
    payload_metadata["bayesian_idata"] = bayesian_result.bayesian_idata

    return _run_visualizations_from_payload(
        payload_frame=payload_frame,
        payload_metadata=payload_metadata,
        output_dir=Path(paths.outputs_figures),
        lead_time_max_lookback_steps=int(lead_time_max_lookback_steps),
        previous_bayesian_convergence=previous_bayesian_convergence,
        effective_seed=int(effective_seed),
    )


def run_visualization_phase_from_payload(
    payload_csv_path: Path,
    *,
    payload_metadata_path: Path | None = None,
    output_dir: Path,
    lead_time_max_lookback_steps: int = 8,
    previous_bayesian_convergence: dict[str, Any] | None = None,
    effective_seed: int = 42,
) -> dict[str, Path]:
    payload_frame = pd.read_csv(payload_csv_path)

    resolved_metadata_path = payload_metadata_path
    if resolved_metadata_path is None:
        candidate_path = payload_csv_path.with_name("visualization_payload_metadata.json")
        resolved_metadata_path = candidate_path if candidate_path.exists() else None

    payload_metadata: dict[str, Any] = {}
    if resolved_metadata_path is not None and resolved_metadata_path.exists():
        payload_metadata = json.loads(resolved_metadata_path.read_text(encoding="utf-8"))

    return _run_visualizations_from_payload(
        payload_frame=payload_frame,
        payload_metadata=payload_metadata,
        output_dir=output_dir,
        lead_time_max_lookback_steps=int(lead_time_max_lookback_steps),
        previous_bayesian_convergence=previous_bayesian_convergence,
        effective_seed=int(effective_seed),
    )
