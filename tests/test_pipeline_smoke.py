"""Smoke tests for pipeline modules."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import run_pipeline as run_pipeline_module

from run_pipeline import _apply_memory_optimization_filters, _load_curated_municipality_contract, run
from src.models.baselines.cv_splitter import TimeSeriesCVConfig
from src.models.baselines.train_baselines import BaselineTrainingConfig
from src.pipeline_runtime.phase_context import SharedPhaseState
from src.pipeline_runtime.phases_baseline import build_model_input_df, run_baseline_phase
from src.pipeline_runtime import config_runtime


@pytest.fixture(autouse=True)
def _use_test_curated_contract(tmp_path, monkeypatch):
    contract_path = tmp_path / "curated_municipalities_test.json"
    contract_path.write_text(
        json.dumps(
            {
                "version": "v1-smoke-tests",
                "source": "tests/test_pipeline_smoke.py",
                "selection": "synthetic_district_labels",
                "municipality_ids": ["A", "B", "C", "D"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(run_pipeline_module, "_DEFAULT_CURATED_MUNICIPALITIES_PATH", contract_path)

    cv_config_path = tmp_path / "cv_config_smoke.yaml"
    cv_config_path.write_text(
        "\n".join(
            [
                "strategy: time_series_split",
                "n_splits: 5",
                "gap: 0",
                "test_size: 12",
                "date_column: date",
                "target_column: outbreak_label",
                "first_valid_year: 2016",
                "last_valid_year: 2020",
                "start_train_year: 2015",
                "thesis_strict: false",
                "train_window_years: 5",
                "skip_single_class_folds: true",
                "minimum_evaluated_folds: 1",
                "fail_on_gate_violation: false",
            ]
        ),
        encoding="utf-8",
    )

    original_run = run_pipeline_module.run

    def run_with_relaxed_feature_gate(*, strict_feature_gate=False, **kwargs):
        kwargs.setdefault("cv_config_path", cv_config_path)
        return original_run(strict_feature_gate=bool(strict_feature_gate), **kwargs)

    monkeypatch.setattr(run_pipeline_module, "run", run_with_relaxed_feature_gate)
    monkeypatch.setitem(globals(), "run", run_with_relaxed_feature_gate)


def test_pipeline_smoke_with_synthetic_dataframe(tmp_path) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=8, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "state": ["S"] * 8,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0],
        }
    )
    raw_path = tmp_path / "synthetic_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_bayesian=True,
        skip_visualizations=True,
    )

    assert artifacts["labeled_data"].exists()
    assert artifacts["feature_matrix"].exists()
    assert artifacts["decision_alerts"].exists()
    assert artifacts["run_manifest"].exists()
    assert artifacts["fold_ledger"].exists()
    assert artifacts["run_metadata"].exists()


def test_model_input_forbidden_columns_are_dropped() -> None:
    frame = pd.DataFrame(
        {
            "outbreak_label": [0, 1],
            "outbreak_label_p75": [0, 1],
            "threshold_p75": [2, 3],
            "cases": [10, 20],
            "case_lag_1": [5, 10],
            "rainfall": [1.0, 2.0],
        }
    )

    sanitized, audit = build_model_input_df(frame)
    assert "outbreak_label" not in sanitized.columns
    assert "outbreak_label_p75" not in sanitized.columns
    assert "threshold_p75" not in sanitized.columns
    assert "cases" not in sanitized.columns
    assert "case_lag_1" in sanitized.columns
    assert "rainfall" in sanitized.columns
    assert set(audit["dropped_forbidden_columns"]) >= {"outbreak_label", "outbreak_label_p75", "threshold_p75", "cases"}


def test_baseline_phase_propagates_custom_splitter_to_train_baselines(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    paths = SimpleNamespace(
        outputs_reports=reports_dir,
        outputs_models=models_dir,
        outputs_metrics=metrics_dir,
    )

    labeled_df = pd.DataFrame(
        {
            "date": ["2016-01-01", "2017-01-01", "2018-01-01"],
            "district": ["A", "A", "A"],
            "cases": [1.0, 2.0, 3.0],
            "outbreak_label": [0, 1, 0],
            "threshold_p75": [1.0, 1.0, 1.0],
        }
    )
    features_df = pd.DataFrame(
        {
            "temp_anomaly": [0.1, 0.2, 0.3],
            "rainfall_4wk": [10.0, 11.0, 12.0],
        },
        index=labeled_df.index,
    )

    splitter_observed = {"used": False}
    train_seen = {"received_callable": False}

    def custom_generate_time_splits(df: pd.DataFrame, cv_cfg: TimeSeriesCVConfig):
        df
        cv_cfg
        splitter_observed["used"] = True
        yield np.array([0, 1]), np.array([2])

    def fake_train_baselines(
        X: pd.DataFrame,
        y: pd.Series,
        *,
        config,
        cv_config,
        model_names,
        build_fold_ledger_fn,
        generate_time_splits_fn,
    ):
        X
        y
        config
        model_names
        build_fold_ledger_fn
        split_fn = generate_time_splits_fn
        assert callable(split_fn)
        train_seen["received_callable"] = bool(split_fn is custom_generate_time_splits)
        training_frame = X.copy()
        training_frame["outbreak_label"] = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
        list(split_fn(training_frame, cv_config))
        return {}

    def fake_predict_baselines(models, X):
        models
        return pd.DataFrame(index=X.index)

    def fake_evaluate_baseline_predictions(*args, **kwargs):
        args
        kwargs
        return {"accuracy": 0.0}

    def fake_collect_oof_scores(*, output_root, expected_index):
        output_root
        return pd.Series(np.nan, index=expected_index, dtype="float64")

    def fake_collect_oof_predictions(*, output_root, expected_index):
        output_root
        return pd.DataFrame(index=expected_index)

    def fake_collect_oof_fold_ids(output_root):
        output_root
        return []

    def safe_write_json(payload: dict, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    run_baseline_phase(
        state=SharedPhaseState(run_id="test-run"),
        paths=paths,
        labeled_df=labeled_df,
        features_df=features_df,
        selected_percentile=75,
        effective_cv_config=TimeSeriesCVConfig(
            date_column="date",
            target_column="outbreak_label",
            start_train_year=2016,
            first_valid_year=2017,
            last_valid_year=2018,
            skip_single_class_folds=False,
        ),
        effective_seed=42,
        strict_feature_gate=True,
        baseline_compute_backend="cpu",
        skip_baselines=False,
        export_detailed_csv=False,
        model_names=["logistic_regression"],
        lead_time_max_lookback_steps=8,
        threshold_scope_audit={"checked": False},
        cv_ledger_callable=lambda df, cv_cfg: [{"status": "yielded", "valid_year": 2018}],
        cv_split_callable=custom_generate_time_splits,
        train_baselines_fn=fake_train_baselines,
        baseline_training_config_cls=BaselineTrainingConfig,
        predict_baselines_fn=fake_predict_baselines,
        evaluate_baseline_predictions_fn=fake_evaluate_baseline_predictions,
        collect_baseline_oof_scores_fn=fake_collect_oof_scores,
        collect_baseline_oof_predictions_fn=fake_collect_oof_predictions,
        collect_baseline_oof_fold_ids_fn=fake_collect_oof_fold_ids,
        safe_write_json_fn=safe_write_json,
    )

    assert train_seen["received_callable"] is True
    assert splitter_observed["used"] is True


def test_baseline_phase_cleans_stale_fold_artifacts_before_training(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    stale_root = models_dir / "baselines"
    stale_fold_dir = stale_root / "fold_99"
    stale_fold_dir.mkdir(parents=True, exist_ok=True)
    (stale_fold_dir / "predictions.csv").write_text("idx,ghost_model\n0,0.9\n", encoding="utf-8")
    (stale_fold_dir / "ghost_model.pkl").write_text("stale", encoding="utf-8")
    (stale_root / "cv_metrics.csv").write_text("fold,model\n99,ghost_model\n", encoding="utf-8")
    (stale_root / "cv_metrics_aggregate.csv").write_text("model,accuracy\nghost_model,1.0\n", encoding="utf-8")
    (stale_root / "fold_ledger.json").write_text("[]", encoding="utf-8")

    paths = SimpleNamespace(
        outputs_reports=reports_dir,
        outputs_models=models_dir,
        outputs_metrics=metrics_dir,
    )
    labeled_df = pd.DataFrame(
        {
            "date": ["2016-01-01", "2017-01-01", "2018-01-01"],
            "district": ["A", "A", "A"],
            "cases": [1.0, 2.0, 3.0],
            "outbreak_label": [0, 1, 0],
            "threshold_p75": [1.0, 1.0, 1.0],
        }
    )
    features_df = pd.DataFrame(
        {
            "temp_anomaly": [0.1, 0.2, 0.3],
            "rainfall_4wk": [10.0, 11.0, 12.0],
        },
        index=labeled_df.index,
    )

    observed = {"cleanup_happened_before_train": False}

    def fake_train_baselines(X, y, *, config, cv_config, model_names):
        X
        y
        config
        cv_config
        model_names
        observed["cleanup_happened_before_train"] = not any(stale_root.glob("fold_*"))
        return {}

    def fake_predict_baselines(models, X):
        models
        return pd.DataFrame(index=X.index)

    def fake_evaluate_baseline_predictions(*args, **kwargs):
        args
        kwargs
        return {"accuracy": 0.0}

    def fake_collect_oof_scores(*, output_root, expected_index):
        output_root
        return pd.Series(np.nan, index=expected_index, dtype="float64")

    def fake_collect_oof_predictions(*, output_root, expected_index):
        output_root
        return pd.DataFrame(index=expected_index)

    def fake_collect_oof_fold_ids(output_root):
        output_root
        return []

    def safe_write_json(payload: dict, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    run_baseline_phase(
        state=SharedPhaseState(run_id="test-run"),
        paths=paths,
        labeled_df=labeled_df,
        features_df=features_df,
        selected_percentile=75,
        effective_cv_config=TimeSeriesCVConfig(
            date_column="date",
            target_column="outbreak_label",
            start_train_year=2016,
            first_valid_year=2017,
            last_valid_year=2018,
            skip_single_class_folds=False,
        ),
        effective_seed=42,
        strict_feature_gate=True,
        baseline_compute_backend="cpu",
        skip_baselines=False,
        export_detailed_csv=False,
        model_names=["random_forest"],
        lead_time_max_lookback_steps=8,
        threshold_scope_audit={"checked": False},
        cv_ledger_callable=lambda df, cv_cfg: [{"status": "yielded", "valid_year": 2018}],
        cv_split_callable=lambda df, cv_cfg: iter(()),
        train_baselines_fn=fake_train_baselines,
        baseline_training_config_cls=BaselineTrainingConfig,
        predict_baselines_fn=fake_predict_baselines,
        evaluate_baseline_predictions_fn=fake_evaluate_baseline_predictions,
        collect_baseline_oof_scores_fn=fake_collect_oof_scores,
        collect_baseline_oof_predictions_fn=fake_collect_oof_predictions,
        collect_baseline_oof_fold_ids_fn=fake_collect_oof_fold_ids,
        safe_write_json_fn=safe_write_json,
    )

    assert observed["cleanup_happened_before_train"] is True
    assert not (stale_root / "fold_99").exists()
    assert not (stale_root / "cv_metrics.csv").exists()
    assert not (stale_root / "cv_metrics_aggregate.csv").exists()
    assert not (stale_root / "fold_ledger.json").exists()


def test_baseline_tracka_model_scores_filters_to_configured_model_names(tmp_path) -> None:
    reports_dir = tmp_path / "reports"
    models_dir = tmp_path / "models"
    metrics_dir = tmp_path / "metrics"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    paths = SimpleNamespace(
        outputs_reports=reports_dir,
        outputs_models=models_dir,
        outputs_metrics=metrics_dir,
    )
    labeled_df = pd.DataFrame(
        {
            "date": ["2016-01-01", "2017-01-01", "2018-01-01"],
            "district": ["A", "A", "A"],
            "cases": [1.0, 2.0, 3.0],
            "outbreak_label": [0, 1, 0],
            "threshold_p75": [1.0, 1.0, 1.0],
        }
    )
    features_df = pd.DataFrame(
        {
            "temp_anomaly": [0.1, 0.2, 0.3],
            "rainfall_4wk": [10.0, 11.0, 12.0],
        },
        index=labeled_df.index,
    )

    def fake_train_baselines(X, y, *, config, cv_config, model_names):
        X
        y
        config
        cv_config
        model_names
        return {"random_forest": object()}

    def fake_predict_baselines(models, X):
        models
        return pd.DataFrame({"random_forest": [0.2, 0.4, 0.6]}, index=X.index)

    def fake_evaluate_baseline_predictions(*args, **kwargs):
        args
        kwargs
        return {"accuracy": 0.5, "f1": 0.5}

    def fake_collect_oof_scores(*, output_root, expected_index):
        output_root
        return pd.Series([0.3, 0.5, 0.7], index=expected_index, dtype="float64")

    def fake_collect_oof_predictions(*, output_root, expected_index):
        output_root
        return pd.DataFrame(
            {
                "random_forest": [0.2, 0.4, 0.6],
                "ghost_removed_model": [0.9, 0.9, 0.9],
            },
            index=expected_index,
        )

    def fake_collect_oof_fold_ids(output_root):
        output_root
        return [1]

    def safe_write_json(payload: dict, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    run_baseline_phase(
        state=SharedPhaseState(run_id="test-run"),
        paths=paths,
        labeled_df=labeled_df,
        features_df=features_df,
        selected_percentile=75,
        effective_cv_config=TimeSeriesCVConfig(
            date_column="date",
            target_column="outbreak_label",
            start_train_year=2016,
            first_valid_year=2017,
            last_valid_year=2018,
            skip_single_class_folds=False,
            minimum_evaluated_folds=1,
        ),
        effective_seed=42,
        strict_feature_gate=True,
        baseline_compute_backend="cpu",
        skip_baselines=False,
        export_detailed_csv=False,
        model_names=["random_forest"],
        lead_time_max_lookback_steps=8,
        threshold_scope_audit={"checked": False},
        cv_ledger_callable=lambda df, cv_cfg: [{"status": "yielded", "valid_year": 2018}],
        cv_split_callable=lambda df, cv_cfg: iter(()),
        train_baselines_fn=fake_train_baselines,
        baseline_training_config_cls=BaselineTrainingConfig,
        predict_baselines_fn=fake_predict_baselines,
        evaluate_baseline_predictions_fn=fake_evaluate_baseline_predictions,
        collect_baseline_oof_scores_fn=fake_collect_oof_scores,
        collect_baseline_oof_predictions_fn=fake_collect_oof_predictions,
        collect_baseline_oof_fold_ids_fn=fake_collect_oof_fold_ids,
        safe_write_json_fn=safe_write_json,
    )

    score_path = metrics_dir / "tracka_model_scores.csv"
    assert score_path.exists()
    score_df = pd.read_csv(score_path)
    assert score_df["model"].tolist() == ["random_forest"]


def test_memory_optimization_default_noop_behavior() -> None:
    labeled = pd.DataFrame(
        {
            "date": ["2016-01-01", "2017-01-01"],
            "district": ["A", "B"],
            "cases": [1, 2],
        }
    )
    features = pd.DataFrame({"rainfall": [1.0, 2.0]}, index=labeled.index)

    filtered_labeled, filtered_features, report = _apply_memory_optimization_filters(
        labeled,
        features,
        config=config_runtime.MemoryOptimizationConfig(),
    )

    assert report["mode"] == "off"
    assert report["active"] is False
    assert len(filtered_labeled) == len(labeled)
    assert len(filtered_features) == len(features)


def test_curated_contract_selection_optional_but_unknown_keys_rejected(tmp_path) -> None:
    contract_path = tmp_path / "curated_ok.json"
    contract_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "source": "tests",
                "selection": "balanced_top300_snapshot",
                "municipality_ids": ["3304557", "3304557", " 2927408 "],
            }
        ),
        encoding="utf-8",
    )

    loaded = _load_curated_municipality_contract(contract_path)
    assert loaded["version"] == "v1"
    assert loaded["source"] == "tests"
    assert loaded["selection"] == "balanced_top300_snapshot"
    assert loaded["count"] == 2
    assert loaded["municipality_ids"] == ["2927408", "3304557"]

    bad_contract_path = tmp_path / "curated_bad.json"
    bad_contract_path.write_text(
        json.dumps(
            {
                "version": "v1",
                "source": "tests",
                "municipality_ids": ["3304557"],
                "unexpected": "should-fail",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown keys"):
        _load_curated_municipality_contract(bad_contract_path)


def test_memory_optimization_year_window_filtering() -> None:
    labeled = pd.DataFrame(
        {
            "date": ["2015-01-01", "2016-01-01", "2017-01-01", "2018-01-01"],
            "district": ["A", "A", "B", "B"],
            "cases": [1, 2, 3, 4],
        }
    )
    features = pd.DataFrame({"rainfall": [1.0, 2.0, 3.0, 4.0]}, index=labeled.index)
    cfg = config_runtime.MemoryOptimizationConfig(mode="year_window", train_year_window=(2016, 2017))

    filtered_labeled, filtered_features, report = _apply_memory_optimization_filters(labeled, features, config=cfg)

    kept_years = pd.to_datetime(filtered_labeled["date"]).dt.year.tolist()
    assert kept_years == [2016, 2017]
    assert len(filtered_features) == 2
    assert report["filters"]["year_window"]["applied"] is True


def test_memory_optimization_district_shard_is_deterministic() -> None:
    labeled = pd.DataFrame(
        {
            "date": ["2016-01-01"] * 6,
            "district": ["A", "B", "C", "D", "E", "F"],
            "cases": [1, 2, 3, 4, 5, 6],
        }
    )
    features = pd.DataFrame({"rainfall": [1.0] * 6}, index=labeled.index)
    cfg0 = config_runtime.MemoryOptimizationConfig(mode="district_shard", district_shard_count=2, district_shard_index=0)
    cfg1 = config_runtime.MemoryOptimizationConfig(mode="district_shard", district_shard_count=2, district_shard_index=1)

    shard0_a, _, _ = _apply_memory_optimization_filters(labeled, features, config=cfg0)
    shard0_b, _, _ = _apply_memory_optimization_filters(labeled, features, config=cfg0)
    shard1, _, _ = _apply_memory_optimization_filters(labeled, features, config=cfg1)

    districts0_a = set(shard0_a["district"].tolist())
    districts0_b = set(shard0_b["district"].tolist())
    districts1 = set(shard1["district"].tolist())

    assert districts0_a == districts0_b
    assert districts0_a.isdisjoint(districts1)


def test_headline_baseline_metrics_are_oof(tmp_path, monkeypatch) -> None:
    rows: list[dict[str, object]] = []
    for year in range(2009, 2020):
        for district, bias in (("A", 0), ("B", 1)):
            cases = (year % 3) + bias
            rows.append(
                {
                    "date": f"{year}-01-01",
                    "district": district,
                    "state": "S",
                    "cases": cases,
                    "rainfall": float((year % 5) + bias),
                    "temperature": float(26 + (year % 4)),
                    "humidity": float(60 + (year % 6)),
                }
            )

    raw_path = tmp_path / "synthetic_oof_raw.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    n_rows = len(rows)
    oof_scores = pd.Series([0.2, 0.8, float("nan")] * ((n_rows // 3) + 1), dtype="float64").iloc[:n_rows]

    def fake_collect_baseline_oof_scores(*, output_root, expected_index):
        output_root
        return pd.Series(oof_scores.to_numpy(), index=expected_index, dtype="float64")

    observed_lengths: list[int] = []

    def fake_evaluate_baseline_predictions(y_true, y_pred_proba, threshold=0.5, **kwargs):
        threshold
        kwargs
        observed_lengths.append(len(y_true))
        return {
            "accuracy": float(len(y_true)),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "kappa": 0.0,
            "false_alarm_rate": 0.0,
            "roc_auc": 0.0,
            "pr_auc": 0.0,
            "lead_time_mean": 0.0,
            "lead_time_utility": 0.0,
        }

    monkeypatch.setattr("run_pipeline._collect_baseline_oof_scores", fake_collect_baseline_oof_scores)
    monkeypatch.setattr("run_pipeline.evaluate_baseline_predictions", fake_evaluate_baseline_predictions)

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2009,
        end_year=2019,
        skip_bayesian=True,
        skip_visualizations=True,
        model_names=["logistic_regression"],
    )

    assert "baseline_metrics" in artifacts
    assert "baseline_metrics_fullfit" in artifacts
    assert len(observed_lengths) >= 2
    assert observed_lengths[0] == n_rows
    assert observed_lengths[1] == int(oof_scores.notna().sum())


def test_bayesian_count_target_and_decision_uses_oof_or_passthrough(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=8, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "state": ["S"] * 8,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0],
        }
    )
    raw_path = tmp_path / "synthetic_bayes_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    observed_count_targets: list[pd.Series] = []

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings):
        features_df
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        observed_count_targets.append(pd.to_numeric(count_target, errors="coerce").fillna(0.0))
        n = len(count_target)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.2] * n,
                "risk_q05": [0.1] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.2] * n,
            }
        )
        return risk_frame, None, {
            "fallback_used": False,
            "degraded_mode": False,
            "mode_used": "full_latent_ar",
            "threshold_basis": "provided_series",
            "threshold_default": 1.0,
            "interval_source": "posterior",
            "posterior_samples_used": 10,
        }

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
    )

    assert observed_count_targets
    assert observed_count_targets[0].tolist() == synthetic["cases"].astype(float).tolist()

    decision = pd.read_csv(artifacts["decision_alerts"])
    assert "risk_q95" in decision.columns
    assert "risk_score_basis" in decision.columns
    assert (decision["risk_score"] == decision["risk_q95"]).all()
    assert (decision["risk_score_basis"] == "target_passthrough").all()


def test_bayesian_subset_metadata_emitted(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2014-01-01", periods=24, freq="MS").astype(str),
            "district": (["A", "B", "C", "D"] * 6),
            "state": ["S"] * 24,
            "cases": list(range(1, 25)),
            "rainfall": [float(i % 7) for i in range(24)],
            "temperature": [28.0 + float(i % 3) for i in range(24)],
            "humidity": [60.0 + float(i % 5) for i in range(24)],
        }
    )
    raw_path = tmp_path / "synthetic_subset_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    model_config_path = tmp_path / "model_config_memory.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "random_seed: 123",
                "memory_optimization:",
                "  mode: off",
                "  bayesian_subset:",
                "    enabled: true",
                "    max_rows: 8",
                "    strategy: top_cases",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings, **kwargs):
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        kwargs
        n = len(features_df)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.4] * n,
                "risk_q05": [0.2] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.4] * n,
            }
        )
        return risk_frame, None, {
            "fallback_used": False,
            "degraded_mode": False,
            "mode_used": "full_latent_ar",
            "threshold_basis": "provided_series",
            "threshold_default": 1.0,
            "interval_source": "posterior",
            "posterior_samples_used": 10,
        }

    def fake_collect_bayesian_oof_scores(**kwargs):
        features = kwargs["features_df"]
        return pd.Series([0.5] * len(features), index=features.index, dtype="float64")

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr("run_pipeline._collect_bayesian_oof_scores", fake_collect_bayesian_oof_scores)

    artifacts = run(
        model_config_path=model_config_path,
        raw_data_path=raw_path,
        start_year=2014,
        end_year=2015,
        skip_baselines=True,
        skip_visualizations=True,
    )

    bayes_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))
    memory_report = json.loads(artifacts["memory_optimization_report"].read_text(encoding="utf-8"))
    subset_meta = bayes_meta.get("bayesian_subset", {})

    assert subset_meta.get("enabled") is True
    assert subset_meta.get("applied") is True
    assert int(subset_meta.get("selected_rows", 0)) <= 8
    assert memory_report.get("mode") == "off"


def test_bayesian_profiles_route_fullfit_and_oof_settings(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2014-01-01", periods=24, freq="MS").astype(str),
            "district": (["A", "B", "C", "D"] * 6),
            "state": ["S"] * 24,
            "cases": list(range(1, 25)),
            "rainfall": [float(i % 7) for i in range(24)],
            "temperature": [28.0 + float(i % 3) for i in range(24)],
            "humidity": [60.0 + float(i % 5) for i in range(24)],
        }
    )
    raw_path = tmp_path / "synthetic_profile_routing_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    model_config_path = tmp_path / "model_config_profiles.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "bayesian_model:",
                "  sampling_backend: auto",
                "  draws: 800",
                "  tune: 1200",
                "  chains: 2",
                "  target_accept: 0.99",
                "bayesian_model_profiles:",
                "  cv:",
                "    draws: 120",
                "    tune: 200",
                "    chains: 1",
                "    bayesian_progress: false",
                "  final:",
                "    draws: 900",
                "    tune: 1300",
                "    chains: 2",
                "    bayesian_progress: true",
            ]
        ),
        encoding="utf-8",
    )

    observed_fullfit: dict[str, object] = {}
    observed_oof: dict[str, object] = {}

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings, **kwargs):
        features_df
        count_target
        outbreak_threshold
        strict_dependencies
        kwargs
        observed_fullfit.update(
            {
                "draws": bayesian_settings.get("draws"),
                "tune": bayesian_settings.get("tune"),
                "chains": bayesian_settings.get("chains"),
            }
        )
        n = len(features_df)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.4] * n,
                "risk_q05": [0.2] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.4] * n,
            }
        )
        return risk_frame, None, {
            "fallback_used": False,
            "degraded_mode": False,
            "mode_used": "full_latent_ar",
            "threshold_basis": "provided_series",
            "threshold_default": 1.0,
            "interval_source": "posterior",
            "posterior_samples_used": 10,
        }

    def fake_collect_bayesian_oof_scores(**kwargs):
        settings = kwargs.get("bayesian_settings", {})
        observed_oof.update(
            {
                "draws": settings.get("draws"),
                "tune": settings.get("tune"),
                "chains": settings.get("chains"),
            }
        )
        features = kwargs["features_df"]
        return pd.Series([0.5] * len(features), index=features.index, dtype="float64")

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr("run_pipeline._collect_bayesian_oof_scores", fake_collect_bayesian_oof_scores)

    run(
        model_config_path=model_config_path,
        raw_data_path=raw_path,
        start_year=2014,
        end_year=2015,
        skip_baselines=True,
        skip_visualizations=True,
    )

    assert int(observed_fullfit.get("draws", 0)) == 900
    assert int(observed_oof.get("draws", 0)) == 120
    assert int(observed_fullfit.get("tune", 0)) == 1300
    assert int(observed_oof.get("tune", 0)) == 200


def test_bayesian_profile_metadata_fields_present_and_consistent(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=16, freq="W").astype(str),
            "district": ["A", "A", "B", "B"] * 4,
            "state": ["S"] * 16,
            "cases": [1, 3, 2, 4, 5, 2, 1, 6, 3, 2, 4, 5, 2, 1, 3, 4],
            "rainfall": [float(i % 5) for i in range(16)],
            "temperature": [27.0 + float(i % 3) for i in range(16)],
            "humidity": [60.0 + float(i % 4) for i in range(16)],
        }
    )
    raw_path = tmp_path / "synthetic_profile_metadata_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    model_config_path = tmp_path / "model_config_profiles_meta.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "bayesian_model_profiles:",
                "  cv:",
                "    draws: 120",
                "    tune: 200",
                "  final:",
                "    draws: 900",
                "    tune: 1300",
                "  dev:",
                "    draws: 60",
                "    tune: 80",
            ]
        ),
        encoding="utf-8",
    )

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings, **kwargs):
        features_df
        count_target
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        kwargs
        n = len(features_df)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.4] * n,
                "risk_q05": [0.2] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.4] * n,
            }
        )
        return risk_frame, None, {
            "fallback_used": False,
            "degraded_mode": False,
            "mode_used": "full_latent_ar",
            "threshold_basis": "provided_series",
            "threshold_default": 1.0,
            "interval_source": "posterior",
            "posterior_samples_used": 10,
        }

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr(
        "run_pipeline._collect_bayesian_oof_scores",
        lambda **kwargs: pd.Series([0.5] * len(kwargs["features_df"]), index=kwargs["features_df"].index, dtype="float64"),
    )

    artifacts = run(
        model_config_path=model_config_path,
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
    )

    run_meta = json.loads(artifacts["run_metadata"].read_text(encoding="utf-8"))
    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))

    profile_usage_run = run_meta.get("bayesian_profile_usage", {})
    profile_usage_risk = risk_meta.get("bayesian_profile_usage", {})
    assert profile_usage_run.get("fullfit_profile_name") == "final"
    assert profile_usage_run.get("oof_profile_name") == "cv"
    assert profile_usage_run.get("cv_profile_differs_from_final") is True
    assert "draws" in profile_usage_run.get("cv_vs_final_diff_keys", [])
    assert profile_usage_risk.get("fullfit_profile_name") == "final"
    assert profile_usage_risk.get("oof_profile_name") == "cv"
    assert "warning_flags" in profile_usage_run
    assert "cv_subset_mode_active" in run_meta.get("bayesian_profile_usage", {})
    assert "cv_subset_mode_active" in risk_meta


def test_degraded_run_suppresses_headline_comparison_on_insufficient_folds(tmp_path, monkeypatch) -> None:
    rows: list[dict[str, object]] = []
    for year in range(2009, 2020):
        rows.append(
            {
                "date": f"{year}-01-01",
                "district": "A",
                "state": "S",
                "cases": year % 3,
                "rainfall": float(year % 5),
                "temperature": float(26 + (year % 4)),
                "humidity": float(60 + (year % 6)),
            }
        )
    raw_path = tmp_path / "synthetic_degraded_raw.csv"
    pd.DataFrame(rows).to_csv(raw_path, index=False)

    oof_scores = pd.Series([0.2, 0.8] * 6, dtype="float64").iloc[: len(rows)]

    def fake_collect_baseline_oof_scores(*, output_root, expected_index):
        output_root
        return pd.Series(oof_scores.to_numpy(), index=expected_index, dtype="float64")

    def fake_collect_baseline_oof_fold_ids(output_root):
        output_root
        return []

    monkeypatch.setattr("run_pipeline._collect_baseline_oof_scores", fake_collect_baseline_oof_scores)
    monkeypatch.setattr("run_pipeline._collect_baseline_oof_fold_ids", fake_collect_baseline_oof_fold_ids)

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2009,
        end_year=2019,
        skip_bayesian=True,
        skip_visualizations=True,
        model_names=["logistic_regression"],
    )

    degraded = json.loads(artifacts["degraded_run"].read_text(encoding="utf-8"))
    manifest = json.loads(artifacts["run_manifest"].read_text(encoding="utf-8"))
    assert degraded["degraded"] is True
    assert degraded["suppress_headline_comparison"] is True
    assert manifest["headline_claims"]["suppressed"] is True
    assert degraded["suppress_headline_comparison"] == manifest["headline_claims"]["suppressed"]
    assert "baseline_metrics" in artifacts
    baseline_payload = json.loads(artifacts["baseline_metrics"].read_text(encoding="utf-8"))
    assert baseline_payload.get("suppressed") is True


def test_manifest_contract_artifacts_and_run_id_parity(tmp_path) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=8, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "state": ["S"] * 8,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0],
        }
    )
    raw_path = tmp_path / "synthetic_contract_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_bayesian=True,
        skip_visualizations=True,
    )

    manifest = json.loads(artifacts["run_manifest"].read_text(encoding="utf-8"))
    metadata = json.loads(artifacts["run_metadata"].read_text(encoding="utf-8"))
    degraded = json.loads(artifacts["degraded_run"].read_text(encoding="utf-8"))
    fold_ledger = json.loads(artifacts["fold_ledger"].read_text(encoding="utf-8"))
    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))

    required = set(manifest["contract_required_artifacts"])
    assert required.issubset(set(manifest["artifacts"].keys()))
    for key in required:
        assert key in artifacts
        assert artifacts[key].exists()

    run_id = manifest["run_id"]
    assert metadata["run_id"] == run_id
    assert degraded["run_id"] == run_id
    assert fold_ledger["run_id"] == run_id
    assert risk_meta.get("climate_covariates") == ["month", "year", "weekofyear"]
    assert metadata.get("bayesian_covariates_effective") == ["month", "year", "weekofyear"]
    assert metadata.get("bayesian_covariates_requested") == ["month", "year", "weekofyear"]
    assert manifest.get("bayesian_covariates", {}).get("effective") == ["month", "year", "weekofyear"]
    assert manifest.get("bayesian_covariates", {}).get("requested") == ["month", "year", "weekofyear"]
    assert fold_ledger.get("bayesian_covariates_effective") == ["month", "year", "weekofyear"]
    assert fold_ledger.get("bayesian_covariates_requested") == ["month", "year", "weekofyear"]
    assert degraded.get("bayesian_covariates_effective") == ["month", "year", "weekofyear"]
    assert degraded.get("bayesian_covariates_requested") == ["month", "year", "weekofyear"]


def test_bayesian_track_suppressed_when_no_viable_covariates_remain(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=8, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "state": ["S"] * 8,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0],
        }
    )
    raw_path = tmp_path / "synthetic_no_viable_covariates.csv"
    synthetic.to_csv(raw_path, index=False)

    def _should_not_run_bayesian_track(*args, **kwargs):
        args
        kwargs
        raise AssertionError("Bayesian model execution should be skipped when no viable covariates remain")

    monkeypatch.setattr("run_pipeline._run_bayesian_track", _should_not_run_bayesian_track)

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
        bayesian_overrides={"climate_covariates": ["missing_covariate_a", "missing_covariate_b"]},
    )

    bayes_metrics = json.loads(artifacts["bayesian_metrics"].read_text(encoding="utf-8"))
    bayes_fullfit_metrics = json.loads(artifacts["bayesian_metrics_fullfit"].read_text(encoding="utf-8"))
    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))
    run_meta = json.loads(artifacts["run_metadata"].read_text(encoding="utf-8"))
    degraded = json.loads(artifacts["degraded_run"].read_text(encoding="utf-8"))
    manifest = json.loads(artifacts["run_manifest"].read_text(encoding="utf-8"))

    assert bayes_metrics.get("suppressed") is True
    assert bayes_metrics.get("reason") == "bayesian_no_viable_covariates"
    assert bayes_fullfit_metrics.get("suppressed") is True
    assert bayes_fullfit_metrics.get("reason") == "bayesian_no_viable_covariates"
    assert risk_meta.get("mode_used") == "suppressed_no_viable_covariates"
    assert risk_meta.get("degraded_mode") is True
    assert risk_meta.get("climate_covariates") == []
    assert risk_meta.get("climate_covariates_requested") == ["missing_covariate_a", "missing_covariate_b"]
    assert run_meta.get("bayesian_covariates_effective") == []
    assert run_meta.get("bayesian_covariates_requested") == ["missing_covariate_a", "missing_covariate_b"]
    assert run_meta.get("bayesian_covariate_selection", {}).get("viable_count") == 0
    assert manifest.get("bayesian_covariates", {}).get("effective") == []
    assert manifest.get("bayesian_covariates", {}).get("requested") == ["missing_covariate_a", "missing_covariate_b"]
    assert manifest.get("bayesian_covariates", {}).get("selection", {}).get("viable_count") == 0
    fold_ledger = json.loads(artifacts["fold_ledger"].read_text(encoding="utf-8"))
    assert fold_ledger.get("bayesian_covariates_effective") == []
    assert fold_ledger.get("bayesian_covariates_requested") == ["missing_covariate_a", "missing_covariate_b"]
    assert degraded.get("bayesian_covariates_effective") == []
    assert degraded.get("bayesian_covariates_requested") == ["missing_covariate_a", "missing_covariate_b"]
    assert any(reason.get("code") == "bayesian_no_viable_covariates" for reason in degraded.get("reasons", []))


def test_stale_headline_contract_files_removed_and_regenerated_on_rerun(tmp_path) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=10, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "state": ["S"] * 10,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1, 2, 3],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0, 4.0, 5.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0, 28.0, 27.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0, 63.0, 61.0],
        }
    )
    raw_path = tmp_path / "synthetic_rerun_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    first_artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_bayesian=True,
        skip_visualizations=True,
    )

    baseline_path = first_artifacts["baseline_metrics"]
    track_csv_path = first_artifacts["track_comparison_csv"]
    track_md_path = first_artifacts["track_comparison_md"]

    baseline_path.write_text('{"sentinel":"stale"}', encoding="utf-8")
    track_csv_path.write_text("sentinel\nold\n", encoding="utf-8")
    track_md_path.write_text("stale", encoding="utf-8")

    second_artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_bayesian=True,
        skip_visualizations=True,
    )

    assert second_artifacts["baseline_metrics"].read_text(encoding="utf-8") != '{"sentinel":"stale"}'
    assert "sentinel" not in second_artifacts["track_comparison_csv"].read_text(encoding="utf-8")
    assert second_artifacts["track_comparison_md"].read_text(encoding="utf-8") != "stale"


def test_bayesian_headline_suppressed_when_convergence_false(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=12, freq="W").astype(str),
            "district": ["A"] * 6 + ["B"] * 6,
            "state": ["S"] * 12,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1, 2, 3, 5, 7],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0, 28.0, 27.0, 26.0, 25.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0, 63.0, 61.0, 60.0, 62.0],
        }
    )
    raw_path = tmp_path / "synthetic_convergence_false_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings):
        features_df
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        n = len(count_target)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.6] * n,
                "risk_q05": [0.4] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.6] * n,
            }
        )
        return risk_frame, object(), {"degraded_mode": False, "fallback_used": False, "mode_used": "full_latent_ar"}

    def fake_collect_bayesian_oof_scores(
        *,
        features_df,
        outbreak_target,
        count_target,
        strict_dependencies,
        bayesian_settings,
        cv_config,
        threshold_series,
        fail_on_error,
        date_column="date",
        target_column="outbreak_label",
    ):
        features_df
        outbreak_target
        count_target
        strict_dependencies
        bayesian_settings
        cv_config
        threshold_series
        fail_on_error
        date_column
        target_column
        return pd.Series([0.2, 0.8] * 6, index=features_df.index, dtype="float64")

    def fake_eval_bayes(y_true, y_score, **kwargs):
        y_true
        y_score
        kwargs
        return {"brier_score": 0.2, "pr_auc": 0.6, "roc_auc": 0.7, "lead_time_mean": 1.0}

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr("run_pipeline._collect_bayesian_oof_scores", fake_collect_bayesian_oof_scores)
    monkeypatch.setattr("run_pipeline.evaluate_bayesian_predictions", fake_eval_bayes)
    monkeypatch.setattr(
        "run_pipeline.check_convergence",
        lambda *args, **kwargs: {
            "converged": False,
            "divergences": 2.0,
            "max_tree_depth": 14.0,
            "r_hat_max": 1.2,
            "ess_min": 20.0,
            "divergence_threshold": 0.0,
            "max_tree_depth_threshold": 12.0,
            "rhat_threshold": 1.05,
            "ess_threshold": 200.0,
        },
    )
    monkeypatch.setattr(
        "run_pipeline.extract_rhat_ess",
        lambda _idata: pd.DataFrame({"parameter": ["x"], "r_hat": [1.2], "ess_bulk": [20.0]}),
    )

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
    )

    bayes_metrics = json.loads(artifacts["bayesian_metrics"].read_text(encoding="utf-8"))
    degraded = json.loads(artifacts["degraded_run"].read_text(encoding="utf-8"))
    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))

    assert bayes_metrics.get("suppressed") is True
    assert bayes_metrics.get("reason") == "bayesian_convergence_not_met"
    assert any(reason.get("code") == "bayesian_convergence_failed" for reason in degraded.get("reasons", []))
    assert risk_meta["headline_eligible"] is False
    assert risk_meta["converged"] is False


def test_bayesian_convergence_warn_mode_prevents_strict_crash(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=12, freq="W").astype(str),
            "district": ["A"] * 6 + ["B"] * 6,
            "state": ["S"] * 12,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1, 2, 3, 5, 7],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0, 28.0, 27.0, 26.0, 25.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0, 63.0, 61.0, 60.0, 62.0],
        }
    )
    raw_path = tmp_path / "synthetic_warn_mode_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings):
        features_df
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        n = len(count_target)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.6] * n,
                "risk_q05": [0.4] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.6] * n,
            }
        )
        return risk_frame, object(), {"degraded_mode": False, "fallback_used": False, "mode_used": "full_latent_ar"}

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr(
        "run_pipeline._collect_bayesian_oof_scores",
        lambda **kwargs: pd.Series([0.2, 0.8] * 6, index=kwargs["features_df"].index, dtype="float64"),
    )
    monkeypatch.setattr(
        "run_pipeline.evaluate_bayesian_predictions",
        lambda *args, **kwargs: {"brier_score": 0.2, "pr_auc": 0.6, "roc_auc": 0.7, "lead_time_mean": 1.0},
    )
    monkeypatch.setattr(
        "run_pipeline.check_convergence",
        lambda *args, **kwargs: {
            "converged": False,
            "divergences": 2.0,
            "max_tree_depth": 14.0,
            "r_hat_max": 1.2,
            "ess_min": 20.0,
            "divergence_threshold": 0.0,
            "max_tree_depth_threshold": 12.0,
            "rhat_threshold": 1.05,
            "ess_threshold": 200.0,
        },
    )
    monkeypatch.setattr(
        "run_pipeline.extract_rhat_ess",
        lambda _idata: pd.DataFrame({"parameter": ["x"], "r_hat": [1.2], "ess_bulk": [20.0]}),
    )

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
        strict_bayesian_deps=True,
        bayesian_overrides={"convergence_failure_mode": "warn"},
    )

    bayes_metrics = json.loads(artifacts["bayesian_metrics"].read_text(encoding="utf-8"))
    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))
    assert bayes_metrics.get("suppressed") is True
    assert bayes_metrics.get("reason") == "bayesian_convergence_not_met"
    assert risk_meta.get("convergence_failure_mode") == "warn"


def test_bayesian_low_sample_auto_downgrade_to_warn(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=12, freq="W").astype(str),
            "district": ["A"] * 6 + ["B"] * 6,
            "state": ["S"] * 12,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1, 2, 3, 5, 7],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0, 28.0, 27.0, 26.0, 25.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0, 63.0, 61.0, 60.0, 62.0],
        }
    )
    raw_path = tmp_path / "synthetic_low_sample_warn_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings):
        features_df
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        n = len(count_target)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.6] * n,
                "risk_q05": [0.4] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.6] * n,
            }
        )
        return risk_frame, object(), {"degraded_mode": False, "fallback_used": False, "mode_used": "full_latent_ar"}

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr(
        "run_pipeline._collect_bayesian_oof_scores",
        lambda **kwargs: pd.Series([0.2, 0.8] * 6, index=kwargs["features_df"].index, dtype="float64"),
    )
    monkeypatch.setattr(
        "run_pipeline.evaluate_bayesian_predictions",
        lambda *args, **kwargs: {"brier_score": 0.2, "pr_auc": 0.6, "roc_auc": 0.7, "lead_time_mean": 1.0},
    )
    monkeypatch.setattr(
        "run_pipeline.check_convergence",
        lambda *args, **kwargs: {
            "converged": False,
            "divergences": 2.0,
            "max_tree_depth": 14.0,
            "r_hat_max": 1.2,
            "ess_min": 20.0,
            "divergence_threshold": 0.0,
            "max_tree_depth_threshold": 12.0,
            "rhat_threshold": 1.05,
            "ess_threshold": 200.0,
        },
    )
    monkeypatch.setattr(
        "run_pipeline.extract_rhat_ess",
        lambda _idata: pd.DataFrame({"parameter": ["x"], "r_hat": [1.2], "ess_bulk": [20.0]}),
    )

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
        strict_bayesian_deps=True,
        bayesian_overrides={"draws": 20, "chains": 1},
    )

    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))
    assert risk_meta.get("convergence_failure_mode") == "warn"
    assert risk_meta.get("convergence_failure_mode_explicit") is False
    assert "auto-downgraded to warn" in str(risk_meta.get("convergence_failure_mode_auto_reason"))


def test_bayesian_backend_metadata_truthful_cpu_fallback(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=8, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "state": ["S"] * 8,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0],
        }
    )
    raw_path = tmp_path / "synthetic_backend_meta_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    model_config_path = tmp_path / "model_config_backend_meta.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "compute_backend:",
                "  mode: macos_metal",
                "  prefer_gpu_for: both",
                "  fallback: cpu",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "run_pipeline.runtime_backend.resolve_backends",
        lambda _cfg: {
            "baseline_backend": "cpu",
            "bayesian_backend": "macos_metal",
            "configured": True,
            "requested": {"mode": "macos_metal", "prefer_gpu_for": "both", "fallback": "cpu", "configured": True},
            "effective": {"baseline_backend": "cpu", "bayesian_backend": "macos_metal"},
            "fallback": "cpu",
            "fallback_used": True,
            "reason_logs": [],
            "capabilities": {"macos_metal": True, "nvidia_cuda": False, "details": {}},
            "tracks": {},
        },
    )

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings):
        features_df
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        n = len(count_target)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.3] * n,
                "risk_q05": [0.1] * n,
                "risk_q95": [0.7] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.3] * n,
            }
        )
        return risk_frame, None, {
            "degraded_mode": False,
            "fallback_used": False,
            "mode_used": "full_latent_ar",
            "threshold_basis": "provided_series",
            "threshold_default": 1.0,
            "interval_source": "posterior",
            "posterior_samples_used": 10,
        }

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr(
        "run_pipeline._collect_bayesian_oof_scores",
        lambda **kwargs: pd.Series([0.4] * len(kwargs["features_df"]), index=kwargs["features_df"].index, dtype="float64"),
    )

    artifacts = run(
        model_config_path=model_config_path,
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
    )

    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))
    run_meta = json.loads(artifacts["run_metadata"].read_text(encoding="utf-8"))

    assert risk_meta.get("requested_backend") == "macos_metal"
    assert risk_meta.get("resolved_backend") == "macos_metal"
    assert risk_meta.get("actual_runtime_backend") == "cpu"
    assert risk_meta.get("backend_implemented") is False
    assert "does not currently support" in str(risk_meta.get("fallback_reason"))
    assert risk_meta.get("sampling_backend_requested") == "auto"
    assert risk_meta.get("sampling_backend_effective") == "pymc"
    assert risk_meta.get("sampling_backend_fallback_reason") == risk_meta.get("fallback_reason")
    assert risk_meta.get("actual_runtime_backend") == "cpu"

    bayesian_backend_meta = run_meta.get("bayesian_backend", {})
    assert bayesian_backend_meta.get("requested_backend") == "macos_metal"
    assert bayesian_backend_meta.get("resolved_backend") == "macos_metal"
    assert bayesian_backend_meta.get("actual_runtime_backend") == "cpu"
    assert bayesian_backend_meta.get("backend_implemented") is False
    assert bayesian_backend_meta.get("sampling_backend_requested") == "auto"
    assert bayesian_backend_meta.get("sampling_backend_effective") == "pymc"
    assert bayesian_backend_meta.get("sampling_backend_fallback_reason") == risk_meta.get("fallback_reason")
    assert run_meta.get("sampling_backend_requested") == "auto"
    assert run_meta.get("sampling_backend_effective") == "pymc"
    assert run_meta.get("sampling_backend_fallback_reason") == risk_meta.get("fallback_reason")


def test_runtime_sampling_backend_is_wired_into_bayesian_settings_and_metadata(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=8, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "state": ["S"] * 8,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0],
        }
    )
    raw_path = tmp_path / "synthetic_runtime_sampling_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    model_config_path = tmp_path / "model_config_runtime_sampling.yaml"
    model_config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  sampling_backend: jax_numpyro",
                "compute_backend:",
                "  mode: cpu",
                "  prefer_gpu_for: both",
                "  fallback: cpu",
            ]
        ),
        encoding="utf-8",
    )

    observed_sampling_backends: list[str] = []

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings):
        features_df
        count_target
        outbreak_threshold
        strict_dependencies
        observed_sampling_backends.append(str(bayesian_settings.get("sampling_backend", "")))
        n = len(features_df)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.3] * n,
                "risk_q05": [0.1] * n,
                "risk_q95": [0.7] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.3] * n,
            }
        )
        return risk_frame, None, {
            "degraded_mode": False,
            "fallback_used": False,
            "mode_used": "full_latent_ar",
            "sampling_backend_requested": "jax_numpyro",
            "sampling_backend_effective": "pymc",
            "sampling_backend_fallback_reason": "JAX sampler unavailable; falling back to PyMC CPU sampler (test)",
            "requested_backend": "cpu",
            "resolved_backend": "cpu",
            "actual_runtime_backend": "cpu",
            "backend_implemented": True,
            "fallback_reason": "JAX sampler unavailable; falling back to PyMC CPU sampler (test)",
            "compute_backend_requested": "cpu",
            "compute_backend_effective": "cpu",
            "compute_backend_runtime": "cpu",
            "compute_backend_fallback_used": True,
            "compute_backend_fallback_reason": "JAX sampler unavailable; falling back to PyMC CPU sampler (test)",
        }

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr(
        "run_pipeline._collect_bayesian_oof_scores",
        lambda **kwargs: pd.Series([0.4] * len(kwargs["features_df"]), index=kwargs["features_df"].index, dtype="float64"),
    )

    artifacts = run(
        model_config_path=model_config_path,
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
    )

    assert observed_sampling_backends
    assert observed_sampling_backends[0] == "jax_numpyro"

    risk_meta = json.loads(artifacts["bayesian_risk_metadata"].read_text(encoding="utf-8"))
    run_meta = json.loads(artifacts["run_metadata"].read_text(encoding="utf-8"))

    assert risk_meta.get("sampling_backend_requested") == "jax_numpyro"
    assert risk_meta.get("sampling_backend_effective") == "pymc"
    assert "falling back" in str(risk_meta.get("sampling_backend_fallback_reason", "")).lower()
    assert risk_meta.get("actual_runtime_backend") == "cpu"
    assert run_meta.get("sampling_backend_requested") == "jax_numpyro"
    assert run_meta.get("sampling_backend_effective") == "pymc"
    assert "falling back" in str(run_meta.get("sampling_backend_fallback_reason", "")).lower()
    assert run_meta.get("bayesian_backend", {}).get("sampling_backend_requested_source") == "runtime"


def test_bayesian_convergence_summary_regenerates_and_updates(tmp_path, monkeypatch) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=12, freq="W").astype(str),
            "district": ["A"] * 6 + ["B"] * 6,
            "state": ["S"] * 12,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1, 2, 3, 5, 7],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0, 4.0, 5.0, 1.0, 3.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0, 28.0, 27.0, 26.0, 25.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0, 63.0, 61.0, 60.0, 62.0],
        }
    )
    raw_path = tmp_path / "synthetic_convergence_summary_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    def fake_run_bayesian_track(features_df, count_target, *, outbreak_threshold, strict_dependencies, bayesian_settings):
        features_df
        outbreak_threshold
        strict_dependencies
        bayesian_settings
        n = len(count_target)
        risk_frame = pd.DataFrame(
            {
                "risk_mean": [0.6] * n,
                "risk_q05": [0.4] * n,
                "risk_q95": [0.8] * n,
                "threshold_cases": [1.0] * n,
                "bayesian_risk": [0.6] * n,
            }
        )
        return risk_frame, object(), {"degraded_mode": False, "fallback_used": False, "mode_used": "full_latent_ar"}

    monkeypatch.setattr("run_pipeline._run_bayesian_track", fake_run_bayesian_track)
    monkeypatch.setattr(
        "run_pipeline._collect_bayesian_oof_scores",
        lambda **kwargs: pd.Series([0.2, 0.8] * 6, index=kwargs["features_df"].index, dtype="float64"),
    )
    monkeypatch.setattr(
        "run_pipeline.evaluate_bayesian_predictions",
        lambda *args, **kwargs: {"brier_score": 0.2, "pr_auc": 0.6, "roc_auc": 0.7, "lead_time_mean": 1.0},
    )
    monkeypatch.setattr(
        "run_pipeline.extract_rhat_ess",
        lambda _idata: pd.DataFrame({"parameter": ["x"], "r_hat": [1.01], "ess_bulk": [300.0]}),
    )

    state = {"converged": False, "divergences": 3.0}

    def fake_check_convergence(*args, **kwargs):
        args
        kwargs
        return {
            "converged": bool(state["converged"]),
            "divergences": float(state["divergences"]),
            "max_tree_depth": 10.0,
            "r_hat_max": 1.01,
            "ess_min": 300.0,
            "divergence_threshold": 25.0,
            "max_tree_depth_threshold": 14.0,
            "rhat_threshold": 1.05,
            "ess_threshold": 200.0,
        }

    monkeypatch.setattr("run_pipeline.check_convergence", fake_check_convergence)

    artifacts_1 = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
    )
    summary_csv_1 = artifacts_1["bayesian_convergence_summary_csv"].read_text(encoding="utf-8")
    summary_md_1 = artifacts_1["bayesian_convergence_summary_md"].read_text(encoding="utf-8")

    state["converged"] = True
    state["divergences"] = 0.0

    artifacts_2 = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_visualizations=True,
    )
    summary_csv_2 = artifacts_2["bayesian_convergence_summary_csv"].read_text(encoding="utf-8")
    summary_md_2 = artifacts_2["bayesian_convergence_summary_md"].read_text(encoding="utf-8")

    assert artifacts_2["bayesian_convergence_summary_csv"].exists()
    assert artifacts_2["bayesian_convergence_summary_md"].exists()
    assert summary_csv_1 != summary_csv_2
    assert summary_md_1 != summary_md_2


def test_decision_alerts_schema_locked_columns(tmp_path) -> None:
    synthetic = pd.DataFrame(
        {
            "date": pd.date_range("2016-01-01", periods=8, freq="W").astype(str),
            "district": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "state": ["S"] * 8,
            "cases": [1, 3, 7, 2, 4, 6, 8, 1],
            "rainfall": [5.0, 2.0, 7.0, 9.0, 1.0, 0.0, 3.0, 2.0],
            "temperature": [28.0, 29.0, 30.0, 31.0, 27.0, 26.0, 25.0, 24.0],
            "humidity": [60.0, 62.0, 58.0, 57.0, 64.0, 66.0, 68.0, 65.0],
        }
    )
    raw_path = tmp_path / "synthetic_schema_raw.csv"
    synthetic.to_csv(raw_path, index=False)

    artifacts = run(
        raw_data_path=raw_path,
        start_year=2016,
        end_year=2016,
        skip_baselines=True,
        skip_bayesian=True,
        skip_visualizations=True,
    )

    decision = pd.read_csv(artifacts["decision_alerts"])
    assert decision.columns.tolist() == [
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
