"""Smoke tests for pipeline modules."""

from __future__ import annotations

import json

import pandas as pd

from run_pipeline import _apply_memory_optimization_filters, _build_model_input_df, run
from src.pipeline_runtime import config_runtime


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

    sanitized, audit = _build_model_input_df(frame)
    assert "outbreak_label" not in sanitized.columns
    assert "outbreak_label_p75" not in sanitized.columns
    assert "threshold_p75" not in sanitized.columns
    assert "cases" not in sanitized.columns
    assert "case_lag_1" in sanitized.columns
    assert "rainfall" in sanitized.columns
    assert set(audit["dropped_forbidden_columns"]) >= {"outbreak_label", "outbreak_label_p75", "threshold_p75", "cases"}


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

    required = set(manifest["contract_required_artifacts"])
    assert required.issubset(set(manifest["artifacts"].keys()))
    for key in required:
        assert key in artifacts
        assert artifacts[key].exists()

    run_id = manifest["run_id"]
    assert metadata["run_id"] == run_id
    assert degraded["run_id"] == run_id
    assert fold_ledger["run_id"] == run_id


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
