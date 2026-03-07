"""Tests for bayesian scaffold model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.bayesian import hierarchical_model
from src.pipeline_runtime import config_runtime
from src.pipeline_runtime.phases_bayesian import _impute_selected_covariates_for_bayesian
from src.pipeline_runtime.phases_bayesian import evaluate_strict_convergence_with_groups
import run_pipeline


def _base_bayesian_fit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "district": ["A", "A", "B"],
            "date": ["2016-01-01", "2016-01-08", "2016-01-15"],
            "rt": [0.9, 1.1, 1.0],
            "p_rt1": [0.4, 0.6, 0.5],
            "receptivo": [0.2, 0.3, 0.1],
            "month": [1.0, 2.0, 3.0],
            "year": [2016.0, 2017.0, 2018.0],
            "weekofyear": [1.0, 2.0, 3.0],
        }
    )


def test_bayesian_fit_raises_on_missing_required_covariate() -> None:
    X = _base_bayesian_fit_frame().drop(columns=["month"])
    y = pd.Series([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="missing required climate covariates"):
        hierarchical_model.HierarchicalBayesianModel().fit(X, y)


def test_bayesian_fit_raises_on_non_numeric_covariate_values() -> None:
    X = _base_bayesian_fit_frame()
    X["month"] = X["month"].astype(object)
    X.loc[1, "month"] = "not-a-number"
    y = pd.Series([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match="contains null/non-numeric values"):
        hierarchical_model.HierarchicalBayesianModel().fit(X, y)


def test_bayesian_fit_raises_on_degenerate_covariate_variance() -> None:
    X = _base_bayesian_fit_frame()
    X["month"] = 1.0
    y = pd.Series([0.0, 1.0, 2.0])

    with pytest.raises(ValueError, match=r"degenerate \(near-zero variance\)"):
        hierarchical_model.HierarchicalBayesianModel().fit(X, y)


def test_bayesian_model_fit_predict() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = pd.DataFrame(
            {
                "rt": [0.9, 1.0, 1.1],
                "p_rt1": [0.4, 0.5, 0.6],
                "receptivo": [0.2, 0.2, 0.3],
                "month": [1.0, 2.0, 3.0],
                "year": [2016.0, 2017.0, 2018.0],
                "weekofyear": [1.0, 2.0, 3.0],
            }
        )
        y = pd.Series([0, 1, 0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)
        preds = model.predict(X)
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader

    assert len(preds) == 3


def test_bayesian_predict_records_covariate_order_reorder_metadata() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = _base_bayesian_fit_frame().copy()
        y = pd.Series([0.0, 1.0, 2.0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)

        shuffled_columns = [
            "district",
            "date",
            "weekofyear",
            "year",
            "month",
            "receptivo",
            "p_rt1",
            "rt",
        ]
        _, metadata = model.predict_with_uncertainty(X[shuffled_columns])
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader

    assert metadata.get("trained_covariate_order") == ["month", "year", "weekofyear"]
    alignment = metadata.get("covariate_alignment", {})
    assert alignment.get("covariate_order_mismatch_reordered") is True
    assert alignment.get("trained_covariate_order_locked") is True


def test_bayesian_predict_raises_on_missing_required_trained_covariate() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = _base_bayesian_fit_frame().copy()
        y = pd.Series([0.0, 1.0, 2.0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)
        missing = X.drop(columns=["month"])
        with pytest.raises(ValueError, match="missing required climate covariates"):
            model.predict_with_uncertainty(missing)
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader


def test_imputation_alignment_safety_preserves_index_and_rows() -> None:
    frame = pd.DataFrame(
        {
            "district": ["B", "A", "A", "B"],
            "date": ["2025-01-08", "2025-01-01", "2025-01-08", "2025-01-01"],
            "rainfall": [np.nan, 1.0, np.nan, 4.0],
        },
        index=[10, 11, 12, 13],
    )
    stats: dict[str, object] = {}
    imputed = _impute_selected_covariates_for_bayesian(
        frame,
        covariates=["rainfall"],
        context="test_alignment",
        imputation_stats=stats,
    )

    assert list(imputed.index) == [10, 11, 12, 13]
    assert len(imputed) == len(frame)
    assert bool(stats.get("alignment_ok")) is True
    assert int(stats.get("covariates_imputed", 0)) == 1


def test_bayesian_predict_with_uncertainty_returns_intervals_and_metadata() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = pd.DataFrame(
            {
                "district": ["A", "A", "B"],
                "date": ["2016-01-01", "2016-01-08", "2016-01-15"],
                "rt": [0.9, 1.1, 1.0],
                "p_rt1": [0.4, 0.6, 0.5],
                "receptivo": [0.2, 0.3, 0.1],
                "month": [1.0, 2.0, 3.0],
                "year": [2016.0, 2017.0, 2018.0],
                "weekofyear": [1.0, 2.0, 3.0],
            }
        )
        y = pd.Series([0.0, 2.0, 1.0])
        thresholds = pd.Series([1.0, 3.0, 1.0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)
        risk_frame, metadata = model.predict_with_uncertainty(X, outbreak_threshold=thresholds)
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader

    assert {"risk_mean", "risk_q05", "risk_q95", "threshold_cases", "bayesian_risk"}.issubset(risk_frame.columns)
    assert len(risk_frame) == len(X)
    assert bool(metadata.get("degraded_mode", False)) is True
    assert metadata.get("threshold_basis") == "provided_series"
    assert metadata.get("climate_covariates") == ["month", "year", "weekofyear"]
    assert float(risk_frame.loc[0, "risk_mean"]) >= float(risk_frame.loc[1, "risk_mean"])


def test_bayesian_predict_with_uncertainty_clamps_nonpositive_threshold_for_risk_only() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = pd.DataFrame(
            {
                "district": ["A", "A", "B"],
                "date": ["2016-01-01", "2016-01-08", "2016-01-15"],
                "month": [1.0, 2.0, 3.0],
                "year": [2016.0, 2017.0, 2018.0],
                "weekofyear": [1.0, 2.0, 3.0],
            }
        )
        y = pd.Series([0.0, 2.0, 1.0])
        thresholds = pd.Series([0.0, -3.0, 2.0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)
        risk_frame, metadata = model.predict_with_uncertainty(X, outbreak_threshold=thresholds)
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader

    assert risk_frame["threshold_cases"].tolist() == [0.0, -3.0, 2.0]
    assert metadata.get("threshold_clamped_for_risk") is True
    assert int(metadata.get("threshold_clamped_count", 0)) == 2


def test_bayesian_model_config_accepts_convergence_failure_mode() -> None:
    strict_cfg = hierarchical_model.BayesianModelConfig(convergence_failure_mode="strict")
    warn_cfg = hierarchical_model.BayesianModelConfig(convergence_failure_mode="warn")

    assert strict_cfg.convergence_failure_mode == "strict"
    assert warn_cfg.convergence_failure_mode == "warn"


def test_sampling_backend_config_normalization() -> None:
    cfg_auto = config_runtime.build_bayesian_config(
        strict_dependencies=False,
        bayesian_settings={"sampling_backend": "AUTO"},
    )
    cfg_jax = config_runtime.build_bayesian_config(
        strict_dependencies=False,
        bayesian_settings={"sampling_backend": "JAX_NUMPYRO"},
    )
    cfg_invalid = config_runtime.build_bayesian_config(
        strict_dependencies=False,
        bayesian_settings={"sampling_backend": "invalid_backend"},
    )

    assert cfg_auto.sampling_backend == "auto"
    assert cfg_jax.sampling_backend == "jax_numpyro"
    assert cfg_invalid.sampling_backend == "auto"


def test_build_bayesian_config_accepts_explicit_climate_covariates() -> None:
    cfg = config_runtime.build_bayesian_config(
        strict_dependencies=False,
        bayesian_settings={"climate_covariates": ["rainfall_4wk", "temp_anomaly"]},
    )

    assert cfg.climate_covariates == ("rainfall_4wk", "temp_anomaly")


def test_select_bayesian_covariates_by_availability_excludes_unusable_covariates() -> None:
    frame = pd.DataFrame(
        {
            "temp_anomaly": [0.1, 0.2, 0.3],
            "bad_string": ["x", "y", "z"],
            "flat_feature": [5.0, 5.0, 5.0],
        }
    )

    selection = config_runtime.select_bayesian_covariates_by_availability(
        frame=frame,
        requested_covariates=["temp_anomaly", "missing_covariate", "bad_string", "flat_feature"],
    )

    assert selection["selected_covariates"] == ["temp_anomaly"]
    excluded = {item["name"]: item["reason"] for item in selection["excluded_covariates"]}
    assert excluded["missing_covariate"] == "missing_column"
    assert excluded["bad_string"] == "null_or_non_numeric_values"
    assert excluded["flat_feature"] == "degenerate_variance"


def test_select_bayesian_covariates_by_availability_prefers_higher_variance_tie_break() -> None:
    frame = pd.DataFrame(
        {
            "cov_low": [0.0, 1.0, 0.0, 1.0],
            "cov_high": [0.0, 4.0, 0.0, 4.0],
        }
    )

    selection = config_runtime.select_bayesian_covariates_by_availability(
        frame=frame,
        requested_covariates=["cov_low", "cov_high"],
    )

    assert selection["selected_covariates"] == ["cov_high", "cov_low"]


def test_select_bayesian_covariates_by_availability_keeps_tiny_null_rate_for_imputation() -> None:
    frame = pd.DataFrame(
        {
            "cov_keep": ([0.1, 0.2] * 50) + [None],
        }
    )

    selection = config_runtime.select_bayesian_covariates_by_availability(
        frame=frame,
        requested_covariates=["cov_keep"],
    )

    assert selection["selected_covariates"] == ["cov_keep"]
    diagnostics = {item["name"]: item for item in selection["covariate_diagnostics"]}
    assert diagnostics["cov_keep"]["imputation_required"] is True
    assert diagnostics["cov_keep"]["imputation_strategy"] == "forward_fill_then_district_median"


def test_resolve_bayesian_profile_settings_profiles_disabled_use_base_settings() -> None:
    fullfit, oof, usage = config_runtime.resolve_bayesian_profile_settings(
        bayesian_settings={"draws": 800, "chains": 2},
        bayesian_profiles={
            "final": {"draws": 1000, "chains": 3},
            "cv": {"draws": 300, "chains": 1},
            "dev": {"draws": 120, "chains": 1},
        },
        profile_mode="dev",
        enable_profiles=False,
    )

    assert int(fullfit["draws"]) == 800
    assert int(oof["draws"]) == 800
    assert usage["profiles_enabled"] is False
    assert usage["fullfit_profile_name"] == "base"
    assert usage["oof_profile_name"] == "base"


def test_parse_memory_optimization_config_false_like_mode_maps_to_off() -> None:
    parsed = config_runtime.parse_memory_optimization_config(
        {
            "memory_optimization": {
                "mode": "False",
            }
        }
    )

    assert parsed.mode == "off"


def test_run_bayesian_track_import_error_keeps_effective_covariates(monkeypatch) -> None:
    def _raise_import_error(*args, **kwargs):
        args
        kwargs
        raise ImportError("optional deps missing in test")

    monkeypatch.setattr(run_pipeline, "_build_bayesian_config", _raise_import_error)

    features = pd.DataFrame(
        {
            "district": ["A", "B"],
            "date": ["2016-01-01", "2016-01-08"],
            "temp_anomaly": [0.1, 0.2],
            "degree_days_20": [1.0, 2.0],
            "rainfall_4wk": [10.0, 12.0],
        }
    )
    counts = pd.Series([1.0, 2.0])

    risk_frame, idata, diagnostics = run_pipeline._run_bayesian_track(
        features,
        counts,
        outbreak_threshold=None,
        strict_dependencies=False,
        bayesian_settings={"climate_covariates": ["rainfall_4wk", "temp_anomaly"]},
    )

    assert risk_frame is None
    assert idata is None
    assert diagnostics.get("degraded_mode") is True
    assert diagnostics.get("climate_covariates") == ["rainfall_4wk", "temp_anomaly"]


def test_resolve_bayesian_profile_settings_default_routing() -> None:
    fullfit, oof, usage = config_runtime.resolve_bayesian_profile_settings(
        bayesian_settings={"draws": 800, "tune": 1200, "chains": 2, "target_accept": 0.99},
        bayesian_profiles={
            "final": {"draws": 1000, "tune": 1500},
            "cv": {"draws": 300, "tune": 500, "bayesian_progress": False},
        },
        profile_mode=None,
        enable_profiles=False,
    )

    assert int(fullfit["draws"]) == 800
    assert int(oof["draws"]) == 800
    assert usage["profiles_enabled"] is False
    assert usage["fullfit_profile_name"] == "base"
    assert usage["oof_profile_name"] == "base"
    assert usage["cv_profile_differs_from_final"] is False


def test_resolve_bayesian_profile_settings_dev_override_applies_to_both() -> None:
    fullfit, oof, usage = config_runtime.resolve_bayesian_profile_settings(
        bayesian_settings={"draws": 800, "chains": 2},
        bayesian_profiles={
            "dev": {"draws": 50, "chains": 1, "bayesian_progress": False},
        },
        profile_mode="dev",
        enable_profiles=True,
    )

    assert int(fullfit["draws"]) == 50
    assert int(oof["draws"]) == 50
    assert int(fullfit["chains"]) == 1
    assert int(oof["chains"]) == 1
    assert usage["profiles_enabled"] is True
    assert usage["profile_mode_override"] == "dev"
    assert usage["fullfit_profile_name"] == "dev"
    assert usage["oof_profile_name"] == "dev"


def test_evaluate_strict_convergence_with_groups_passes_within_thresholds() -> None:
    grouped = pd.DataFrame(
        [
            {"group": "random_effects", "n_parameters": 4, "ess_min": 250.0, "fail_rhat_count": 0, "fail_ess_count": 0},
            {"group": "fixed_effects", "n_parameters": 3, "ess_min": 300.0, "fail_rhat_count": 0, "fail_ess_count": 0},
            {"group": "likelihood", "n_parameters": 1, "ess_min": 260.0, "fail_rhat_count": 0, "fail_ess_count": 0},
            {"group": "temporal_state", "n_parameters": 10, "ess_min": 210.0, "fail_rhat_count": 1, "fail_ess_count": 2},
            {"group": "other", "n_parameters": 2, "ess_min": 220.0, "fail_rhat_count": 0, "fail_ess_count": 0},
        ]
    )
    converged, details = evaluate_strict_convergence_with_groups(
        {
            "converged": True,
            "divergences": 0.0,
            "divergence_threshold": 25.0,
            "max_tree_depth": 8.0,
            "max_tree_depth_threshold": 12.0,
            "ess_threshold": 200.0,
        },
        grouped,
        temporal_fail_fraction_threshold=0.30,
        temporal_rhat_fail_fraction_threshold=0.30,
    )

    assert converged is True
    assert details["hard_fail_global"] is False
    assert details["strict_fail_reasons"] == []


def test_evaluate_strict_convergence_with_groups_fails_when_thresholds_violated() -> None:
    grouped = pd.DataFrame(
        [
            {"group": "random_effects", "n_parameters": 3, "ess_min": 100.0, "fail_rhat_count": 1, "fail_ess_count": 1},
            {"group": "fixed_effects", "n_parameters": 2, "ess_min": 150.0, "fail_rhat_count": 0, "fail_ess_count": 1},
            {"group": "likelihood", "n_parameters": 1, "ess_min": 190.0, "fail_rhat_count": 0, "fail_ess_count": 1},
            {"group": "temporal_state", "n_parameters": 10, "ess_min": 120.0, "fail_rhat_count": 4, "fail_ess_count": 5},
            {"group": "other", "n_parameters": 4, "ess_min": 100.0, "fail_rhat_count": 3, "fail_ess_count": 3},
        ]
    )
    converged, details = evaluate_strict_convergence_with_groups(
        {
            "converged": False,
            "divergences": 30.0,
            "divergence_threshold": 25.0,
            "max_tree_depth": 13.0,
            "max_tree_depth_threshold": 12.0,
            "ess_threshold": 200.0,
        },
        grouped,
        temporal_fail_fraction_threshold=0.20,
        temporal_rhat_fail_fraction_threshold=0.20,
    )

    assert converged is False
    assert details["hard_fail_global"] is True
    assert details["strict_fail_reasons"]


def test_auto_sampling_backend_on_metal_attempts_jax_then_falls_back(monkeypatch) -> None:
    original_loader = hierarchical_model._require_pymc_dependencies
    original_try_import = hierarchical_model._try_import

    class _FakePM:
        pass

    def _fake_require_pymc_dependencies():
        return _FakePM(), object()

    class _FakeJax:
        @staticmethod
        def devices():
            return []

    def _fake_try_import(module_name: str):
        if module_name == "jax":
            return _FakeJax()
        if module_name == "numpyro":
            return object()
        return None

    jax_called = {"value": False}
    pymc_called = {"value": False}

    def _fake_fit_jax_numpyro_model(self, **kwargs):
        self
        kwargs
        jax_called["value"] = True
        raise RuntimeError("jax unavailable in test")

    def _fake_fit_pymc_model(self, **kwargs):
        self
        kwargs
        pymc_called["value"] = True
        return object(), object(), [0.0]

    def _fake_extract_sampler_diagnostics(self, idata):
        self
        idata
        return {"divergences": 0.0, "max_tree_depth": 0.0, "r_hat_max": 1.0, "ess_min": 300.0}

    def _fake_finalize_from_posterior(self, **kwargs):
        self.fitted_ = True
        self.diagnostics_summary_ = dict(kwargs.get("diagnostics_summary", {}))
        self.simplified_used_ = bool(kwargs.get("simplified_mode", False))
        self.idata_ = kwargs.get("idata")

    hierarchical_model._require_pymc_dependencies = _fake_require_pymc_dependencies
    hierarchical_model._try_import = _fake_try_import
    monkeypatch.setattr(
        hierarchical_model.HierarchicalBayesianModel,
        "_fit_jax_numpyro_model",
        _fake_fit_jax_numpyro_model,
    )
    monkeypatch.setattr(
        hierarchical_model.HierarchicalBayesianModel,
        "_fit_pymc_model",
        _fake_fit_pymc_model,
    )
    monkeypatch.setattr(
        hierarchical_model.HierarchicalBayesianModel,
        "_extract_sampler_diagnostics",
        _fake_extract_sampler_diagnostics,
    )
    monkeypatch.setattr(
        hierarchical_model.HierarchicalBayesianModel,
        "_finalize_from_posterior",
        _fake_finalize_from_posterior,
    )
    try:
        X = pd.DataFrame(
            {
                "district": ["A", "A", "B"],
                "date": ["2016-01-01", "2016-01-08", "2016-01-15"],
                "temp_anomaly": [0.1, 0.0, -0.1],
                "degree_days_20": [1.0, 2.0, 3.0],
                "rainfall_4wk": [11.0, 13.0, 12.0],
            }
        )
        y = pd.Series([0.0, 1.0, 2.0])
        model = hierarchical_model.HierarchicalBayesianModel(
            config=hierarchical_model.BayesianModelConfig(
                sampling_backend="auto",
                climate_covariates=("temp_anomaly", "degree_days_20", "rainfall_4wk"),
            )
        )
        model.fit(X, y, compute_backend_effective="macos_metal")
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader
        hierarchical_model._try_import = original_try_import

    assert jax_called["value"] is False
    assert pymc_called["value"] is True
    assert model.sampling_diagnostics_.get("sampling_backend_requested") == "auto"
    assert model.sampling_diagnostics_.get("sampling_backend_effective") == "pymc"
    assert model.sampling_diagnostics_.get("sampling_backend_fallback_reason") is None
    assert model.sampling_diagnostics_.get("actual_runtime_backend") == "cpu"
    assert model.sampling_diagnostics_.get("backend_implemented") is False
