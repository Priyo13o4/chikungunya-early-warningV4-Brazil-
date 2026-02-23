"""Tests for bayesian scaffold model."""

from __future__ import annotations

import pandas as pd

from src.models.bayesian import hierarchical_model
from src.pipeline_runtime import config_runtime


def test_bayesian_model_fit_predict() -> None:
    original_loader = hierarchical_model._require_pymc_dependencies

    def _missing_optional_dependencies():
        raise ImportError("optional deps missing in test")

    hierarchical_model._require_pymc_dependencies = _missing_optional_dependencies
    try:
        X = pd.DataFrame({"x": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        model = hierarchical_model.HierarchicalBayesianModel().fit(X, y)
        preds = model.predict(X)
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader

    assert len(preds) == 3


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
                "temp_anomaly": [0.1, 0.2, -0.1],
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
    assert float(risk_frame.loc[0, "risk_mean"]) >= float(risk_frame.loc[1, "risk_mean"])


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


def test_resolve_bayesian_profile_settings_default_routing() -> None:
    fullfit, oof, usage = config_runtime.resolve_bayesian_profile_settings(
        bayesian_settings={"draws": 800, "tune": 1200, "chains": 2, "target_accept": 0.99},
        bayesian_profiles={
            "final": {"draws": 1000, "tune": 1500},
            "cv": {"draws": 300, "tune": 500, "bayesian_progress": False},
        },
        profile_mode=None,
    )

    assert int(fullfit["draws"]) == 1000
    assert int(oof["draws"]) == 300
    assert usage["fullfit_profile_name"] == "final"
    assert usage["oof_profile_name"] == "cv"
    assert usage["cv_profile_differs_from_final"] is True
    assert "draws" in usage["cv_vs_final_diff_keys"]


def test_resolve_bayesian_profile_settings_dev_override_applies_to_both() -> None:
    fullfit, oof, usage = config_runtime.resolve_bayesian_profile_settings(
        bayesian_settings={"draws": 800, "chains": 2},
        bayesian_profiles={
            "final": {"draws": 1000},
            "cv": {"draws": 400},
            "dev": {"draws": 50, "chains": 1, "bayesian_progress": False},
        },
        profile_mode="dev",
    )

    assert int(fullfit["draws"]) == 50
    assert int(oof["draws"]) == 50
    assert int(fullfit["chains"]) == 1
    assert int(oof["chains"]) == 1
    assert usage["profile_mode_override"] == "dev"
    assert usage["fullfit_profile_name"] == "dev"
    assert usage["oof_profile_name"] == "dev"


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
            }
        )
        y = pd.Series([0.0, 1.0, 2.0])
        model = hierarchical_model.HierarchicalBayesianModel(
            config=hierarchical_model.BayesianModelConfig(sampling_backend="auto")
        )
        model.fit(X, y, compute_backend_effective="macos_metal")
    finally:
        hierarchical_model._require_pymc_dependencies = original_loader
        hierarchical_model._try_import = original_try_import

    assert jax_called["value"] is True
    assert pymc_called["value"] is True
    assert model.sampling_diagnostics_.get("sampling_backend_requested") == "auto"
    assert model.sampling_diagnostics_.get("sampling_backend_effective") == "pymc"
    assert "JAX sampler unavailable" in str(model.sampling_diagnostics_.get("sampling_backend_fallback_reason"))
    assert model.sampling_diagnostics_.get("actual_runtime_backend") == "cpu"
    assert model.sampling_diagnostics_.get("backend_implemented") is False
