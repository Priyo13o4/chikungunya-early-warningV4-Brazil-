"""Hierarchical Bayesian model for latent chikungunya outbreak risk."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

from src.threshold_resolution import resolve_threshold_series

LOGGER = logging.getLogger(__name__)


def _try_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _require_pymc_dependencies() -> tuple[Any, Any]:
    pm = _try_import("pymc")
    az = _try_import("arviz")
    if pm is None or az is None:
        raise ImportError(
            "Hierarchical Bayesian model requires optional dependencies 'pymc' and 'arviz'. "
            "Install them to run Track B full inference."
        )
    return pm, az


@dataclass(frozen=True)
class BayesianModelConfig:
    """Configuration for hierarchical NB + AR(1) model fitting."""

    district_column: str = "district"
    date_column: str = "date"
    target_column: str = "cases"
    climate_covariates: tuple[str, ...] = ("month", "year", "weekofyear")
    draws: int = 1200
    tune: int = 1500
    chains: int = 2
    bayesian_progress: bool = True
    target_accept: float = 0.95
    max_treedepth: int = 12
    max_convergence_retries: int = 1
    bayesian_simplified_mode: bool = False
    divergence_warn_threshold: int = 25
    rhat_warn_threshold: float = 1.05
    ess_warn_threshold: float = 200.0
    random_seed: int = 42
    strict_dependencies: bool = False
    force_full_bayesian: bool = False
    convergence_failure_mode: str = "strict"
    sampling_backend: str = "auto"
    outbreak_threshold_default_cases: float = 1.0
    posterior_sample_cap: int = 400
    predictive_chunk_rows: int = 20000


@dataclass
class HierarchicalBayesianModel:
    """Hierarchical negative-binomial Bayesian model with AR(1) latent risk state."""

    config: BayesianModelConfig = field(default_factory=BayesianModelConfig)
    fitted_: bool = False
    model_: Any = None
    idata_: Any = None
    district_effects_: dict[str, float] = field(default_factory=dict)
    beta_effects_: dict[str, float] = field(default_factory=dict)
    latent_by_state_time_: dict[tuple[str, pd.Timestamp], float] = field(default_factory=dict)
    global_intercept_: float = 0.0
    fallback_rate_: float = 0.0
    covariate_means_: dict[str, float] = field(default_factory=dict)
    covariate_scales_: dict[str, float] = field(default_factory=dict)
    diagnostics_summary_: dict[str, float] = field(default_factory=dict)
    sampling_diagnostics_: dict[str, Any] = field(default_factory=dict)
    sampling_backend_requested_: str = "auto"
    sampling_backend_effective_: str = "pymc"
    sampling_backend_fallback_reason_: str | None = None
    sampling_runtime_backend_: str = "cpu"
    simplified_used_: bool = False
    alpha_nb_: float = 1.0
    trained_covariate_order_: tuple[str, ...] = field(default_factory=tuple)
    prediction_covariate_alignment_: dict[str, Any] = field(default_factory=dict)

    def _resolve_state_codes(self, frame: pd.DataFrame) -> pd.Series:
        district_values = frame.get(self.config.district_column, pd.Series("unknown", index=frame.index, dtype="object"))
        district_prefix = district_values.astype(str).str.split(r"[_\-\s]", n=1).str[0].str.strip().str.upper()
        district_prefix = district_prefix.str.slice(0, 2)
        district_prefix = district_prefix.where(district_prefix.str.len() == 2, "unknown")

        geocode_state = pd.Series("unknown", index=frame.index, dtype="object")
        if "municipio_geocodigo" in frame.columns:
            geocodigo = frame["municipio_geocodigo"].astype(str).str.replace(r"\D", "", regex=True)
            geocode_state = geocodigo.str.slice(0, 2)
            geocode_state = geocode_state.where(geocode_state.str.len() == 2, "unknown")

        state_series = pd.Series("unknown", index=frame.index, dtype="object")
        if "state" in frame.columns:
            state_series = frame["state"].astype(str).str.strip()
            state_series = state_series.replace({"": "unknown", "nan": "unknown", "None": "unknown"}).str.upper()

        state_series = state_series.where(state_series != "unknown", geocode_state)
        state_series = state_series.where(state_series != "unknown", district_prefix)
        return state_series

    def _validate_covariate_contract(
        self,
        frame: pd.DataFrame,
        *,
        context: str,
        require_variance: bool,
    ) -> pd.DataFrame:
        missing_covariates = [covariate for covariate in self.config.climate_covariates if covariate not in frame.columns]
        if missing_covariates:
            raise ValueError(
                f"{context}: missing required climate covariates {missing_covariates}. "
                "Pipeline must provide the configured Bayesian covariate set explicitly."
            )

        validated = frame.copy()
        for covariate in self.config.climate_covariates:
            validated[covariate] = pd.to_numeric(validated[covariate], errors="coerce")
            if validated[covariate].isna().any():
                raise ValueError(
                    f"{context}: covariate '{covariate}' contains null/non-numeric values after coercion. "
                    "Silent Bayesian covariate imputation is disabled."
                )
            if require_variance and float(validated[covariate].std(ddof=0)) <= 1e-12:
                raise ValueError(
                    f"{context}: covariate '{covariate}' is degenerate (near-zero variance). "
                    "Bayesian fit requires informative covariates."
                )
        return validated

    def _prepare_design(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        frame = X.copy()
        frame["__target__"] = pd.to_numeric(y, errors="coerce").fillna(0.0)

        if self.config.district_column not in frame.columns:
            frame[self.config.district_column] = "unknown"

        if self.config.date_column in frame.columns:
            frame[self.config.date_column] = pd.to_datetime(frame[self.config.date_column], errors="coerce")
        elif "year" in frame.columns:
            years = pd.to_numeric(frame["year"], errors="coerce").fillna(0).astype(int)
            frame[self.config.date_column] = pd.to_datetime(years.astype(str) + "-01-01", errors="coerce")
        else:
            frame[self.config.date_column] = pd.date_range("2009-01-01", periods=len(frame), freq="W")

        frame = self._validate_covariate_contract(frame, context="fit", require_variance=True)

        frame[self.config.district_column] = frame[self.config.district_column].astype(str).fillna("unknown")
        frame["__state_code__"] = self._resolve_state_codes(frame).astype(str).fillna("unknown")
        frame = frame.sort_values([self.config.date_column, self.config.district_column]).copy()
        return frame

    def _fit_fallback(self, frame: pd.DataFrame, reason: str) -> None:
        LOGGER.warning("Using Bayesian fallback mode: %s", reason)
        requested_sampling_backend = self._normalize_sampling_backend(self.config.sampling_backend)
        self.sampling_backend_requested_ = requested_sampling_backend
        self.sampling_backend_effective_ = "pymc"
        self.sampling_backend_fallback_reason_ = str(reason)
        self.sampling_runtime_backend_ = "cpu"
        self.fallback_rate_ = float(frame["__target__"].mean()) if len(frame) else 0.0
        self.global_intercept_ = float(np.log1p(self.fallback_rate_))
        self.district_effects_ = (
            frame.groupby(self.config.district_column, dropna=False)["__target__"].mean().apply(np.log1p).to_dict()
        )
        self.beta_effects_ = {covariate: 0.0 for covariate in self.config.climate_covariates}
        self.alpha_nb_ = 1.0
        self.covariate_means_ = {
            covariate: float(pd.to_numeric(frame[covariate], errors="coerce").mean())
            for covariate in self.config.climate_covariates
        }
        self.covariate_scales_ = {covariate: 1.0 for covariate in self.config.climate_covariates}
        if "__state_code__" not in frame.columns:
            frame["__state_code__"] = self._resolve_state_codes(frame).astype(str).fillna("unknown")
        state_time_mean = (
            frame.groupby(["__state_code__", self.config.date_column], dropna=False)["__target__"].mean().apply(np.log1p)
        )
        self.latent_by_state_time_ = {
            (str(state_code), pd.Timestamp(timestamp)): float(value)
            for (state_code, timestamp), value in state_time_mean.items()
            if pd.notna(timestamp)
        }
        self.diagnostics_summary_ = {
            "divergences": float("nan"),
            "max_tree_depth": float("nan"),
            "r_hat_max": float("nan"),
            "ess_min": float("nan"),
            "simplified_mode": 0.0,
            "fallback": 1.0,
        }
        self.trained_covariate_order_ = tuple(self.config.climate_covariates)
        self.fitted_ = True

    @staticmethod
    def _nb_exceedance_probability(mu: np.ndarray, alpha: np.ndarray, threshold_counts: np.ndarray) -> np.ndarray:
        """Compute NB exceedance probability P(Y >= k) for each row.

        Uses scipy when available, otherwise degrades to a Poisson tail approximation.
        """
        mu_safe = np.clip(np.asarray(mu, dtype=float), a_min=1e-12, a_max=1e12)
        alpha_safe = np.clip(np.asarray(alpha, dtype=float), a_min=1e-9, a_max=1e12)
        k = np.ceil(np.clip(np.asarray(threshold_counts, dtype=float), a_min=0.0, a_max=None)).astype(int)

        scipy_stats = _try_import("scipy.stats")
        if scipy_stats is not None and hasattr(scipy_stats, "nbinom"):
            p = alpha_safe / (alpha_safe + mu_safe)
            sf = scipy_stats.nbinom.sf(k - 1, alpha_safe, p)
            return np.asarray(sf, dtype=float)

        scipy_special = _try_import("scipy.special")
        if scipy_special is not None and hasattr(scipy_special, "gammaincc"):
            poisson_sf = scipy_special.gammaincc(k, mu_safe)
            return np.asarray(poisson_sf, dtype=float)

        LOGGER.warning("Neither scipy.stats.nbinom nor scipy.special.gammaincc available; using coarse approximation.")
        return np.clip(1.0 - np.exp(-mu_safe), 0.0, 1.0)

    def _resolve_outbreak_thresholds(
        self,
        frame: pd.DataFrame,
        outbreak_threshold: pd.Series | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        threshold_series, threshold_meta = resolve_threshold_series(
            frame=frame,
            default_threshold=float(self.config.outbreak_threshold_default_cases),
            outbreak_threshold=outbreak_threshold,
        )
        return threshold_series.to_numpy(dtype=float), threshold_meta

    def _compute_linear_components(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        district_values = frame[self.config.district_column].astype(str)
        date_values = pd.to_datetime(frame[self.config.date_column], errors="coerce")
        state_values = self._resolve_state_codes(frame).astype(str)

        district_default = float(np.log1p(max(self.fallback_rate_, 0.0)))
        district_effect = np.asarray(
            [self.district_effects_.get(value, district_default) for value in district_values],
            dtype=float,
        )

        global_time_default = float(list(self.latent_by_state_time_.values())[-1]) if self.latent_by_state_time_ else 0.0
        state_defaults: dict[str, float] = {}
        for (state_code, _timestamp), latent_value in self.latent_by_state_time_.items():
            state_defaults[str(state_code)] = float(latent_value)
        time_effect = np.asarray(
            [
                (
                    self.latent_by_state_time_.get(
                        (str(state_code), pd.Timestamp(value)),
                        state_defaults.get(str(state_code), global_time_default),
                    )
                    if pd.notna(value)
                    else 0.0
                )
                for state_code, value in zip(state_values, date_values)
            ],
            dtype=float,
        )

        feature_matrix = self._validate_covariate_contract(
            frame.loc[:, list(self.config.climate_covariates)].copy(),
            context="predict",
            require_variance=False,
        )

        means = np.asarray([self.covariate_means_.get(covariate, 0.0) for covariate in self.config.climate_covariates])
        scales = np.asarray([self.covariate_scales_.get(covariate, 1.0) for covariate in self.config.climate_covariates])
        scales = np.where(np.abs(scales) > 1e-12, scales, 1.0)
        beta = np.asarray([self.beta_effects_.get(covariate, 0.0) for covariate in self.config.climate_covariates])

        x_scaled = (feature_matrix.to_numpy(dtype=float) - means) / scales
        covariate_effect = np.dot(x_scaled, beta)
        return district_effect, covariate_effect, time_effect, x_scaled

    def predict_with_uncertainty(
        self,
        X: pd.DataFrame,
        *,
        outbreak_threshold: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Predict Bayesian risk with uncertainty intervals.

        Risk is defined as posterior predictive exceedance probability:
        P(Y >= outbreak_threshold_row).
        """
        if not self.fitted_:
            raise RuntimeError("Bayesian model must be fit before prediction")

        frame = X.copy()
        if self.config.district_column not in frame.columns:
            frame[self.config.district_column] = "unknown"
        frame[self.config.district_column] = frame[self.config.district_column].astype(str)

        if self.config.date_column in frame.columns:
            frame[self.config.date_column] = pd.to_datetime(frame[self.config.date_column], errors="coerce")
        else:
            frame[self.config.date_column] = pd.NaT

        frame = self._validate_covariate_contract(frame, context="predict", require_variance=False)
        expected_covariate_order = tuple(self.trained_covariate_order_ or tuple(self.config.climate_covariates))
        provided_covariate_order = [column for column in frame.columns if column in expected_covariate_order]
        missing_covariates = [column for column in expected_covariate_order if column not in frame.columns]
        if missing_covariates:
            raise ValueError(
                "predict: missing required trained covariates "
                f"{missing_covariates}; expected order={list(expected_covariate_order)}"
            )
        covariate_order_mismatch = provided_covariate_order != list(expected_covariate_order)
        if covariate_order_mismatch:
            LOGGER.info(
                "Prediction covariate order mismatch detected; expected=%s provided=%s. Reordering to training order.",
                list(expected_covariate_order),
                provided_covariate_order,
            )
        self.prediction_covariate_alignment_ = {
            "expected_covariate_order": list(expected_covariate_order),
            "provided_covariate_order": provided_covariate_order,
            "covariate_order_mismatch_reordered": bool(covariate_order_mismatch),
            "missing_required_covariates": [],
            "trained_covariate_order_locked": bool(len(expected_covariate_order) > 0),
        }
        frame["__state_code__"] = self._resolve_state_codes(frame).astype(str).fillna("unknown")

        threshold_values, threshold_meta = self._resolve_outbreak_thresholds(frame, outbreak_threshold)
        threshold_values_for_risk = np.where(np.asarray(threshold_values, dtype=float) <= 0.0, 1.0, threshold_values)
        threshold_clamped_count = int(np.sum(np.asarray(threshold_values, dtype=float) <= 0.0))
        threshold_meta = {
            **threshold_meta,
            "threshold_clamped_for_risk": bool(threshold_clamped_count > 0),
            "threshold_clamped_count": threshold_clamped_count,
            "threshold_clamped_min_cases": 1.0,
        }
        district_effect, covariate_effect, time_effect, x_scaled = self._compute_linear_components(frame)
        linear_mean = district_effect + covariate_effect + time_effect
        mu_mean = np.exp(np.clip(linear_mean, a_min=-20.0, a_max=20.0))
        risk_mean_point = self._nb_exceedance_probability(
            mu_mean,
            np.full(len(frame), self.alpha_nb_),
            threshold_values_for_risk,
        )

        if self.idata_ is None:
            risk_frame = pd.DataFrame(
                {
                    "cases_mean": mu_mean,
                    "cases_q05": mu_mean,
                    "cases_q95": mu_mean,
                    "risk_mean": risk_mean_point,
                    "risk_q05": risk_mean_point,
                    "risk_q95": risk_mean_point,
                    "threshold_cases": threshold_values,
                },
                index=frame.index,
            )
            risk_frame["bayesian_risk"] = risk_frame["risk_mean"]
            metadata = {
                **threshold_meta,
                "interval_source": "point_estimate",
                "posterior_samples_used": 0,
                "degraded_mode": True,
                "climate_covariates": list(self.config.climate_covariates),
                "trained_covariate_order": list(expected_covariate_order),
                "covariate_alignment": dict(self.prediction_covariate_alignment_),
            }
            return risk_frame, metadata

        try:
            posterior = self.idata_.posterior
            alpha_draws = np.asarray(posterior["alpha_district"].to_numpy(), dtype=float)
            beta_draws = np.asarray(posterior["beta"].to_numpy(), dtype=float)
            alpha_nb_draws = np.asarray(posterior["alpha_nb"].to_numpy(), dtype=float)

            alpha_draws = alpha_draws.reshape(-1, alpha_draws.shape[-1])
            beta_draws = beta_draws.reshape(-1, beta_draws.shape[-1])
            alpha_nb_draws = alpha_nb_draws.reshape(-1)

            n_samples_total = alpha_nb_draws.shape[0]
            sample_cap = max(1, int(self.config.posterior_sample_cap))
            if n_samples_total > sample_cap:
                sample_idx = np.linspace(0, n_samples_total - 1, num=sample_cap, dtype=int)
                alpha_draws = alpha_draws[sample_idx]
                beta_draws = beta_draws[sample_idx]
                alpha_nb_draws = alpha_nb_draws[sample_idx]

            district_coord = posterior["alpha_district"].coords["district"].to_numpy()
            district_to_idx = {str(name): idx for idx, name in enumerate(district_coord.tolist())}
            district_idx = np.asarray(
                [district_to_idx.get(str(value), -1) for value in frame[self.config.district_column].astype(str)],
                dtype=int,
            )
            district_idx = np.where(district_idx >= 0, district_idx, 0)

            if "z_state" in posterior:
                z_draws = np.asarray(posterior["z_state"].to_numpy(), dtype=np.float32).reshape(
                    -1,
                    posterior["z_state"].shape[-2],
                    posterior["z_state"].shape[-1],
                )
                if n_samples_total > sample_cap:
                    z_draws = z_draws[sample_idx]

                state_coord = posterior["z_state"].coords["state"].to_numpy()
                state_to_idx = {str(state_value): idx for idx, state_value in enumerate(state_coord.tolist())}
                obs_state_idx = np.asarray(
                    [state_to_idx.get(str(state_code), -1) for state_code in frame["__state_code__"].astype(str)],
                    dtype=int,
                )
                obs_state_idx = np.where(obs_state_idx >= 0, obs_state_idx, 0)

                time_coord = posterior["z_state"].coords["time"].to_numpy()
                time_to_idx = {pd.Timestamp(str(time_value)): idx for idx, time_value in enumerate(time_coord.tolist())}
                if len(time_to_idx):
                    last_idx = len(time_to_idx) - 1
                    obs_time_idx = np.asarray(
                        [
                            time_to_idx.get(pd.Timestamp(value), last_idx) if pd.notna(value) else last_idx
                            for value in frame[self.config.date_column]
                        ],
                        dtype=int,
                    )
                else:
                    obs_time_idx = np.full(len(frame), 0, dtype=int)
            else:
                z_draws = None
                obs_state_idx = np.zeros(len(frame), dtype=int)
                obs_time_idx = np.zeros(len(frame), dtype=int)

            n_rows = len(frame)
            n_samples = int(alpha_nb_draws.shape[0])
            element_budget = 4_000_000
            configured_chunk_rows = max(1, int(self.config.predictive_chunk_rows))
            max_rows_by_budget = max(1, int(element_budget // max(1, n_samples)))
            chunk_rows = min(configured_chunk_rows, max_rows_by_budget)

            LOGGER.info(
                "Posterior predictive uncertainty in chunks: rows=%d, samples=%d, chunk_rows=%d",
                n_rows,
                n_samples,
                chunk_rows,
            )

            risk_mean = np.empty(n_rows, dtype=np.float64)
            risk_q05 = np.empty(n_rows, dtype=np.float64)
            risk_q95 = np.empty(n_rows, dtype=np.float64)
            cases_mean = np.empty(n_rows, dtype=np.float64)
            cases_q05 = np.empty(n_rows, dtype=np.float64)
            cases_q95 = np.empty(n_rows, dtype=np.float64)

            alpha_draws_f32 = alpha_draws.astype(np.float32, copy=False)
            beta_draws_f32 = beta_draws.astype(np.float32, copy=False)
            alpha_nb_draws_f32 = alpha_nb_draws.astype(np.float32, copy=False)
            x_scaled_f32 = x_scaled.astype(np.float32, copy=False)
            threshold_values_f32 = threshold_values_for_risk.astype(np.float32, copy=False)

            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                row_slice = slice(start, end)

                district_component = alpha_draws_f32[:, district_idx[row_slice]].T
                beta_component = np.dot(x_scaled_f32[row_slice], beta_draws_f32.T)
                if z_draws is not None:
                    time_component = z_draws[:, obs_state_idx[row_slice], obs_time_idx[row_slice]].T
                else:
                    time_component = 0.0

                linear_draws = district_component + beta_component + time_component
                mu_draws = np.exp(np.clip(linear_draws, a_min=-20.0, a_max=20.0)).astype(np.float32, copy=False)

                risk_draws = np.empty_like(mu_draws, dtype=np.float32)
                threshold_chunk = threshold_values_f32[row_slice]
                for sample_id in range(n_samples):
                    risk_draws[:, sample_id] = self._nb_exceedance_probability(
                        mu_draws[:, sample_id],
                        np.full(end - start, alpha_nb_draws_f32[sample_id], dtype=np.float32),
                        threshold_chunk,
                    ).astype(np.float32, copy=False)

                cases_mean[row_slice] = np.mean(mu_draws, axis=1, dtype=np.float64)
                cases_q05[row_slice] = np.quantile(mu_draws, 0.05, axis=1)
                cases_q95[row_slice] = np.quantile(mu_draws, 0.95, axis=1)
                risk_mean[row_slice] = np.mean(risk_draws, axis=1, dtype=np.float64)
                risk_q05[row_slice] = np.quantile(risk_draws, 0.05, axis=1)
                risk_q95[row_slice] = np.quantile(risk_draws, 0.95, axis=1)

            risk_frame = pd.DataFrame(
                {
                    "cases_mean": cases_mean,
                    "cases_q05": cases_q05,
                    "cases_q95": cases_q95,
                    "risk_mean": risk_mean,
                    "risk_q05": risk_q05,
                    "risk_q95": risk_q95,
                    "threshold_cases": threshold_values,
                },
                index=frame.index,
            )
            risk_frame["bayesian_risk"] = risk_frame["risk_mean"]
            metadata = {
                **threshold_meta,
                "interval_source": "posterior",
                "posterior_samples_used": int(n_samples),
                "degraded_mode": False,
                "climate_covariates": list(self.config.climate_covariates),
                "trained_covariate_order": list(expected_covariate_order),
                "covariate_alignment": dict(self.prediction_covariate_alignment_),
            }
            return risk_frame, metadata
        except Exception as predictive_error:
            LOGGER.warning("Posterior uncertainty prediction degraded to point estimate: %s", predictive_error)
            risk_frame = pd.DataFrame(
                {
                    "cases_mean": mu_mean,
                    "cases_q05": mu_mean,
                    "cases_q95": mu_mean,
                    "risk_mean": risk_mean_point,
                    "risk_q05": risk_mean_point,
                    "risk_q95": risk_mean_point,
                    "threshold_cases": threshold_values,
                },
                index=frame.index,
            )
            risk_frame["bayesian_risk"] = risk_frame["risk_mean"]
            metadata = {
                **threshold_meta,
                "interval_source": "point_estimate",
                "posterior_samples_used": 0,
                "degraded_mode": True,
                "climate_covariates": list(self.config.climate_covariates),
                "trained_covariate_order": list(expected_covariate_order),
                "covariate_alignment": dict(self.prediction_covariate_alignment_),
            }
            return risk_frame, metadata

    @staticmethod
    def _compute_scaling(feature_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        means = np.nanmean(feature_matrix, axis=0)
        scales = np.nanstd(feature_matrix, axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-6), scales, 1.0)
        means = np.where(np.isfinite(means), means, 0.0)
        return means.astype(float), scales.astype(float)

    @staticmethod
    def _extract_sampler_diagnostics(idata: Any) -> dict[str, float]:
        summary: dict[str, float] = {
            "divergences": 0.0,
            "max_tree_depth": 0.0,
            "r_hat_max": float("nan"),
            "ess_min": float("nan"),
        }
        try:
            sample_stats = getattr(idata, "sample_stats", None)
            if sample_stats is None:
                return summary

            if "diverging" in sample_stats:
                divergences = np.asarray(sample_stats["diverging"].to_numpy(), dtype=float)
                summary["divergences"] = float(np.nansum(divergences))

            if "tree_depth" in sample_stats:
                tree_depth = np.asarray(sample_stats["tree_depth"].to_numpy(), dtype=float)
                summary["max_tree_depth"] = float(np.nanmax(tree_depth))
            elif "depth" in sample_stats:
                tree_depth = np.asarray(sample_stats["depth"].to_numpy(), dtype=float)
                summary["max_tree_depth"] = float(np.nanmax(tree_depth))
        except Exception as diagnostics_error:
            LOGGER.warning("Unable to extract sampler diagnostics from InferenceData: %s", diagnostics_error)
        return summary

    def _needs_simplification(self, diagnostics: dict[str, float]) -> bool:
        divergence_count = diagnostics.get("divergences", 0.0)
        max_tree_depth = diagnostics.get("max_tree_depth", float("nan"))
        rhat_max = diagnostics.get("r_hat_max", float("nan"))
        ess_min = diagnostics.get("ess_min", float("nan"))
        return bool(
            divergence_count > float(self.config.divergence_warn_threshold)
            or (np.isfinite(max_tree_depth) and max_tree_depth >= float(self.config.max_treedepth))
            or (np.isfinite(rhat_max) and rhat_max > float(self.config.rhat_warn_threshold))
            or (np.isfinite(ess_min) and ess_min < float(self.config.ess_warn_threshold))
        )

    @staticmethod
    def _normalize_sampling_backend(raw_backend: Any) -> str:
        normalized = str(raw_backend if raw_backend is not None else "auto").strip().lower()
        if normalized not in {"auto", "pymc", "jax_numpyro"}:
            LOGGER.warning("Invalid sampling_backend '%s'; defaulting to 'auto'", raw_backend)
            return "auto"
        return normalized

    @staticmethod
    def _resolve_sampling_backend(requested_backend: str, compute_backend_effective: str) -> str:
        if requested_backend == "auto":
            if str(compute_backend_effective or "cpu").strip().lower() == "nvidia_cuda":
                return "jax_numpyro"
            return "pymc"
        return requested_backend

    @staticmethod
    def _expected_runtime_backend(compute_backend_effective: str) -> str:
        normalized = str(compute_backend_effective or "cpu").strip().lower()
        if normalized == "macos_metal":
            return "metal"
        if normalized == "nvidia_cuda":
            return "cuda"
        return "cpu"

    def _infer_jax_runtime_backend(self, jax_module: Any) -> str:
        try:
            devices = jax_module.devices()
        except Exception:
            return "cpu"
        if not devices:
            return "cpu"
        device_platform = str(getattr(devices[0], "platform", "cpu")).strip().lower()
        if device_platform == "metal":
            return "metal"
        if device_platform in {"gpu", "cuda", "rocm"}:
            return "cuda"
        return "cpu"

    def _choose_sampling_backend(self, compute_backend_effective: str) -> tuple[str, str]:
        requested_sampling_backend = self._normalize_sampling_backend(self.config.sampling_backend)
        effective_sampling_backend = self._resolve_sampling_backend(
            requested_sampling_backend,
            str(compute_backend_effective or "cpu"),
        )
        return requested_sampling_backend, effective_sampling_backend

    def _fit_jax_numpyro_model(
        self,
        *,
        pm: Any,
        jax: Any,
        districts: pd.Index,
        states: pd.Index,
        times: pd.Index,
        district_idx: np.ndarray,
        state_idx: np.ndarray,
        time_idx: np.ndarray,
        feature_matrix_scaled: np.ndarray,
        observed: np.ndarray,
        simplified_mode: bool,
    ) -> tuple[Any, Any, np.ndarray, str]:
        sampling_jax = getattr(pm, "sampling_jax", None)
        sample_numpyro_nuts = getattr(sampling_jax, "sample_numpyro_nuts", None) if sampling_jax is not None else None
        if sample_numpyro_nuts is None:
            pymc_sampling_jax = _try_import("pymc.sampling.jax")
            sample_numpyro_nuts = (
                getattr(pymc_sampling_jax, "sample_numpyro_nuts", None) if pymc_sampling_jax is not None else None
            )
        if sample_numpyro_nuts is None:
            LOGGER.warning("PyMC JAX sampler is unavailable. Falling back to CPU.")
            raise ImportError("PyMC JAX sampler is unavailable (missing pymc.sampling.jax.sample_numpyro_nuts)")

        coords: dict[str, Any] = {
            "district": districts.astype(str).tolist(),
            "state": states.astype(str).tolist(),
            "covariate": list(self.config.climate_covariates),
            "obs": np.arange(len(observed)),
        }
        if not simplified_mode:
            coords["time"] = [str(timestamp) for timestamp in times.tolist()]
            coords["time_inner"] = [str(timestamp) for timestamp in times.tolist()[1:]]

        with pm.Model(coords=coords) as model:
            mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.0)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
            alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0, dims="district")
            alpha_district = pm.Deterministic(
                "alpha_district",
                mu_alpha + alpha_raw * sigma_alpha,
                dims="district",
            )

            beta = pm.Normal("beta", mu=0.0, sigma=0.4, dims="covariate")

            if simplified_mode:
                z_state_values = np.zeros((len(states), len(times)), dtype=float)
                linear = alpha_district[district_idx] + pm.math.dot(feature_matrix_scaled, beta)
            else:
                rho = pm.TruncatedNormal(
                    "rho",
                    mu=0.85,
                    sigma=0.15,
                    lower=0.0,
                    upper=0.999,
                )
                sigma_z_state = pm.HalfNormal("sigma_z_state", sigma=0.25)
                u_init_s = pm.Normal("u_init_s", mu=0.0, sigma=1.0, dims="state")
                z_innov_s = pm.Normal("z_innov_s", mu=0.0, sigma=1.0, dims=("state", "time_inner"))

                pytensor = _try_import("pytensor")
                if pytensor is None:
                    raise ImportError("pytensor is required for non-centered AR(1) latent recursion")

                def _ar1_step(innov_t: Any, prev_u: Any, rho_param: Any) -> Any:
                    return rho_param * prev_u + innov_t

                u_scan_t, _ = pytensor.scan(
                    fn=_ar1_step,
                    sequences=[z_innov_s.T],
                    outputs_info=[u_init_s],
                    non_sequences=[rho],
                    strict=True,
                )
                u_t = pm.math.concatenate([u_init_s[:, None], u_scan_t.T], axis=1)
                z_state = pm.Deterministic("z_state", sigma_z_state * u_t, dims=("state", "time"))
                linear = alpha_district[district_idx] + pm.math.dot(feature_matrix_scaled, beta) + z_state[state_idx, time_idx]
                z_state_values = np.full((len(states), len(times)), np.nan, dtype=float)

            mu = pm.math.exp(pm.math.clip(linear, -10.0, 10.0))
            alpha_nb = pm.HalfNormal("alpha_nb", sigma=1.0)
            pm.NegativeBinomial("cases_obs", mu=mu, alpha=alpha_nb, observed=observed, dims="obs")

            LOGGER.info(
                "Starting Bayesian JAX sampling: obs=%d, districts=%d, times=%d, n_covariates=%d, chains=%d, draws=%d, tune=%d, target_accept=%.3f, max_treedepth=%d, simplified_mode=%s, progressbar=%s",
                int(len(observed)),
                int(len(districts)),
                int(len(times)),
                int(feature_matrix_scaled.shape[1]),
                int(self.config.chains),
                int(self.config.draws),
                int(self.config.tune),
                float(self.config.target_accept),
                int(self.config.max_treedepth),
                bool(simplified_mode),
                bool(self.config.bayesian_progress),
            )

            sampling_kwargs: dict[str, Any] = {
                "draws": self.config.draws,
                "tune": self.config.tune,
                "chains": self.config.chains,
                "target_accept": self.config.target_accept,
                "random_seed": self.config.random_seed,
                "progressbar": self.config.bayesian_progress,
                "chain_method": "vectorized",  # Run multiple chains in parallel on a single GPU
                "nuts_kwargs": {"max_tree_depth": self.config.max_treedepth},
                "idata_kwargs": {"log_likelihood": False},
            }
            try:
                idata = sample_numpyro_nuts(**sampling_kwargs)
            except TypeError:
                sampling_kwargs.pop("nuts_kwargs", None)
                sampling_kwargs.pop("idata_kwargs", None)
                idata = sample_numpyro_nuts(**sampling_kwargs)

        if not hasattr(idata, "posterior"):
            raise RuntimeError("JAX sampler did not return posterior inference data")
        posterior = getattr(idata, "posterior", None)
        if posterior is None or "alpha_district" not in posterior or "beta" not in posterior:
            raise RuntimeError("JAX sampler posterior missing required variables for downstream diagnostics")

        if not simplified_mode and "z_state" in idata.posterior:
            z_state_values = np.asarray(idata.posterior["z_state"].mean(dim=("chain", "draw")).to_numpy(), dtype=float)

        runtime_backend = self._infer_jax_runtime_backend(jax)
        return model, idata, z_state_values, runtime_backend

    def _fit_pymc_model(
        self,
        *,
        pm: Any,
        districts: pd.Index,
        states: pd.Index,
        times: pd.Index,
        district_idx: np.ndarray,
        state_idx: np.ndarray,
        time_idx: np.ndarray,
        feature_matrix_scaled: np.ndarray,
        observed: np.ndarray,
        simplified_mode: bool,
    ) -> tuple[Any, Any, np.ndarray]:
        coords: dict[str, Any] = {
            "district": districts.astype(str).tolist(),
            "state": states.astype(str).tolist(),
            "covariate": list(self.config.climate_covariates),
            "obs": np.arange(len(observed)),
        }
        if not simplified_mode:
            coords["time"] = [str(timestamp) for timestamp in times.tolist()]
            coords["time_inner"] = [str(timestamp) for timestamp in times.tolist()[1:]]

        with pm.Model(coords=coords) as model:
            mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.0)
            sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=0.5)
            alpha_raw = pm.Normal("alpha_raw", mu=0.0, sigma=1.0, dims="district")
            alpha_district = pm.Deterministic(
                "alpha_district",
                mu_alpha + alpha_raw * sigma_alpha,
                dims="district",
            )

            beta = pm.Normal("beta", mu=0.0, sigma=0.4, dims="covariate")

            if simplified_mode:
                z_state_values = np.zeros((len(states), len(times)), dtype=float)
                linear = alpha_district[district_idx] + pm.math.dot(feature_matrix_scaled, beta)
            else:
                rho = pm.TruncatedNormal(
                    "rho",
                    mu=0.85,
                    sigma=0.15,
                    lower=0.0,
                    upper=0.999,
                )
                sigma_z_state = pm.HalfNormal("sigma_z_state", sigma=0.25)
                u_init_s = pm.Normal("u_init_s", mu=0.0, sigma=1.0, dims="state")
                z_innov_s = pm.Normal("z_innov_s", mu=0.0, sigma=1.0, dims=("state", "time_inner"))

                pytensor = _try_import("pytensor")
                if pytensor is None:
                    raise ImportError("pytensor is required for non-centered AR(1) latent recursion")

                def _ar1_step(innov_t: Any, prev_u: Any, rho_param: Any) -> Any:
                    return rho_param * prev_u + innov_t

                u_scan_t, _ = pytensor.scan(
                    fn=_ar1_step,
                    sequences=[z_innov_s.T],
                    outputs_info=[u_init_s],
                    non_sequences=[rho],
                    strict=True,
                )
                u_t = pm.math.concatenate([u_init_s[:, None], u_scan_t.T], axis=1)
                z_state = pm.Deterministic("z_state", sigma_z_state * u_t, dims=("state", "time"))
                linear = alpha_district[district_idx] + pm.math.dot(feature_matrix_scaled, beta) + z_state[state_idx, time_idx]
                z_state_values = np.full((len(states), len(times)), np.nan, dtype=float)

            mu = pm.math.exp(pm.math.clip(linear, -10.0, 10.0))
            alpha_nb = pm.HalfNormal("alpha_nb", sigma=1.0)
            pm.NegativeBinomial("cases_obs", mu=mu, alpha=alpha_nb, observed=observed, dims="obs")

            LOGGER.info(
                "Starting Bayesian sampling: obs=%d, districts=%d, times=%d, n_covariates=%d, chains=%d, draws=%d, tune=%d, target_accept=%.3f, max_treedepth=%d, simplified_mode=%s, progressbar=%s",
                int(len(observed)),
                int(len(districts)),
                int(len(times)),
                int(feature_matrix_scaled.shape[1]),
                int(self.config.chains),
                int(self.config.draws),
                int(self.config.tune),
                float(self.config.target_accept),
                int(self.config.max_treedepth),
                bool(simplified_mode),
                bool(self.config.bayesian_progress),
            )

            idata = pm.sample(
                draws=self.config.draws,
                tune=self.config.tune,
                chains=self.config.chains,
                target_accept=self.config.target_accept,
                nuts={"max_treedepth": self.config.max_treedepth},
                random_seed=self.config.random_seed,
                progressbar=self.config.bayesian_progress,
                return_inferencedata=True,
            )

        if not simplified_mode:
            z_state_values = np.asarray(idata.posterior["z_state"].mean(dim=("chain", "draw")).to_numpy(), dtype=float)

        return model, idata, z_state_values

    def _fit_with_selected_backend(
        self,
        *,
        pm: Any,
        districts: pd.Index,
        states: pd.Index,
        times: pd.Index,
        district_idx: np.ndarray,
        state_idx: np.ndarray,
        time_idx: np.ndarray,
        feature_matrix_scaled: np.ndarray,
        observed: np.ndarray,
        simplified_mode: bool,
        requested_sampling_backend: str,
        effective_sampling_backend: str,
    ) -> tuple[Any, Any, np.ndarray, str, str, str | None, str]:
        sampling_fallback_reason: str | None = None
        actual_runtime_backend = "cpu"

        if effective_sampling_backend == "jax_numpyro":
            try:
                jax_module = _try_import("jax")
                numpyro_module = _try_import("numpyro")
                if jax_module is None or numpyro_module is None:
                    raise ImportError("missing optional JAX dependencies 'jax' and/or 'numpyro'")
                model, idata, z_state_values, actual_runtime_backend = self._fit_jax_numpyro_model(
                    pm=pm,
                    jax=jax_module,
                    districts=districts,
                    states=states,
                    times=times,
                    district_idx=district_idx,
                    state_idx=state_idx,
                    time_idx=time_idx,
                    feature_matrix_scaled=feature_matrix_scaled,
                    observed=observed,
                    simplified_mode=simplified_mode,
                )
                return (
                    model,
                    idata,
                    z_state_values,
                    requested_sampling_backend,
                    effective_sampling_backend,
                    sampling_fallback_reason,
                    actual_runtime_backend,
                )
            except Exception as jax_error:
                sampling_fallback_reason = f"JAX sampler unavailable; falling back to PyMC CPU sampler ({jax_error})"
                LOGGER.warning("============================================================")
                LOGGER.warning("WARNING: %s", sampling_fallback_reason)
                error_text = str(jax_error)
                if "default_memory_space" in error_text:
                    LOGGER.warning(
                        "Detected JAX-Metal runtime incompatibility (default_memory_space). "
                        "On macOS, ensure a supported JAX/jaxlib/jax-metal version matrix and consider Python 3.12/3.13 for Metal runs."
                    )
                    LOGGER.warning(
                        "Optional probe: set ENABLE_PJRT_COMPATIBILITY=1 for newer jaxlib compatibility on Metal."
                    )
                else:
                    LOGGER.warning("Ensure jax, jaxlib, and numpyro are installed in your environment.")
                LOGGER.warning("============================================================")
                effective_sampling_backend = "pymc"
                actual_runtime_backend = "cpu"

        model, idata, z_state_values = self._fit_pymc_model(
            pm=pm,
            districts=districts,
            states=states,
            times=times,
            district_idx=district_idx,
            state_idx=state_idx,
            time_idx=time_idx,
            feature_matrix_scaled=feature_matrix_scaled,
            observed=observed,
            simplified_mode=simplified_mode,
        )
        return (
            model,
            idata,
            z_state_values,
            requested_sampling_backend,
            effective_sampling_backend,
            sampling_fallback_reason,
            actual_runtime_backend,
        )

    def _finalize_from_posterior(
        self,
        *,
        idata: Any,
        states: pd.Index,
        districts: pd.Index,
        times: pd.Index,
        z_state_values: np.ndarray,
        simplified_mode: bool,
        diagnostics_summary: dict[str, float],
    ) -> None:
        self.model_ = None
        self.idata_ = idata

        posterior = idata.posterior
        self.global_intercept_ = float(posterior["mu_alpha"].mean().item())

        district_means = posterior["alpha_district"].mean(dim=("chain", "draw")).to_series()
        self.district_effects_ = {
            str(district): float(district_means.loc[district])
            for district in district_means.index
        }

        beta_means = posterior["beta"].mean(dim=("chain", "draw")).to_series()
        self.beta_effects_ = {
            str(covariate): float(beta_means.loc[covariate])
            for covariate in beta_means.index
        }
        self.alpha_nb_ = float(posterior["alpha_nb"].mean().item()) if "alpha_nb" in posterior else 1.0

        if "rho" in posterior:
            rho_mean = float(posterior["rho"].mean().item())
            if abs(rho_mean) > 0.98:
                LOGGER.warning("Posterior mean rho is near AR(1) boundary: rho=%.4f", rho_mean)

        if simplified_mode:
            self.latent_by_state_time_ = {
                (str(state_code), pd.Timestamp(timestamp)): 0.0
                for state_code in states
                for timestamp in times
            }
        else:
            self.latent_by_state_time_ = {
                (str(state_code), pd.Timestamp(timestamp)): float(z_state_values[state_idx, time_idx])
                for state_idx, state_code in enumerate(states)
                for time_idx, timestamp in enumerate(times)
            }

        self.simplified_used_ = simplified_mode
        self.diagnostics_summary_ = diagnostics_summary
        self.trained_covariate_order_ = tuple(self.config.climate_covariates)
        self.fitted_ = True

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        compute_backend_effective: str = "cpu",
    ) -> "HierarchicalBayesianModel":
        """Fit the hierarchical model using PyMC when available."""
        frame = self._prepare_design(X, y)
        self.fallback_rate_ = float(frame["__target__"].mean()) if len(frame) else 0.0

        try:
            pm, _ = _require_pymc_dependencies()
        except ImportError as import_error:
            if self.config.strict_dependencies:
                raise
            self._fit_fallback(frame, str(import_error))
            return self

        districts = pd.Index(sorted(frame[self.config.district_column].dropna().unique()), dtype="object")
        district_to_idx = {district: idx for idx, district in enumerate(districts)}
        district_idx = frame[self.config.district_column].map(district_to_idx).astype(int).to_numpy(dtype=np.int32)

        state_codes = frame["__state_code__"].astype(str)
        states = pd.Index(sorted(state_codes.dropna().unique()), dtype="object")
        state_to_idx = {state_code: idx for idx, state_code in enumerate(states)}
        state_idx = state_codes.map(state_to_idx).astype(int).to_numpy(dtype=np.int32)

        times = pd.Index(sorted(frame[self.config.date_column].dropna().unique()))
        time_to_idx = {timestamp: idx for idx, timestamp in enumerate(times)}
        time_idx = frame[self.config.date_column].map(time_to_idx).fillna(len(times) - 1).astype(int).to_numpy(dtype=np.int32)

        feature_matrix = frame.loc[:, list(self.config.climate_covariates)].to_numpy(dtype=np.float32)
        means, scales = self._compute_scaling(feature_matrix)
        feature_matrix_scaled = ((feature_matrix - means) / scales).astype(np.float32, copy=False)
        self.covariate_means_ = {
            covariate: float(means[idx])
            for idx, covariate in enumerate(self.config.climate_covariates)
        }
        self.covariate_scales_ = {
            covariate: float(scales[idx])
            for idx, covariate in enumerate(self.config.climate_covariates)
        }
        observed = frame["__target__"].to_numpy(dtype=np.float32)

        requested_sampling_backend, effective_sampling_backend = self._choose_sampling_backend(
            str(compute_backend_effective or "cpu")
        )
        expected_runtime_backend = self._expected_runtime_backend(str(compute_backend_effective or "cpu"))

        simplified_mode = bool(self.config.bayesian_simplified_mode)
        if self.config.force_full_bayesian:
            if simplified_mode:
                LOGGER.warning(
                    "force_full_bayesian=True overrides bayesian_simplified_mode=True; running full latent AR(1) model."
                )
            simplified_mode = False

        (
            model,
            idata,
            z_state_values,
            requested_sampling_backend,
            effective_sampling_backend,
            sampling_fallback_reason,
            actual_runtime_backend,
        ) = self._fit_with_selected_backend(
            pm=pm,
            districts=districts,
            states=states,
            times=times,
            district_idx=district_idx,
            state_idx=state_idx,
            time_idx=time_idx,
            feature_matrix_scaled=feature_matrix_scaled,
            observed=observed,
            simplified_mode=simplified_mode,
            requested_sampling_backend=requested_sampling_backend,
            effective_sampling_backend=effective_sampling_backend,
        )

        self.sampling_backend_requested_ = requested_sampling_backend
        self.sampling_backend_effective_ = effective_sampling_backend
        self.sampling_backend_fallback_reason_ = sampling_fallback_reason
        self.sampling_runtime_backend_ = actual_runtime_backend

        self.sampling_diagnostics_ = {
            "sampling_backend_requested": self.sampling_backend_requested_,
            "sampling_backend_effective": self.sampling_backend_effective_,
            "sampling_backend_fallback_reason": self.sampling_backend_fallback_reason_,
            "actual_runtime_backend": self.sampling_runtime_backend_,
            "backend_implemented": bool(actual_runtime_backend == expected_runtime_backend),
            "compute_backend_effective": str(compute_backend_effective or "cpu"),
        }

        diagnostics_summary = self._extract_sampler_diagnostics(idata)
        arviz_diagnostics_status = "ok"
        try:
            from src.models.bayesian.diagnostics import summarize_diagnostics

            convergence = summarize_diagnostics(idata)
            diagnostics_summary["r_hat_max"] = float(convergence.get("r_hat_max", float("nan")))
            diagnostics_summary["ess_min"] = float(convergence.get("ess_min", float("nan")))
        except Exception as diagnostics_error:
            arviz_diagnostics_status = "unavailable"
            LOGGER.warning("Unable to compute R-hat/ESS diagnostics from ArviZ: %s", diagnostics_error)

        rhat_value = pd.to_numeric(pd.Series([diagnostics_summary.get("r_hat_max")]), errors="coerce").iloc[0]
        ess_value = pd.to_numeric(pd.Series([diagnostics_summary.get("ess_min")]), errors="coerce").iloc[0]
        diagnostics_nan = bool(pd.isna(rhat_value) or pd.isna(ess_value))
        if diagnostics_nan:
            if arviz_diagnostics_status != "unavailable":
                arviz_diagnostics_status = "nan_after_fit"
            LOGGER.warning(
                "Bayesian post-fit diagnostics contain NaN values: r_hat_max=%s ess_min=%s",
                diagnostics_summary.get("r_hat_max"),
                diagnostics_summary.get("ess_min"),
            )

        self.sampling_diagnostics_["arviz_diagnostics_status"] = str(arviz_diagnostics_status)
        self.sampling_diagnostics_["arviz_diagnostics_available"] = bool(arviz_diagnostics_status == "ok")
        self.sampling_diagnostics_["arviz_diagnostics_nan"] = bool(diagnostics_nan)

        diagnostics_summary["simplified_mode"] = 1.0 if simplified_mode else 0.0
        diagnostics_summary["fallback"] = 0.0

        self.model_ = model
        self._finalize_from_posterior(
            idata=idata,
            states=states,
            districts=districts,
            times=times,
            z_state_values=z_state_values,
            simplified_mode=simplified_mode,
            diagnostics_summary=diagnostics_summary,
        )

        LOGGER.info(
            "Bayesian fit diagnostics: backend=%s, simplified_mode=%s, divergences=%.0f, max_tree_depth=%.0f, r_hat_max=%.4f, ess_min=%.1f",
            self.sampling_backend_effective_,
            bool(simplified_mode),
            diagnostics_summary.get("divergences", float("nan")),
            diagnostics_summary.get("max_tree_depth", float("nan")),
            diagnostics_summary.get("r_hat_max", float("nan")),
            diagnostics_summary.get("ess_min", float("nan")),
        )

        if self._needs_simplification(diagnostics_summary):
            LOGGER.warning(
                "Bayesian fit completed with convergence concerns: backend=%s, simplified_mode=%s, divergences=%.0f, max_tree_depth=%.0f, r_hat_max=%.4f, ess_min=%.1f",
                self.sampling_backend_effective_,
                bool(simplified_mode),
                diagnostics_summary.get("divergences", float("nan")),
                diagnostics_summary.get("max_tree_depth", float("nan")),
                diagnostics_summary.get("r_hat_max", float("nan")),
                diagnostics_summary.get("ess_min", float("nan")),
            )
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Backward-compatible risk mean prediction as a Series."""
        risk_frame, _ = self.predict_with_uncertainty(X)
        return pd.Series(
            pd.to_numeric(risk_frame["risk_mean"], errors="coerce").fillna(0.0).clip(0.0, 1.0),
            index=risk_frame.index,
            name="bayesian_risk",
            dtype="float64",
        )


def build_hierarchical_nb_model(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    config: BayesianModelConfig = BayesianModelConfig(strict_dependencies=True),
) -> HierarchicalBayesianModel:
    """Build and fit a strict-dependency hierarchical NB model.

    Raises informative ImportError when ``pymc``/``arviz`` are missing.
    """
    model = HierarchicalBayesianModel(config=config)
    return model.fit(X, y)
