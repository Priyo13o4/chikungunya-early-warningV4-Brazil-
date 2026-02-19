"""Hierarchical Bayesian model for latent chikungunya outbreak risk."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

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
    climate_covariates: tuple[str, ...] = ("temp_anomaly", "degree_days_20", "rainfall_4wk", "lai_anomaly")
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
    outbreak_threshold_default_cases: float = 1.0
    posterior_sample_cap: int = 400


@dataclass
class HierarchicalBayesianModel:
    """Hierarchical negative-binomial Bayesian model with AR(1) latent risk state."""

    config: BayesianModelConfig = field(default_factory=BayesianModelConfig)
    fitted_: bool = False
    model_: Any = None
    idata_: Any = None
    district_effects_: dict[str, float] = field(default_factory=dict)
    beta_effects_: dict[str, float] = field(default_factory=dict)
    latent_by_time_: dict[pd.Timestamp, float] = field(default_factory=dict)
    global_intercept_: float = 0.0
    fallback_rate_: float = 0.0
    covariate_means_: dict[str, float] = field(default_factory=dict)
    covariate_scales_: dict[str, float] = field(default_factory=dict)
    diagnostics_summary_: dict[str, float] = field(default_factory=dict)
    simplified_used_: bool = False
    alpha_nb_: float = 1.0

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

        for covariate in self.config.climate_covariates:
            if covariate not in frame.columns:
                frame[covariate] = 0.0
            frame[covariate] = pd.to_numeric(frame[covariate], errors="coerce").fillna(0.0)

        frame[self.config.district_column] = frame[self.config.district_column].astype(str).fillna("unknown")
        frame = frame.sort_values([self.config.date_column, self.config.district_column]).copy()
        return frame

    def _fit_fallback(self, frame: pd.DataFrame, reason: str) -> None:
        LOGGER.warning("Using Bayesian fallback mode: %s", reason)
        self.fallback_rate_ = float(frame["__target__"].mean()) if len(frame) else 0.0
        self.global_intercept_ = float(np.log1p(self.fallback_rate_))
        self.district_effects_ = (
            frame.groupby(self.config.district_column, dropna=False)["__target__"].mean().apply(np.log1p).to_dict()
        )
        self.beta_effects_ = {covariate: 0.0 for covariate in self.config.climate_covariates}
        self.alpha_nb_ = 1.0
        self.covariate_means_ = {
            covariate: float(pd.to_numeric(frame[covariate], errors="coerce").fillna(0.0).mean())
            for covariate in self.config.climate_covariates
        }
        self.covariate_scales_ = {covariate: 1.0 for covariate in self.config.climate_covariates}
        time_mean = frame.groupby(self.config.date_column, dropna=False)["__target__"].mean().apply(np.log1p)
        self.latent_by_time_ = {pd.Timestamp(key): float(value) for key, value in time_mean.items() if pd.notna(key)}
        self.diagnostics_summary_ = {
            "divergences": float("nan"),
            "max_tree_depth": float("nan"),
            "r_hat_max": float("nan"),
            "ess_min": float("nan"),
            "simplified_mode": 0.0,
            "fallback": 1.0,
        }
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
        if outbreak_threshold is not None:
            threshold_series = pd.to_numeric(pd.Series(outbreak_threshold, index=frame.index), errors="coerce")
            threshold_series = threshold_series.fillna(float(self.config.outbreak_threshold_default_cases))
            return threshold_series.to_numpy(dtype=float), {
                "threshold_basis": "provided_series",
                "threshold_default": float(self.config.outbreak_threshold_default_cases),
                "threshold_column": None,
            }

        threshold_candidates = [
            column
            for column in frame.columns
            if isinstance(column, str) and (column.startswith("threshold_p") or column == "outbreak_threshold")
        ]
        if threshold_candidates:
            selected_column = sorted(threshold_candidates)[0]
            threshold_series = pd.to_numeric(frame[selected_column], errors="coerce").fillna(
                float(self.config.outbreak_threshold_default_cases)
            )
            return threshold_series.to_numpy(dtype=float), {
                "threshold_basis": f"column:{selected_column}",
                "threshold_default": float(self.config.outbreak_threshold_default_cases),
                "threshold_column": selected_column,
            }

        return np.full(len(frame), float(self.config.outbreak_threshold_default_cases), dtype=float), {
            "threshold_basis": "default",
            "threshold_default": float(self.config.outbreak_threshold_default_cases),
            "threshold_column": None,
        }

    def _compute_linear_components(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        district_values = frame[self.config.district_column].astype(str)
        date_values = pd.to_datetime(frame[self.config.date_column], errors="coerce")

        district_default = float(np.log1p(max(self.fallback_rate_, 0.0)))
        district_effect = np.asarray(
            [self.district_effects_.get(value, district_default) for value in district_values],
            dtype=float,
        )

        time_default = float(list(self.latent_by_time_.values())[-1]) if self.latent_by_time_ else 0.0
        time_effect = np.asarray(
            [
                self.latent_by_time_.get(pd.Timestamp(value), time_default) if pd.notna(value) else 0.0
                for value in date_values
            ],
            dtype=float,
        )

        feature_matrix = frame.loc[:, list(self.config.climate_covariates)].copy()
        for covariate in self.config.climate_covariates:
            feature_matrix[covariate] = pd.to_numeric(feature_matrix[covariate], errors="coerce").fillna(0.0)

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

        for covariate in self.config.climate_covariates:
            if covariate not in frame.columns:
                frame[covariate] = 0.0

        threshold_values, threshold_meta = self._resolve_outbreak_thresholds(frame, outbreak_threshold)
        district_effect, covariate_effect, time_effect, x_scaled = self._compute_linear_components(frame)
        linear_mean = district_effect + covariate_effect + time_effect
        mu_mean = np.exp(np.clip(linear_mean, a_min=-20.0, a_max=20.0))
        risk_mean_point = self._nb_exceedance_probability(mu_mean, np.full(len(frame), self.alpha_nb_), threshold_values)

        if self.idata_ is None:
            risk_frame = pd.DataFrame(
                {
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

            district_component = alpha_draws[:, district_idx].T
            beta_component = np.dot(x_scaled, beta_draws.T)

            if "z_t" in posterior:
                z_draws = np.asarray(posterior["z_t"].to_numpy(), dtype=float).reshape(-1, posterior["z_t"].shape[-1])
                if n_samples_total > sample_cap:
                    z_draws = z_draws[sample_idx]
                time_coord = posterior["z_t"].coords["time"].to_numpy()
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
                    time_component = z_draws[:, obs_time_idx].T
                else:
                    time_component = np.zeros_like(beta_component)
            else:
                time_component = np.zeros_like(beta_component)

            linear_draws = district_component + beta_component + time_component
            mu_draws = np.exp(np.clip(linear_draws, a_min=-20.0, a_max=20.0))

            risk_draws = np.empty_like(mu_draws)
            for sample_id in range(mu_draws.shape[1]):
                risk_draws[:, sample_id] = self._nb_exceedance_probability(
                    mu_draws[:, sample_id],
                    np.full(len(frame), alpha_nb_draws[sample_id], dtype=float),
                    threshold_values,
                )

            risk_frame = pd.DataFrame(
                {
                    "risk_mean": np.mean(risk_draws, axis=1),
                    "risk_q05": np.quantile(risk_draws, 0.05, axis=1),
                    "risk_q95": np.quantile(risk_draws, 0.95, axis=1),
                    "threshold_cases": threshold_values,
                },
                index=frame.index,
            )
            risk_frame["bayesian_risk"] = risk_frame["risk_mean"]
            metadata = {
                **threshold_meta,
                "interval_source": "posterior",
                "posterior_samples_used": int(mu_draws.shape[1]),
                "degraded_mode": False,
            }
            return risk_frame, metadata
        except Exception as predictive_error:
            LOGGER.warning("Posterior uncertainty prediction degraded to point estimate: %s", predictive_error)
            risk_frame = pd.DataFrame(
                {
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
        rhat_max = diagnostics.get("r_hat_max", float("nan"))
        ess_min = diagnostics.get("ess_min", float("nan"))
        return bool(
            divergence_count > float(self.config.divergence_warn_threshold)
            or (np.isfinite(rhat_max) and rhat_max > float(self.config.rhat_warn_threshold))
            or (np.isfinite(ess_min) and ess_min < float(self.config.ess_warn_threshold))
        )

    def _fit_pymc_model(
        self,
        *,
        pm: Any,
        districts: pd.Index,
        times: pd.Index,
        district_idx: np.ndarray,
        time_idx: np.ndarray,
        feature_matrix_scaled: np.ndarray,
        observed: np.ndarray,
        simplified_mode: bool,
    ) -> tuple[Any, Any, np.ndarray]:
        coords: dict[str, Any] = {
            "district": districts.astype(str).tolist(),
            "covariate": list(self.config.climate_covariates),
            "obs": np.arange(len(observed)),
        }
        if not simplified_mode:
            coords["time"] = [str(timestamp) for timestamp in times.tolist()]

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
                z_t_values = np.zeros(len(times), dtype=float)
                linear = alpha_district[district_idx] + pm.math.dot(feature_matrix_scaled, beta)
            else:
                rho_raw = pm.Normal("rho_raw", mu=0.0, sigma=0.8)
                rho = pm.Deterministic("rho", 0.95 * pm.math.tanh(rho_raw))
                sigma_z = pm.HalfNormal("sigma_z", sigma=0.35)
                z_t = pm.AR(
                    "z_t",
                    rho=rho,
                    sigma=sigma_z,
                    init_dist=pm.Normal.dist(mu=0.0, sigma=0.35),
                    dims="time",
                )
                linear = alpha_district[district_idx] + pm.math.dot(feature_matrix_scaled, beta) + z_t[time_idx]
                z_t_values = np.full(len(times), np.nan, dtype=float)

            mu = pm.math.exp(pm.math.clip(linear, -10.0, 10.0))
            alpha_nb = pm.Exponential("alpha_nb", lam=1.0)
            pm.NegativeBinomial("cases_obs", mu=mu, alpha=alpha_nb, observed=observed, dims="obs")

            LOGGER.info(
                "Starting Bayesian sampling: chains=%d, draws=%d, tune=%d, target_accept=%.3f, max_treedepth=%d, simplified_mode=%s, progressbar=%s",
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
            z_t_values = np.asarray(idata.posterior["z_t"].mean(dim=("chain", "draw")).to_numpy(), dtype=float)

        return model, idata, z_t_values

    def _finalize_from_posterior(
        self,
        *,
        idata: Any,
        districts: pd.Index,
        times: pd.Index,
        z_t_values: np.ndarray,
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

        if simplified_mode:
            self.latent_by_time_ = {pd.Timestamp(timestamp): 0.0 for timestamp in times}
        else:
            self.latent_by_time_ = {
                pd.Timestamp(timestamp): float(z_t_values[idx])
                for idx, timestamp in enumerate(times)
            }

        self.simplified_used_ = simplified_mode
        self.diagnostics_summary_ = diagnostics_summary
        self.fitted_ = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "HierarchicalBayesianModel":
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
        district_idx = frame[self.config.district_column].map(district_to_idx).astype(int).to_numpy()

        times = pd.Index(sorted(frame[self.config.date_column].dropna().unique()))
        time_to_idx = {timestamp: idx for idx, timestamp in enumerate(times)}
        time_idx = frame[self.config.date_column].map(time_to_idx).fillna(len(times) - 1).astype(int).to_numpy()

        feature_matrix = frame.loc[:, list(self.config.climate_covariates)].to_numpy(dtype=float)
        means, scales = self._compute_scaling(feature_matrix)
        feature_matrix_scaled = (feature_matrix - means) / scales
        self.covariate_means_ = {
            covariate: float(means[idx])
            for idx, covariate in enumerate(self.config.climate_covariates)
        }
        self.covariate_scales_ = {
            covariate: float(scales[idx])
            for idx, covariate in enumerate(self.config.climate_covariates)
        }
        observed = frame["__target__"].to_numpy(dtype=float)

        simplified_mode = bool(self.config.bayesian_simplified_mode)
        if self.config.force_full_bayesian:
            if simplified_mode:
                LOGGER.warning(
                    "force_full_bayesian=True overrides bayesian_simplified_mode=True; running full latent AR(1) model."
                )
            simplified_mode = False
        model, idata, z_t_values = self._fit_pymc_model(
            pm=pm,
            districts=districts,
            times=times,
            district_idx=district_idx,
            time_idx=time_idx,
            feature_matrix_scaled=feature_matrix_scaled,
            observed=observed,
            simplified_mode=simplified_mode,
        )

        diagnostics_summary = self._extract_sampler_diagnostics(idata)
        try:
            from src.models.bayesian.diagnostics import summarize_diagnostics

            convergence = summarize_diagnostics(idata)
            diagnostics_summary["r_hat_max"] = float(convergence.get("r_hat_max", float("nan")))
            diagnostics_summary["ess_min"] = float(convergence.get("ess_min", float("nan")))
        except Exception as diagnostics_error:
            LOGGER.warning("Unable to compute R-hat/ESS diagnostics from ArviZ: %s", diagnostics_error)

        diagnostics_summary["simplified_mode"] = 1.0 if simplified_mode else 0.0
        diagnostics_summary["fallback"] = 0.0

        should_retry_simple = (
            not simplified_mode
            and not self.config.force_full_bayesian
            and self.config.max_convergence_retries > 0
            and self._needs_simplification(diagnostics_summary)
        )
        if should_retry_simple:
            LOGGER.warning(
                "Bayesian diagnostics poor in full mode (divergences=%s, r_hat_max=%s, ess_min=%s). "
                "Retrying with bayesian_simplified_mode=True (tradeoff: latent AR(1) temporal state removed for stability).",
                diagnostics_summary.get("divergences"),
                diagnostics_summary.get("r_hat_max"),
                diagnostics_summary.get("ess_min"),
            )
            model, idata, z_t_values = self._fit_pymc_model(
                pm=pm,
                districts=districts,
                times=times,
                district_idx=district_idx,
                time_idx=time_idx,
                feature_matrix_scaled=feature_matrix_scaled,
                observed=observed,
                simplified_mode=True,
            )
            diagnostics_summary = self._extract_sampler_diagnostics(idata)
            try:
                from src.models.bayesian.diagnostics import summarize_diagnostics

                convergence = summarize_diagnostics(idata)
                diagnostics_summary["r_hat_max"] = float(convergence.get("r_hat_max", float("nan")))
                diagnostics_summary["ess_min"] = float(convergence.get("ess_min", float("nan")))
            except Exception:
                pass
            diagnostics_summary["simplified_mode"] = 1.0
            diagnostics_summary["fallback"] = 0.0
            simplified_mode = True

        self.model_ = model
        self._finalize_from_posterior(
            idata=idata,
            districts=districts,
            times=times,
            z_t_values=z_t_values,
            simplified_mode=simplified_mode,
            diagnostics_summary=diagnostics_summary,
        )

        if self._needs_simplification(diagnostics_summary):
            LOGGER.warning(
                "Bayesian fit completed with convergence concerns: divergences=%.0f, max_tree_depth=%.0f, r_hat_max=%.4f, ess_min=%.1f",
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
