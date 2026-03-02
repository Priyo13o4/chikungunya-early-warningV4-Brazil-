"""Diagnostics for Bayesian model fitting quality."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

_GROUP_PREFIXES: dict[str, tuple[str, ...]] = {
    "random_effects": ("mu_alpha", "sigma_alpha", "alpha_raw", "alpha_district"),
    "fixed_effects": ("beta",),
    "temporal_state": ("rho_raw", "rho", "sigma_z", "z_t", "sigma_z_state", "z_state"),
    "likelihood": ("alpha_nb",),
}
_GROUP_ORDER: tuple[str, ...] = (
    "random_effects",
    "fixed_effects",
    "temporal_state",
    "likelihood",
    "other",
)


def _try_import_arviz() -> Any | None:
    try:
        return importlib.import_module("arviz")
    except Exception:
        return None


def extract_rhat_ess(idata: Any) -> pd.DataFrame:
    """Extract per-parameter R-hat and ESS diagnostics from InferenceData."""
    az = _try_import_arviz()
    if az is None:
        raise ImportError("Diagnostics require optional dependency 'arviz'.")

    rhat_ds = az.rhat(idata)
    ess_ds = az.ess(idata, method="bulk")

    def _dataset_metric_to_parameter_map(dataset: Any) -> dict[str, float]:
        metric_values: dict[str, float] = {}
        for var_name, data_array in dataset.data_vars.items():
            if int(getattr(data_array, "ndim", 0)) == 0:
                scalar_value = np.asarray(data_array.to_numpy(), dtype=float).item()
                metric_values[str(var_name)] = float(scalar_value)
                continue

            series = data_array.to_series()
            for index_key, metric_value in series.items():
                if isinstance(index_key, tuple):
                    path_parts = (str(var_name), *(str(part) for part in index_key))
                else:
                    path_parts = (str(var_name), str(index_key))
                metric_values["|".join(path_parts)] = float(metric_value)
        return metric_values

    rhat_values = _dataset_metric_to_parameter_map(rhat_ds)
    ess_values = _dataset_metric_to_parameter_map(ess_ds)

    parameters = sorted(set(rhat_values.keys()) | set(ess_values.keys()))
    diagnostics = pd.DataFrame({"parameter": parameters})
    diagnostics["r_hat"] = diagnostics["parameter"].map(rhat_values)
    diagnostics["ess_bulk"] = diagnostics["parameter"].map(ess_values)
    return diagnostics.reset_index(drop=True)


def _resolve_parameter_group(parameter_name: str) -> str:
    base_name = str(parameter_name).split("|", 1)[0]
    for group, prefixes in _GROUP_PREFIXES.items():
        if any(base_name.startswith(prefix) for prefix in prefixes):
            return group
    return "other"


def summarize_diagnostics_by_group(
    idata: Any,
    *,
    rhat_threshold: float = 1.05,
    ess_threshold: float = 200.0,
) -> pd.DataFrame:
    """Aggregate R-hat and ESS diagnostics by Bayesian parameter group."""
    diagnostics = extract_rhat_ess(idata)
    columns = [
        "group",
        "n_parameters",
        "r_hat_max",
        "r_hat_p95",
        "ess_min",
        "ess_p05",
        "ess_median",
        "fail_rhat_count",
        "fail_ess_count",
        "rhat_threshold",
        "ess_threshold",
    ]
    if diagnostics.empty:
        empty_group_rows = [
            {
                "group": group_name,
                "n_parameters": 0,
                "r_hat_max": np.nan,
                "r_hat_p95": np.nan,
                "ess_min": np.nan,
                "ess_p05": np.nan,
                "ess_median": np.nan,
                "fail_rhat_count": 0,
                "fail_ess_count": 0,
                "rhat_threshold": float(rhat_threshold),
                "ess_threshold": float(ess_threshold),
            }
            for group_name in _GROUP_ORDER
        ]
        return pd.DataFrame(empty_group_rows, columns=columns)

    diagnostics = diagnostics.copy()
    diagnostics["group"] = diagnostics["parameter"].map(_resolve_parameter_group)

    grouped = []
    for group_name in _GROUP_ORDER:
        subset = diagnostics.loc[diagnostics["group"] == group_name]
        if subset.empty:
            grouped.append(
                {
                    "group": group_name,
                    "n_parameters": 0,
                    "r_hat_max": np.nan,
                    "r_hat_p95": np.nan,
                    "ess_min": np.nan,
                    "ess_p05": np.nan,
                    "ess_median": np.nan,
                    "fail_rhat_count": 0,
                    "fail_ess_count": 0,
                    "rhat_threshold": float(rhat_threshold),
                    "ess_threshold": float(ess_threshold),
                }
            )
            continue

        grouped.append(
            {
                "group": group_name,
                "n_parameters": int(len(subset)),
                "r_hat_max": float(subset["r_hat"].max()),
                "r_hat_p95": float(subset["r_hat"].quantile(0.95)),
                "ess_min": float(subset["ess_bulk"].min()),
                "ess_p05": float(subset["ess_bulk"].quantile(0.05)),
                "ess_median": float(subset["ess_bulk"].median()),
                "fail_rhat_count": int((subset["r_hat"] > float(rhat_threshold)).sum()),
                "fail_ess_count": int((subset["ess_bulk"] < float(ess_threshold)).sum()),
                "rhat_threshold": float(rhat_threshold),
                "ess_threshold": float(ess_threshold),
            }
        )

    return pd.DataFrame(grouped, columns=columns)


def summarize_diagnostics(idata: Any | None = None) -> dict[str, float]:
    """Return summary diagnostics including divergences, tree depth, R-hat and ESS."""
    if idata is None:
        return {
            "divergences": 0.0,
            "max_tree_depth": 0.0,
            "r_hat_max": 1.0,
            "ess_min": 1000.0,
            "n_parameters": 0.0,
        }

    divergences = 0.0
    max_tree_depth = 0.0
    sample_stats = getattr(idata, "sample_stats", None)
    if sample_stats is not None:
        if "diverging" in sample_stats:
            divergences = float(np.asarray(sample_stats["diverging"].to_numpy(), dtype=float).sum())
        if "tree_depth" in sample_stats:
            max_tree_depth = float(np.asarray(sample_stats["tree_depth"].to_numpy(), dtype=float).max())
        elif "depth" in sample_stats:
            max_tree_depth = float(np.asarray(sample_stats["depth"].to_numpy(), dtype=float).max())

    diagnostics = extract_rhat_ess(idata)
    if diagnostics.empty:
        return {
            "divergences": divergences,
            "max_tree_depth": max_tree_depth,
            "r_hat_max": 1.0,
            "ess_min": 0.0,
            "n_parameters": 0.0,
        }

    return {
        "divergences": divergences,
        "max_tree_depth": max_tree_depth,
        "r_hat_max": float(diagnostics["r_hat"].max()),
        "ess_min": float(diagnostics["ess_bulk"].min()),
        "n_parameters": float(len(diagnostics)),
    }


def check_convergence(
    idata: Any,
    *,
    divergence_threshold: float = 0.0,
    rhat_threshold: float = 1.01,
    ess_threshold: float = 400.0,
    max_tree_depth_threshold: float = 12.0,
) -> dict[str, Any]:
    """Evaluate convergence status from divergences, tree depth, R-hat and ESS thresholds."""
    summary = summarize_diagnostics(idata)
    converged = bool(
        summary["divergences"] <= divergence_threshold
        and summary["max_tree_depth"] <= max_tree_depth_threshold
        and summary["r_hat_max"] <= rhat_threshold
        and summary["ess_min"] >= ess_threshold
    )
    if not converged:
        LOGGER.warning(
            "Convergence check failed: divergences=%.0f, max_tree_depth=%.0f, r_hat_max=%.4f, ess_min=%.1f",
            summary["divergences"],
            summary["max_tree_depth"],
            summary["r_hat_max"],
            summary["ess_min"],
        )
    return {
        "converged": converged,
        "divergence_threshold": float(divergence_threshold),
        "max_tree_depth_threshold": float(max_tree_depth_threshold),
        "rhat_threshold": float(rhat_threshold),
        "ess_threshold": float(ess_threshold),
        **summary,
    }
