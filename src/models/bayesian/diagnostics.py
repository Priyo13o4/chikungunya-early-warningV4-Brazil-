"""Diagnostics for Bayesian model fitting quality."""

from __future__ import annotations

import importlib
import logging
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


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

    rhat_series = rhat_ds.to_array().to_series()
    ess_series = ess_ds.to_array().to_series()
    diagnostics = pd.DataFrame({"r_hat": rhat_series, "ess_bulk": ess_series})
    diagnostics.index = diagnostics.index.map(lambda item: "|".join(str(part) for part in item))
    diagnostics = diagnostics.reset_index().rename(columns={"index": "parameter"})
    return diagnostics.sort_values("parameter").reset_index(drop=True)


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
        and summary["max_tree_depth"] < max_tree_depth_threshold
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
