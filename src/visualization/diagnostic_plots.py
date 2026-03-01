"""Model diagnostics plotting utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config.paths import get_paths

logger = logging.getLogger(__name__)

try:
    import arviz as az
except ImportError:
    az = None


def _get_output_dir(output_dir: Path | None) -> Path:
    target = output_dir or get_paths().outputs_figures
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_figure(fig: plt.Figure, output_dir: Path | None, filename: str) -> Path:
    target_dir = _get_output_dir(output_dir)
    path = target_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", path)
    return path


def plot_trace(
    inference_data: object,
    var_names: Sequence[str] | None = None,
    filename: str = "diagnostic_trace_plot.png",
    output_dir: Path | None = None,
) -> Path | None:
    """Plot MCMC trace using ArviZ if available."""
    if az is None:
        logger.warning("ArviZ unavailable; trace plot skipped")
        return None

    target_dir = _get_output_dir(output_dir)
    axes = az.plot_trace(inference_data, var_names=var_names)
    fig = axes.ravel()[0].figure
    path = target_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved trace plot to %s", path)
    return path


def plot_posterior_predictive_check(
    y_true: Sequence[float] | pd.Series,
    posterior_predictive_samples: np.ndarray,
    filename: str = "diagnostic_posterior_predictive_check.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot posterior predictive distribution against observed outcomes."""
    observed = pd.Series(y_true, copy=False).astype(float).dropna()
    samples = np.asarray(posterior_predictive_samples, dtype=float)
    if samples.ndim == 1:
        samples = samples.reshape(1, -1)
    predictive_mean = np.nanmean(samples, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(predictive_mean, bins=30, alpha=0.55, label="Posterior Predictive Mean")
    ax.hist(observed.to_numpy(), bins=30, alpha=0.55, label="Observed")
    ax.set_title("Posterior Predictive Check")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Frequency")
    ax.legend()
    return _save_figure(fig, output_dir, filename)


def plot_residuals(
    y_true: Sequence[float] | pd.Series,
    y_pred: Sequence[float] | pd.Series,
    filename: str = "diagnostic_residuals.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot residual diagnostics for point predictions."""
    observed = pd.Series(y_true, copy=False).astype(float).reset_index(drop=True)
    predicted = pd.Series(y_pred, copy=False).astype(float).reset_index(drop=True)
    if len(observed) != len(predicted):
        raise ValueError("y_true and y_pred must have equal length.")

    residuals = observed - predicted
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].scatter(predicted, residuals, alpha=0.7)
    axes[0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[0].set_title("Residuals vs Predictions")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=30, alpha=0.8)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    return _save_figure(fig, output_dir, filename)


def plot_convergence_comparison(
    current_diagnostics: Mapping[str, float | int | bool],
    previous_diagnostics: Mapping[str, float | int | bool] | None = None,
    filename: str = "bayesian_convergence_comparison.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot convergence diagnostics for current run vs optional previous run."""
    metric_keys = ["divergences", "r_hat_max", "ess_min", "max_tree_depth"]
    metric_labels = {
        "divergences": "Divergences",
        "r_hat_max": "R-hat (max)",
        "ess_min": "ESS bulk (min)",
        "max_tree_depth": "Max tree depth",
    }

    def _value(diag: Mapping[str, float | int | bool], key: str) -> float:
        raw = diag.get(key, float("nan"))
        try:
            return float(raw)
        except Exception:
            return float("nan")

    current_values = np.asarray([_value(current_diagnostics, key) for key in metric_keys], dtype=float)

    if previous_diagnostics is None:
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        x = np.arange(len(metric_keys))
        bars = ax.bar(x, current_values, color="tab:blue", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[key] for key in metric_keys], rotation=15, ha="right")
        ax.set_title("Bayesian Convergence Diagnostics (Current Run)")
        ax.set_ylabel("Value")
        for bar, value in zip(bars, current_values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom")
        logger.info("No previous convergence diagnostics found; saving single-run summary figure.")
        return _save_figure(fig, output_dir, filename)

    previous_values = np.asarray([_value(previous_diagnostics, key) for key in metric_keys], dtype=float)
    x = np.arange(len(metric_keys))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    ax.bar(x - width / 2, previous_values, width=width, label="Previous", color="tab:gray", alpha=0.85)
    ax.bar(x + width / 2, current_values, width=width, label="Current", color="tab:blue", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[key] for key in metric_keys], rotation=15, ha="right")
    ax.set_title("Bayesian Convergence Comparison")
    ax.set_ylabel("Value")
    ax.legend(loc="best")
    return _save_figure(fig, output_dir, filename)
