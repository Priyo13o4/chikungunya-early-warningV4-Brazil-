"""Feature diagnostics and interpretability plots."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.visualization._common import _get_output_dir, _save_figure, _sparsify_labels

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    import arviz as az
except ImportError:
    az = None


def plot_correlation_heatmap(
    feature_frame: pd.DataFrame,
    filename: str = "features_correlation_heatmap.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot feature-feature correlation heatmap."""
    corr = feature_frame.corr(numeric_only=True)
    if corr.empty:
        raise ValueError("No numeric features available for correlation heatmap.")

    fig, ax = plt.subplots(figsize=(10, 8))
    if sns is not None:
        sns.heatmap(corr, cmap="coolwarm", center=0.0, ax=ax)
        ax.set_xticklabels(_sparsify_labels(corr.columns.astype(str).tolist(), max_labels=25), rotation=60, ha="right")
        ax.set_yticklabels(_sparsify_labels(corr.index.astype(str).tolist(), max_labels=25))
    else:
        im = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax)
        x_labels = _sparsify_labels(corr.columns.astype(str).tolist(), max_labels=25)
        y_labels = _sparsify_labels(corr.index.astype(str).tolist(), max_labels=25)
        x_positions = [idx for idx, label in enumerate(x_labels) if label]
        y_positions = [idx for idx, label in enumerate(y_labels) if label]
        ax.set_xticks(x_positions)
        ax.set_xticklabels([x_labels[idx] for idx in x_positions], rotation=60, ha="right")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([y_labels[idx] for idx in y_positions])
    ax.set_title("Feature Correlation Heatmap")
    return _save_figure(fig, output_dir, filename)


def plot_feature_importance(
    importances: Mapping[str, float] | pd.Series,
    top_k: int = 20,
    filename: str = "features_importance.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot top-K feature importances."""
    series = pd.Series(importances, dtype=float).sort_values(ascending=False)
    if series.empty:
        raise ValueError("Feature importances are empty.")
    top = series.head(top_k).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top.index.astype(str), top.values, color="tab:purple")
    ax.set_title(f"Top {len(top)} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_yticklabels(_sparsify_labels(top.index.astype(str).tolist(), max_labels=30))
    return _save_figure(fig, output_dir, filename)


def plot_shap_summary(
    shap_values: np.ndarray | Sequence[Sequence[float]],
    feature_frame: pd.DataFrame,
    filename: str = "features_shap_summary.png",
    output_dir: Path | None = None,
) -> Path | None:
    """Plot SHAP summary if `shap` is available, else log and skip."""
    try:
        import shap
    except ImportError:
        logger.warning("SHAP unavailable; SHAP summary plot skipped")
        return None

    target_dir = _get_output_dir(output_dir)
    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 1:
        shap_array = shap_array.reshape(-1, 1)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_array, feature_frame, show=False)
    path = target_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved SHAP summary figure to %s", path)
    return path


def plot_covariate_forest_hdi(
    inference_data: object,
    *,
    covariates: Sequence[str] = ("temperature", "rainfall", "humidity"),
    filename: str = "thesis_covariate_forest_hdi.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot posterior coefficient means with 95% HDI intervals for selected covariates."""
    posterior = getattr(inference_data, "posterior", None)
    if posterior is None or "beta" not in posterior:
        raise ValueError("InferenceData does not contain posterior beta coefficients.")

    beta = posterior["beta"]
    coord_name = None
    for candidate in ("covariate", "beta_dim_0"):
        if candidate in getattr(beta, "coords", {}):
            coord_name = candidate
            break
    if coord_name is None:
        raise ValueError("Unable to find beta covariate coordinate in posterior.")

    available_covariates = [str(value) for value in beta.coords[coord_name].to_numpy().tolist()]
    target_covariates = [cov for cov in covariates if cov in available_covariates]
    if not target_covariates:
        raise ValueError(f"None of requested covariates found in posterior beta: {covariates}")

    rows: list[dict[str, float | str]] = []
    for covariate in target_covariates:
        values = np.asarray(beta.sel({coord_name: covariate}).to_numpy(), dtype=float).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        if az is not None:
            hdi_bounds = az.hdi(values, hdi_prob=0.95)
            low = float(np.asarray(hdi_bounds)[0])
            high = float(np.asarray(hdi_bounds)[1])
        else:
            low = float(np.quantile(values, 0.025))
            high = float(np.quantile(values, 0.975))
        rows.append(
            {
                "covariate": str(covariate),
                "mean": float(np.mean(values)),
                "hdi_low": low,
                "hdi_high": high,
            }
        )

    if not rows:
        raise ValueError("No finite posterior samples available for selected covariates.")

    forest = pd.DataFrame(rows).sort_values("mean", ascending=True)
    y = np.arange(len(forest))
    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 1.25 * len(forest))))
    ax.hlines(y=y, xmin=forest["hdi_low"], xmax=forest["hdi_high"], color="tab:blue", alpha=0.85, linewidth=2.0)
    ax.scatter(forest["mean"], y, color="tab:orange", zorder=3, s=45)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(forest["covariate"].astype(str).tolist())
    ax.set_xlabel("Posterior coefficient")
    ax.set_title("Covariate Forest Plot (95% HDI)")
    return _save_figure(fig, output_dir, filename)


def plot_covariate_forest_hdi_from_summary(
    summary_records: Sequence[Mapping[str, float | str]],
    *,
    filename: str = "thesis_covariate_forest_hdi.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot covariate forest using pre-computed mean/HDI summary records."""
    frame = pd.DataFrame(summary_records)
    required = {"covariate", "mean", "hdi_low", "hdi_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"summary_records missing required fields: {sorted(required.difference(frame.columns))}")

    frame["mean"] = pd.to_numeric(frame["mean"], errors="coerce")
    frame["hdi_low"] = pd.to_numeric(frame["hdi_low"], errors="coerce")
    frame["hdi_high"] = pd.to_numeric(frame["hdi_high"], errors="coerce")
    frame = frame.dropna(subset=["covariate", "mean", "hdi_low", "hdi_high"])
    if frame.empty:
        raise ValueError("No valid covariate summary rows for forest plot.")

    forest = frame.sort_values("mean", ascending=True)
    y = np.arange(len(forest))
    fig, ax = plt.subplots(figsize=(8.0, max(3.8, 1.25 * len(forest))))
    ax.hlines(y=y, xmin=forest["hdi_low"], xmax=forest["hdi_high"], color="tab:blue", alpha=0.85, linewidth=2.0)
    ax.scatter(forest["mean"], y, color="tab:orange", zorder=3, s=45)
    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(forest["covariate"].astype(str).tolist())
    ax.set_xlabel("Posterior coefficient")
    ax.set_title("Covariate Forest Plot (95% HDI)")
    return _save_figure(fig, output_dir, filename)
