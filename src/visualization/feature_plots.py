"""Feature diagnostics and interpretability plots."""

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
    import seaborn as sns
except ImportError:
    sns = None


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


def _sparsify_labels(labels: Sequence[str], max_labels: int) -> list[str]:
    if max_labels <= 0:
        return [""] * len(labels)
    if len(labels) <= max_labels:
        return [str(label) for label in labels]
    step = int(np.ceil(len(labels) / max_labels))
    return [str(label) if idx % step == 0 else "" for idx, label in enumerate(labels)]


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
