"""Exploratory visualizations for epidemiology and climate covariates."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from config.paths import get_paths
from src.visualization.promising_districts import filter_to_promising_districts

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


def _sparsify_labels(labels: list[str], max_labels: int) -> list[str]:
    if max_labels <= 0:
        return [""] * len(labels)
    if len(labels) <= max_labels:
        return labels
    step = int(np.ceil(len(labels) / max_labels))
    return [label if idx % step == 0 else "" for idx, label in enumerate(labels)]


def plot_temporal_coverage_heatmap(
    frame: pd.DataFrame,
    date_col: str,
    district_col: str,
    top_k_districts: int = 20,
    filename: str = "exploratory_temporal_coverage_heatmap.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot district-by-month temporal coverage heatmap."""
    data = filter_to_promising_districts(frame.copy(), district_col=district_col, top_k=top_k_districts)
    data = data[[date_col, district_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, district_col])
    if data.empty:
        raise ValueError("No valid rows available for temporal coverage heatmap.")

    data["period"] = data[date_col].dt.to_period("M").astype(str)
    matrix = pd.crosstab(data[district_col], data["period"]).sort_index(axis=0).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    if sns is not None:
        sns.heatmap(matrix, cmap="viridis", ax=ax)
        ax.set_xticklabels(_sparsify_labels(matrix.columns.astype(str).tolist(), max_labels=24), rotation=45, ha="right")
        ax.set_yticklabels(_sparsify_labels(matrix.index.astype(str).tolist(), max_labels=24))
    else:
        im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax)
        x_labels = _sparsify_labels(matrix.columns.astype(str).tolist(), max_labels=24)
        y_labels = _sparsify_labels(matrix.index.astype(str).tolist(), max_labels=24)
        x_positions = [idx for idx, label in enumerate(x_labels) if label]
        y_positions = [idx for idx, label in enumerate(y_labels) if label]
        ax.set_xticks(x_positions)
        ax.set_xticklabels([x_labels[idx] for idx in x_positions], rotation=45, ha="right")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([y_labels[idx] for idx in y_positions])
    ax.set_title("Temporal Data Coverage by District and Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("District")
    return _save_figure(fig, output_dir, filename)


def plot_case_distribution(
    frame: pd.DataFrame,
    case_col: str,
    filename: str = "exploratory_case_distribution.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot the distribution of case counts."""
    values = frame[case_col].dropna().astype(float)
    if values.empty:
        raise ValueError("No non-null case values available for case distribution plot.")

    fig, ax = plt.subplots(figsize=(9, 5))
    if sns is not None:
        sns.histplot(values, bins=30, kde=True, ax=ax)
    else:
        ax.hist(values, bins=30, alpha=0.8)
    ax.set_title("Case Count Distribution")
    ax.set_xlabel(case_col)
    ax.set_ylabel("Frequency")
    return _save_figure(fig, output_dir, filename)


def plot_climate_kde_by_season(
    frame: pd.DataFrame,
    climate_col: str,
    season_col: str,
    filename: str = "exploratory_climate_kde_by_season.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot climate variable distribution stratified by season."""
    data = frame[[climate_col, season_col]].dropna().copy()
    if data.empty:
        raise ValueError("No valid rows available for climate KDE plot.")

    fig, ax = plt.subplots(figsize=(10, 5))
    if sns is not None:
        for season, subset in data.groupby(season_col):
            sns.kdeplot(subset[climate_col].astype(float), ax=ax, label=str(season), fill=False)
        ax.legend(title=season_col)
    else:
        for season, subset in data.groupby(season_col):
            ax.hist(subset[climate_col].astype(float), bins=25, density=True, alpha=0.3, label=str(season))
        ax.legend(title=season_col)
    ax.set_title(f"{climate_col} Distribution by Season")
    ax.set_xlabel(climate_col)
    ax.set_ylabel("Density")
    return _save_figure(fig, output_dir, filename)


def plot_missingness_summary(
    frame: pd.DataFrame,
    filename: str = "exploratory_missingness_summary.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot variable-wise missing percentage summary."""
    missing_pct = frame.isna().mean().sort_values(ascending=False) * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    missing_pct.plot(kind="bar", ax=ax, color="tab:blue")
    ax.set_title("Missingness Summary by Feature")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Missing (%)")
    labels = [str(label) for label in missing_pct.index.tolist()]
    ax.set_xticklabels(_sparsify_labels(labels, max_labels=30), rotation=70, ha="right")
    return _save_figure(fig, output_dir, filename)
