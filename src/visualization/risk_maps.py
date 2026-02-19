"""Risk trajectory and mapping utilities for district-level alerts."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from config.paths import get_paths
from src.decision_layer.cost_loss import AlertThresholds, assign_alert_levels
from src.visualization.promising_districts import filter_to_promising_districts

logger = logging.getLogger(__name__)

try:
    import geopandas as gpd
except ImportError:
    gpd = None


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


def plot_risk_trajectory(
    frame: pd.DataFrame,
    date_col: str,
    risk_col: str,
    district_col: str,
    top_n: int = 10,
    filename: str = "risk_trajectory.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot risk trajectories over time for highest-risk districts."""
    data = frame[[date_col, risk_col, district_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, risk_col, district_col])
    if data.empty:
        raise ValueError("No valid rows available for risk trajectory plot.")
    subset = filter_to_promising_districts(
        data,
        district_col=district_col,
        top_k=top_n,
        risk_candidates=(risk_col, "risk_score", "bayesian_risk", "risk", "probability"),
        fallback_case_candidates=("cases", "case_count"),
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    for district, group in subset.groupby(district_col):
        series = group.sort_values(date_col)
        ax.plot(series[date_col], series[risk_col], label=str(district), linewidth=1.8)
    ax.set_title("Risk Trajectory (Top Districts)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    return _save_figure(fig, output_dir, filename)


def plot_top_risk_districts(
    frame: pd.DataFrame,
    district_col: str,
    risk_col: str,
    top_n: int = 20,
    filename: str = "risk_top_districts.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot top-N districts by latest available risk."""
    latest = frame[[district_col, risk_col]].dropna().copy()
    latest = filter_to_promising_districts(
        latest,
        district_col=district_col,
        top_k=top_n,
        risk_candidates=(risk_col, "risk_score", "bayesian_risk", "risk", "probability"),
    )
    summary = latest.groupby(district_col)[risk_col].mean().sort_values(ascending=False).head(top_n).sort_values()
    if summary.empty:
        raise ValueError("No valid rows available for top risk districts plot.")

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(summary.index.astype(str), summary.values, color="tab:red")
    ax.set_title(f"Top {len(summary)} Risk Districts")
    ax.set_xlabel("Average Risk")
    ax.set_ylabel("District")
    return _save_figure(fig, output_dir, filename)


def plot_decision_alert_trend(
    decision_frame: pd.DataFrame,
    date_col: str = "date",
    alert_col: str = "alert_level",
    filename: str = "decision_alert_levels_over_time.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot stacked alert-level counts over time from decision alerts."""
    required_cols = {date_col, alert_col}
    if not required_cols.issubset(decision_frame.columns):
        raise ValueError(f"decision_frame must include columns: {date_col}, {alert_col}.")

    data = decision_frame.loc[:, [date_col, alert_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.dropna(subset=[date_col, alert_col])

    alert_order = ["NO_ACTION", "YELLOW", "ORANGE", "RED"]
    data[alert_col] = data[alert_col].astype(str).str.strip().str.upper()
    data = data[data[alert_col].isin(alert_order)]
    if data.empty:
        raise ValueError("No valid rows available for decision alert trend plot.")

    data[date_col] = data[date_col].dt.normalize()
    counts = (
        data.groupby([date_col, alert_col]).size().unstack(fill_value=0).reindex(columns=alert_order, fill_value=0).sort_index()
    )
    if counts.empty:
        raise ValueError("No alert counts available for decision alert trend plot.")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.stackplot(
        counts.index,
        [counts[level].to_numpy() for level in alert_order],
        labels=alert_order,
        alpha=0.85,
    )
    ax.set_title("Decision Alert Levels Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("District Count")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    return _save_figure(fig, output_dir, filename)


def plot_alerts_map(
    risk_frame: pd.DataFrame,
    district_col: str,
    risk_col: str,
    shapefile_path: Path | None,
    shapefile_district_col: str,
    thresholds: AlertThresholds = AlertThresholds(),
    filename: str = "risk_alerts_map.png",
    output_dir: Path | None = None,
) -> Path | None:
    """Plot district choropleth map for risk and staged alerts.

    If geopandas or shapefile is unavailable, a warning is logged and the
    function returns `None`.
    """
    if gpd is None:
        logger.warning("geopandas unavailable; choropleth alerts map skipped")
        return None
    if shapefile_path is None or not shapefile_path.exists():
        logger.warning("Shapefile missing; choropleth alerts map skipped")
        return None

    map_frame = gpd.read_file(shapefile_path)
    latest_risk = risk_frame[[district_col, risk_col]].dropna().groupby(district_col, as_index=False).mean()
    latest_risk["alert_level"] = assign_alert_levels(latest_risk[risk_col], thresholds=thresholds)

    merged = map_frame.merge(
        latest_risk,
        left_on=shapefile_district_col,
        right_on=district_col,
        how="left",
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    merged.plot(column=risk_col, ax=axes[0], legend=True, cmap="YlOrRd", missing_kwds={"color": "lightgrey"})
    axes[0].set_title("District Risk Choropleth")
    axes[0].axis("off")

    alert_order = ["NO_ACTION", "YELLOW", "ORANGE", "RED"]
    merged["alert_level"] = pd.Categorical(merged["alert_level"], categories=alert_order, ordered=True)
    merged.plot(column="alert_level", ax=axes[1], legend=True, cmap="RdYlGn_r", missing_kwds={"color": "lightgrey"})
    axes[1].set_title("District Alert Levels")
    axes[1].axis("off")

    return _save_figure(fig, output_dir, filename)


def save_risk_map_stub(output_dir: Path) -> Path:
    """Preserved compatibility helper; no placeholder file is written."""
    target_dir = _get_output_dir(output_dir)
    logger.warning("Risk map stub called; no placeholder .txt is written")
    return target_dir
