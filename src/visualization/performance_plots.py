"""Performance visualization utilities for classification and warning quality."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

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


def plot_roc_curve(
    y_true: Sequence[int] | pd.Series,
    y_score: Sequence[float] | pd.Series,
    filename: str = "performance_roc_curve.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot ROC curve with AUC annotation."""
    y = pd.Series(y_true, copy=False).astype(int)
    score = pd.Series(y_score, copy=False).astype(float)
    fpr, tpr, _ = roc_curve(y, score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    return _save_figure(fig, output_dir, filename)


def plot_pr_curve(
    y_true: Sequence[int] | pd.Series,
    y_score: Sequence[float] | pd.Series,
    filename: str = "performance_pr_curve.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot precision-recall curve with AUC-PR annotation."""
    y = pd.Series(y_true, copy=False).astype(int)
    score = pd.Series(y_score, copy=False).astype(float)
    precision, recall, _ = precision_recall_curve(y, score)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")
    return _save_figure(fig, output_dir, filename)


def plot_calibration_curve(
    y_true: Sequence[int] | pd.Series,
    y_score: Sequence[float] | pd.Series,
    n_bins: int = 10,
    filename: str = "performance_calibration_curve.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot calibration reliability curve."""
    y = pd.Series(y_true, copy=False).astype(int)
    score = pd.Series(y_score, copy=False).astype(float)
    prob_true, prob_pred = calibration_curve(y, score, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Frequency")
    ax.set_title("Calibration Curve")
    ax.legend(loc="upper left")
    return _save_figure(fig, output_dir, filename)


def plot_lead_time_boxplot(
    lead_time_data: Mapping[str, Sequence[float]] | pd.DataFrame,
    filename: str = "performance_lead_time_boxplot.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot lead-time distribution by model/track."""
    if isinstance(lead_time_data, pd.DataFrame):
        data = lead_time_data.copy()
        if set(data.columns) >= {"track", "lead_time"}:
            grouped = [group["lead_time"].dropna().to_numpy() for _, group in data.groupby("track")]
            labels = [str(name) for name in data.groupby("track").groups]
        else:
            grouped = [data[col].dropna().to_numpy() for col in data.columns]
            labels = [str(col) for col in data.columns]
    else:
        labels = list(lead_time_data.keys())
        grouped = [pd.Series(values, copy=False).dropna().astype(float).to_numpy() for values in lead_time_data.values()]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(grouped, tick_labels=labels, vert=True)
    ax.set_title("Lead-Time Distribution")
    ax.set_xlabel("Track")
    ax.set_ylabel("Lead Time (steps)")
    if labels:
        capped_labels = _sparsify_labels([str(label) for label in labels], max_labels=20)
        ax.set_xticklabels(capped_labels, rotation=45 if len(labels) > 10 else 0, ha="right" if len(labels) > 10 else "center")
    return _save_figure(fig, output_dir, filename)


def plot_confusion_matrix_grid(
    y_true: Sequence[int] | pd.Series,
    predictions: Mapping[str, Sequence[int] | pd.Series],
    filename: str = "performance_confusion_matrix_grid.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot confusion matrix grid for multiple tracks/models."""
    y = pd.Series(y_true, copy=False).astype(int)
    if not predictions:
        raise ValueError("predictions mapping cannot be empty.")

    names = list(predictions.keys())
    n_models = len(names)
    n_cols = min(3, n_models)
    n_rows = int(np.ceil(n_models / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for idx, name in enumerate(names):
        pred = pd.Series(predictions[name], copy=False).astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax = axes_array[idx]
        im = ax.imshow(cm, cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(str(name))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center")

    for idx in range(n_models, len(axes_array)):
        axes_array[idx].axis("off")

    fig.suptitle("Confusion Matrix Grid")
    return _save_figure(fig, output_dir, filename)


def plot_track_delta_heatmap(
    comparison_table: pd.DataFrame,
    filename: str = "performance_track_delta_heatmap.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot a compact heatmap of Bayesian-minus-baseline metric deltas."""
    required_cols = {"metric", "delta"}
    if not required_cols.issubset(comparison_table.columns):
        raise ValueError("comparison_table must include metric and delta columns.")

    data = comparison_table.loc[:, ["metric", "delta"]].copy()
    data["delta"] = pd.to_numeric(data["delta"], errors="coerce")
    data = data.dropna(subset=["delta"])
    if data.empty:
        raise ValueError("comparison_table has no numeric delta rows.")

    matrix = data.set_index("metric")["delta"].to_frame(name="Bayesian - Baseline")
    fig, ax = plt.subplots(figsize=(6.5, max(3.2, len(matrix) * 0.45)))
    annotate_cells = len(matrix.index) <= 40
    if sns is not None:
        sns.heatmap(matrix, cmap="coolwarm", center=0.0, annot=annotate_cells, fmt=".3f", cbar=True, ax=ax)
        ax.set_yticklabels(_sparsify_labels(matrix.index.astype(str).tolist(), max_labels=25))
    else:
        values = matrix.to_numpy()
        im = ax.imshow(values, cmap="coolwarm", aspect="auto")
        fig.colorbar(im, ax=ax)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(_sparsify_labels(matrix.index.astype(str).tolist(), max_labels=25))
        ax.set_xticks([0])
        ax.set_xticklabels(matrix.columns.astype(str))
        if annotate_cells:
            for ridx in range(values.shape[0]):
                ax.text(0, ridx, f"{values[ridx, 0]:.3f}", ha="center", va="center")
    ax.set_title("Metric Delta Heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("Metric")
    return _save_figure(fig, output_dir, filename)


def plot_track_comparison_shared_metrics_bar(
    comparison_table: pd.DataFrame,
    filename: str = "track_comparison_shared_metrics_bar.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot only metrics available for both Track A and Track B (no empty cells)."""
    required_cols = {"metric", "baseline", "bayesian"}
    if not required_cols.issubset(comparison_table.columns):
        raise ValueError("comparison_table must include metric, baseline, bayesian columns.")

    data = comparison_table.loc[:, ["metric", "baseline", "bayesian"]].copy()
    data["baseline"] = pd.to_numeric(data["baseline"], errors="coerce")
    data["bayesian"] = pd.to_numeric(data["bayesian"], errors="coerce")
    data = data.dropna(subset=["baseline", "bayesian"])
    if data.empty:
        raise ValueError("No shared metrics with numeric values for both tracks.")

    metric_names = data["metric"].astype(str).to_numpy()
    x = np.arange(len(metric_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8.2, len(metric_names) * 0.95), 5.2))
    ax.bar(x - width / 2, data["baseline"].to_numpy(), width=width, label="Track A", color="tab:blue", alpha=0.88)
    ax.bar(x + width / 2, data["bayesian"].to_numpy(), width=width, label="Track B", color="tab:orange", alpha=0.88)
    ax.set_xticks(x)
    ax.set_xticklabels(_sparsify_labels(metric_names.tolist(), max_labels=24), rotation=45, ha="right")
    ax.set_ylabel("Metric Value")
    ax.set_title("Track Comparison (Shared Metrics)")
    ax.legend(loc="best")
    return _save_figure(fig, output_dir, filename)


def plot_brier_lead_time_summary(
    *,
    brier_score: float,
    lead_time_mean: float,
    filename: str = "trackb_brier_leadtime_summary.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot compact Track B summary for Brier score and mean lead time."""
    metrics = ["Brier Score", "Lead Time Mean"]
    values = [float(brier_score), float(lead_time_mean)]
    colors = ["tab:orange", "tab:green"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars = ax.bar(metrics, values, color=colors, alpha=0.9)
    ax.set_title("Track B Summary: Brier and Lead Time")
    ax.set_ylabel("Value")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}", ha="center", va="bottom")
    return _save_figure(fig, output_dir, filename)


def plot_tracka_model_score_comparison(
    model_scores: pd.DataFrame,
    filename: str = "tracka_models_all_scores.png",
    output_dir: Path | None = None,
) -> Path:
    """Plot Track A per-model metric comparison as bar-chart panels."""
    if "model" not in model_scores.columns:
        raise ValueError("model_scores must include a 'model' column.")

    candidate_metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "kappa",
        "false_alarm_rate",
        "lead_time_mean",
        "lead_time_utility",
    ]
    present_metrics = [metric for metric in candidate_metrics if metric in model_scores.columns]
    if not present_metrics:
        raise ValueError("model_scores does not contain any plottable metric columns.")

    matrix = model_scores.loc[:, ["model", *present_metrics]].copy()
    for metric in present_metrics:
        matrix[metric] = pd.to_numeric(matrix[metric], errors="coerce")
    matrix = matrix.dropna(subset=["model"]).set_index("model")
    if matrix.empty:
        raise ValueError("model_scores has no rows to plot.")

    n_metrics = len(present_metrics)
    n_cols = min(3, n_metrics)
    n_rows = int(np.ceil(n_metrics / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max(11, n_cols * 5.2), max(4.2, n_rows * 3.9)),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for idx, metric in enumerate(present_metrics):
        ax = axes_flat[idx]
        metric_series = matrix[metric].astype(float)
        model_names = metric_series.index.astype(str)
        x = np.arange(len(model_names))

        ax.bar(x, metric_series.to_numpy(), color="tab:blue", alpha=0.88)
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(_sparsify_labels(model_names.tolist(), max_labels=24), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Value")
        if len(model_names) <= 30:
            for pos, value in enumerate(metric_series.to_numpy()):
                label = "nan" if np.isnan(value) else f"{value:.3f}"
                ax.text(pos, value if np.isfinite(value) else 0.0, label, ha="center", va="bottom", fontsize=7)

    for idx in range(n_metrics, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Track A Model Comparison (All Scores)")
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.96))
    return _save_figure(fig, output_dir, filename)
