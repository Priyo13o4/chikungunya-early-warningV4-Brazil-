"""Utilities to compare baseline and Bayesian model tracks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd

from config.paths import get_paths

logger = logging.getLogger(__name__)


def _coerce_metric_dict(metrics: Mapping[str, float], label: str) -> Dict[str, float]:
    converted: Dict[str, float] = {}
    for key, value in metrics.items():
        try:
            converted[str(key)] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Metric '{key}' in {label} is not numeric: {value!r}") from exc
    return converted


def compare_tracks(baseline_metrics: Mapping[str, float], bayesian_metrics: Mapping[str, float]) -> Dict[str, float]:
    """Combine track metrics into a single prefixed dictionary."""
    baseline = _coerce_metric_dict(baseline_metrics, "baseline_metrics")
    bayesian = _coerce_metric_dict(bayesian_metrics, "bayesian_metrics")

    comparison: Dict[str, float] = {}
    for key, value in baseline.items():
        comparison[f"baseline_{key}"] = value
    for key, value in bayesian.items():
        comparison[f"bayesian_{key}"] = value
    return comparison


def build_comparison_table(
    baseline_metrics: Mapping[str, float],
    bayesian_metrics: Mapping[str, float],
) -> pd.DataFrame:
    """Build a metric comparison table with absolute and relative deltas."""
    baseline = _coerce_metric_dict(baseline_metrics, "baseline_metrics")
    bayesian = _coerce_metric_dict(bayesian_metrics, "bayesian_metrics")

    metric_names = sorted(set(baseline).union(bayesian))
    frame = pd.DataFrame(
        {
            "metric": metric_names,
            "baseline": [baseline.get(name, float("nan")) for name in metric_names],
            "bayesian": [bayesian.get(name, float("nan")) for name in metric_names],
        }
    )
    frame["delta"] = frame["bayesian"] - frame["baseline"]
    denominator = frame["baseline"].replace(0.0, pd.NA)
    frame["relative_delta_pct"] = (frame["delta"] / denominator) * 100.0
    return frame


def export_track_comparison(
    baseline_metrics: Mapping[str, float],
    bayesian_metrics: Mapping[str, float],
    output_dir: Path | None = None,
    filename_prefix: str = "track_comparison",
    include_wide_csv: bool = False,
) -> Dict[str, Path]:
    """Export baseline-vs-Bayesian comparison tables to `outputs/metrics`."""
    if not filename_prefix:
        raise ValueError("filename_prefix cannot be empty.")

    target_dir = output_dir or get_paths().outputs_metrics
    target_dir.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(baseline_metrics=baseline_metrics, bayesian_metrics=bayesian_metrics)

    csv_path = target_dir / f"{filename_prefix}.csv"
    md_path = target_dir / f"{filename_prefix}.md"

    table.to_csv(csv_path, index=False)
    table.to_markdown(md_path, index=False)
    outputs: Dict[str, Path] = {"long_csv": csv_path, "markdown": md_path}

    if include_wide_csv:
        combined = compare_tracks(baseline_metrics=baseline_metrics, bayesian_metrics=bayesian_metrics)
        combined_table = pd.DataFrame([combined])
        wide_csv_path = target_dir / f"{filename_prefix}_wide.csv"
        combined_table.to_csv(wide_csv_path, index=False)
        outputs["wide_csv"] = wide_csv_path

    logger.info("Exported track comparison files to %s", target_dir)
    return outputs
