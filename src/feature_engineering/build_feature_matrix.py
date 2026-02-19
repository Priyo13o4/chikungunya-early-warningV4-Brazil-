"""Feature matrix assembly pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from config.paths import ensure_directories
from src.feature_engineering.mechanistic_features import build_mechanistic_features
from src.feature_engineering.spatial_features import build_spatial_features
from src.feature_engineering.temporal_features import build_temporal_features
from src.feature_engineering.validate_features import gate_mechanistic_features, validate_feature_matrix

LOGGER = logging.getLogger(__name__)

DEFAULT_REQUIRED_FEATURES: tuple[str, ...] = (
    "month",
    "case_rolling_mean",
    "case_rolling_std",
    "case_velocity",
    "case_acceleration",
)


def build_feature_matrix(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    district_column: str = "district",
    validate: bool = True,
    strict_validation: bool = False,
    enable_quality_gate: bool = True,
    quality_gate_variance_threshold: float = 1e-10,
    quality_gate_uniqueness_threshold: float = 0.01,
    quality_gate_min_non_missing_samples: int = 24,
    quality_gate_report_path: Path | None = None,
    required_features: Sequence[str] = DEFAULT_REQUIRED_FEATURES,
    write_output: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run feature engineering stages and optionally persist output.

    The default output path is ``data/features/feature_matrix.csv``.
    """
    LOGGER.info("Building feature matrix")
    output = build_temporal_features(
        df,
        date_column=date_column,
        district_column=district_column,
    )
    output = build_spatial_features(
        output,
        date_column=date_column,
        district_column=district_column,
    )
    output = build_mechanistic_features(
        output,
        date_column=date_column,
        district_column=district_column,
    )

    if enable_quality_gate:
        output, quality_gate_report = gate_mechanistic_features(
            output,
            variance_threshold=quality_gate_variance_threshold,
            uniqueness_threshold=quality_gate_uniqueness_threshold,
            min_non_missing_samples=quality_gate_min_non_missing_samples,
            strict=strict_validation,
        )
        LOGGER.info(
            "Mechanistic quality gate passed=%s dropped=%s",
            quality_gate_report.get("passed"),
            quality_gate_report.get("dropped_features", []),
        )

        paths = ensure_directories()
        report_path = quality_gate_report_path or (paths.outputs_reports / "feature_quality_gate_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(quality_gate_report, indent=2), encoding="utf-8")
        LOGGER.info("Feature quality gate report written to %s", report_path)

    if validate:
        report = validate_feature_matrix(
            output,
            required_columns=required_features,
            strict=strict_validation,
        )
        LOGGER.info("Feature validation passed=%s", report["passed"])

    if write_output:
        final_output_path = output_path
        if final_output_path is None:
            paths = ensure_directories()
            final_output_path = paths.data_features / "feature_matrix.csv"
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(final_output_path, index=False)
        LOGGER.info("Feature matrix written to %s", final_output_path)

    return output


def run(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    district_column: str = "district",
    validate: bool = True,
    strict_validation: bool = False,
    enable_quality_gate: bool = True,
    quality_gate_variance_threshold: float = 1e-10,
    quality_gate_uniqueness_threshold: float = 0.01,
    quality_gate_min_non_missing_samples: int = 24,
    quality_gate_report_path: Path | None = None,
    required_features: Sequence[str] = DEFAULT_REQUIRED_FEATURES,
    write_output: bool = True,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Entrypoint for feature matrix assembly."""
    return build_feature_matrix(
        df,
        date_column=date_column,
        district_column=district_column,
        validate=validate,
        strict_validation=strict_validation,
        enable_quality_gate=enable_quality_gate,
        quality_gate_variance_threshold=quality_gate_variance_threshold,
        quality_gate_uniqueness_threshold=quality_gate_uniqueness_threshold,
        quality_gate_min_non_missing_samples=quality_gate_min_non_missing_samples,
        quality_gate_report_path=quality_gate_report_path,
        required_features=required_features,
        write_output=write_output,
        output_path=output_path,
    )
