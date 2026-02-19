"""Posterior export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_posterior(summary: pd.DataFrame, output_path: Path) -> None:
    """Write posterior summary table to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


def export_posterior_artifacts(
    summary: pd.DataFrame,
    diagnostics: dict[str, Any],
    *,
    output_dir: Path,
    prefix: str = "bayesian",
) -> dict[str, Path]:
    """Export posterior summary and diagnostics as CSV/JSON artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / f"{prefix}_posterior_summary.csv"
    diagnostics_json_path = output_dir / f"{prefix}_diagnostics.json"
    diagnostics_csv_path = output_dir / f"{prefix}_diagnostics.csv"

    summary.to_csv(summary_path, index=False)
    with diagnostics_json_path.open("w", encoding="utf-8") as file_obj:
        json.dump(diagnostics, file_obj, indent=2, default=str)

    diagnostics_frame = pd.DataFrame([diagnostics])
    diagnostics_frame.to_csv(diagnostics_csv_path, index=False)

    return {
        "summary_csv": summary_path,
        "diagnostics_json": diagnostics_json_path,
        "diagnostics_csv": diagnostics_csv_path,
    }
