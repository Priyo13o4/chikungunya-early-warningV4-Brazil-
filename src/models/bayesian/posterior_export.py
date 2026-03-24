"""Posterior export utilities."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _try_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


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


def export_inferencedata_artifacts(
    idata: Any,
    *,
    output_dir: Path,
    prefix: str = "fullfit",
    export_idata_netcdf: bool = True,
    export_summary_csv: bool = True,
) -> dict[str, Path]:
    """Persist InferenceData and summary artifacts for downstream plotting."""
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, Path] = {}
    errors: list[str] = []
    manifest: dict[str, Any] = {
        "prefix": prefix,
        "export_idata_netcdf": bool(export_idata_netcdf),
        "export_summary_csv": bool(export_summary_csv),
        "artifacts": {},
        "errors": errors,
    }

    if idata is None:
        errors.append("idata_is_none")
    else:
        if export_idata_netcdf:
            idata_path = output_dir / f"{prefix}_idata.nc"
            try:
                if hasattr(idata, "to_netcdf"):
                    idata.to_netcdf(str(idata_path))
                    artifacts["bayesian_posterior_idata"] = idata_path
                    manifest["artifacts"]["bayesian_posterior_idata"] = str(idata_path)
                else:
                    errors.append("idata_missing_to_netcdf")
            except Exception as export_error:
                errors.append(f"idata_netcdf_export_failed:{export_error}")

        if export_summary_csv:
            az = _try_import("arviz")
            if az is None:
                errors.append("arviz_not_available_for_summary")
            else:
                summary_path = output_dir / f"{prefix}_posterior_summary.csv"
                try:
                    summary = az.summary(idata, round_to=None)
                    summary_frame = pd.DataFrame(summary).reset_index().rename(columns={"index": "parameter"})
                    summary_frame.to_csv(summary_path, index=False)
                    artifacts["bayesian_posterior_summary"] = summary_path
                    manifest["artifacts"]["bayesian_posterior_summary"] = str(summary_path)
                except Exception as summary_error:
                    errors.append(f"posterior_summary_export_failed:{summary_error}")

    manifest_path = output_dir / f"{prefix}_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    artifacts["bayesian_posterior_manifest"] = manifest_path
    return artifacts
