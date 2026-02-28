from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

_CONTRACT_REPORT_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "run_metadata.json",
    "fold_ledger.json",
    "degraded_run.json",
    "feature_quality_gate_report.json",
    "model_input_leakage_audit.json",
)
_CONTRACT_METRIC_FILES: tuple[str, ...] = (
    "baseline_backend_metadata.json",
    "baseline_metrics.json",
    "baseline_metrics_fullfit.json",
    "bayesian_metrics.json",
    "bayesian_metrics_fullfit.json",
    "bayesian_risk_metadata.json",
    "track_comparison.csv",
    "track_comparison.md",
    "decision_alerts.csv",
    "bayesian_convergence_summary.csv",
    "bayesian_convergence_summary.md",
)


def safe_write_json(data: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_compatible(data)
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def clear_headline_artifacts(metrics_dir: Path) -> None:
    for filename in (
        "baseline_metrics.json",
        "bayesian_metrics.json",
        "track_comparison.csv",
        "track_comparison.md",
        "bayesian_convergence_summary.csv",
        "bayesian_convergence_summary.md",
    ):
        path = metrics_dir / filename
        if path.exists():
            path.unlink()


def clear_contract_artifacts(*, reports_dir: Path, metrics_dir: Path) -> None:
    for filename in _CONTRACT_REPORT_FILES:
        path = reports_dir / filename
        if path.exists():
            path.unlink()
    for filename in _CONTRACT_METRIC_FILES:
        path = metrics_dir / filename
        if path.exists():
            path.unlink()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit_sha(project_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit if commit else None


def json_compatible(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    return value


def build_suppressed_metric_payload(*, run_id: str, track: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "track": track,
        "suppressed": True,
        "reason": reason,
    }


def build_contract_track_comparison_placeholder(*, run_id: str, reason: str) -> tuple[pd.DataFrame, str]:
    frame = pd.DataFrame(
        [
            {
                "metric": "headline_comparison",
                "baseline": None,
                "bayesian": None,
                "delta": None,
                "status": "suppressed",
                "reason": reason,
                "run_id": run_id,
            }
        ]
    )
    markdown = (
        "# Track Comparison\n\n"
        "| metric | baseline | bayesian | delta | status | reason | run_id |\n"
        "|---|---:|---:|---:|---|---|---|\n"
        f"| headline_comparison |  |  |  | suppressed | {reason} | {run_id} |\n"
    )
    return frame, markdown


def write_bayesian_convergence_summary(
    *,
    metrics_dir: Path,
    run_id: str,
    diagnostics: dict[str, Any],
    convergence_artifact_path: Path | None = None,
    convergence: dict[str, Any] | None,
) -> tuple[Path, Path]:
    if convergence_artifact_path is not None and convergence_artifact_path.exists():
        convergence_payload = json.loads(convergence_artifact_path.read_text(encoding="utf-8"))
    else:
        convergence_payload = convergence or {}
    summary_row = {
        "run_id": run_id,
        "mode_used": str(diagnostics.get("mode_used", "not_run")),
        "degraded_mode": bool(diagnostics.get("degraded_mode", False)),
        "fallback_used": bool(diagnostics.get("fallback_used", False)),
        "converged": bool(convergence_payload.get("converged", False)),
        "divergences": float(convergence_payload.get("divergences", 0.0) or 0.0),
        "divergence_threshold": float(convergence_payload.get("divergence_threshold", float("nan"))),
        "max_tree_depth": float(convergence_payload.get("max_tree_depth", 0.0) or 0.0),
        "max_tree_depth_threshold": float(convergence_payload.get("max_tree_depth_threshold", float("nan"))),
        "r_hat_max": float(convergence_payload.get("r_hat_max", 0.0) or 0.0),
        "rhat_threshold": float(convergence_payload.get("rhat_threshold", float("nan"))),
        "ess_min": float(convergence_payload.get("ess_min", 0.0) or 0.0),
        "ess_threshold": float(convergence_payload.get("ess_threshold", float("nan"))),
    }
    summary_frame = pd.DataFrame([summary_row])
    csv_path = metrics_dir / "bayesian_convergence_summary.csv"
    md_path = metrics_dir / "bayesian_convergence_summary.md"
    summary_frame.to_csv(csv_path, index=False)
    md_path.write_text(
        "# Bayesian Convergence Summary\n\n" + summary_frame.to_markdown(index=False),
        encoding="utf-8",
    )
    return csv_path, md_path


def validate_manifest_contract(
    *,
    run_manifest_path: Path,
    artifacts: dict[str, Path],
    required_artifact_keys: set[str],
    expected_run_id: str,
) -> None:
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_artifacts = manifest.get("artifacts", {})

    missing_in_manifest = sorted(required_artifact_keys.difference(manifest_artifacts.keys()))
    missing_in_artifacts = sorted(required_artifact_keys.difference(artifacts.keys()))
    if missing_in_manifest or missing_in_artifacts:
        raise RuntimeError(
            "Contract artifact key mismatch "
            f"(manifest_missing={missing_in_manifest}, artifacts_missing={missing_in_artifacts})"
        )

    mismatched_paths: list[str] = []
    missing_files: list[str] = []
    for key in sorted(required_artifact_keys):
        artifact_path = artifacts[key]
        manifest_path = Path(str(manifest_artifacts[key]))
        if str(artifact_path) != str(manifest_path):
            mismatched_paths.append(key)
        if not artifact_path.exists():
            missing_files.append(str(artifact_path))
    if mismatched_paths or missing_files:
        raise RuntimeError(
            "Contract artifact parity failed "
            f"(path_mismatch={mismatched_paths}, missing_files={missing_files})"
        )

    metadata = json.loads(Path(str(manifest_artifacts["run_metadata"])).read_text(encoding="utf-8"))
    fold_ledger = json.loads(Path(str(manifest_artifacts["fold_ledger"])).read_text(encoding="utf-8"))
    degraded_run = json.loads(Path(str(manifest_artifacts["degraded_run"])).read_text(encoding="utf-8"))

    run_id_values = {
        "manifest": str(manifest.get("run_id")),
        "run_metadata": str(metadata.get("run_id")),
        "fold_ledger": str(fold_ledger.get("run_id")),
        "degraded_run": str(degraded_run.get("run_id")),
    }
    if any(value != expected_run_id for value in run_id_values.values()):
        raise RuntimeError(f"run_id parity mismatch: {run_id_values} expected={expected_run_id}")

    suppressed_manifest = bool(manifest.get("headline_claims", {}).get("suppressed", False))
    suppressed_degraded = bool(degraded_run.get("suppress_headline_comparison", False))
    if suppressed_manifest != suppressed_degraded:
        raise RuntimeError(
            "degraded_run parity mismatch "
            f"(manifest_suppressed={suppressed_manifest}, degraded_suppressed={suppressed_degraded})"
        )

    manifest_curated = manifest.get("curated_municipalities")
    metadata_curated = metadata.get("curated_municipalities")
    if not isinstance(manifest_curated, dict) or not isinstance(metadata_curated, dict):
        raise RuntimeError(
            "curated_municipalities block must exist as object in both run_manifest and run_metadata "
            f"(manifest_type={type(manifest_curated).__name__}, run_metadata_type={type(metadata_curated).__name__})"
        )

    parity_fields = ("path", "version", "sha256")
    mismatched_fields = [
        field
        for field in parity_fields
        if str(manifest_curated.get(field)) != str(metadata_curated.get(field))
    ]
    if mismatched_fields:
        raise RuntimeError(
            "curated_municipalities parity mismatch "
            f"(fields={mismatched_fields}, manifest={manifest_curated}, run_metadata={metadata_curated})"
        )
