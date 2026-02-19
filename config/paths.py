"""Centralized project path definitions and directory bootstrap utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ProjectPaths:
    """Container for project paths."""

    root: Path
    data_raw: Path
    data_processed: Path
    data_features: Path
    data_external: Path
    outputs_figures: Path
    outputs_metrics: Path
    outputs_models: Path
    outputs_reports: Path
    config_dir: Path


def _infer_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_paths(root: Path | None = None) -> ProjectPaths:
    """Build a `ProjectPaths` object using a project root path."""
    root_path = root or _infer_project_root()
    data_dir = root_path / "data"
    outputs_dir = root_path / "outputs"
    return ProjectPaths(
        root=root_path,
        data_raw=data_dir / "raw",
        data_processed=data_dir / "processed",
        data_features=data_dir / "features",
        data_external=data_dir / "external",
        outputs_figures=outputs_dir / "figures",
        outputs_metrics=outputs_dir / "metrics",
        outputs_models=outputs_dir / "models",
        outputs_reports=outputs_dir / "reports",
        config_dir=root_path / "config",
    )


def _required_directories(paths: ProjectPaths) -> Iterable[Path]:
    return (
        paths.data_processed,
        paths.data_features,
        paths.data_external,
        paths.outputs_figures,
        paths.outputs_metrics,
        paths.outputs_models,
        paths.outputs_reports,
    )


def ensure_directories(root: Path | None = None) -> ProjectPaths:
    """Create required directories if they do not exist."""
    paths = get_paths(root=root)
    for directory in _required_directories(paths):
        directory.mkdir(parents=True, exist_ok=True)
    return paths
