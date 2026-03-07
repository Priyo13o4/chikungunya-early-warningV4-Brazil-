"""Private shared helpers for visualization modules."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from config.paths import get_paths

logger = logging.getLogger(__name__)


def _get_output_dir(output_dir: Path | None) -> Path:
    target = output_dir or get_paths().outputs_figures
    target.mkdir(parents=True, exist_ok=True)
    return target


def _save_figure(
    fig: plt.Figure,
    output_dir: Path | None,
    filename: str,
) -> Path:
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