"""Run visualization generation from persisted payload artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from config.paths import ensure_directories
from src.pipeline_runtime.phases_visualization import run_visualization_phase_from_payload

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone visualization phase from persisted payload artifacts")
    parser.add_argument(
        "--payload-csv",
        type=Path,
        default=Path("outputs/reports/visualization_payload.csv"),
        help="Path to visualization payload CSV",
    )
    parser.add_argument(
        "--payload-metadata",
        type=Path,
        default=Path("outputs/reports/visualization_payload_metadata.json"),
        help="Path to visualization payload metadata JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures"),
        help="Directory where figures are written",
    )
    parser.add_argument(
        "--lead-time-max-lookback-steps",
        type=int,
        default=8,
        help="Max lookback steps for lead-time figure",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for any deterministic sampling in plotting",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )

    paths = ensure_directories()
    payload_csv_path = args.payload_csv
    payload_metadata_path = args.payload_metadata
    output_dir = args.output_dir

    previous_bayesian_convergence = None
    previous_convergence_path = paths.outputs_models / "bayesian" / "diagnostics" / "convergence.json"
    if previous_convergence_path.exists():
        try:
            previous_bayesian_convergence = json.loads(previous_convergence_path.read_text(encoding="utf-8"))
        except Exception as read_error:
            LOGGER.warning("Unable to load previous Bayesian convergence diagnostics: %s", read_error)

    generated = run_visualization_phase_from_payload(
        payload_csv_path=payload_csv_path,
        payload_metadata_path=payload_metadata_path,
        output_dir=output_dir,
        lead_time_max_lookback_steps=args.lead_time_max_lookback_steps,
        previous_bayesian_convergence=previous_bayesian_convergence,
        effective_seed=args.seed,
    )

    LOGGER.info("Standalone visualization phase completed | generated=%d", len(generated))


if __name__ == "__main__":
    main()
