from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_JSONL_DIR = RAW_DIR / "infodengue_raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROBES_RAW_DIR = PROJECT_ROOT / "data" / "external" / "probes" / "raw"

UF_PARQUET_RE = re.compile(r"^infodengue_chik_[A-Z]{2}_\d{4}_\d{4}\.parquet$")
MERGED_PARQUET_RE = re.compile(r"^infodengue_chik_merged_\d{4}_\d{4}\.parquet$")
PANEL_PARQUET_RE = re.compile(r"^infodengue_chik_panel_\d{4}_\d{4}\.parquet$")


def _remove_path(path: Path, removed: list[str]) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()
    removed.append(str(path.relative_to(PROJECT_ROOT)))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cleanup generated Brazil pipeline outputs")
    parser.add_argument(
        "--include-raw",
        action="store_true",
        help="Also delete raw Infodengue artifacts (protected by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    removed: list[str] = []

    if args.include_raw:
        _remove_path(RAW_JSONL_DIR, removed)

    if args.include_raw and RAW_DIR.exists():
        for path in RAW_DIR.iterdir():
            if not path.is_file():
                continue
            name = path.name
            if UF_PARQUET_RE.match(name) or MERGED_PARQUET_RE.match(name):
                _remove_path(path, removed)

    if PROCESSED_DIR.exists():
        for path in PROCESSED_DIR.iterdir():
            if not path.is_file():
                continue
            name = path.name
            if PANEL_PARQUET_RE.match(name) or name in {"epiclim_labeled.csv", "Epiclim_Final_data.csv"}:
                _remove_path(path, removed)

    if args.include_raw and RAW_DIR.exists():
        for candidate in (RAW_DIR / "epiclim_labeled.csv", RAW_DIR / "Epiclim_Final_data.csv"):
            _remove_path(candidate, removed)

    _remove_path(PROBES_RAW_DIR, removed)

    for directory in (RAW_DIR, RAW_JSONL_DIR, PROCESSED_DIR, PROBES_RAW_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    print("CLEANUP_STATUS=SUCCESS")
    print(f"INCLUDE_RAW={args.include_raw}")
    print(f"REMOVED_COUNT={len(removed)}")
    for item in removed:
        print(f"REMOVED={item}")


if __name__ == "__main__":
    main()
