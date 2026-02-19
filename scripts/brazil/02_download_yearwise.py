from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from projects.brazil_chik.data_adapter import DEFAULT_UFS, run


def _parse_bool(raw: str) -> bool:
    value = str(raw).strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")


def _year_chunks(start_year: int, end_year: int, year_chunk_size: int) -> list[tuple[int, int]]:
    if year_chunk_size <= 0:
        raise ValueError("year_chunk_size must be >= 1")
    chunks: list[tuple[int, int]] = []
    current = start_year
    while current <= end_year:
        chunk_end = min(current + year_chunk_size - 1, end_year)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


def _parse_ufs(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    cleaned = [token.strip().upper() for token in raw.split(",") if token.strip()]
    return tuple(cleaned) if cleaned else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Brazil Infodengue data year-wise with resumable raw cache")
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--year-chunk-size", type=int, default=1)
    parser.add_argument("--ufs", type=str, default=None, help="Comma-separated UF list (e.g. RJ,SP)")
    parser.add_argument("--force-refresh", nargs="?", const="true", default="false", type=_parse_bool)
    parser.add_argument("--build-panel", nargs="?", const="true", default="true", type=_parse_bool)
    parser.add_argument("--max-workers", type=int, default=1)
    args = parser.parse_args()

    start_year = int(args.start_year)
    end_year = int(args.end_year)
    year_chunk_size = int(args.year_chunk_size)
    selected_ufs = _parse_ufs(args.ufs)

    started = time.perf_counter()
    first_output, _ = run(
        epiclim_path=Path("data/raw/Epiclim_Final_data.csv"),
        census_path=None,
        discovery_dir=Path("data/raw"),
        start_year=start_year,
        end_year=end_year,
        ufs=selected_ufs,
        force_refresh=bool(args.force_refresh),
        year_chunk_size=year_chunk_size,
        max_workers=int(args.max_workers),
        build_panel=bool(args.build_panel),
    )
    elapsed = time.perf_counter() - started

    effective_ufs = selected_ufs or DEFAULT_UFS
    chunk_ranges = _year_chunks(start_year, end_year, year_chunk_size)

    raw_dir = Path("data/raw")
    raw_jsonl_dir = raw_dir / "infodengue_raw"
    processed_dir = Path("data/processed")

    raw_chunk_file_count = sum(
        int((raw_jsonl_dir / f"infodengue_chik_{uf}_{y0}_{y1}.jsonl").exists())
        for uf in effective_ufs
        for y0, y1 in chunk_ranges
    )

    per_uf_parquet_count = sum(
        int((raw_dir / f"infodengue_chik_{uf}_{start_year}_{end_year}.parquet").exists())
        for uf in effective_ufs
    )

    merged_path = raw_dir / f"infodengue_chik_merged_{start_year}_{end_year}.parquet"
    merged_rows = len(pd.read_parquet(merged_path)) if merged_path.exists() else -1

    yearly_merged_file_count = sum(
        int((raw_dir / f"infodengue_chik_merged_{year}_{year}.parquet").exists())
        for year in range(start_year, end_year + 1)
    )

    panel_rows_text = "SKIPPED"
    if bool(args.build_panel):
        panel_rows_text = str(len(first_output))
        panel_path = processed_dir / f"infodengue_chik_panel_{start_year}_{end_year}.parquet"
        if not panel_path.exists():
            panel_rows_text = "0"

    print("RUN_STATUS=SUCCESS")
    print(f"START_YEAR={start_year}")
    print(f"END_YEAR={end_year}")
    print(f"YEAR_CHUNK_SIZE={year_chunk_size}")
    print(f"UFS={','.join(effective_ufs)}")
    print(f"RAW_CHUNK_FILE_COUNT={raw_chunk_file_count}")
    print(f"PER_UF_PARQUET_COUNT={per_uf_parquet_count}")
    print(f"MERGED_ROWS={merged_rows}")
    print(f"YEARLY_MERGED_FILE_COUNT={yearly_merged_file_count}")
    print(f"PANEL_ROWS={panel_rows_text}")
    print(f"ELAPSED_SECONDS={elapsed:.2f}")


if __name__ == "__main__":
    main()
