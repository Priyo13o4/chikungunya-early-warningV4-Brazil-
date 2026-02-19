from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
RAW_JSONL_DIR = RAW_DIR / "infodengue_raw"
PROC_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "reports"


def _count_jsonl_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def _profile_raw_coverage() -> dict[str, Any]:
    pattern = re.compile(r"^infodengue_chik_([A-Z]{2})_(\d{4})_(\d{4})\.jsonl$")
    files = sorted(RAW_JSONL_DIR.glob("infodengue_chik_*_*.jsonl"))

    coverage: dict[int, set[str]] = {}
    total_rows = 0
    invalid_name: list[str] = []

    for file_path in files:
        match = pattern.match(file_path.name)
        if not match:
            invalid_name.append(file_path.name)
            continue
        uf, start_year, end_year = match.group(1), int(match.group(2)), int(match.group(3))
        if start_year == end_year:
            coverage.setdefault(start_year, set()).add(uf)
        total_rows += _count_jsonl_lines(file_path)

    years = list(range(2019, 2024))
    coverage_counts = {str(year): len(coverage.get(year, set())) for year in years}
    missing_ufs = {
        str(year): sorted(set(DEFAULT_UFS) - coverage.get(year, set()))
        for year in years
    }
    return {
        "jsonl_files": len(files),
        "jsonl_rows_total": total_rows,
        "coverage_counts": coverage_counts,
        "missing_ufs": missing_ufs,
        "invalid_filenames": invalid_name,
    }


def _load_available_parquet() -> pd.DataFrame:
    parquet_paths = sorted(RAW_DIR.glob("infodengue_chik_*_*.parquet"))
    parquet_paths = [path for path in parquet_paths if "merged" not in path.name]
    frames: list[pd.DataFrame] = []
    for path in parquet_paths:
        try:
            frame = pd.read_parquet(path)
        except Exception:
            continue
        frame["_source_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _profile_parquet_integrity(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "available": False,
        }

    key_cols = [column for column in ("municipality_id", "date") if column in df.columns]
    duplicates = 0
    if len(key_cols) == 2:
        duplicates = int(df.duplicated(subset=key_cols).sum())

    missingness = {}
    for column in ("municipality_id", "date", "cases", "rt", "p_rt1", "receptivo"):
        if column in df.columns:
            missingness[column] = float(df[column].isna().mean())

    cases = pd.to_numeric(df.get("cases", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    top_ufs: list[dict[str, Any]] = []
    if "state" in df.columns:
        by_state = (
            pd.DataFrame({"state": df["state"], "cases": cases})
            .groupby("state", dropna=False)["cases"]
            .sum()
            .sort_values(ascending=False)
        )
        top_ufs = [
            {"state": str(state), "cases": float(value)}
            for state, value in by_state.head(10).items()
        ]

    return {
        "available": True,
        "rows": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "duplicate_key_rows": duplicates,
        "duplicate_key_rate": float(duplicates / len(df)) if len(df) else 0.0,
        "total_cases": float(cases.sum()),
        "median_weekly_cases": float(cases.median()) if len(cases) else 0.0,
        "top_ufs_by_cases": top_ufs,
        "missingness": missingness,
    }


def _profile_artifacts() -> dict[str, Any]:
    merged = RAW_DIR / "infodengue_chik_merged_2019_2023.parquet"
    panel = PROC_DIR / "infodengue_chik_panel_2019_2023.parquet"
    epiclim = RAW_DIR / "Epiclim_Final_data.csv"
    log_path = RAW_DIR / "infodengue_2019_2023.log"

    return {
        "merged_2019_2023_exists": merged.exists(),
        "panel_2019_2023_exists": panel.exists(),
        "epiclim_exists": epiclim.exists(),
        "download_log_exists": log_path.exists(),
    }


def _profile_chik_dataset() -> dict[str, Any]:
    chik_path = ROOT / "chikungunya.csv"
    if not chik_path.exists():
        return {"available": False}

    frame = pd.read_csv(chik_path, low_memory=False)
    join_cols = ["ID_MUNICIP", "ID_MN_RESI", "SEM_NOT", "NU_ANO", "SG_UF", "SG_UF_NOT"]
    present_join_cols = [column for column in join_cols if column in frame.columns]

    missingness = {
        column: float(frame[column].isna().mean())
        for column in present_join_cols
    }

    muni_col = "ID_MN_RESI" if "ID_MN_RESI" in frame.columns else ("ID_MUNICIP" if "ID_MUNICIP" in frame.columns else None)
    municipality_unique = None
    if muni_col is not None:
        municipality_unique = int(pd.to_numeric(frame[muni_col], errors="coerce").dropna().astype("Int64").nunique())

    return {
        "available": True,
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "join_columns_present": present_join_cols,
        "join_key_missingness": missingness,
        "municipality_column_used": muni_col,
        "municipality_unique_count": municipality_unique,
        "climate_merge_feasible": bool(muni_col is not None and "SEM_NOT" in frame.columns and "NU_ANO" in frame.columns),
        "climate_merge_note": "Use climate weekly endpoint and join on municipality code + epiweek (YYYYWW).",
    }


DEFAULT_UFS = (
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    raw_profile = _profile_raw_coverage()
    parquet_df = _load_available_parquet()
    parquet_profile = _profile_parquet_integrity(parquet_df)
    artifacts = _profile_artifacts()
    chik_profile = _profile_chik_dataset()

    report = {
        "raw_coverage": raw_profile,
        "parquet_integrity": parquet_profile,
        "artifacts": artifacts,
        "chik_dataset": chik_profile,
    }

    output_path = REPORT_DIR / "brazil_data_integrity_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("INTEGRITY_REPORT", output_path)
    print("RAW_COVERAGE", raw_profile["coverage_counts"])
    print("RAW_ROWS_TOTAL", raw_profile["jsonl_rows_total"])
    print("PARQUET_ROWS", parquet_profile.get("rows", 0))
    print("TOTAL_CASES", parquet_profile.get("total_cases", 0.0))
    print("FULL_ARTIFACTS", artifacts)
    print("CHIK_DATASET", {k: chik_profile[k] for k in ("available", "rows", "columns", "climate_merge_feasible") if k in chik_profile})


if __name__ == "__main__":
    main()
