from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing.clean_data import run as run_clean_data
from src.data_preprocessing.impute_climate import run as run_impute_climate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build top-N municipality filtered final dataset + metadata snapshot",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/brazil_chik_dataset_final.csv"),
        help="Input dataset CSV",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("data/processed/brazil_chik_dataset_top300_final.csv"),
        help="Filtered top-N output CSV",
    )
    parser.add_argument(
        "--output-meta",
        type=Path,
        default=Path("outputs/reports/top300_dataset_snapshot.json"),
        help="Output metadata JSON path",
    )
    parser.add_argument(
        "--output-meta-md",
        type=Path,
        default=Path("outputs/reports/top300_dataset_snapshot.md"),
        help="Output metadata markdown path",
    )
    parser.add_argument("--top-n", type=int, default=300, help="Number of municipalities to keep")
    parser.add_argument("--start-year", type=int, default=2015, help="Cleaning lower year bound")
    parser.add_argument("--end-year", type=int, default=2025, help="Cleaning upper year bound")
    return parser.parse_args()


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_json_compatible(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return _to_json_compatible(value.item())
        except Exception:
            pass
    return str(value)


def _load_csv_robust(input_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(input_path, low_memory=False)
    frame.columns = [str(column).strip().lower() for column in frame.columns]
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame


def _canonical_dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    output = df.copy()
    if "district" in output.columns and "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
        before_rows = int(len(output))
        output = (
            output.sort_values(["district", "date"], ascending=[True, True], na_position="last")
            .drop_duplicates(subset=["district", "date"], keep="last")
            .reset_index(drop=True)
        )
        return output, int(before_rows - len(output))
    return output, 0


def _resolve_municipality_key(df: pd.DataFrame) -> str:
    if "municipality_id" in df.columns:
        return "municipality_id"
    if "district" in df.columns:
        return "district"
    raise ValueError("Neither 'municipality_id' nor 'district' columns are available.")


def _safe_top_n_summary(df: pd.DataFrame, key_column: str, top_n: int) -> tuple[pd.DataFrame, list[str], str]:
    working = df.copy()
    key_as_text = working[key_column].astype("string")
    working["__municipality_key"] = key_as_text

    summary = working.groupby("__municipality_key", dropna=False).size().rename("row_count").reset_index()
    summary = summary.rename(columns={"__municipality_key": "id"})

    metric_mode = "row_count_fallback"
    if "cases" in working.columns:
        working["__cases_numeric"] = pd.to_numeric(working["cases"], errors="coerce")
        cases_summary = (
            working.groupby("__municipality_key", dropna=False)["__cases_numeric"]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={"__municipality_key": "id", "__cases_numeric": "total_cases"})
        )
        summary = summary.merge(cases_summary, on="id", how="left")
        if summary["total_cases"].notna().any():
            metric_mode = "total_cases"
            summary = summary.sort_values(["total_cases", "row_count", "id"], ascending=[False, False, True])
        else:
            summary["total_cases"] = summary["row_count"].astype(float)
            summary = summary.sort_values(["row_count", "id"], ascending=[False, True])
    else:
        summary["total_cases"] = summary["row_count"].astype(float)
        summary = summary.sort_values(["row_count", "id"], ascending=[False, True])

    summary = summary.reset_index(drop=True)
    top_summary = summary.head(int(top_n)).copy()
    top_ids = [str(value) for value in top_summary["id"].astype("string").tolist()]
    return top_summary, top_ids, metric_mode


def _numeric_describe(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}
    described = numeric_df.describe(include="all").to_dict()
    return _to_json_compatible(described)


def _date_min_max(df: pd.DataFrame) -> dict[str, Any]:
    if "date" not in df.columns:
        return {"present": False, "min": None, "max": None, "non_null_count": 0}
    parsed = pd.to_datetime(df["date"], errors="coerce")
    non_null = parsed.dropna()
    return {
        "present": True,
        "min": non_null.min().isoformat() if not non_null.empty else None,
        "max": non_null.max().isoformat() if not non_null.empty else None,
        "non_null_count": int(non_null.size),
    }


def _shape(df: pd.DataFrame) -> dict[str, int]:
    return {"rows": int(len(df)), "columns": int(df.shape[1])}


def _build_markdown_report(metadata: dict[str, Any]) -> str:
    paths = metadata["paths"]
    steps = metadata["step_shapes"]
    quality = metadata["quality"]
    top_summary = metadata["top_n_summary"]

    lines: list[str] = []
    lines.append("# Top Municipality Dataset Snapshot")
    lines.append("")
    lines.append(f"- Timestamp (UTC): {metadata['timestamp_utc']}")
    lines.append(f"- Input: {paths['input']}")
    lines.append(f"- Output data: {paths['output_data']}")
    lines.append(f"- Output metadata JSON: {paths['output_meta_json']}")
    lines.append(f"- Top-N requested: {metadata['top_n_requested']}")
    lines.append(f"- Municipality key used: {metadata['municipality_key_used']}")
    lines.append(f"- Top-N ranking metric: {quality['top_n_metric_mode']}")
    lines.append("")
    lines.append("## Step Shapes")
    lines.append("")
    for step_name in ["loaded", "cleaned", "imputed", "canonical_deduped", "top_n_filtered"]:
        step = steps[step_name]
        lines.append(f"- {step_name}: {step['rows']} rows, {step['columns']} columns")
    lines.append("")
    lines.append("## Quality")
    lines.append("")
    lines.append(f"- Duplicate district/date rows removed: {quality['canonical_dedup_removed_rows']}")
    lines.append(f"- Missing municipality keys in final output: {quality['final_missing_municipality_keys']}")
    lines.append(f"- Final duplicate district/date rows: {quality['final_duplicate_district_date_rows']}")
    lines.append("")
    lines.append("## Top Municipalities (first 20)")
    lines.append("")
    lines.append("| id | total_cases | row_count |")
    lines.append("|---|---:|---:|")
    for row in top_summary[:20]:
        lines.append(f"| {row['id']} | {row['total_cases']} | {row['row_count']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()

    input_path = Path(args.input)
    output_data_path = Path(args.output_data)
    output_meta_path = Path(args.output_meta)
    output_meta_md_path = Path(args.output_meta_md)

    raw_df = _load_csv_robust(input_path)
    cleaned_df = run_clean_data(raw_df, start_year=int(args.start_year), end_year=int(args.end_year))
    imputed_df = run_impute_climate(cleaned_df)
    deduped_df, dedup_removed_rows = _canonical_dedupe(imputed_df)

    municipality_key = _resolve_municipality_key(deduped_df)
    top_summary_df, top_ids, metric_mode = _safe_top_n_summary(deduped_df, municipality_key, int(args.top_n))

    top_id_set = set(top_ids)
    filtered_df = deduped_df.loc[
        deduped_df[municipality_key].astype("string").isin(top_id_set)
    ].copy()

    _ensure_parent(output_data_path)
    filtered_df.to_csv(output_data_path, index=False)

    final_missing_counts = {
        str(column): int(count)
        for column, count in filtered_df.isna().sum().to_dict().items()
    }
    final_dtypes = {str(column): str(dtype) for column, dtype in filtered_df.dtypes.to_dict().items()}
    district_date_dupes = 0
    if {"district", "date"}.issubset(filtered_df.columns):
        district_date_dupes = int(
            filtered_df.duplicated(subset=["district", "date"], keep=False).sum()
        )

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paths": {
            "input": str(input_path),
            "output_data": str(output_data_path),
            "output_meta_json": str(output_meta_path),
            "output_meta_markdown": str(output_meta_md_path),
        },
        "top_n_requested": int(args.top_n),
        "start_year": int(args.start_year),
        "end_year": int(args.end_year),
        "municipality_key_used": municipality_key,
        "top_n_ids": top_ids,
        "step_shapes": {
            "loaded": _shape(raw_df),
            "cleaned": _shape(cleaned_df),
            "imputed": _shape(imputed_df),
            "canonical_deduped": _shape(deduped_df),
            "top_n_filtered": _shape(filtered_df),
        },
        "final_columns": [str(column) for column in filtered_df.columns],
        "final_dtypes": final_dtypes,
        "final_missing_counts": final_missing_counts,
        "final_date_range": _date_min_max(filtered_df),
        "top_n_summary": _to_json_compatible(top_summary_df.to_dict(orient="records")),
        "numeric_summary_stats": _numeric_describe(filtered_df),
        "quality": {
            "top_n_metric_mode": metric_mode,
            "canonical_dedup_removed_rows": int(dedup_removed_rows),
            "final_missing_municipality_keys": int(filtered_df[municipality_key].isna().sum()),
            "final_duplicate_district_date_rows": district_date_dupes,
            "final_any_missing_cells": int(filtered_df.isna().sum().sum()),
        },
    }

    _ensure_parent(output_meta_path)
    output_meta_path.write_text(json.dumps(_to_json_compatible(metadata), indent=2), encoding="utf-8")

    report_markdown = _build_markdown_report(metadata)
    _ensure_parent(output_meta_md_path)
    output_meta_md_path.write_text(report_markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
