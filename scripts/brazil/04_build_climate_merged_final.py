from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BASE_URL = "https://api.mosqlimate.org"
DEFAULT_CLIMATE_ENDPOINT = "/api/datastore/climate/weekly/"


def _load_env_file(env_path: Path = Path(".env")) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, value.strip())


def _resolve_api_key() -> str:
    _load_env_file()
    for name in (
        "INFODENGUE_X_UID_KEY",
        "X_UID_KEY",
        "MOSQLIMATE_API_KEY",
        "MOSQLIENT_API_KEY",
        "api_key",
        "API_KEY",
    ):
        value = os.getenv(name)
        if value and str(value).strip():
            return str(value).strip()
    raise RuntimeError(
        "No API key found. Set INFODENGUE_X_UID_KEY (or X_UID_KEY / MOSQLIMATE_API_KEY) in environment or .env."
    )


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "X-UID-Key": api_key,
        "Authorization": f"Bearer {api_key}",
    }


def _derive_epiweek(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    iso = dates.dt.isocalendar()
    return (iso.year.astype("Int64") * 100 + iso.week.astype("Int64")).astype("Int64")


def _normalize_epiweek_for_api(epiweek_value: int) -> int:
    year = int(epiweek_value) // 100
    week = int(epiweek_value) % 100
    if week < 1:
        week = 1
    if week > 52:
        week = 52
    return (year * 100) + week


def _extract_items(payload: Any) -> tuple[list[dict[str, Any]], str | None, bool | None]:
    if isinstance(payload, list):
        normalized = [item for item in payload if isinstance(item, dict)]
        return normalized, None, None

    if not isinstance(payload, dict):
        return [], None, False

    items: list[dict[str, Any]] = []
    for key in ("results", "items", "data", "records"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            items = [item for item in candidate if isinstance(item, dict)]
            break
    if not items and isinstance(payload.get("result"), list):
        items = [item for item in payload["result"] if isinstance(item, dict)]

    links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
    next_url = payload.get("next") or payload.get("next_page") or links.get("next")
    has_more = payload.get("has_more")
    if has_more is None:
        has_more = payload.get("has_next")
    return items, str(next_url) if isinstance(next_url, str) and next_url else None, bool(has_more) if has_more is not None else None


def _fetch_climate_by_uf(
    api_key: str,
    ufs: list[str],
    epiweek_windows: list[tuple[str, str]],
    *,
    base_url: str,
    endpoint: str,
    page_size: int,
    timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
    cache_dir: Path,
) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(_auth_headers(api_key))

    endpoint_normalized = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    base_request_url = f"{base_url.rstrip('/')}{endpoint_normalized}"

    frames: list[pd.DataFrame] = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    for uf in ufs:
        for chunk_start, chunk_end in epiweek_windows:
            source_year = int(str(chunk_start)[:4])
            cache_path = cache_dir / f"climate_weekly_{uf}_{chunk_start}_{chunk_end}.parquet"
            if cache_path.exists():
                print(f"[CACHE] uf={uf} window={chunk_start}-{chunk_end}", flush=True)
                cached = pd.read_parquet(cache_path)
                if not cached.empty:
                    frames.append(cached)
                continue

            chunk_frames: list[pd.DataFrame] = []
            print(f"[FETCH] uf={uf} window={chunk_start}-{chunk_end}", flush=True)
            page = 1
            next_url: str | None = None

            while True:
                request_url = next_url or base_request_url
                params: dict[str, Any] | None
                if next_url:
                    params = None
                else:
                    params = {
                        "uf": uf,
                        "start": chunk_start,
                        "end": chunk_end,
                        "page": page,
                        "per_page": int(page_size),
                    }

                response: requests.Response | None = None
                last_error: Exception | None = None
                for attempt in range(1, int(max_retries) + 1):
                    try:
                        response = session.get(request_url, params=params, timeout=float(timeout_seconds))
                        response.raise_for_status()
                        last_error = None
                        break
                    except requests.RequestException as error:
                        last_error = error
                        if attempt < int(max_retries):
                            time.sleep(float(retry_sleep_seconds) * attempt)

                if last_error is not None:
                    raise RuntimeError(
                        f"Climate pull failed for uf={uf} window={chunk_start}-{chunk_end} page={page}: {last_error}"
                    ) from last_error

                if response is None:
                    break

                payload = response.json()
                rows, extracted_next_url, has_more = _extract_items(payload)
                if not rows:
                    break
                if rows:
                    frame = pd.DataFrame.from_records(rows)
                    frame["state"] = str(uf)
                    frame["source_year"] = int(source_year)
                    chunk_frames.append(frame)

                if extracted_next_url:
                    next_url = extracted_next_url
                    page += 1
                    continue

                if has_more is True:
                    next_url = None
                    page += 1
                    continue

                if has_more is None and len(rows) >= int(page_size):
                    next_url = None
                    page += 1
                    continue

                break

            if chunk_frames:
                chunk_df = pd.concat(chunk_frames, ignore_index=True)
                chunk_df.to_parquet(cache_path, index=False)
                frames.append(chunk_df)
                print(f"[SAVED] uf={uf} window={chunk_start}-{chunk_end} rows={len(chunk_df)}", flush=True)

    if not frames:
        raise RuntimeError("Climate API returned no rows for requested UF and epiweek range")

    combined = pd.concat(frames, ignore_index=True)
    combined["geocodigo"] = pd.to_numeric(combined.get("geocodigo"), errors="coerce").astype("Int64")
    combined["epiweek"] = pd.to_numeric(combined.get("epiweek"), errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["geocodigo", "epiweek"]).copy()
    combined = combined.drop_duplicates(subset=["geocodigo", "epiweek"], keep="last")
    return combined


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pull climate data into a separate dataset and build merged brazil_chik_dataset_final"
    )
    parser.add_argument("--raw-path", type=Path, default=Path("data/raw/Epiclim_Final_data.csv"))
    parser.add_argument(
        "--climate-output",
        type=Path,
        default=Path("data/processed/brazil_climate_weekly_dataset.csv"),
    )
    parser.add_argument(
        "--final-output",
        type=Path,
        default=Path("data/processed/brazil_chik_dataset_final.csv"),
    )
    parser.add_argument("--start-epiweek", type=str, default=None, help="Optional YYYYWW override")
    parser.add_argument("--end-epiweek", type=str, default=None, help="Optional YYYYWW override")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--endpoint", type=str, default=DEFAULT_CLIMATE_ENDPOINT)
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/climate_weekly_cache"),
        help="Directory for per-UF/year cached climate chunks",
    )
    args = parser.parse_args()

    raw_path: Path = args.raw_path
    climate_output: Path = args.climate_output
    final_output: Path = args.final_output

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    api_key = _resolve_api_key()

    raw = pd.read_csv(raw_path)
    required_columns = {"municipality_id", "date", "state"}
    missing = sorted(list(required_columns - set(raw.columns)))
    if missing:
        raise ValueError(f"Raw dataset missing required columns for merge: {missing}")

    raw = raw.copy()
    raw["epiweek"] = _derive_epiweek(raw["date"])
    raw["municipality_id_num"] = pd.to_numeric(raw["municipality_id"], errors="coerce").astype("Int64")

    if args.start_epiweek is not None:
        start_epiweek = str(args.start_epiweek)
    else:
        min_epiweek = raw["epiweek"].dropna().min()
        if pd.isna(min_epiweek):
            raise ValueError("Cannot infer start epiweek from raw date column")
        start_epiweek = f"{int(min_epiweek):06d}"

    if args.end_epiweek is not None:
        end_epiweek = str(args.end_epiweek)
    else:
        max_epiweek = raw["epiweek"].dropna().max()
        if pd.isna(max_epiweek):
            raise ValueError("Cannot infer end epiweek from raw date column")
        end_epiweek = f"{int(max_epiweek):06d}"

    ufs = sorted(str(value).strip().upper() for value in raw["state"].dropna().unique() if str(value).strip())
    if not ufs:
        raise ValueError("No valid UF values found in raw state column")

    epiweeks = pd.to_numeric(raw["epiweek"], errors="coerce").dropna().astype(int)
    if epiweeks.empty:
        raise ValueError("No valid epiweek values could be derived from raw date column")
    raw_windows = pd.DataFrame({"epiweek": epiweeks})
    raw_windows["epi_year"] = raw_windows["epiweek"] // 100
    window_rows = (
        raw_windows.groupby("epi_year", as_index=False)["epiweek"]
        .agg(start="min", end="max")
        .sort_values("epi_year")
    )
    epiweek_windows = [
        (
            f"{_normalize_epiweek_for_api(int(row.start)):06d}",
            f"{_normalize_epiweek_for_api(int(row.end)):06d}",
        )
        for row in window_rows.itertuples(index=False)
    ]

    climate = _fetch_climate_by_uf(
        api_key=api_key,
        ufs=ufs,
        epiweek_windows=epiweek_windows,
        base_url=str(args.base_url),
        endpoint=str(args.endpoint),
        page_size=int(args.page_size),
        timeout_seconds=float(args.timeout_seconds),
        max_retries=int(args.max_retries),
        retry_sleep_seconds=float(args.retry_sleep_seconds),
        cache_dir=Path(args.cache_dir),
    )

    climate_output.parent.mkdir(parents=True, exist_ok=True)
    climate.to_csv(climate_output, index=False)

    merged = raw.merge(
        climate,
        how="left",
        left_on=["municipality_id_num", "epiweek"],
        right_on=["geocodigo", "epiweek"],
        suffixes=("", "_climate"),
    )

    matched_rows = int(merged["geocodigo"].notna().sum())
    total_rows = int(len(merged))

    final_output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(final_output, index=False)

    summary = {
        "raw_path": str(raw_path),
        "climate_output": str(climate_output),
        "final_output": str(final_output),
        "start_epiweek": start_epiweek,
        "end_epiweek": end_epiweek,
        "uf_count": len(ufs),
        "raw_rows": total_rows,
        "climate_rows": int(len(climate)),
        "climate_unique_keys": int(climate[["geocodigo", "epiweek"]].drop_duplicates().shape[0]),
        "matched_rows": matched_rows,
        "match_rate": round((matched_rows / total_rows), 6) if total_rows else None,
        "raw_unique_keys": int(raw[["municipality_id_num", "epiweek"]].drop_duplicates().shape[0]),
    }
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
