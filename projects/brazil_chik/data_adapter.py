from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Iterable

import pandas as pd
import requests

from src.data_preprocessing.load_data import load_census_data

LOGGER = logging.getLogger(__name__)

DEFAULT_INFODENGUE_API_BASE_URL = "https://api.mosqlimate.org"
DEFAULT_INFODENGUE_API_ENDPOINT = "/api/datastore/infodengue/"
DEFAULT_INFODENGUE_API_MAX_PER_PAGE = 100
DEFAULT_INFODENGUE_API_DISEASE = "chik"
DEFAULT_START_YEAR = 2015
DEFAULT_END_YEAR = 2025
DEFAULT_WEEK_ANCHOR = "SUN"

FORBIDDEN_ENGINEERED_COLUMN_PREFIXES: tuple[str, ...] = (
    "case_lag",
    "cases_lag",
    "case_rolling",
    "cases_rolling",
)
FORBIDDEN_ENGINEERED_COLUMNS: tuple[str, ...] = (
    "outbreak_label",
    "outbreak_target",
    "month_sin",
    "month_cos",
    "weekofyear",
)

DEFAULT_UFS: tuple[str, ...] = (
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


def _safe_positive_int(raw: Any, default: int) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return value if value > 0 else int(default)


def _safe_float(raw: Any, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _env_str(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and str(value).strip() != "":
            return str(value).strip()
    return default


def _env_positive_int(*names: str, default: int) -> int:
    for name in names:
        if name in os.environ:
            return _safe_positive_int(os.getenv(name), default)
    return int(default)


def _env_float(*names: str, default: float) -> float:
    for name in names:
        if name in os.environ:
            return _safe_float(os.getenv(name), default)
    return float(default)


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


def _auth_headers() -> dict[str, str]:
    _load_env_file()
    token = (
        os.getenv("INFODENGUE_X_UID_KEY")
        or os.getenv("X_UID_KEY")
        or os.getenv("INFODENGUE_TOKEN")
        or os.getenv("INFODENGUE_API_TOKEN")
    )
    bearer = os.getenv("INFODENGUE_BEARER") or os.getenv("INFODENGUE_AUTHORIZATION")

    headers: dict[str, str] = {"Accept": "application/json"}
    if token:
        headers["X-UID-Key"] = token
    if bearer:
        auth_value = bearer if bearer.lower().startswith("bearer ") else f"Bearer {bearer}"
        headers["Authorization"] = auth_value
    elif token:
        headers["Authorization"] = f"Bearer {token}"

    return headers


def _extract_items(payload: Any) -> tuple[list[dict[str, Any]], str | None, bool | None]:
    if isinstance(payload, list):
        normalized = [item for item in payload if isinstance(item, dict)]
        return normalized, None, None

    if not isinstance(payload, dict):
        return [], None, False

    container_keys = ("results", "items", "data", "records")
    items: list[dict[str, Any]] = []
    for key in container_keys:
        candidate = payload.get(key)
        if isinstance(candidate, list):
            items = [item for item in candidate if isinstance(item, dict)]
            break

    if not items:
        nested = payload.get("result")
        if isinstance(nested, list):
            items = [item for item in nested if isinstance(item, dict)]

    next_url = payload.get("next") or payload.get("next_page") or payload.get("links", {}).get("next")
    has_more = payload.get("has_more")
    if has_more is None:
        has_more = payload.get("has_next")
    if has_more is None and isinstance(payload.get("pagination"), dict):
        pagination = payload.get("pagination", {})
        page_value = pagination.get("page")
        total_pages = pagination.get("total_pages")
        try:
            has_more = int(page_value) < int(total_pages)
        except Exception:
            has_more = None

    return items, str(next_url) if isinstance(next_url, str) and next_url else None, bool(has_more) if has_more is not None else None


def _safe_epiweek_to_date(year_value: Any, week_value: Any) -> pd.Timestamp | None:
    try:
        year_num = int(float(year_value))
        week_num = int(float(week_value))
        if week_num < 1 or week_num > 53:
            return pd.NaT
        return pd.Timestamp.fromisocalendar(year_num, week_num, 1)
    except Exception:
        return pd.NaT


def _resolve_week_anchor(anchor: str | None) -> int:
    resolved = str(anchor or DEFAULT_WEEK_ANCHOR).strip().upper()
    weekday_map = {
        "MON": 0,
        "TUE": 1,
        "WED": 2,
        "THU": 3,
        "FRI": 4,
        "SAT": 5,
        "SUN": 6,
    }
    if resolved not in weekday_map:
        raise ValueError(f"Invalid week_anchor '{anchor}'. Supported values: {', '.join(weekday_map)}")
    return weekday_map[resolved]


def _align_to_week_anchor(dates: pd.Series, *, anchor_weekday: int) -> pd.Series:
    normalized = pd.to_datetime(dates, errors="coerce").dt.normalize()
    if normalized.empty:
        return normalized

    weekday = normalized.dt.weekday
    days_to_prev_anchor = (weekday - int(anchor_weekday)) % 7
    shift_days = days_to_prev_anchor.where(days_to_prev_anchor <= 3, days_to_prev_anchor - 7)
    aligned = normalized - pd.to_timedelta(shift_days, unit="D")
    return aligned


def _assert_minimal_cleaning_only(frame: pd.DataFrame, *, context: str) -> None:
    if frame.empty:
        return

    lowered = [str(column).strip().lower() for column in frame.columns]
    blocked: list[str] = []
    forbidden_exact = set(FORBIDDEN_ENGINEERED_COLUMNS)
    for original, column in zip(frame.columns, lowered, strict=False):
        if column in forbidden_exact or any(column.startswith(prefix) for prefix in FORBIDDEN_ENGINEERED_COLUMN_PREFIXES):
            blocked.append(str(original))

    if blocked:
        raise AssertionError(
            f"Adapter minimal-cleaning contract violated in {context}; found engineered columns: {sorted(set(blocked))}"
        )


def _normalize_columns(frame: pd.DataFrame, uf: str, *, anchor_weekday: int) -> pd.DataFrame:
    output = frame.copy()
    output.columns = [str(column).strip().lower() for column in output.columns]

    rename_map: dict[str, str] = {}
    if "uf" in output.columns:
        rename_map["uf"] = "state"
    if "estado" in output.columns and "state" not in output.columns:
        rename_map["estado"] = "state"
    if "municipio" in output.columns:
        rename_map["municipio"] = "district"
    if "municipio_nome" in output.columns and "district" not in output.columns:
        rename_map["municipio_nome"] = "district"
    if "mun_name" in output.columns and "district" not in output.columns:
        rename_map["mun_name"] = "district"
    if "geocode" in output.columns:
        rename_map["geocode"] = "municipality_id"
    if "municipio_geocodigo" in output.columns and "municipality_id" not in output.columns:
        rename_map["municipio_geocodigo"] = "municipality_id"
    if "id_municipio" in output.columns:
        rename_map["id_municipio"] = "municipality_id"
    if "casos_est" in output.columns:
        rename_map["casos_est"] = "cases"
    elif "casos" in output.columns:
        rename_map["casos"] = "cases"
    if "data_ini_se" in output.columns:
        rename_map["data_ini_se"] = "date"
    if "data_inise" in output.columns and "date" not in output.columns:
        rename_map["data_inise"] = "date"
    if "data_ini" in output.columns and "date" not in output.columns:
        rename_map["data_ini"] = "date"
    if "se" in output.columns and "epiweek" not in output.columns:
        rename_map["se"] = "epiweek"

    if rename_map:
        output = output.rename(columns=rename_map)

    if "state" not in output.columns:
        output["state"] = uf

    if "cases" not in output.columns:
        output["cases"] = 0.0
    output["cases"] = pd.to_numeric(output["cases"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce")
    elif {"ano", "se"}.issubset(output.columns):
        output["date"] = [
            _safe_epiweek_to_date(year_value, week_value)
            for year_value, week_value in zip(output["ano"], output["se"], strict=False)
        ]
    elif {"year", "week"}.issubset(output.columns):
        output["date"] = [
            _safe_epiweek_to_date(year_value, week_value)
            for year_value, week_value in zip(output["year"], output["week"], strict=False)
        ]
    else:
        output["date"] = pd.NaT

    output["date"] = _align_to_week_anchor(output["date"], anchor_weekday=anchor_weekday)

    if "district" not in output.columns:
        if "municipality_id" in output.columns:
            output["district"] = output["municipality_id"].astype("string")
        else:
            output["district"] = "UNKNOWN"

    output["district"] = output["district"].astype("string")
    if "municipality_id" in output.columns:
        output["municipality_id"] = output["municipality_id"].astype("string")
    else:
        output["municipality_id"] = output["district"].astype("string")

    optional_columns = ("rt", "p_rt1", "receptivo", "pop")
    for column in optional_columns:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")

    return output


def _write_jsonl_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _read_jsonl_rows(input_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not input_path.exists():
        return rows

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _year_chunks(start_year: int, end_year: int, year_chunk_size: int) -> list[tuple[int, int]]:
    if year_chunk_size <= 0:
        raise ValueError("year_chunk_size must be >= 1")
    chunks: list[tuple[int, int]] = []
    current = int(start_year)
    while current <= int(end_year):
        chunk_end = min(current + int(year_chunk_size) - 1, int(end_year))
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


def _build_uf_frame_from_chunks(
    *,
    chunk_paths: Iterable[Path],
    uf: str,
    start_year: int,
    end_year: int,
    anchor_weekday: int,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    for chunk_path in chunk_paths:
        all_rows.extend(_read_jsonl_rows(chunk_path))

    if not all_rows:
        return pd.DataFrame()

    state_frame = pd.DataFrame(all_rows)
    state_frame = _normalize_columns(state_frame, uf=uf, anchor_weekday=anchor_weekday)
    state_frame = state_frame.loc[state_frame["date"].notna()].copy()
    state_frame["year"] = state_frame["date"].dt.year.astype("Int64")
    state_frame = state_frame.loc[state_frame["year"].between(start_year, end_year, inclusive="both")].copy()
    state_frame = state_frame.sort_values(["municipality_id", "date"]).drop_duplicates(
        subset=["municipality_id", "date"],
        keep="last",
    )
    state_frame = state_frame.sort_values(["state", "district", "date"]).reset_index(drop=True)
    return state_frame


def _fetch_chunk_rows(
    *,
    base_url: str,
    endpoint: str,
    uf: str,
    disease: str,
    chunk_start_year: int,
    chunk_end_year: int,
    page_size: int,
    rate_limit_seconds: float,
    timeout: float,
) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers.update(_auth_headers())
    return _fetch_uf_rows(
        session=session,
        base_url=base_url,
        endpoint=endpoint,
        uf=uf,
        disease=disease,
        start_year=chunk_start_year,
        end_year=chunk_end_year,
        page_size=page_size,
        rate_limit_seconds=rate_limit_seconds,
        timeout=timeout,
    )


def _fetch_uf_rows(
    *,
    session: requests.Session,
    base_url: str,
    endpoint: str,
    uf: str,
    disease: str,
    start_year: int,
    end_year: int,
    page_size: int,
    rate_limit_seconds: float,
    timeout: float,
) -> list[dict[str, Any]]:
    normalized_endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{base_url.rstrip('/')}{normalized_endpoint}"
    params: dict[str, Any] = {
        "uf": uf,
        "disease": disease,
        "start": f"{int(start_year)}-01-01",
        "end": f"{int(end_year)}-12-31",
        "page": 1,
        "per_page": int(page_size),
    }

    rows: list[dict[str, Any]] = []
    page = 0
    while True:
        response = session.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        items, next_url, has_more = _extract_items(payload)
        if not items:
            break

        rows.extend(items)
        page += 1
        LOGGER.info("Fetched UF=%s page=%d rows=%d", uf, page, len(items))

        if next_url:
            url = next_url
            params = {}
        else:
            params["page"] = int(params.get("page", 1)) + 1

        if has_more is False:
            break

        time.sleep(max(rate_limit_seconds, 0.0))

    return rows


def _build_complete_panel(
    df: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    anchor_weekday: int,
    enforce_anchor_assertion: bool,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    output = df.copy()
    output["date"] = pd.to_datetime(output["date"], errors="coerce")
    output = output.loc[output["date"].notna()].copy()
    source_key_count = int(output[["municipality_id", "district", "state", "date"]].drop_duplicates().shape[0])
    output["date"] = _align_to_week_anchor(output["date"], anchor_weekday=anchor_weekday)
    aligned_key_count = int(output[["municipality_id", "district", "state", "date"]].drop_duplicates().shape[0])
    if aligned_key_count < source_key_count:
        collapsed = source_key_count - aligned_key_count
        message = (
            "Week anchor alignment collapsed municipality-week keys "
            f"(before={source_key_count}, after={aligned_key_count}, collapsed={collapsed}, anchor_weekday={anchor_weekday})"
        )
        if enforce_anchor_assertion:
            raise AssertionError(message)
        LOGGER.warning(message)

    weekday_to_freq = {
        0: "W-MON",
        1: "W-TUE",
        2: "W-WED",
        3: "W-THU",
        4: "W-FRI",
        5: "W-SAT",
        6: "W-SUN",
    }
    inferred_weekday: int | None = None
    inferred_freq = weekday_to_freq.get(anchor_weekday, "W-SUN")
    weekday_counts = output["date"].dt.weekday.value_counts(dropna=True)
    if not weekday_counts.empty:
        try:
            inferred_weekday = int(weekday_counts.idxmax())
        except (TypeError, ValueError):
            inferred_weekday = None

    LOGGER.info(
        "Panel weekly anchor target=%s inferred=%s freq=%s",
        anchor_weekday,
        inferred_weekday if inferred_weekday is not None else "unknown",
        inferred_freq,
    )
    if inferred_weekday is not None and inferred_weekday != int(anchor_weekday):
        message = (
            "Panel anchor mismatch detected "
            f"(target={anchor_weekday}, inferred={inferred_weekday}, freq={inferred_freq})"
        )
        if enforce_anchor_assertion:
            raise AssertionError(message)
        LOGGER.warning(message)

    weeks = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq=inferred_freq)
    ids = output[["municipality_id", "district", "state"]].drop_duplicates()

    panel = ids.assign(_key=1).merge(
        pd.DataFrame({"date": weeks, "_key": 1}),
        on="_key",
        how="inner",
    ).drop(columns=["_key"])

    keep_columns = ["municipality_id", "district", "state", "date", "cases"]
    for candidate in ("rt", "p_rt1", "receptivo"):
        if candidate in output.columns:
            keep_columns.append(candidate)

    merged = panel.merge(
        output[keep_columns],
        on=["municipality_id", "district", "state", "date"],
        how="left",
    )

    matched_rows = int(merged["cases"].notna().sum())
    total_rows = int(len(merged))
    matched_ratio = (matched_rows / total_rows) if total_rows else 0.0
    if matched_rows == 0:
        LOGGER.warning(
            "Panel merge produced zero matched rows before case fill (matched_ratio=%.6f)",
            matched_ratio,
        )

    merged["cases"] = pd.to_numeric(merged["cases"], errors="coerce").fillna(0.0)
    for candidate in ("rt", "p_rt1", "receptivo"):
        if candidate in merged.columns:
            merged[candidate] = pd.to_numeric(merged[candidate], errors="coerce")

    merged = merged.sort_values(["state", "district", "date"]).reset_index(drop=True)
    return merged


def run(
    epiclim_path: Path,
    census_path: Path | None = None,
    date_columns: Iterable[str] | None = None,
    discovery_dir: Path = Path("data/raw"),
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    ufs: Iterable[str] | None = None,
    page_size: int | None = None,
    rate_limit_seconds: float | None = None,
    timeout: float | None = None,
    base_url: str | None = None,
    endpoint: str | None = None,
    disease: str | None = None,
    week_anchor: str | None = None,
    enforce_anchor_assertion: bool = True,
    force_refresh: bool = False,
    year_chunk_size: int = 1,
    max_workers: int = 1,
    build_panel: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    date_columns
    _load_env_file()
    raw_dir = Path("data/raw")
    raw_jsonl_dir = raw_dir / "infodengue_raw"
    processed_dir = Path("data/processed")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_jsonl_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    resolved_start_year = int(start_year) if start_year is not None else _env_positive_int(
        "INFODENGUE_START_YEAR",
        "BRAZIL_CHIK_START_YEAR",
        default=DEFAULT_START_YEAR,
    )
    resolved_end_year = int(end_year) if end_year is not None else _env_positive_int(
        "INFODENGUE_END_YEAR",
        "BRAZIL_CHIK_END_YEAR",
        default=DEFAULT_END_YEAR,
    )
    resolved_week_anchor = _env_str("INFODENGUE_WEEK_ANCHOR", default=DEFAULT_WEEK_ANCHOR)
    if week_anchor is not None:
        resolved_week_anchor = str(week_anchor)
    anchor_weekday = _resolve_week_anchor(resolved_week_anchor)
    resolved_disease = (disease or _env_str("INFODENGUE_API_DISEASE", default=DEFAULT_INFODENGUE_API_DISEASE)).strip()
    resolved_rate_limit_seconds = float(
        rate_limit_seconds if rate_limit_seconds is not None else _env_float("INFODENGUE_RATE_LIMIT_SECONDS", default=0.25)
    )
    resolved_timeout = float(timeout if timeout is not None else _env_float("INFODENGUE_API_TIMEOUT", default=45.0))

    resolved_base_url = base_url or os.getenv("INFODENGUE_API_BASE_URL", DEFAULT_INFODENGUE_API_BASE_URL)
    resolved_endpoint = endpoint or os.getenv("INFODENGUE_API_ENDPOINT", DEFAULT_INFODENGUE_API_ENDPOINT)

    api_max_per_page = _safe_positive_int(
        os.getenv("INFODENGUE_API_MAX_PER_PAGE"),
        DEFAULT_INFODENGUE_API_MAX_PER_PAGE,
    )
    configured_page_size = _safe_positive_int(
        os.getenv("INFODENGUE_API_PAGE_SIZE"),
        api_max_per_page,
    )
    requested_page_size = _safe_positive_int(
        configured_page_size if page_size is None else page_size,
        configured_page_size,
    )
    resolved_page_size = min(requested_page_size, api_max_per_page)
    if requested_page_size > api_max_per_page:
        LOGGER.warning(
            "Requested page_size=%d exceeds INFODENGUE_API_MAX_PER_PAGE=%d; clamped to %d",
            requested_page_size,
            api_max_per_page,
            resolved_page_size,
        )

    selected_ufs = tuple(str(uf).upper() for uf in (ufs or DEFAULT_UFS))
    if int(resolved_start_year) > int(resolved_end_year):
        raise ValueError("start_year must be <= end_year")
    if int(max_workers) < 1:
        raise ValueError("max_workers must be >= 1")

    chunk_ranges = _year_chunks(int(resolved_start_year), int(resolved_end_year), int(year_chunk_size))
    if not resolved_disease:
        raise ValueError("disease must be a non-empty string")

    per_state_frames: list[pd.DataFrame] = []
    uf_chunk_row_counts: dict[str, dict[str, int]] = {}
    for uf in selected_ufs:
        chunk_paths = [
            raw_jsonl_dir / f"infodengue_chik_{uf}_{chunk_start}_{chunk_end}.jsonl"
            for chunk_start, chunk_end in chunk_ranges
        ]

        missing_chunks: list[tuple[int, int, Path]] = []
        for (chunk_start, chunk_end), chunk_path in zip(chunk_ranges, chunk_paths, strict=False):
            if chunk_path.exists() and not force_refresh:
                LOGGER.info("[CACHE-HIT] UF=%s %d-%d raw chunk found", uf, chunk_start, chunk_end)
            else:
                LOGGER.info("[FETCH] UF=%s %d-%d raw chunk", uf, chunk_start, chunk_end)
                missing_chunks.append((chunk_start, chunk_end, chunk_path))

        if missing_chunks:
            if int(max_workers) == 1:
                chunk_failures = False
                for chunk_start, chunk_end, chunk_path in missing_chunks:
                    try:
                        chunk_rows = _fetch_chunk_rows(
                            base_url=resolved_base_url,
                            endpoint=resolved_endpoint,
                            uf=uf,
                            disease=resolved_disease,
                            chunk_start_year=chunk_start,
                            chunk_end_year=chunk_end,
                            page_size=resolved_page_size,
                            rate_limit_seconds=resolved_rate_limit_seconds,
                            timeout=resolved_timeout,
                        )
                    except requests.HTTPError as http_error:
                        LOGGER.warning("Skipping UF=%s due to API HTTP error for %d-%d: %s", uf, chunk_start, chunk_end, http_error)
                        chunk_failures = True
                        break
                    except requests.RequestException as request_error:
                        LOGGER.warning(
                            "Skipping UF=%s due to API request error for %d-%d: %s",
                            uf,
                            chunk_start,
                            chunk_end,
                            request_error,
                        )
                        chunk_failures = True
                        break

                    _write_jsonl_rows(chunk_rows, chunk_path)
                    LOGGER.info("[SAVED] UF=%s %d-%d rows=%d", uf, chunk_start, chunk_end, len(chunk_rows))
                    uf_chunk_row_counts.setdefault(uf, {})[f"{chunk_start}-{chunk_end}"] = int(len(chunk_rows))
                if chunk_failures:
                    continue
            else:
                chunk_failures = False
                with ThreadPoolExecutor(max_workers=min(int(max_workers), len(missing_chunks))) as executor:
                    futures = {
                        executor.submit(
                            _fetch_chunk_rows,
                            base_url=resolved_base_url,
                            endpoint=resolved_endpoint,
                            uf=uf,
                            disease=resolved_disease,
                            chunk_start_year=chunk_start,
                            chunk_end_year=chunk_end,
                            page_size=resolved_page_size,
                            rate_limit_seconds=resolved_rate_limit_seconds,
                            timeout=resolved_timeout,
                        ): (chunk_start, chunk_end, chunk_path)
                        for chunk_start, chunk_end, chunk_path in missing_chunks
                    }
                    for future in as_completed(futures):
                        chunk_start, chunk_end, chunk_path = futures[future]
                        try:
                            chunk_rows = future.result()
                        except requests.HTTPError as http_error:
                            LOGGER.warning(
                                "Skipping UF=%s due to API HTTP error for %d-%d: %s",
                                uf,
                                chunk_start,
                                chunk_end,
                                http_error,
                            )
                            chunk_failures = True
                            continue
                        except requests.RequestException as request_error:
                            LOGGER.warning(
                                "Skipping UF=%s due to API request error for %d-%d: %s",
                                uf,
                                chunk_start,
                                chunk_end,
                                request_error,
                            )
                            chunk_failures = True
                            continue

                        _write_jsonl_rows(chunk_rows, chunk_path)
                        LOGGER.info("[SAVED] UF=%s %d-%d rows=%d", uf, chunk_start, chunk_end, len(chunk_rows))
                        uf_chunk_row_counts.setdefault(uf, {})[f"{chunk_start}-{chunk_end}"] = int(len(chunk_rows))

                if chunk_failures and not all(path.exists() for _, _, path in missing_chunks):
                    continue

        try:
            state_frame = _build_uf_frame_from_chunks(
                chunk_paths=chunk_paths,
                uf=uf,
                start_year=resolved_start_year,
                end_year=resolved_end_year,
                anchor_weekday=anchor_weekday,
            )
        except (OSError, json.JSONDecodeError, ValueError) as parse_error:
            LOGGER.warning("Skipping UF=%s due to raw JSONL parsing error: %s", uf, parse_error)
            continue

        if state_frame.empty:
            LOGGER.info("No records available for UF=%s across requested year chunks", uf)
            continue

        _assert_minimal_cleaning_only(state_frame, context=f"UF={uf}")

        state_path = raw_dir / f"infodengue_chik_{uf}_{resolved_start_year}_{resolved_end_year}.parquet"
        state_frame.to_parquet(state_path, index=False)
        LOGGER.info("Saved UF=%s parquet rows=%d to %s", uf, len(state_frame), state_path)
        per_state_frames.append(state_frame)

    if not per_state_frames:
        raise RuntimeError("Infodengue adapter returned no records for requested UF/year window")

    merged = pd.concat(per_state_frames, ignore_index=True)
    merged = merged.sort_values(["state", "district", "date"]).reset_index(drop=True)
    _assert_minimal_cleaning_only(merged, context="merged")
    merged_path = raw_dir / f"infodengue_chik_merged_{resolved_start_year}_{resolved_end_year}.parquet"
    merged.to_parquet(merged_path, index=False)

    for year in range(int(resolved_start_year), int(resolved_end_year) + 1):
        yearly = merged.loc[merged["date"].dt.year == year].copy()
        yearly = yearly.sort_values(["municipality_id", "state", "date"]).drop_duplicates(
            subset=["municipality_id", "state", "date"],
            keep="last",
        )
        yearly = yearly.sort_values(["state", "district", "date"]).reset_index(drop=True)
        yearly_path = raw_dir / f"infodengue_chik_merged_{year}_{year}.parquet"
        yearly.to_parquet(yearly_path, index=False)

    if build_panel:
        panel = _build_complete_panel(
            merged,
            start_year=resolved_start_year,
            end_year=resolved_end_year,
            anchor_weekday=anchor_weekday,
            enforce_anchor_assertion=bool(enforce_anchor_assertion),
        )
        panel_path = processed_dir / f"infodengue_chik_panel_{resolved_start_year}_{resolved_end_year}.parquet"
        panel.to_parquet(panel_path, index=False)

        epiclim_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(epiclim_path, index=False)
        first_output = panel
    else:
        first_output = merged

    merged_with_year = merged.assign(year=merged["date"].dt.year.astype("Int64"))
    coverage = merged_with_year.groupby(["state", "year"], dropna=True).size().reset_index(name="rows")
    observed_pairs = {
        (str(row.state), int(row.year))
        for row in coverage.itertuples(index=False)
        if row.year is not None
    }
    missing_uf_years = [
        {"uf": uf, "year": int(year)}
        for uf in selected_ufs
        for year in range(int(resolved_start_year), int(resolved_end_year) + 1)
        if (uf, int(year)) not in observed_pairs
    ]

    raw_chunk_file_count = int(
        sum(
            int((raw_jsonl_dir / f"infodengue_chik_{uf}_{chunk_start}_{chunk_end}.jsonl").exists())
            for uf in selected_ufs
            for chunk_start, chunk_end in chunk_ranges
        )
    )
    state_row_counts = {
        str(frame["state"].iloc[0]): int(len(frame))
        for frame in per_state_frames
        if not frame.empty and "state" in frame.columns
    }
    provenance_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "range": {"start_year": int(resolved_start_year), "end_year": int(resolved_end_year)},
        "api": {
            "base_url": resolved_base_url,
            "endpoint": resolved_endpoint,
            "disease": resolved_disease,
            "page_size": int(resolved_page_size),
            "api_max_per_page": int(api_max_per_page),
            "rate_limit_seconds": float(resolved_rate_limit_seconds),
            "timeout_seconds": float(resolved_timeout),
        },
        "panel": {
            "build_panel": bool(build_panel),
            "week_anchor": str(resolved_week_anchor).upper(),
            "anchor_weekday": int(anchor_weekday),
            "enforce_anchor_assertion": bool(enforce_anchor_assertion),
        },
        "fetch": {
            "selected_ufs": list(selected_ufs),
            "year_chunk_size": int(year_chunk_size),
            "max_workers": int(max_workers),
            "force_refresh": bool(force_refresh),
            "raw_chunk_file_count": int(raw_chunk_file_count),
            "chunk_row_counts": uf_chunk_row_counts,
        },
        "counts": {
            "rows_merged": int(len(merged)),
            "rows_output": int(len(first_output)),
            "rows_by_uf": state_row_counts,
            "missing_uf_year_count": int(len(missing_uf_years)),
        },
        "missing_uf_years": missing_uf_years,
        "artifacts": {
            "merged_parquet": str(merged_path),
            "epiclim_csv": str(epiclim_path),
        },
    }
    manifest_path = raw_dir / f"infodengue_chik_manifest_{resolved_start_year}_{resolved_end_year}.json"
    manifest_path.write_text(json.dumps(provenance_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved ingestion provenance manifest to %s", manifest_path)

    census_df = load_census_data(census_path, date_columns=date_columns, discovery_dir=discovery_dir)
    return first_output, census_df
