from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import time
from typing import Any

import pandas as pd
import requests


DEFAULT_INFODENGUE_API_BASE_URL = "https://api.mosqlimate.org"
DEFAULT_INFODENGUE_API_ENDPOINT = "/api/datastore/infodengue/"
DEFAULT_INFODENGUE_PROBE_UF = "RJ"
DEFAULT_INFODENGUE_PROBE_DISEASE = "chik"
DEFAULT_INFODENGUE_API_MAX_PER_PAGE = 100
DEFAULT_INFODENGUE_PROBE_MAX_PAGES = 2
DEFAULT_INFODENGUE_PROBE_TIMEOUT_SECONDS = 30.0
DEFAULT_INFODENGUE_PROBE_SAMPLE_ROWS = 200
DEFAULT_INFODENGUE_PROBE_WINDOW_DAYS = 14
DEFAULT_INFODENGUE_PROBE_PAGE_SLEEP_SECONDS = 0.2


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


def _env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned if cleaned else default


def _safe_positive_int(raw: Any, default: int) -> int:
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _env_positive_int(name: str, default: int) -> int:
    return _safe_positive_int(os.getenv(name), default)


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        return float(default)


def _env_optional_int(name: str) -> int | None:
    raw = _env_str(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _parse_iso_date(raw: str, field_name: str) -> datetime.date:
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: '{raw}'. Expected format YYYY-MM-DD.") from exc


def _resolve_page_limit(requested_limit: int | None) -> int:
    api_max_per_page = _env_positive_int("INFODENGUE_API_MAX_PER_PAGE", DEFAULT_INFODENGUE_API_MAX_PER_PAGE)
    configured_limit = _env_positive_int("INFODENGUE_API_PAGE_SIZE", api_max_per_page)
    chosen = configured_limit if requested_limit is None else _safe_positive_int(requested_limit, configured_limit)
    return min(chosen, api_max_per_page)


def _resolve_date_window(
    *,
    start_date_raw: str | None,
    end_date_raw: str | None,
    start_year: int | None,
    end_year: int | None,
    default_window_days: int,
) -> tuple[str, str, str]:
    if bool(start_date_raw) ^ bool(end_date_raw):
        raise ValueError("Provide both --start-date and --end-date, or neither.")

    if start_date_raw and end_date_raw:
        start_date = _parse_iso_date(start_date_raw, "--start-date")
        end_date = _parse_iso_date(end_date_raw, "--end-date")
        if start_date > end_date:
            raise ValueError("--start-date must be <= --end-date.")
        return start_date.isoformat(), end_date.isoformat(), "explicit_dates"

    if start_year is not None or end_year is not None:
        if start_year is None or end_year is None:
            raise ValueError("Provide both --start-year and --end-year when using year fallback.")
        if start_year > end_year:
            raise ValueError("--start-year must be <= --end-year.")
        return f"{start_year}-01-01", f"{end_year}-12-31", "year_fallback"

    effective_window_days = _safe_positive_int(default_window_days, DEFAULT_INFODENGUE_PROBE_WINDOW_DAYS)
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=effective_window_days - 1)
    return start_date.isoformat(), end_date.isoformat(), "default_short_window"


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


def _auth_headers() -> dict[str, str]:
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
        headers["Authorization"] = bearer if bearer.lower().startswith("bearer ") else f"Bearer {bearer}"
    elif token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _require_auth(headers: dict[str, str]) -> None:
    if headers.get("Authorization") or headers.get("X-UID-Key"):
        return
    raise RuntimeError(
        "Authentication not configured. Set credentials in .env or environment using "
        "INFODENGUE_BEARER / INFODENGUE_AUTHORIZATION or INFODENGUE_X_UID_KEY / X_UID_KEY."
    )


def _extract_items(payload: Any) -> tuple[list[dict[str, Any]], str | None, bool | None]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)], None, None

    if not isinstance(payload, dict):
        return [], None, False

    items: list[dict[str, Any]] = []
    for key in ("results", "items", "data", "records"):
        candidate = payload.get(key)
        if isinstance(candidate, list):
            items = [row for row in candidate if isinstance(row, dict)]
            break

    if not items and isinstance(payload.get("result"), list):
        items = [row for row in payload["result"] if isinstance(row, dict)]

    links = payload.get("links") if isinstance(payload.get("links"), dict) else {}
    next_url = payload.get("next") or payload.get("next_page") or links.get("next")

    has_more = payload.get("has_more")
    if has_more is None:
        has_more = payload.get("has_next")

    return items, str(next_url) if isinstance(next_url, str) and next_url else None, bool(has_more) if has_more is not None else None


def _extract_pagination_meta(payload: Any, response: requests.Response, requested_page_index: int, requested_per_page: int) -> dict[str, Any]:
    payload_page = None
    payload_per_page = None
    payload_total = None
    payload_total_pages = None

    if isinstance(payload, dict):
        payload_page = payload.get("page")
        payload_per_page = payload.get("per_page") or payload.get("page_size")
        payload_total = payload.get("total") or payload.get("count")
        payload_total_pages = payload.get("total_pages")

    return {
        "http_status": response.status_code,
        "requested_page_index": requested_page_index,
        "requested_per_page": requested_per_page,
        "payload_page": payload_page,
        "payload_per_page": payload_per_page,
        "payload_total": payload_total,
        "payload_total_pages": payload_total_pages,
        "header_link_present": bool(response.headers.get("Link")),
        "header_x_total_count": response.headers.get("X-Total-Count"),
    }


def _probe_pages(
    *,
    base_url: str,
    endpoint: str,
    uf: str,
    disease: str,
    start_date: str,
    end_date: str,
    limit: int,
    max_pages: int,
    timeout: float,
    page_sleep_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = requests.Session()
    headers = _auth_headers()
    _require_auth(headers)
    session.headers.update(headers)

    endpoint_normalized = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{base_url.rstrip('/')}{endpoint_normalized}"
    params: dict[str, Any] = {
        "uf": uf,
        "disease": disease,
        "start": start_date,
        "end": end_date,
        "page": 1,
        "per_page": limit,
    }

    all_rows: list[dict[str, Any]] = []
    page_summaries: list[dict[str, Any]] = []
    visited_next_urls: set[str] = set()
    stop_reason = "max_pages_reached"

    for page_index in range(max_pages):
        page_number = page_index + 1
        response = session.get(url, params=params, timeout=timeout)

        if response.status_code in {401, 403}:
            raise RuntimeError(
                "Authentication failed (HTTP 401/403). Verify token in .env or environment variables "
                "INFODENGUE_BEARER / INFODENGUE_AUTHORIZATION / INFODENGUE_X_UID_KEY / X_UID_KEY."
            )

        response.raise_for_status()
        payload = response.json()

        rows, next_url, has_more = _extract_items(payload)
        page_meta = _extract_pagination_meta(payload, response, page_number, limit)
        page_summaries.append(
            {
                "page": page_number,
                "rows": len(rows),
                "has_more": has_more,
                "next_url": next_url,
                "meta": page_meta,
            }
        )

        print(f"page={page_number} rows={len(rows)} has_more={has_more} next_url={'yes' if next_url else 'no'}")
        print("pagination_meta:", json.dumps(page_meta, ensure_ascii=False))

        if page_index == 0:
            preview_keys = sorted(rows[0].keys()) if rows else []
            print("available_fields:", ", ".join(preview_keys) if preview_keys else "<none>")
            if isinstance(payload, dict):
                print("top_level_payload_keys:", ", ".join(sorted(payload.keys())))

        if not rows:
            stop_reason = "empty_page"
            break

        all_rows.extend(rows)

        if next_url:
            if next_url in visited_next_urls:
                stop_reason = "pagination_loop_detected"
                print("pagination_validation=loop_detected next_url_repeated=true")
                break
            visited_next_urls.add(next_url)
            url = next_url
            params = {}
        else:
            if len(rows) < limit and has_more is None:
                stop_reason = "short_page_without_next_url"
                break
            params["page"] = int(params.get("page", 1)) + 1

        if has_more is False:
            stop_reason = "has_more_false"
            break

        if page_sleep_seconds > 0:
            time.sleep(page_sleep_seconds)

    print(f"pagination_stop_reason={stop_reason}")
    pagination_summary = {
        "stop_reason": stop_reason,
        "pages_fetched": len(page_summaries),
        "max_pages": max_pages,
        "pagination_validation": "ok" if stop_reason != "pagination_loop_detected" else "failed",
        "page_summaries": page_summaries,
    }
    return all_rows, pagination_summary


def main() -> None:
    _load_env_file()

    parser = argparse.ArgumentParser(description="Probe Infodengue API pagination and schema (small-scope defaults)")
    parser.add_argument("--base-url", default=_env_str("INFODENGUE_API_BASE_URL", DEFAULT_INFODENGUE_API_BASE_URL))
    parser.add_argument("--endpoint", default=_env_str("INFODENGUE_API_ENDPOINT", DEFAULT_INFODENGUE_API_ENDPOINT))
    parser.add_argument("--uf", default=_env_str("INFODENGUE_PROBE_UF", DEFAULT_INFODENGUE_PROBE_UF))
    parser.add_argument("--disease", default=_env_str("INFODENGUE_PROBE_DISEASE", DEFAULT_INFODENGUE_PROBE_DISEASE))

    parser.add_argument("--start-date", default=_env_str("INFODENGUE_PROBE_START_DATE"))
    parser.add_argument("--end-date", default=_env_str("INFODENGUE_PROBE_END_DATE"))
    parser.add_argument("--start-year", type=int, default=_env_optional_int("INFODENGUE_PROBE_START_YEAR"))
    parser.add_argument("--end-year", type=int, default=_env_optional_int("INFODENGUE_PROBE_END_YEAR"))

    parser.add_argument("--limit", type=int, default=_env_optional_int("INFODENGUE_API_PAGE_SIZE"))
    parser.add_argument("--max-pages", type=int, default=_env_positive_int("INFODENGUE_PROBE_MAX_PAGES", DEFAULT_INFODENGUE_PROBE_MAX_PAGES))
    parser.add_argument("--timeout", type=float, default=_env_float("INFODENGUE_PROBE_TIMEOUT", DEFAULT_INFODENGUE_PROBE_TIMEOUT_SECONDS))
    parser.add_argument("--sample-rows", type=int, default=_env_positive_int("INFODENGUE_PROBE_SAMPLE_ROWS", DEFAULT_INFODENGUE_PROBE_SAMPLE_ROWS))
    parser.add_argument(
        "--default-window-days",
        type=int,
        default=_env_positive_int("INFODENGUE_PROBE_DEFAULT_WINDOW_DAYS", DEFAULT_INFODENGUE_PROBE_WINDOW_DAYS),
        help="Used only when no explicit dates and no year fallback are provided.",
    )
    parser.add_argument(
        "--page-sleep-seconds",
        type=float,
        default=_env_float("INFODENGUE_PROBE_PAGE_SLEEP_SECONDS", DEFAULT_INFODENGUE_PROBE_PAGE_SLEEP_SECONDS),
    )

    args = parser.parse_args()

    effective_limit = _resolve_page_limit(args.limit)
    effective_max_pages = _safe_positive_int(args.max_pages, DEFAULT_INFODENGUE_PROBE_MAX_PAGES)
    effective_sample_rows = _safe_positive_int(args.sample_rows, DEFAULT_INFODENGUE_PROBE_SAMPLE_ROWS)

    start_date, end_date, date_source = _resolve_date_window(
        start_date_raw=args.start_date,
        end_date_raw=args.end_date,
        start_year=args.start_year,
        end_year=args.end_year,
        default_window_days=args.default_window_days,
    )

    print(
        f"probe_scope uf={str(args.uf).upper()} disease={str(args.disease)} start={start_date} end={end_date} source={date_source}"
    )
    print(
        f"probe_limits per_page={effective_limit} max_pages={effective_max_pages} sample_rows={effective_sample_rows}"
    )

    rows, pagination_summary = _probe_pages(
        base_url=str(args.base_url),
        endpoint=str(args.endpoint),
        uf=str(args.uf).upper(),
        disease=str(args.disease),
        start_date=start_date,
        end_date=end_date,
        limit=effective_limit,
        max_pages=effective_max_pages,
        timeout=float(args.timeout),
        page_sleep_seconds=max(0.0, float(args.page_sleep_seconds)),
    )

    output_dir = Path("data/external/probes")
    raw_output_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    raw_path = raw_output_dir / f"infodengue_probe_{str(args.uf).upper()}_{stamp}.jsonl"
    _write_jsonl_rows(rows, raw_path)

    raw_rows = _read_jsonl_rows(raw_path)
    frame = pd.DataFrame(raw_rows)
    print(f"total_rows_collected={len(frame)}")

    sample_size = min(len(frame), effective_sample_rows)
    sample_frame = frame.head(sample_size)
    out_path = output_dir / f"infodengue_probe_{str(args.uf).upper()}_{stamp}_sample.parquet"
    sample_frame.to_parquet(out_path, index=False)

    print(f"saved_probe_raw_jsonl={raw_path}")
    print(f"saved_probe_sample_parquet={out_path}")

    meta_path = output_dir / f"infodengue_probe_{str(args.uf).upper()}_{stamp}.json"
    meta_path.write_text(
        json.dumps(
            {
                "base_url": args.base_url,
                "endpoint": args.endpoint,
                "uf": str(args.uf).upper(),
                "disease": str(args.disease),
                "start_date": start_date,
                "end_date": end_date,
                "date_source": date_source,
                "effective_limit": int(effective_limit),
                "effective_max_pages": int(effective_max_pages),
                "rows": int(len(frame)),
                "sample_rows": int(sample_size),
                "raw_jsonl": str(raw_path),
                "sample_parquet": str(out_path),
                "pagination": pagination_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved_probe_meta={meta_path}")


if __name__ == "__main__":
    main()
