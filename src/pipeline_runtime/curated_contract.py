"""Curated municipality contract loading and filtering helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_curated_municipality_contract(contract_path: Path) -> dict[str, Any]:
    if not contract_path.exists():
        raise FileNotFoundError(
            f"Curated municipality contract not found at {contract_path}. "
            "Pipeline requires a frozen, versioned municipality list."
        )

    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Curated municipality contract at {contract_path} must be a versioned object schema, "
            f"not {type(payload).__name__}"
        )

    required_keys = {"version", "source", "municipality_ids"}
    optional_keys = {"selection"}
    allowed_keys = required_keys.union(optional_keys)
    unknown_keys = sorted(set(payload.keys()).difference(allowed_keys))
    if unknown_keys:
        raise ValueError(
            f"Curated municipality contract {contract_path} has unknown keys: {unknown_keys}. "
            f"Allowed keys: {sorted(allowed_keys)}"
        )
    missing_keys = sorted(required_keys.difference(payload.keys()))
    if missing_keys:
        raise ValueError(
            f"Curated municipality contract {contract_path} missing required keys: {missing_keys}"
        )

    raw_ids = payload.get("municipality_ids")
    if not isinstance(raw_ids, list):
        raise ValueError(
            f"Curated municipality contract {contract_path} must contain list field 'municipality_ids'"
        )
    municipality_ids = [str(value).strip() for value in raw_ids if str(value).strip()]
    version = str(payload.get("version", "")).strip()
    source = str(payload.get("source", "")).strip()
    selection_raw = payload.get("selection", None)
    selection = None if selection_raw is None else str(selection_raw).strip()
    if not version:
        raise ValueError(f"Curated municipality contract {contract_path} has empty required field 'version'")
    if not source:
        raise ValueError(f"Curated municipality contract {contract_path} has empty required field 'source'")
    if selection_raw is not None and not selection:
        raise ValueError(f"Curated municipality contract {contract_path} has empty optional field 'selection'")

    unique_ids = sorted(set(municipality_ids))
    if not unique_ids:
        raise ValueError(f"Curated municipality contract at {contract_path} has no municipality IDs")

    curated_contract = {
        "path": contract_path,
        "version": version,
        "source": source,
        "count": int(len(unique_ids)),
        "municipality_ids": unique_ids,
        "sha256": _sha256_file(contract_path),
    }
    if selection is not None:
        curated_contract["selection"] = selection
    return curated_contract


def resolve_municipality_column(df: pd.DataFrame) -> str:
    for candidate in ("municipality_id", "district"):
        if candidate in df.columns:
            return candidate
    raise ValueError("Municipality filtering requires 'municipality_id' or 'district' column")


def apply_curated_municipality_filter(
    df: pd.DataFrame,
    *,
    curated_ids: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    municipality_column = resolve_municipality_column(df)
    id_series = df[municipality_column].astype("string").fillna("").str.strip()
    keep_mask = id_series.isin(curated_ids)
    filtered = df.loc[keep_mask].copy()
    report = {
        "municipality_column": municipality_column,
        "rows_before": int(len(df)),
        "rows_after": int(len(filtered)),
        "districts_before": int(df.get("district", pd.Series(dtype=object)).nunique(dropna=True)),
        "districts_after": int(filtered.get("district", pd.Series(dtype=object)).nunique(dropna=True)),
    }
    return filtered, report