"""Spatial feature engineering helpers."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

LOGGER = logging.getLogger(__name__)


def _pick_coordinate_columns(columns: Sequence[str]) -> tuple[str | None, str | None]:
    lat_candidates = ("lat", "latitude", "district_lat", "centroid_lat")
    lon_candidates = ("lon", "longitude", "lng", "district_lon", "centroid_lon")
    lat_col = next((column for column in lat_candidates if column in columns), None)
    lon_col = next((column for column in lon_candidates if column in columns), None)
    return lat_col, lon_col


def _resolve_case_column(df: pd.DataFrame) -> str | None:
    for candidate in ("cases", "case_count", "weekly_cases"):
        if candidate in df.columns:
            return candidate
    return None


def _build_neighbor_map(
    centroid_table: pd.DataFrame,
    *,
    district_column: str,
    lat_column: str,
    lon_column: str,
    k_neighbors: int,
) -> dict[object, list[object]]:
    neighbor_map: dict[object, list[object]] = {}
    if len(centroid_table) <= 1:
        return neighbor_map

    coords = centroid_table[[lat_column, lon_column]].astype(float)
    districts = centroid_table[district_column]
    for idx, district in districts.items():
        deltas = coords - coords.loc[idx]
        distances = (deltas[lat_column] ** 2 + deltas[lon_column] ** 2) ** 0.5
        ordered = distances.sort_values()
        nearest_idx = [row_idx for row_idx in ordered.index if row_idx != idx][:k_neighbors]
        neighbor_map[district] = districts.loc[nearest_idx].tolist()
    return neighbor_map


def build_spatial_features(
    df: pd.DataFrame,
    *,
    district_column: str = "district",
    date_column: str = "date",
    k_neighbors: int = 3,
) -> pd.DataFrame:
    """Build spatial proxy features without shapefile dependencies.

    Uses district centroids (lat/lon) when available to derive nearest-neighbor
    districts and compute ``neighbor_mean_cases`` per date.

    If coordinates are unavailable, this function performs a safe no-op for
    neighbor features and logs warnings.
    """
    LOGGER.info("Building spatial features")
    output = df.copy()

    if district_column in output.columns:
        output["district_code"] = output[district_column].astype("category").cat.codes
    else:
        LOGGER.warning("District column '%s' missing; spatial features limited", district_column)
        return output

    lat_column, lon_column = _pick_coordinate_columns(output.columns)
    case_column = _resolve_case_column(output)

    if lat_column is None or lon_column is None:
        LOGGER.warning("Latitude/longitude columns not found; skipping neighbor_mean_cases")
        return output
    if case_column is None:
        LOGGER.warning("Case column not found; skipping neighbor_mean_cases")
        return output

    centroid_table = (
        output[[district_column, lat_column, lon_column]]
        .dropna(subset=[district_column, lat_column, lon_column])
        .groupby(district_column, as_index=False)
        .agg({lat_column: "mean", lon_column: "mean"})
    )
    if len(centroid_table) < 2:
        LOGGER.warning("Insufficient centroid coverage for neighbors; skipping neighbor_mean_cases")
        return output

    neighbor_map = _build_neighbor_map(
        centroid_table,
        district_column=district_column,
        lat_column=lat_column,
        lon_column=lon_column,
        k_neighbors=max(1, int(k_neighbors)),
    )
    if not neighbor_map:
        LOGGER.warning("Neighbor map empty; skipping neighbor_mean_cases")
        return output

    has_date_column = date_column in output.columns
    if has_date_column:
        output[date_column] = pd.to_datetime(output[date_column], errors="coerce")
        date_series = output[date_column]
    else:
        LOGGER.warning("Date column '%s' missing; using global-date proxy for neighbors", date_column)
        date_series = pd.Series("all_dates", index=output.index)

    temp_df = pd.DataFrame(
        {
            "_district": output[district_column],
            "_date": date_series,
            "_cases": pd.to_numeric(output[case_column], errors="coerce"),
        },
        index=output.index,
    )
    neighbor_means: list[float | None] = []

    if has_date_column:
        lagged_cases = (
            temp_df.groupby(["_district", "_date"], dropna=False)["_cases"]
            .mean()
            .reset_index()
            .sort_values(["_district", "_date"])
        )
        lagged_cases["_cases_lag_1"] = lagged_cases.groupby("_district", dropna=False)["_cases"].shift(1)
        lagged_lookup = lagged_cases.set_index(["_date", "_district"])["_cases_lag_1"]

        for _, row in temp_df.iterrows():
            district_value = row["_district"]
            date_value = row["_date"]
            neighbors = neighbor_map.get(district_value, [])
            if not neighbors:
                neighbor_means.append(None)
                continue
            values: list[float] = []
            for neighbor in neighbors:
                if (date_value, neighbor) in lagged_lookup.index:
                    value = lagged_lookup.loc[(date_value, neighbor)]
                    if pd.notna(value):
                        values.append(float(value))
            neighbor_means.append(float(sum(values) / len(values)) if values else None)
    else:
        latest_neighbor_cases: dict[object, float] = {}
        for _, row in temp_df.iterrows():
            district_value = row["_district"]
            neighbors = neighbor_map.get(district_value, [])
            if not neighbors:
                neighbor_means.append(None)
            else:
                values = [latest_neighbor_cases[n] for n in neighbors if n in latest_neighbor_cases]
                neighbor_means.append(float(sum(values) / len(values)) if values else None)

            current_cases = row["_cases"]
            if pd.notna(current_cases):
                latest_neighbor_cases[district_value] = float(current_cases)

    output["neighbor_mean_cases"] = pd.Series(neighbor_means, index=output.index, dtype="Float64")
    return output


def run(
    df: pd.DataFrame,
    *,
    district_column: str = "district",
    date_column: str = "date",
    k_neighbors: int = 3,
) -> pd.DataFrame:
    """Entrypoint for spatial feature generation."""
    return build_spatial_features(
        df,
        district_column=district_column,
        date_column=date_column,
        k_neighbors=k_neighbors,
    )
