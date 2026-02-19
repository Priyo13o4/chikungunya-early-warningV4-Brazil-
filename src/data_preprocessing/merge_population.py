"""Population merge helpers."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import pandas as pd

LOGGER = logging.getLogger(__name__)

DISTRICT_COLUMN_ALIASES: tuple[str, ...] = (
    "district",
    "district_name",
    "dist_name",
    "dist",
    "dt_name",
    "name",
)
STATE_COLUMN_ALIASES: tuple[str, ...] = (
    "state",
    "state_name",
    "st_name",
    "province",
    "region",
)
LEVEL_COLUMN_ALIASES: tuple[str, ...] = ("level", "admin_level")
NAME_COLUMN_ALIASES: tuple[str, ...] = ("name", "district_name", "area_name")
TRU_COLUMN_ALIASES: tuple[str, ...] = ("tru", "area_type", "settlement_type")
POPULATION_COLUMN_ALIASES: tuple[str, ...] = (
    "population",
    "total_population",
    "tot_population",
    "tot_p",
    "pop_total",
    "persons",
    "persons_total",
)

DISTRICT_LEVEL_TOKENS: tuple[str, ...] = ("district", "dist")
STATE_LEVEL_TOKENS: tuple[str, ...] = ("state", "ut", "stateut")
TOTAL_LEVEL_TOKENS: tuple[str, ...] = ("total", "all")

DISTRICT_SUFFIX_PATTERN = re.compile(r"\b(district|dist|distt|dt)\b$", re.IGNORECASE)
TOKEN_STOPWORDS = {"district", "dist", "distt", "dt", "the", "of"}


def _normalize_identifier(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def _column_has_textual_values(series: pd.Series, sample_size: int = 250) -> bool:
    sample = series.dropna().astype(str).head(sample_size)
    if sample.empty:
        return False
    alpha_ratio = sample.map(lambda value: bool(re.search(r"[a-zA-Z]", value))).mean()
    return bool(alpha_ratio >= 0.5)


def _find_best_column(
    df: pd.DataFrame,
    aliases: tuple[str, ...],
    *,
    require_textual_values: bool,
    excluded_columns: set[str] | None = None,
) -> str | None:
    excluded = excluded_columns or set()
    if df.empty:
        return None

    alias_ids = [_normalize_identifier(alias) for alias in aliases]
    candidates: list[tuple[int, str]] = []
    for column in df.columns:
        if column in excluded:
            continue
        column_id = _normalize_identifier(column)
        score = 0
        for alias_id in alias_ids:
            if not alias_id:
                continue
            if column_id == alias_id:
                score = max(score, 100)
            elif column_id.startswith(alias_id):
                score = max(score, 90)
            elif alias_id in column_id:
                score = max(score, 80)

        if score == 0:
            continue
        if require_textual_values and not _column_has_textual_values(df[column]):
            continue
        candidates.append((score, str(column)))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1]))
    return candidates[0][1]


def _resolve_main_columns(
    main_df: pd.DataFrame,
    district_column: str,
    state_column: str | None,
) -> tuple[str, str | None]:
    resolved_district = district_column if district_column in main_df.columns else None
    if resolved_district is None:
        resolved_district = _find_best_column(
            main_df,
            DISTRICT_COLUMN_ALIASES,
            require_textual_values=True,
        )
    if resolved_district is None:
        raise ValueError(
            f"Unable to resolve district column in main dataframe. Available columns: {list(main_df.columns)}"
        )

    resolved_state: str | None = None
    if state_column and state_column in main_df.columns:
        resolved_state = state_column
    elif state_column is not None:
        resolved_state = _find_best_column(
            main_df,
            STATE_COLUMN_ALIASES,
            require_textual_values=True,
            excluded_columns={resolved_district},
        )

    LOGGER.info(
        "Main dataframe merge columns selected: district='%s', state='%s'",
        resolved_district,
        resolved_state,
    )
    return resolved_district, resolved_state


def _select_population_metric_columns(population_df: pd.DataFrame) -> list[str]:
    alias_ids = [_normalize_identifier(alias) for alias in POPULATION_COLUMN_ALIASES]
    selected: list[str] = []
    for column in population_df.columns:
        column_id = _normalize_identifier(column)
        if any(alias_id and alias_id in column_id for alias_id in alias_ids):
            selected.append(str(column))

    if selected:
        return list(dict.fromkeys(selected))

    fallback_columns = [
        str(column)
        for column in population_df.columns
        if _normalize_identifier(str(column)).startswith("tot")
    ]
    return fallback_columns[:1]


def _prepare_population_canonical_frame(
    population_df: pd.DataFrame,
    *,
    requested_district_column: str,
    requested_state_column: str | None,
) -> tuple[pd.DataFrame, str, str | None, list[str], dict[str, str | None]]:
    level_column = _find_best_column(
        population_df,
        LEVEL_COLUMN_ALIASES,
        require_textual_values=True,
    )
    name_column = _find_best_column(
        population_df,
        NAME_COLUMN_ALIASES,
        require_textual_values=True,
    )
    tru_column = _find_best_column(
        population_df,
        TRU_COLUMN_ALIASES,
        require_textual_values=True,
    )

    direct_district_column = (
        requested_district_column
        if requested_district_column in population_df.columns and _column_has_textual_values(population_df[requested_district_column])
        else _find_best_column(
            population_df,
            DISTRICT_COLUMN_ALIASES,
            require_textual_values=True,
        )
    )
    direct_state_column = None
    if requested_state_column and requested_state_column in population_df.columns:
        if _column_has_textual_values(population_df[requested_state_column]):
            direct_state_column = requested_state_column
    if direct_state_column is None:
        direct_state_column = _find_best_column(
            population_df,
            STATE_COLUMN_ALIASES,
            require_textual_values=True,
            excluded_columns={direct_district_column} if direct_district_column else None,
        )

    metric_columns = _select_population_metric_columns(population_df)
    if not metric_columns:
        LOGGER.warning("No plausible population metric columns detected; merge will proceed with key-only diagnostics")

    selected_columns: dict[str, str | None] = {
        "population_level_column": level_column,
        "population_name_column": name_column,
        "population_tru_column": tru_column,
        "population_district_column": direct_district_column,
        "population_state_column": direct_state_column,
    }

    if level_column and name_column:
        level_normalized = population_df[level_column].map(normalize_text_key)
        district_mask = level_normalized.map(
            lambda value: any(token in value for token in DISTRICT_LEVEL_TOKENS)
        )
        state_mask = level_normalized.map(
            lambda value: any(token in value for token in STATE_LEVEL_TOKENS)
        )

        if district_mask.any():
            district_rows = population_df.loc[district_mask].copy()

            if tru_column and tru_column in district_rows.columns:
                tru_normalized = district_rows[tru_column].map(normalize_text_key)
                total_mask = tru_normalized.map(
                    lambda value: any(token in value for token in TOTAL_LEVEL_TOKENS)
                )
                if total_mask.any():
                    district_rows = district_rows.loc[total_mask].copy()

            district_rows["_population_district"] = district_rows[name_column]
            population_state_output_column: str | None = None

            code_state_column = requested_state_column if requested_state_column in population_df.columns else "state"
            if (
                code_state_column in district_rows.columns
                and state_mask.any()
            ):
                state_rows = population_df.loc[state_mask].copy()
                if tru_column and tru_column in state_rows.columns:
                    state_tru_normalized = state_rows[tru_column].map(normalize_text_key)
                    state_total_mask = state_tru_normalized.map(
                        lambda value: any(token in value for token in TOTAL_LEVEL_TOKENS)
                    )
                    if state_total_mask.any():
                        state_rows = state_rows.loc[state_total_mask].copy()

                state_rows = state_rows.dropna(subset=[code_state_column, name_column])
                state_map = (
                    state_rows.drop_duplicates(subset=[code_state_column])
                    .set_index(code_state_column)[name_column]
                    .to_dict()
                )
                district_rows["_population_state"] = district_rows[code_state_column].map(state_map)

                if district_rows["_population_state"].notna().any():
                    population_state_output_column = "_population_state"

            if population_state_output_column is None and direct_state_column is not None:
                district_rows["_population_state"] = district_rows[direct_state_column]
                if district_rows["_population_state"].notna().any():
                    population_state_output_column = "_population_state"

            selected_columns["population_district_column"] = f"{name_column} (from level=district)"
            if population_state_output_column:
                selected_columns["population_state_column"] = f"{code_state_column if 'code_state_column' in locals() else direct_state_column} -> {name_column}"

            payload_columns = [column for column in metric_columns if column in district_rows.columns]
            return (
                district_rows,
                "_population_district",
                population_state_output_column,
                payload_columns,
                selected_columns,
            )

    resolved_district = direct_district_column or requested_district_column
    if resolved_district not in population_df.columns:
        raise ValueError(
            "Unable to resolve district column in population dataframe. "
            f"Available columns: {list(population_df.columns)}"
        )

    resolved_state = direct_state_column
    payload_columns = [column for column in metric_columns if column in population_df.columns]
    return population_df.copy(), resolved_district, resolved_state, payload_columns, selected_columns


def normalize_text_key(value: Any) -> str:
    """Normalize text into a merge-safe key.

    The normalization is conservative: accents and punctuation are removed,
    whitespace collapsed, and text lower-cased.
    """
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = DISTRICT_SUFFIX_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_token_key(value: Any) -> str:
    base = normalize_text_key(value)
    if not base:
        return ""
    tokens = [token for token in base.split() if token and token not in TOKEN_STOPWORDS]
    return " ".join(sorted(set(tokens)))


def _prepare_merge_keys(
    df: pd.DataFrame,
    *,
    district_column: str,
    state_column: str | None,
) -> pd.DataFrame:
    """Build normalized merge keys for district and optional state."""
    output = df.copy()
    if district_column not in output.columns:
        raise ValueError(f"Missing district column: {district_column}")
    output["_district_key"] = output[district_column].map(normalize_text_key)
    if state_column and state_column in output.columns:
        output["_state_key"] = output[state_column].map(normalize_text_key)
    else:
        output["_state_key"] = ""
    output["_merge_key"] = output["_state_key"] + "|" + output["_district_key"]
    return output


def _coverage_mask(merged: pd.DataFrame, population_columns_added: list[str]) -> pd.Series:
    if not population_columns_added:
        return pd.Series(False, index=merged.index, dtype=bool)
    return merged[population_columns_added].notna().any(axis=1)


def _log_key_counts(main_with_keys: pd.DataFrame, pop_with_keys: pd.DataFrame) -> None:
    LOGGER.info(
        "Merge key diagnostics: main_unique_state_district=%d main_unique_district=%d pop_unique_state_district=%d pop_unique_district=%d",
        int(main_with_keys["_merge_key"].nunique(dropna=True)),
        int(main_with_keys["_district_key"].nunique(dropna=True)),
        int(pop_with_keys["_merge_key"].nunique(dropna=True)),
        int(pop_with_keys["_district_key"].nunique(dropna=True)),
    )


def _fill_with_fallback(
    merged: pd.DataFrame,
    fallback_df: pd.DataFrame,
    *,
    on_column: str,
    population_payload_cols: list[str],
    suffix: str,
) -> pd.DataFrame:
    if not population_payload_cols or fallback_df.empty:
        return merged

    fallback_merge = merged.merge(
        fallback_df[[on_column, *population_payload_cols]],
        on=on_column,
        how="left",
        suffixes=("", suffix),
    )

    for column in population_payload_cols:
        fallback_column = f"{column}{suffix}"
        if fallback_column in fallback_merge.columns:
            fallback_merge[column] = fallback_merge[column].combine_first(fallback_merge[fallback_column])
            fallback_merge = fallback_merge.drop(columns=[fallback_column])

    return fallback_merge


def _log_merge_diagnostics(
    main_rows: int,
    merged: pd.DataFrame,
    population_columns_added: list[str],
) -> None:
    """Emit diagnostics for merge quality and coverage."""
    if len(merged) != main_rows:
        LOGGER.warning("Row count changed during left merge: %d -> %d", main_rows, len(merged))
    if not population_columns_added:
        LOGGER.info("No population columns were added")
        return

    population_non_null = _coverage_mask(merged, population_columns_added)
    coverage = float(population_non_null.mean()) if len(merged) else 0.0
    LOGGER.info(
        "Population merge coverage: %.2f%% (%d/%d rows)",
        coverage * 100.0,
        int(population_non_null.sum()),
        len(merged),
    )


def merge_population(
    main_df: pd.DataFrame,
    population_df: pd.DataFrame,
    *,
    district_column: str = "district",
    state_column: str | None = "state",
) -> pd.DataFrame:
    """Merge census/population attributes into epidemiological records.

    Uses normalized text keys (district + optional state) and a safe left join.
    """
    LOGGER.info("Merging population data: main_shape=%s population_shape=%s", main_df.shape, population_df.shape)
    resolved_main_district, resolved_main_state = _resolve_main_columns(main_df, district_column, state_column)
    (
        canonical_population_df,
        resolved_population_district,
        resolved_population_state,
        population_payload_cols,
        selected_population_columns,
    ) = _prepare_population_canonical_frame(
        population_df,
        requested_district_column=district_column,
        requested_state_column=state_column,
    )

    LOGGER.info(
        "Population dataframe columns selected: district='%s', state='%s', metrics=%s, level='%s', name='%s', tru='%s'",
        selected_population_columns.get("population_district_column") or resolved_population_district,
        selected_population_columns.get("population_state_column") or resolved_population_state,
        population_payload_cols,
        selected_population_columns.get("population_level_column"),
        selected_population_columns.get("population_name_column"),
        selected_population_columns.get("population_tru_column"),
    )

    main_with_keys = _prepare_merge_keys(
        main_df,
        district_column=resolved_main_district,
        state_column=resolved_main_state,
    )
    pop_with_keys = _prepare_merge_keys(
        canonical_population_df,
        district_column=resolved_population_district,
        state_column=resolved_population_state,
    )

    _log_key_counts(main_with_keys, pop_with_keys)

    dedup_population = pop_with_keys.drop_duplicates(subset=["_merge_key"])
    merged = main_with_keys.merge(
        dedup_population[["_merge_key", *population_payload_cols]],
        on="_merge_key",
        how="left",
        validate="many_to_one",
        suffixes=("", "_population"),
    )

    primary_coverage_mask = _coverage_mask(merged, population_payload_cols)
    primary_coverage = float(primary_coverage_mask.mean()) if len(merged) else 0.0
    LOGGER.info(
        "Population merge primary coverage (state+district): %.2f%% (%d/%d rows)",
        primary_coverage * 100.0,
        int(primary_coverage_mask.sum()),
        len(merged),
    )

    unmatched_mask = ~primary_coverage_mask
    if unmatched_mask.any() and population_payload_cols:
        district_key_unique_counts = pop_with_keys.groupby("_district_key")["_merge_key"].nunique(dropna=True)
        unique_district_keys = district_key_unique_counts[district_key_unique_counts == 1].index
        district_unique_population = (
            pop_with_keys[pop_with_keys["_district_key"].isin(unique_district_keys)]
            .drop_duplicates(subset=["_district_key"])
        )
        if not district_unique_population.empty:
            merged = _fill_with_fallback(
                merged,
                district_unique_population,
                on_column="_district_key",
                population_payload_cols=population_payload_cols,
                suffix="_district_fallback",
            )
            district_fallback_coverage_mask = _coverage_mask(merged, population_payload_cols)
            district_fallback_coverage = float(district_fallback_coverage_mask.mean()) if len(merged) else 0.0
            LOGGER.info(
                "Population merge coverage after district-only unique fallback: %.2f%% (%d/%d rows)",
                district_fallback_coverage * 100.0,
                int(district_fallback_coverage_mask.sum()),
                len(merged),
            )

    post_fallback_coverage_mask = _coverage_mask(merged, population_payload_cols)
    post_fallback_coverage = float(post_fallback_coverage_mask.mean()) if len(merged) else 0.0

    if post_fallback_coverage <= 0.0 and population_payload_cols:
        LOGGER.info(
            "Primary and district-only fallback coverage is 0%%; running token-normalized exact-key fallback"
        )
        token_main = merged.copy()
        token_population = pop_with_keys.copy()

        token_main["_district_token_key"] = token_main[resolved_main_district].map(_normalize_token_key)
        token_main["_state_token_key"] = (
            token_main[resolved_main_state].map(_normalize_token_key) if resolved_main_state else ""
        )
        token_main["_token_merge_key"] = token_main["_state_token_key"] + "|" + token_main["_district_token_key"]

        token_population["_district_token_key"] = token_population[resolved_population_district].map(_normalize_token_key)
        token_population["_state_token_key"] = (
            token_population[resolved_population_state].map(_normalize_token_key) if resolved_population_state else ""
        )
        token_population["_token_merge_key"] = (
            token_population["_state_token_key"] + "|" + token_population["_district_token_key"]
        )

        token_population_unique = token_population.drop_duplicates(subset=["_token_merge_key"])
        token_main = _fill_with_fallback(
            token_main,
            token_population_unique,
            on_column="_token_merge_key",
            population_payload_cols=population_payload_cols,
            suffix="_token_fallback",
        )

        token_district_counts = token_population.groupby("_district_token_key")["_token_merge_key"].nunique(dropna=True)
        token_unique_district_keys = token_district_counts[token_district_counts == 1].index
        token_unique_district_population = token_population[
            token_population["_district_token_key"].isin(token_unique_district_keys)
        ].drop_duplicates(subset=["_district_token_key"])

        token_main = _fill_with_fallback(
            token_main,
            token_unique_district_population,
            on_column="_district_token_key",
            population_payload_cols=population_payload_cols,
            suffix="_token_district_fallback",
        )
        merged = token_main.drop(columns=["_district_token_key", "_state_token_key", "_token_merge_key"])
        token_coverage_mask = _coverage_mask(merged, population_payload_cols)
        token_coverage = float(token_coverage_mask.mean()) if len(merged) else 0.0
        LOGGER.info(
            "Population merge coverage after token-normalized fallback: %.2f%% (%d/%d rows)",
            token_coverage * 100.0,
            int(token_coverage_mask.sum()),
            len(merged),
        )

    _log_merge_diagnostics(
        main_rows=len(main_df),
        merged=merged,
        population_columns_added=population_payload_cols,
    )

    return merged.drop(columns=["_district_key", "_state_key", "_merge_key"])


def run(
    main_df: pd.DataFrame,
    population_df: pd.DataFrame,
    *,
    district_column: str = "district",
    state_column: str | None = "state",
) -> pd.DataFrame:
    """Entrypoint for the population merge phase."""
    return merge_population(
        main_df,
        population_df,
        district_column=district_column,
        state_column=state_column,
    )
