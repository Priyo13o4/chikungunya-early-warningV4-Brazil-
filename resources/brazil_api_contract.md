# Brazil Infodengue API Contract (Phase 0)

## Scope and Sources

This contract documents the API behavior used by the Brazil adapter/probe in this repository and reconciles it with thesis/PRD references:

- `projects/brazil_chik/data_adapter.py`
- `scripts/brazil/01_probe_api.py`
- `resources/Revised-Thesis-Chik-Brazil.md`
- `resources/PRD-Chik-EWS-Brazil.md`

Primary target API:

- Base host: `https://api.mosqlimate.org`
- Endpoint path: `/api/datastore/infodengue/`

Runtime resolution used by code:

- `INFODENGUE_API_BASE_URL` (fallback `https://api.mosqlimate.org`)
- `INFODENGUE_API_ENDPOINT` (fallback `/api/datastore/infodengue/`)

## Endpoint(s)

### 1) Infodengue datastore endpoint (used for ingestion)

`GET {INFODENGUE_API_BASE_URL}{INFODENGUE_API_ENDPOINT}`

Typical resolved URL:

`GET https://api.mosqlimate.org/api/datastore/infodengue/`

### 2) Follow-up pagination URL (if provided by API)

If response includes `next`/`next_page`/`links.next`, the client follows that URL directly.

## Authentication contract

Headers sent by adapter/probe:

- Always: `Accept: application/json`
- Optional key header: `X-UID-Key: <token>`
- Authorization header:
  - `Authorization: Bearer <token>` when bearer/token available
  - if bearer value already starts with `Bearer `, it is used as-is

Credential env lookup order:

- Token candidates: `INFODENGUE_X_UID_KEY`, `X_UID_KEY`, `INFODENGUE_TOKEN`, `INFODENGUE_API_TOKEN`
- Bearer candidates: `INFODENGUE_BEARER`, `INFODENGUE_AUTHORIZATION`

## Query parameters

### Required in current ingestion behavior

- `disease` (defaults to `chik`, configurable via env/adapter args)
- `start` (`YYYY-01-01` for each requested chunk)
- `end` (`YYYY-12-31` for each requested chunk)
- `page` (starts at `1`)
- `per_page` (resolved/clamped; see pagination section)

### Optional in current ingestion behavior

- `uf` (adapter/probe always sets UF in current design; API itself can support broader queries)

### Not used by current adapter/probe (but documented in thesis/PRD)

- `geocode`

## Pagination behavior and limits

Client behavior implemented in both adapter and probe:

1. Start with `page=1` and `per_page=<resolved_page_size>`.
2. Parse rows from one of: `results`, `items`, `data`, `records`, fallback `result`.
3. Determine continuation using any available signal:
   - `next` / `next_page` / `links.next`
   - `has_more` / `has_next`
   - adapter additionally infers from `pagination.page < pagination.total_pages` if present.
4. If `next` URL exists, request it directly.
5. Else increment `page` by 1.
6. Stop when rows are empty, or `has_more/has_next` is explicitly false.

### Limits and conflicts

- Thesis/PRD references commonly state `per_page` max `100`.
- Historical code had larger defaults (`500` adapter, `200` probe).
- **Current code now enforces safe clamping**:
  - `INFODENGUE_API_MAX_PER_PAGE` (default `100`)
  - requested page size is clamped to this maximum
  - default page size source: `INFODENGUE_API_PAGE_SIZE` (fallback to max)

Result: behavior is now aligned with the documented max-100 expectation while remaining configurable.

## Response schema (observed/handled contract)

The API may return either:

- A top-level list of records, or
- An object containing records under one of `results|items|data|records|result`, optionally with pagination metadata.

### Common raw fields consumed/recognized

- Geography:
  - `uf` / `estado`
  - `municipio` / `municipio_nome` / `mun_name`
  - `municipio_geocodigo` / `geocode` / `id_municipio`
- Time:
  - `data_iniSE` (or variants normalized to date)
  - `SE` (`YYYYWW` epidemiological week)
  - optional fallback pairs: `ano+se`, `year+week`
- Epidemiology/signals:
  - `casos`, `casos_est`
  - `Rt`, `p_rt1`, `receptivo`, `pop`

### Normalized adapter output schema (core columns)

After normalization (`_normalize_columns`), downstream frames use:

- Always present:
  - `state` (string)
  - `district` (string)
  - `municipality_id` (string)
  - `date` (`datetime64[ns]`, may be dropped if invalid)
  - `cases` (float, non-negative; filled to `0.0` when absent/non-numeric)
- Optional numeric columns when available in source:
  - `rt`
  - `p_rt1`
  - `receptivo`
  - `pop`

## Field meanings (operational)

- `cases`: normalized case count used by pipeline (`casos_est` preferred, else `casos`).
- `municipality_id`: IBGE municipality identifier from source aliases.
- `date`: week anchor date (aligned to Sunday-like weekly anchor for stable panel merge).
- `rt`: effective reproduction number estimate.
- `p_rt1`: probability that `Rt > 1`.
- `receptivo`: climate receptivity signal from Infodengue.
- `pop`: population value when present.

## Always-present vs optional/null

### Raw API payload

- No strict guarantee of all keys in all records.
- Container shape can vary (list vs object with nested rows).
- Pagination keys may be missing.

### Adapter-normalized frame

- Always present by construction: `state`, `district`, `municipality_id`, `date`, `cases`.
- Optional and nullable: `rt`, `p_rt1`, `receptivo`, `pop`.
- Invalid or absent dates are coerced to `NaT` and filtered before year filtering/panel build.

## Pitfalls and mitigation rules used in code

1. **Payload shape drift (list vs object, different row containers)**
   - Mitigation: multi-key extraction (`results/items/data/records/result`) with defensive fallback.

2. **Pagination metadata inconsistency**
   - Mitigation: support `next` URL, `has_more/has_next`, and `pagination.page/total_pages` inference (adapter).

3. **Date/week representation variability**
   - Mitigation: map multiple date aliases and derive date from `ano+se` or `year+week` when needed.

4. **Week anchor mismatch causing merge sparsity**
   - Mitigation: explicit weekly anchor normalization (`_align_to_sunday_week_anchor`) before panel merge.

5. **Schema alias drift in municipality/state names**
   - Mitigation: alias mapping for `uf/estado`, `municipio/municipio_nome/mun_name`, `municipio_geocodigo/geocode/id_municipio`.

6. **Missing or malformed numeric fields**
   - Mitigation: `to_numeric(..., errors='coerce')`; `cases` filled and clipped to non-negative.

7. **Duplicate municipality-week records across chunks/pages**
   - Mitigation: deduplication at UF level and yearly merged outputs (`drop_duplicates(..., keep='last')`).

8. **Overly large page requests vs API limits**
   - Mitigation: env-configurable max page size clamp (`INFODENGUE_API_MAX_PER_PAGE`, default `100`).

9. **Chunk-level API failure causing partial UF loads**
   - Mitigation: skip UF when chunk fetch fails; cached chunk JSONL supports resumable reruns.

## Environment controls (current)

- Year window defaults (adapter):
  - `INFODENGUE_START_YEAR` / `BRAZIL_CHIK_START_YEAR` (default `2015`)
  - `INFODENGUE_END_YEAR` / `BRAZIL_CHIK_END_YEAR` (default `2025`)
- API host/path:
  - `INFODENGUE_API_BASE_URL`
  - `INFODENGUE_API_ENDPOINT`
- API query tuning:
  - `INFODENGUE_API_DISEASE` (default `chik`)
  - `INFODENGUE_RATE_LIMIT_SECONDS` (default `0.25`)
  - `INFODENGUE_API_TIMEOUT` (default `45`)
- Pagination limits:
  - `INFODENGUE_API_MAX_PER_PAGE` (default `100`)
  - `INFODENGUE_API_PAGE_SIZE` (default falls back to max)
- Weekly panel anchor:
  - `INFODENGUE_WEEK_ANCHOR` (one of `MON..SUN`, default `SUN`)
- Auth:
  - `INFODENGUE_X_UID_KEY`, `X_UID_KEY`, `INFODENGUE_TOKEN`, `INFODENGUE_API_TOKEN`
  - `INFODENGUE_BEARER`, `INFODENGUE_AUTHORIZATION`

## Provenance manifest artifact

For each adapter run range, a lightweight ingestion manifest is written to:

- `data/raw/infodengue_chik_manifest_<start_year>_<end_year>.json`

Manifest fields include:

- effective year range and API settings,
- selected UFs and chunk fetch settings,
- merged/output row counts,
- missing UF-year pairs,
- key output artifact paths.

## Alignment note (Phase 0 result)

Contract and implementation are aligned after applying a minimal safe patch:

- page-size defaults are no longer hardcoded above documented limits;
- both adapter and probe now apply env-driven max clamping;
- architecture and dataflow remain unchanged.
