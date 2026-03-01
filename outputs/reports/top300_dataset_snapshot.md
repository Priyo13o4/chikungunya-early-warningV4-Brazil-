# Top Municipality Dataset Snapshot

- Timestamp (UTC): 2026-03-01T05:10:40.188430+00:00
- Input: data/processed/brazil_chik_dataset_final.csv
- Output data: data/processed/brazil_chik_dataset_top300_final.csv
- Output metadata JSON: outputs/reports/top300_dataset_snapshot.json
- Top-N requested: 300
- Municipality key used: municipality_id
- Top-N ranking metric: total_cases

## Step Shapes

- loaded: 3197180 rows, 21 columns
- cleaned: 3197180 rows, 21 columns
- imputed: 3197180 rows, 22 columns
- canonical_deduped: 3041052 rows, 22 columns
- top_n_filtered: 172200 rows, 22 columns

## Quality

- Duplicate district/date rows removed: 156128
- Missing municipality keys in final output: 0
- Final duplicate district/date rows: 0

## Top Municipalities (first 20)

| id | total_cases | row_count |
|---|---:|---:|
| 2304400 | 134907.0 | 574 |
| 3304557 | 88669.0 | 574 |
| 2611606 | 55514.0 | 574 |
| 3131307 | 36624.0 | 574 |
| 3127701 | 34986.0 | 574 |
| 3167202 | 29652.0 | 574 |
| 3106200 | 27766.0 | 574 |
| 2408102 | 26851.0 | 574 |
| 3548500 | 26781.0 | 574 |
| 3170206 | 25784.5 | 574 |
| 2927408 | 24425.0 | 574 |
| 3143302 | 24386.0 | 574 |
| 3119401 | 23954.0 | 574 |
| 2914802 | 19607.0 | 574 |
| 3301009 | 19590.0 | 574 |
| 2704302 | 18116.0 | 574 |
| 2931350 | 18092.0 | 574 |
| 2507507 | 17989.0 | 574 |
| 5103403 | 17029.0 | 574 |
| 1721000 | 15970.0 | 574 |
