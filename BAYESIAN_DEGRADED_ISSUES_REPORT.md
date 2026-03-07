# BAYESIAN_DEGRADED_ISSUES_REPORT

## Executive summary
- Current run artifacts show a split status, not a hard degraded fallback:
  - `convergence.json` reports `converged=false` because `ess_min=171.0746 < ess_threshold=200.0`.
  - The same artifact reports `strict_converged=true`, and downstream metadata reports `converged=true`.
- The strongest root cause is a strict-convergence bypass in group-aware checks, which effectively marks strict convergence as passed even when global ESS fails.
- This mismatch can allow headline Bayesian outputs to be treated as converged in `warn` mode, while summary tables/logs still show non-convergence flags.
- Additional risk signals in this run include threshold clamping (`7773` rows) and an optimized decision threshold of `0.0`, both of which can mask poor calibration behavior.

## Ranked root causes for degraded Bayesian runs
1. **Strict convergence bypass in group-aware checker (highest impact)**
- `evaluate_strict_convergence_with_groups` initializes `strict_checks_disabled=True` and returns `strict_converged=True` without enforcing fail conditions.
- This can override global convergence failure into an apparent strict pass.

2. **Global ESS threshold failure in full-fit posterior diagnostics**
- Full-fit diagnostics fail global ESS threshold (`171.0746 < 200.0`) even with no divergences and acceptable tree depth.
- This creates a non-converged global status and warning path.

3. **Warn-mode policy allows continuation after strict/global mismatch**
- With `convergence_failure_mode=warn`, pipeline continues and can set final Bayesian convergence state to true despite global convergence warning fields.

4. **Cross-artifact convergence inconsistency**
- `outputs/metrics/bayesian_convergence_summary.csv` reports `converged=False`, while `outputs/metrics/bayesian_risk_metadata.json` reports `converged=true`.
- This can mislead downstream reporting and governance checks.

5. **Decision threshold and clamp behavior may hide Bayesian quality issues**
- `decision_threshold_optimization.threshold=0.0` and `threshold_cases_clamp_count=7773` indicate high intervention from guardrails, increasing risk of optimistic recall-heavy metrics.

## Evidence table (numeric, from current artifacts/logs)

| Evidence item | Source | Value(s) | Interpretation |
|---|---|---:|---|
| Global convergence flag | `outputs/models/bayesian/diagnostics/convergence.json` | `converged=false` | Global convergence failed |
| Strict convergence flag | `outputs/models/bayesian/diagnostics/convergence.json` | `strict_converged=true` | Group-aware strict path passed |
| ESS minimum vs threshold | `outputs/models/bayesian/diagnostics/convergence.json` | `ess_min=171.074577061912`, `ess_threshold=200.0` | ESS failed by `28.9254` |
| R-hat and tree depth | `outputs/models/bayesian/diagnostics/convergence.json` | `r_hat_max=1.0152074973`, `rhat_threshold=1.05`, `max_tree_depth=10`, `max_tree_depth_threshold=12` | Non-ESS criteria passed |
| Divergences | `outputs/models/bayesian/diagnostics/convergence.json` | `divergences=0.0`, `divergence_threshold=25.0` | Divergence criterion passed |
| Grouped ESS fails | `outputs/models/bayesian/diagnostics/rhat_ess_grouped.csv` | `random_effects fail_ess_count=15 of 296`, total grouped ESS fails `15` | Localized low ESS in random effects |
| Convergence summary CSV | `outputs/metrics/bayesian_convergence_summary.csv` | `converged=False`, `mode_used=full_latent_ar`, `fallback_used=False` | CSV reflects global non-convergence |
| Risk metadata convergence | `outputs/metrics/bayesian_risk_metadata.json` | `converged=true`, `convergence_failure_mode=warn`, `degraded=false` | Downstream metadata reports converged |
| Decision threshold optimization | `outputs/metrics/bayesian_risk_metadata.json` | `threshold=0.0`, `sample_size=30723` | Aggressive threshold selection |
| Threshold clamping | `outputs/metrics/bayesian_risk_metadata.json` and `logs.txt:156` | `threshold_cases_clamp_count=7773`, `threshold_cases_clamp_applied=true` | Significant non-positive threshold correction |
| Full-fit metrics | `outputs/metrics/bayesian_metrics_fullfit.json` | `f1=0.3141`, `precision=0.2022`, `recall=0.7032` | Moderate full-fit quality |
| OOF metrics | `outputs/metrics/bayesian_metrics.json` | `f1=0.4354`, `precision=0.2783`, `recall=1.0`, `threshold_used=0.0` | Recall-saturated OOF behavior |

### Evidence snippets

`outputs/models/bayesian/diagnostics/convergence.json`:
```json
{
  "converged": false,
  "strict_converged": true,
  "ess_min": 171.074577061912,
  "ess_threshold": 200.0,
  "r_hat_max": 1.0152074973046845,
  "rhat_threshold": 1.05,
  "divergences": 0.0,
  "divergence_threshold": 25.0,
  "convergence_failure_mode": "warn"
}
```

`outputs/models/bayesian/diagnostics/rhat_ess_grouped.csv`:
```csv
group,n_parameters,r_hat_max,ess_min,fail_rhat_count,fail_ess_count,rhat_threshold,ess_threshold
random_effects,296,1.0152074973046845,171.074577061912,0,15,1.05,200.0
```

`outputs/metrics/bayesian_convergence_summary.csv`:
```csv
run_id,mode_used,degraded_mode,fallback_used,converged,divergences,divergence_threshold,max_tree_depth,max_tree_depth_threshold,r_hat_max,rhat_threshold,ess_min,ess_threshold
20260305T083716Z-c2100624,full_latent_ar,False,False,False,0.0,25.0,10.0,12.0,1.0152074973046845,1.05,171.074577061912,200.0
```

`outputs/metrics/bayesian_risk_metadata.json`:
```json
{
  "converged": true,
  "degraded": false,
  "convergence_failure_mode": "warn",
  "threshold_cases_clamp_count": 7773,
  "decision_threshold_optimization": {
    "threshold": 0.0,
    "sample_size": 30723
  }
}
```

`logs.txt`:
```text
logs.txt:149  Bayesian convergence summary: converged=False strict_converged=True mode=full_latent_ar divergences=0 r_hat_max=1.0152 ess_min=171.1 grouped_fails(rhat=0,ess=15)
logs.txt:150  Bayesian phase summary | headline_eligible=True converged=True mode=full_latent_ar oof_mode=simplified oof_ess_first=None oof_ess_last=None oof_ess_improved=None
logs.txt:156  Applied sentinel clamp for non-positive threshold_cases in decision/evaluation layer: clamped_rows=7773
```

## Code-path verification (why outcomes occur)
- Global convergence is computed from hard thresholds in `check_convergence`:
  - `src/models/bayesian/diagnostics.py:251`
  - `src/models/bayesian/diagnostics.py:252`
  - `src/models/bayesian/diagnostics.py:253`
  - `src/models/bayesian/diagnostics.py:254`
  - `src/models/bayesian/diagnostics.py:255`
- Group-aware strict checker currently does not enforce fails and returns strict pass:
  - `src/pipeline_runtime/phases_bayesian.py:391` (`strict_checks_disabled = True`)
  - `src/pipeline_runtime/phases_bayesian.py:468` (`strict_converged = True`)
  - `src/pipeline_runtime/phases_bayesian.py:483` (`return strict_converged, details`)
- Convergence payload is merged and written, then logs emit both flags:
  - `src/pipeline_runtime/phases_bayesian.py:1483`
  - `src/pipeline_runtime/phases_bayesian.py:1550`
  - `src/pipeline_runtime/phases_bayesian.py:1570`
- Final pipeline-level convergence state can be set true/false from `strict_converged` branching:
  - `src/pipeline_runtime/phases_bayesian.py:1600`
  - `src/pipeline_runtime/phases_bayesian.py:1630`
- Risk metadata `converged` is sourced from `bayesian_converged` (final pipeline state), not directly from raw global `convergence["converged"]`:
  - `src/pipeline_runtime/phases_eval_decision.py:440`
  - `src/pipeline_runtime/phases_eval_decision.py:466`
- Threshold guardrail clamp is explicitly applied in evaluation/decision phase:
  - `src/pipeline_runtime/phases_eval_decision.py:383`
  - `src/pipeline_runtime/phases_eval_decision.py:384`
  - `src/pipeline_runtime/phases_eval_decision.py:386`
  - `src/pipeline_runtime/phases_eval_decision.py:388`
  - `src/pipeline_runtime/phases_eval_decision.py:527`

## Additional runaway/logical bug candidates (beyond Bayesian degradation)

### 1) Convergence contract drift across artifacts
- Severity: **High**
- Signal: `bayesian_convergence_summary.csv` and `convergence.json` indicate non-convergence, while `bayesian_risk_metadata.json` records `converged=true`.
- Guardrails:
  - Add a single-source-of-truth field pair (`global_converged`, `strict_converged`) in all downstream metadata.
  - Add artifact consistency test asserting equality/mapping rules.

### 2) Strict checker appears partially stubbed
- Severity: **High**
- Signal: `strict_checks_disabled=True` and unconditional `strict_converged=True` path.
- Guardrails:
  - Fail strict when any configured core-group rule is violated.
  - Emit explicit reason codes when strict checks are intentionally bypassed.

### 3) Threshold clamp scale may hide data contract issues
- Severity: **Medium**
- Signal: `clamped_rows=7773` and optimized decision threshold `0.0`.
- Guardrails:
  - Add max clamp ratio alarm and fail-open/fail-closed policy toggle.
  - Require run report to include clamp ratio and source threshold column diagnostics.

### 4) OOF simplified rerun ESS telemetry missing (`None`)
- Severity: **Medium**
- Signal: logs show `oof_ess_first=None oof_ess_last=None oof_ess_improved=None`.
- Guardrails:
  - Validate OOF ESS extraction; write `NaN` with reason code instead of `None`.
  - Add test that rerun branches populate telemetry fields.

## Suggested fixes and tests

### Priority fixes
1. Implement strict group-aware convergence checks in `evaluate_strict_convergence_with_groups` and remove unconditional strict pass.
2. Standardize convergence fields across artifacts:
- `global_converged` from `check_convergence`
- `strict_converged` from group-aware policy
- `effective_converged` from policy mode (`strict` vs `warn`)
3. Add clamp-ratio reporting and upper-bound alerting for `threshold_cases` clamping.

### Test additions
1. Unit test: strict checker returns `False` when `core_ess_fail_groups` non-empty.
2. Integration test: `convergence.json`, `bayesian_convergence_summary.csv`, and `bayesian_risk_metadata.json` are contract-consistent.
3. Integration test: when `threshold_cases_clamp_count > 0`, metadata and logs both include identical counts.
4. Regression test: `warn` mode keeps pipeline running but must not overwrite `global_converged` semantics.

## Notes
- This report is based only on existing artifacts/logs in the workspace.
- No pipeline run or output-regenerating command was executed.

## Mechanistic Input Path Addendum (Verified)

1. **Mechanistic shift-induced leading NaNs with downstream imputation evidence**
- Severity: **High**
- Verdict: **CONFIRMED**
- Explanation: Mechanistic climate features are explicitly shifted by one week, which creates leading NaNs at sequence starts; this is expected and validated by tests. The latest run also logs Bayesian covariate imputation for selected climate covariates in `fullfit_subset`, confirming missing values were present and filled.
- File references: `src/feature_engineering/mechanistic_features.py:141`, `src/feature_engineering/mechanistic_features.py:206`, `tests/test_feature_engineering.py:120`, `tests/test_feature_engineering.py:132`, `tests/test_feature_engineering.py:135`, `src/pipeline_runtime/phases_bayesian.py:230`
- Concrete artifact/log values: `logs.txt:66` (`humidity missing 294 -> 0`), `logs.txt:67` (`temperature missing 294 -> 0`), both using `forward_fill_then_district_median`.

2. **Temporal lag/rolling alignment risk across mechanistic and temporal features (double-lag style mismatch risk)**
- Severity: **High**
- Verdict: **CONFIRMED**
- Explanation: Temporal predictors first lag cases (`shift(1)`) and then compute rolling statistics, while mechanistic predictors independently apply lag/rolling transforms (for example, rainfall uses `series.shift(1).rolling(...)`). These parallel lag pipelines can produce horizon-style misalignment risk if treated as directly comparable at the same prediction step.
- File references: `src/feature_engineering/temporal_features.py:61`, `src/feature_engineering/temporal_features.py:63`, `src/feature_engineering/temporal_features.py:74`, `src/feature_engineering/mechanistic_features.py:141`, `src/feature_engineering/mechanistic_features.py:206`
- Concrete artifact/log values: no dedicated numeric alignment metric is emitted in current artifacts; evidence is code-path level.

3. **Full-fit Bayesian future leakage risk from extended years + `shift(-1)` target construction**
- Severity: **High**
- Verdict: **CONFIRMED**
- Explanation: Target construction uses next-step (`shift(-1)`) future cases, and full-fit selection supports `recent_years`/latest-row subsetting for Bayesian training. In the latest run, CV folds are bounded to 2017-2020 validation years, while Bayesian full-fit sampling runs on a much larger panel, confirming a broader temporal scope in full-fit where future-defined labels can increase leakage risk if interpreted as strict prospective evaluation.
- File references: `projects/brazil_chik/label_adapter.py:139`, `src/pipeline_runtime/phases_bayesian.py:556`, `src/pipeline_runtime/phases_bayesian.py:606`, `src/pipeline_runtime/phases_bayesian.py:1098`
- Concrete artifact/log values: `logs.txt:40` (`valid_year=2017`), `logs.txt:55` (`valid_year=2020`), `logs.txt:69` (`obs=84378`, `times=574`), `outputs/reports/degraded_run.json:45` (`row_count": 84378`).

4. **Covariate order reordering risk between train/predict paths**
- Severity: **Medium**
- Verdict: **PARTIAL**
- Explanation: Covariates are explicitly re-sorted by null-rate/variance rule before use, and the latest degraded metadata shows requested vs effective order changed. This confirms reordering exists. Uncertainty remains on whether this causes an actual train/predict order mismatch in all runs, because runtime also propagates selected order into settings.
- File references: `src/pipeline_runtime/config_runtime.py:328`, `src/pipeline_runtime/config_runtime.py:344`, `src/pipeline_runtime/config_runtime.py:347`, `src/pipeline_runtime/phases_bayesian.py:1148`, `src/pipeline_runtime/phases_bayesian.py:1149`
- Concrete artifact/log values: `outputs/reports/degraded_run.json:8` requested `['month_sin','month_cos','rainfall','temperature','humidity']` vs `outputs/reports/degraded_run.json:15` effective `['rainfall','month_cos','month_sin','humidity','temperature']`.

5. **Imputation reorder/desync risk (with existing restore logic)**
- Severity: **Medium**
- Verdict: **PARTIAL**
- Explanation: Bayesian covariate imputation intentionally reorders rows by district/date, imputes, then restores original order via `__row_order__`. This confirms a reorder stage exists; uncertainty is residual desync risk under edge cases, because explicit restore logic is present and intended to prevent misalignment.
- File references: `src/pipeline_runtime/phases_bayesian.py:188`, `src/pipeline_runtime/phases_bayesian.py:200`, `src/pipeline_runtime/phases_bayesian.py:237`, `src/pipeline_runtime/phases_bayesian.py:238`
- Concrete artifact/log values: `logs.txt:66` and `logs.txt:67` show imputation executed on `fullfit_subset` with missing-to-filled transitions; no direct artifact field reports index-desync failures.

6. **Forbidden-column leakage into Bayesian path**
- Severity: **Low**
- Verdict: **NOT DETECTED IN CURRENT RUN**
- Explanation: Pre-assert guard logic blocks forbidden/leaky columns before pipeline continuation, and latest leakage audit artifact reports no dropped forbidden columns and no threshold-scope violations. In this run, forbidden-column leakage into Bayesian inputs was not detected. This run-level absence is not universal proof of safety, though pre-assert guards are in place.
- File references: `run_pipeline.py:116`, `run_pipeline.py:125`, `run_pipeline.py:981`, `outputs/reports/model_input_leakage_audit.json:4`, `outputs/reports/model_input_leakage_audit.json:18`
- Concrete artifact/log values: `outputs/reports/model_input_leakage_audit.json:4` (`"dropped_forbidden_columns": []`), `outputs/reports/model_input_leakage_audit.json:18` (`"violation_count": 0`).

## Proposed Fixes (High-Impact Snippets)

### 1) Strict convergence contract inconsistency (strict_converged and metadata divergence)
- Why it matters: Mixed convergence semantics across artifacts can mark a run as headline-eligible when global diagnostics failed. This weakens governance and can hide degraded Bayesian quality under `warn` mode.
- Target files: `src/pipeline_runtime/phases_bayesian.py`, `src/pipeline_runtime/phases_eval_decision.py`

```python
global_converged = bool(convergence.get("converged", False))
strict_converged, strict_details = evaluate_strict_convergence_with_groups(...)
effective_converged = strict_converged if failure_mode == "strict" else global_converged

convergence.update({"strict_converged": strict_converged, "strict_details": strict_details})
risk_metadata.update({
  "global_converged": global_converged,
  "strict_converged": strict_converged,
  "effective_converged": effective_converged,
})
```

- Validation:
- Add contract test: `convergence.json`, `bayesian_convergence_summary.csv`, and `bayesian_risk_metadata.json` must agree on `global_converged` semantics.
- Add regression test: in `warn` mode, pipeline may continue, but `global_converged=False` is never overwritten.
- Acceptance criterion: no artifact uses a single ambiguous `converged` field without mode context.

### 2) Covariate order stability between fit and predict
- Why it matters: If fit and predict consume covariates in different order, coefficient alignment can drift silently and distort predictions, especially with reordered effective covariate sets.
- Target files: `src/pipeline_runtime/config_runtime.py`, `src/pipeline_runtime/phases_bayesian.py`

```python
# freeze once at fit time
trained_covariate_order = tuple(effective_covariates)
model_bundle["trained_covariate_order"] = trained_covariate_order

# enforce at predict time
predict_order = tuple(current_covariates)
if predict_order != model_bundle["trained_covariate_order"]:
  raise ValueError(f"Covariate order mismatch: fit={trained_covariate_order} predict={predict_order}")
```

- Validation:
- Unit test: shuffled request order must be normalized to the persisted trained order before prediction.
- Negative test: deliberate order mismatch raises deterministic error with both orders in message.
- Acceptance criterion: `requested_covariates`, `effective_covariates`, and `trained_covariate_order` are all emitted in run metadata.

### 3) Full-fit future leakage interpretation guardrails
- Why it matters: Full-fit uses future-defined labels (`shift(-1)` target) and broad temporal scope; these outputs are useful for retrospective fit, not strict prospective performance claims.
- Target files: `src/pipeline_runtime/phases_bayesian.py`, `src/pipeline_runtime/phases_eval_decision.py`, `outputs/reports/run_metadata.json`

```python
is_full_fit = training_mode == "fullfit"
leakage_guardrail = {
  "label_uses_future_shift": True,
  "full_fit_retrospective_only": is_full_fit,
  "prospective_claim_allowed": not is_full_fit,
}
run_metadata["leakage_interpretation"] = leakage_guardrail
```

- Validation:
- Add metadata assertion: when `training_mode=fullfit`, `prospective_claim_allowed` must be `False`.
- Add report test: dashboards must display a retrospective-only banner when full-fit artifacts are shown.
- Acceptance criterion: no full-fit metric is labeled as OOS/prospective without explicit override and audit trace.

### 4) Mechanistic leading-NaN handling and explicit boundary policy
- Why it matters: Leading NaNs from lag/rolling transforms are expected at sequence boundaries; without explicit policy, downstream imputation can blur causality and temporal interpretation.
- Target files: `src/feature_engineering/mechanistic_features.py`, `config/model_config.yaml`

```python
boundary_policy = cfg.get("mechanistic_boundary_policy", "drop")  # drop|keep_nan|impute
leading_nan_mask = engineered_cols.isna().all(axis=1)

if boundary_policy == "drop":
  df = df.loc[~leading_nan_mask].copy()
elif boundary_policy == "impute":
  df.loc[leading_nan_mask, engineered_cols.columns] = np.nan
df["mechanistic_boundary_rows"] = leading_nan_mask.astype(int)
```

- Validation:
- Unit test: first valid index after lag/rolling is stable and policy-dependent behavior is deterministic.
- Add assertion: boundary row count is logged and exported to run metadata.
- Acceptance criterion: policy value is explicit in config and cannot silently default by code path drift.

### 5) Imputation alignment safety checks
- Why it matters: Reorder-impute-restore workflows are fragile; missing index restoration checks can cause silent row-feature desynchronization.
- Target files: `src/pipeline_runtime/phases_bayesian.py`

```python
df["__row_id__"] = np.arange(len(df), dtype=np.int64)
ordered = df.sort_values(["district", "date", "__row_id__"])
imputed = impute_covariates(ordered, covariates)
restored = imputed.sort_values("__row_id__").drop(columns=["__row_id__"])

if not np.array_equal(restored.index.to_numpy(), df.index.to_numpy()):
  raise RuntimeError("Imputation alignment failure: row order changed")
```

- Validation:
- Add test with duplicate district/date rows to prove stable tie-breaking and exact restoration.
- Add assertion: pre/post row counts and index hashes must match.
- Acceptance criterion: imputation logs include `alignment_ok=true` and row count parity.

### 6) Threshold clamping consistency plus observability
- Why it matters: Large clamp volumes can hide upstream threshold data quality issues; inconsistent propagation across logs/metadata reduces debuggability.
- Target files: `src/pipeline_runtime/phases_eval_decision.py`, `outputs/metrics/bayesian_risk_metadata.json`

```python
clamped_mask = threshold_cases <= 0
clamp_count = int(clamped_mask.sum())
clamp_ratio = clamp_count / max(len(threshold_cases), 1)
threshold_cases = threshold_cases.mask(clamped_mask, 1)

risk_metadata.update({"threshold_cases_clamp_count": clamp_count, "threshold_cases_clamp_ratio": clamp_ratio})
logger.warning("threshold clamp applied", extra={"clamped_rows": clamp_count, "clamp_ratio": clamp_ratio})
```

- Validation:
- Add consistency test: metadata clamp count/ratio must equal log-emitted values.
- Add policy test: alert threshold (for example, `clamp_ratio > 0.05`) triggers run warning or quality gate failure.
- Acceptance criterion: clamping is measured, reported, and bounded by a configurable threshold.

## Latest Run Analysis Snapshot
- Run scope/profile/settings highlights:
- Latest run id in artifacts: `20260305T083716Z-c2100624` (`outputs/reports/run_metadata.json`, `outputs/reports/degraded_run.json`, `outputs/metrics/bayesian_convergence_summary.csv`).
- CV profile in run metadata: `time_series_split`, `n_splits=5`, `test_size=12`, strict rolling CV enabled (`thesis_strict=true`, `thesis_strict_mode=rolling`, `train_window_years=4`).
- Bayesian backend resolved to GPU path with no fallback: requested `auto`, resolved `nvidia_cuda`, runtime backend `cuda`, sampler `jax_numpyro`.
- Sampling settings in logs/diagnostics align: `chains=4`, `draws=1000`, `tune=1000`, `target_accept=0.9`, `max_treedepth=12`, `obs=84378`, `districts=147`, `times=574`, `n_covariates=5`.

- Bayesian diagnostics observed values (`outputs/models/bayesian/diagnostics/convergence.json`, `outputs/models/bayesian/diagnostics/rhat_ess_grouped.csv`):
- `divergences=0.0` (threshold `25.0`), `max_tree_depth=10.0` (threshold `12.0`), `r_hat_max=1.0152` (threshold `1.05`).
- `ess_min=171.0746` vs threshold `200.0`; grouped summary localizes ESS failures to `random_effects` (`fail_ess_count=15`, `n_parameters=296`).
- Diagnostic flags show split semantics in same artifact: `converged=false` and `strict_converged=true`.

- Decision/threshold behavior highlights:
- Decision alerts output uses `risk_score_basis=bayesian_oof_risk_mean`; early rows show `decision_threshold_used=0.0` with `alert_level=NO_ACTION` when risk is `0.0`.
- Bayesian metrics report `threshold_used=0.0`, with balanced threshold optimization metadata showing `balanced_threshold_used=0.16` (`balanced_threshold_optimized=true`, sample size `30723`).
- Threshold sentinel clamp activity is material in this run: `threshold_cases_clamp_count=7773` with warning in logs and matching metadata fields.

- Artifact consistency notes (agreements/disagreements):
- Agreement: run id, mode (`full_latent_ar`), and fallback status (`fallback_used=false`, `degraded=false`) are consistent across `run_metadata.json`, `degraded_run.json`, and convergence summary artifacts.
- Agreement: covariate set after availability selection is consistently recorded as `['rainfall', 'month_cos', 'month_sin', 'humidity', 'temperature']` across degraded/metrics metadata.
- Disagreement to interpret explicitly: convergence status is `converged=False` in `bayesian_convergence_summary.csv` and `convergence.json`, while `bayesian_risk_metadata.json` reports `converged=true` under `convergence_failure_mode=warn`.

- Interpretation guardrails for thesis wording:
- Report global and strict convergence states separately; avoid phrasing that implies a single unambiguous convergence verdict for this run.
- Frame threshold and decision metrics as policy-conditioned outputs (not pure posterior quality metrics) because threshold optimization selected `0.0` and clamp corrections were applied.
- Keep prospective-performance claims tied to CV/OOF metrics and avoid conflating them with full-fit posterior diagnostics.
- When discussing Bayesian reliability, pair favorable diagnostics (R-hat/divergence/tree depth) with the ESS shortfall (`171.07 < 200`) in the same sentence.
