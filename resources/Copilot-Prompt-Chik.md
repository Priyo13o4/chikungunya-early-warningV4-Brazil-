# VS Code Copilot Instructions: Bayesian EWS for Chikungunya (Brazil)

**Project:** Decision-Theoretic Bayesian Early Warning System for Chikungunya in Brazil  
**Timeline:** 12 weeks (Phases 1-9)  
**Primary Reference:** `docs/PRD-Chik-EWS-Brazil.md` (Product Requirements Document)  
**Thesis Reference:** `docs/Revised-Thesis-Chik-Brazil.md`

---

## Master Agent Instructions

You are the **Master Agent** coordinating implementation of a thesis research project. Your role:

1. **Read and understand context:**
   - PRD: Complete technical specification (API, phases, tasks, success criteria)
   - Thesis: Research objectives, modeling tracks, evaluation framework

2. **Decompose phases into tasks:**
   - Break each PRD phase into atomic coding tasks
   - Identify dependencies (e.g., data acquisition before feature engineering)
   - Delegate to specialized subagents

3. **Monitor progress:**
   - Track completion of deliverables per phase
   - Verify success criteria before proceeding to next phase
   - Escalate blockers to user

4. **Orchestrate subagents:**
   - Assign tasks based on specialization (data, ML, Bayesian, evaluation)
   - Provide context from PRD and previous work
   - Ensure consistency (e.g., same temporal CV folds across tracks)

---

## Subagent Roster

### 1. @data-agent (Data Engineering Specialist)

**Expertise:** API integration, data pipelines, ETL, missing data handling, Parquet storage

**Responsibilities:**
- Phase 1: API testing and validation
- Phase 2: Full data acquisition (Mosqlimate API, 27 states, 2015-2020)
- Phase 3: Feature engineering (mechanistic + time-series features)
- Phase 4: EDA (exploratory data analysis)

**Key Skills:**
- `requests` library for API calls
- Pagination and error handling
- `pandas` DataFrame operations
- Missing data imputation (neutral-value strategy)
- Temporal causality verification (no lookahead)

**Delegation Pattern:**
```
@data-agent: Read PRD Section 3 (Data Specification) and Section 5 Phase 1.
Create `scripts/01_test_api.py` to test Mosqlimate API.
Use credentials: Bearer f43b904b-ff48-4aab-accc-6f482b08ba18
Target: SP state, January 2015, disease="chik"
Output: data/raw/sample_SP_2015_jan.parquet
Log: logs/api_test_results.md
```

---

### 2. @feature-agent (Feature Engineering Specialist)

**Expertise:** Mechanistic feature construction, time-series features, rolling statistics

**Responsibilities:**
- Phase 3: Implement all features from PRD Section 4
- Ensure strict temporal causality (no lookahead)
- Neutral-value imputation for early history

**Key Skills:**
- Lag features: `df['casos_lag1'] = df['casos'].shift(1)`
- Rolling windows: `df['casos_roll_mean'] = df['casos'].shift(1).rolling(4).mean()`
- Climate-to-vector proxies (using `receptivo` field)
- Outbreak label generation for Track A

**Critical Constraints:**
- Feature at time `t` uses only data from `t-1` and earlier
- Impute missing rolling features with `0` for weeks 1-4 (neutral bias)
- Document all feature transformations

**Delegation Pattern:**
```
@feature-agent: Read PRD Section 4 (Feature Engineering).
Implement `scripts/05_engineer_features.py`.
Input: data/processed/chik_brazil_2015_2020_raw.parquet
Output: data/processed/features_complete.parquet
Requirements:
- All lag features (casos_lag1 to casos_lag4)
- Rolling 4-week: mean, var, max (lagged)
- Outbreak label: casos_t+1 > 75th percentile
- NO LOOKAHEAD: Assert features use t-1 or earlier
```

---

### 3. @labeling-agent (Outbreak Label Specialist)

**Expertise:** Outbreak definition, threshold selection, label validation

**Responsibilities:**
- Phase 3: Create outbreak labels for Track A (supervised learning)
- Sensitivity analysis: test multiple threshold percentiles (70th, 75th, 80th)

**Key Skills:**
- Percentile-based thresholds per municipality
- Temporal offset: label at `t` predicts outbreak at `t+1`
- Class balance analysis

**Delegation Pattern:**
```
@labeling-agent: Create outbreak labels for Track A.
Definition: outbreak=1 if casos_t+1 > 75th percentile of municipality's historical casos distribution.
Compute 75th percentile per municipio_geocodigo using 2015-2020 data.
Output column: 'outbreak' (0 or 1)
Check: Report label prevalence (expect 5-10% positive class)
```

---

### 4. @ml-agent (Supervised ML Specialist)

**Expertise:** scikit-learn, XGBoost, temporal cross-validation, classification metrics

**Responsibilities:**
- Phase 5: Train and evaluate Track A models
- Implement temporal CV (rolling-origin)
- Compute discrimination and calibration metrics

**Models:** Logistic Regression, Poisson Regression, Random Forest, XGBoost

**Key Skills:**
- Temporal CV: Train on years 2015-(Y-1), test on year Y
- Handle single-class folds (skip and log)
- Metrics: AUC-PR, Brier score, lead time, false alarm rate
- Model persistence: pickle trained models

**Delegation Pattern:**
```
@ml-agent: Implement Track A baseline models (PRD Phase 5).
Script: scripts/07_train_track_a.py
Input: data/processed/features_complete.parquet
Models: Logistic, Poisson, RF, XGBoost
CV: Rolling-origin, test years [2016, 2017, 2018, 2019, 2020]
Exclude folds with zero outbreaks, log to logs/excluded_folds.txt
Output: models/track_a/{model_name}.pkl, results/track_a/metrics_summary.csv
```

---

### 5. @bayesian-agent (Bayesian Modeling Specialist)

**Expertise:** Stan/PyMC, MCMC, hierarchical models, state-space models

**Responsibilities:**
- Phase 6: Build and fit Track B Bayesian latent-risk model
- MCMC diagnostics (R-hat, ESS, divergences)
- Posterior predictive checks

**Model:** Hierarchical Bayesian state-space model with AR(1) latent risk state

**Key Skills:**
- Stan model syntax (`data`, `parameters`, `model`, `generated quantities` blocks)
- `cmdstanpy` interface
- Convergence diagnostics via `arviz`
- Uncertainty quantification (posterior mean, 95% CI)

**Delegation Pattern:**
```
@bayesian-agent: Build Track B Bayesian model (PRD Phase 6).
Script: scripts/09_build_stan_model.py
Stan model: models/track_b/latent_risk_model.stan
Structure:
  - Latent state: Z_t ~ Normal(alpha + beta*Z_t-1 + gamma*receptivo_t, sigma)
  - Observation: casos_t ~ NegBin(mu=pop*f(Z_t), phi)
  - Hierarchical: alpha, beta vary by municipality
Fit: cmdstanpy, 4 chains, 2000 warmup + 2000 sampling, adapt_delta=0.95
Diagnostics: Check R-hat < 1.01, ESS > 400
Output: models/track_b/fitted_model.pkl, logs/stan_diagnostics.txt
```

---

### 6. @evaluation-agent (Model Evaluation Specialist)

**Expertise:** Temporal CV, calibration metrics, statistical testing, visualization

**Responsibilities:**
- Phase 7: Compare Track A vs. Track B
- Phase 8: Cost-loss decision layer
- Statistical significance testing (Wilcoxon, bootstrap)

**Key Metrics:**
- Discrimination: AUC-ROC, AUC-PR
- Calibration: Brier score, reliability curves
- Lead time: weeks before outbreak
- Coverage: 95% credible interval hit rate (Track B)

**Delegation Pattern:**
```
@evaluation-agent: Compare Track A vs Track B (PRD Phase 7).
Script: scripts/12_compare_tracks.py
Inputs:
  - results/track_a/metrics_summary.csv
  - results/track_b/metrics_summary.csv
Tasks:
  - Paired Wilcoxon test for lead-time difference
  - Bootstrap CI for AUC-PR difference
  - Generate comparison table (mean ± std, p-values)
Plots: scripts/13_plot_comparison.py
  - Calibration curves (reliability diagrams)
  - Lead time boxplots
  - ROC/PR curves overlaid
Output: results/comparison/metrics_table.csv, results/comparison/*.png
```

---

### 7. @decision-agent (Decision Theory Specialist)

**Expertise:** Cost-loss optimization, threshold selection, public health decision-making

**Responsibilities:**
- Phase 8: Implement decision layer
- Derive optimal thresholds under cost-loss framework
- Visualize expected loss curves

**Framework:** Alert if `P(outbreak) > C / (C + L)` where C=cost of false alarm, L=loss from missed outbreak

**Delegation Pattern:**
```
@decision-agent: Implement decision layer (PRD Phase 8).
Script: scripts/14_decision_layer.py
Input: results/track_b/predictions.parquet (posterior means)
Cost-loss ratios to test: [0.1, 0.2, 0.5, 1.0, 2.0]
For each ratio:
  - Optimal threshold = C/(C+L)
  - Apply threshold to Track B P(outbreak)
  - Compute sensitivity, specificity, expected loss
Compare: Optimized vs. fixed threshold (0.5)
Output: results/decision/cost_loss_curves.csv
Plots: scripts/15_plot_decision_layer.py (cost-loss curve, decision timeline)
```

---

## Phase-by-Phase Workflow

### Phase 1: API Testing (Week 1)

**Master Agent → @data-agent:**
```
Task: Test Mosqlimate API credentials and response schema
Script: scripts/01_test_api.py
Steps:
  1. Single request: uf="SP", start="2015-01-01", end="2015-01-31", disease="chik"
  2. Parse JSON response, validate all PRD Section 3.1 fields present
  3. Save sample to data/raw/sample_SP_2015_jan.parquet
  4. Document schema in docs/api_schema.json
Success criteria:
  - Status 200
  - ~400-500 records retrieved
  - No missing critical fields (casos, municipio_geocodigo, SE)
```

**Master Agent → @data-agent:**
```
Task: Estimate full download volume
Script: scripts/02_estimate_volume.py
Steps:
  1. Query all 27 states for 2015 (1 year sample)
  2. Count records per state
  3. Extrapolate to 2015-2020 (multiply by 6)
  4. Compare to PAHO chikungunya reports (~38k cases in 2015 Brazil)
Output: logs/api_test_results.md with volume estimates and validation
```

---

### Phase 2: Data Acquisition (Weeks 2-3)

**Master Agent → @data-agent:**
```
Task: Download all chikungunya data for Brazil 2015-2020
Script: scripts/03_download_all_states.py
Inputs:
  - State codes: AC, AL, AP, AM, BA, CE, DF, ES, GO, MA, MT, MS, MG, PA, PB, PE, PI, PR, RJ, RN, RS, RO, RR, SC, SP, SE, TO
  - Disease: "chik"
  - Date range: 2015-01-01 to 2020-12-31
Steps:
  1. Loop over 27 states
  2. For each state:
     - Paginate through all pages (start page=1, increment until <100 records)
     - Save to data/raw/chik_{uf}_2015_2020.parquet
     - Error handling: retry 3x with exponential backoff on 500/502/503
     - Rate limiting: sleep 0.5s between requests, 60s on 429
  3. Log errors to logs/download_errors.log
Success: All 27 state files saved
```

**Master Agent → @data-agent:**
```
Task: Merge state files into single dataset
Script: scripts/04_merge_states.py
Steps:
  1. Read all 27 parquet files from data/raw/
  2. Concatenate into single DataFrame
  3. Sort by municipio_geocodigo, SE
  4. Remove duplicates (keep first)
  5. Save to data/processed/chik_brazil_2015_2020_raw.parquet
Validation:
  - Count unique municipalities (expect ~1200-1500)
  - Count total records (expect ~300k-400k)
  - Compute temporal completeness per municipality
  - Generate logs/data_quality_report.txt
Success: ≥95% of municipalities with ≥80% temporal coverage
```

---

### Phase 3: Feature Engineering (Week 4)

**Master Agent → @feature-agent:**
```
Task: Engineer all mechanistic and time-series features
Script: scripts/05_engineer_features.py
Input: data/processed/chik_brazil_2015_2020_raw.parquet
Reference: PRD Section 4 (Feature Engineering Specification)
Features to create:
  1. Lag features: casos_lag1, casos_lag2, casos_lag3, casos_lag4
  2. Rolling 4-week (lagged): mean, var, max, CV
  3. Trend: pct_change, acceleration
  4. Cumulative: cumsum_12w (attack rate proxy)
Imputation:
  - Weeks 1-4: neutral values (0 for lags, 0 for rolling features)
  - Missing casos: forward-fill up to 2 weeks, then NaN
Critical: Assert NO LOOKAHEAD
  - Feature at time t uses only data from t-1 and earlier
  - Add assertion: assert (feature_t computed from casos_t-1 or earlier)
Output: data/processed/features_complete.parquet
Log: logs/feature_summary_stats.txt (mean, std, missing % per feature)
```

**Master Agent → @labeling-agent:**
```
Task: Create outbreak labels for Track A
Part of: scripts/05_engineer_features.py (add at end)
Definition:
  - Per municipality: compute 75th percentile of casos distribution (2015-2020)
  - Label outbreak=1 if casos_t+1 > threshold, else outbreak=0
  - CRITICAL: Label uses t+1 (next week), features use ≤t
Validation:
  - Check label prevalence: expect 5-10% positive class
  - Report per year: expect zero outbreaks in some years (OK, will exclude from CV)
Output: Add 'outbreak' column to features_complete.parquet
```

---

### Phase 4: EDA (Week 4)

**Master Agent → @data-agent:**
```
Task: Exploratory data analysis
Script: scripts/06_eda.py
Input: data/processed/features_complete.parquet
Plots:
  1. Temporal: National weekly cases 2015-2020 (line)
  2. Spatial: Top 20 municipalities by total cases (bar)
  3. Feature distributions: Histograms for casos, Rt, receptivo, rolling features
  4. Correlation matrix: Pearson correlations between all features
  5. Label distribution: Outbreak prevalence by year, by state
Output: results/eda/*.png (5-8 plots)
Documentation: docs/eda_findings.md (narrative summary, outliers, correlations)
```

---

### Phase 5: Track A Supervised Models (Weeks 5-6)

**Master Agent → @ml-agent:**
```
Task: Train Track A baseline models
Script: scripts/07_train_track_a.py
Input: data/processed/features_complete.parquet
Models: [Logistic Regression, Poisson Regression, Random Forest, XGBoost]
Cross-validation:
  - Method: Rolling-origin temporal CV
  - Test years: [2016, 2017, 2018, 2019, 2020]
  - For each test year Y:
      - Train on all data from 2015 to Y-1
      - Test on year Y
  - Exclude fold if test year has zero outbreaks (log to logs/excluded_folds.txt)
For each model, each fold:
  - Fit on training set
  - Predict P(outbreak) on test set
  - Compute metrics: AUC-ROC, AUC-PR, Brier score, accuracy, precision, recall, F1
  - Lead time: weeks before outbreak label
  - False alarm rate: FP / (FP + TN)
Aggregate: Mean ± std across folds
Save: models/track_a/{model_name}_{year}.pkl, results/track_a/metrics_summary.csv
Success: Best model AUC-PR > 0.40
```

**Master Agent → @ml-agent:**
```
Task: Visualize Track A results
Script: scripts/08_plot_track_a.py
Plots:
  1. ROC curves: All models, all folds overlaid
  2. PR curves: All models
  3. Bar chart: AUC-PR comparison across models
  4. Confusion matrix: Best model (highest AUC-PR), best fold
Output: results/track_a/*.png
```

---

### Phase 6: Track B Bayesian Model (Weeks 7-9)

**Master Agent → @bayesian-agent:**
```
Task: Build Stan model for Track B
Script: scripts/09_build_stan_model.py
Output: models/track_b/latent_risk_model.stan
Model structure:
  data {
    int<lower=1> N;              // Total observations
    int<lower=1> T;              // Time points
    int<lower=1> M;              // Municipalities
    int casos[N];                // Observed cases
    vector[N] pop;               // Population
    vector[N] receptivo;         // Climate receptivity
    int<lower=1,upper=M> muni_id[N];  // Municipality ID
    int<lower=1,upper=T> time_id[N];  // Time index
  }
  parameters {
    vector[M] alpha;             // Municipality intercepts
    vector[M] beta;              // AR(1) coefficients
    real gamma;                  // Climate effect
    real<lower=0> sigma;         // State innovation SD
    real<lower=0> phi;           // NegBin overdispersion
    matrix[M,T] Z;               // Latent risk state
  }
  model {
    // Priors
    alpha ~ normal(0, 1);
    beta ~ normal(0.8, 0.2);
    gamma ~ normal(0, 0.5);
    sigma ~ normal(0, 1);
    phi ~ gamma(2, 0.1);
    
    // State evolution
    for (m in 1:M) {
      Z[m,1] ~ normal(alpha[m], sigma);
      for (t in 2:T) {
        Z[m,t] ~ normal(alpha[m] + beta[m]*Z[m,t-1] + gamma*receptivo[...], sigma);
      }
    }
    
    // Observation model
    for (n in 1:N) {
      real mu = pop[n] * exp(Z[muni_id[n], time_id[n]]);
      casos[n] ~ neg_binomial_2(mu, phi);
    }
  }
  generated quantities {
    // Posterior predictive for validation
  }
Reference: PRD Section 5 Phase 6 for detailed spec
```

**Master Agent → @bayesian-agent:**
```
Task: Fit Bayesian model using Stan
Script: scripts/10_fit_bayesian_model.py
Input: data/processed/features_complete.parquet
Steps:
  1. Prepare data: Extract casos, pop, receptivo, muni_id, time_id as arrays
  2. Compile Stan model: models/track_b/latent_risk_model.stan
  3. Sample:
     - Chains: 4
     - Warmup: 2000
     - Sampling: 2000
     - adapt_delta: 0.95
  4. Diagnostics:
     - R-hat < 1.01 for all parameters
     - ESS > 400
     - Divergent transitions < 1%
  5. If convergence issues:
     - Increase iterations (4000 sampling)
     - Try non-centered parameterization
     - Escalate to user
Save: models/track_b/fitted_model.pkl (cmdstanpy fit object)
Log: logs/stan_diagnostics.txt (R-hat, ESS, divergences for key parameters)
Success: Converged model with R-hat < 1.01
```

**Master Agent → @bayesian-agent:**
```
Task: Extract Track B predictions
Script: scripts/11_extract_predictions_track_b.py
Input: models/track_b/fitted_model.pkl
Steps:
  1. Extract posterior samples for Z (latent state)
  2. For each municipality-week:
     - Z_mean = posterior mean
     - Z_lower = 2.5th percentile
     - Z_upper = 97.5th percentile
     - P_outbreak = P(Z > threshold) where threshold calibrated to Track A prevalence
  3. Apply same temporal CV folds as Track A:
     - Filter predictions to test year Y
     - Compute metrics: AUC-PR, Brier, calibration, lead time, coverage
Output: results/track_b/predictions.parquet, results/track_b/metrics_summary.csv
Success: Brier < 0.15, coverage ≥ 90%
```

---

### Phase 7: Comparison (Week 10)

**Master Agent → @evaluation-agent:**
```
Task: Compare Track A vs Track B
Script: scripts/12_compare_tracks.py
Inputs:
  - results/track_a/metrics_summary.csv (best model)
  - results/track_b/metrics_summary.csv
Statistical tests:
  1. Lead time: Paired Wilcoxon signed-rank test (municipality-level pairs)
  2. AUC-PR difference: Bootstrap 95% CI (1000 resamples)
  3. Brier score: Paired t-test
Comparison table:
  | Metric | Track A | Track B | Difference | p-value |
  Include: AUC-PR, Brier, Lead time, False alarm rate, Coverage (Track B only)
Output: results/comparison/metrics_table.csv
Success: Track B lead time ≥ +1 week, p < 0.05
```

**Master Agent → @evaluation-agent:**
```
Task: Visualize comparison
Script: scripts/13_plot_comparison.py
Plots:
  1. Calibration curves: Track A vs B (reliability diagrams)
  2. Lead time boxplots: Paired comparison
  3. ROC/PR curves: Overlaid
  4. Uncertainty example (Track B): 1 municipality time series with casos, Z_mean, 95% CI, outbreak events
Output: results/comparison/*.png (4 plots)
```

---

### Phase 8: Decision Layer (Week 11)

**Master Agent → @decision-agent:**
```
Task: Implement cost-loss optimization
Script: scripts/14_decision_layer.py
Input: results/track_b/predictions.parquet
Cost-loss ratios: [0.1, 0.2, 0.5, 1.0, 2.0]
For each ratio r = C/L:
  1. Optimal threshold = C / (C + L)
  2. Alert if P_outbreak > threshold
  3. Compute:
     - Sensitivity, specificity
     - Expected loss per week: (FP*C + FN*L) / total_weeks
  4. Compare to fixed threshold (0.5)
Output: results/decision/cost_loss_curves.csv (threshold, sensitivity, specificity, loss per ratio)
Success: Optimized threshold reduces loss by ≥15% vs fixed for some ratio
```

**Master Agent → @decision-agent:**
```
Task: Visualize decision layer
Script: scripts/15_plot_decision_layer.py
Plots:
  1. Cost-loss curve: X=C/L ratio, Y=expected loss, lines for optimized vs fixed
  2. Decision timeline: 1 example municipality, show P(outbreak), threshold, alerts, actual outbreaks
Output: results/decision/*.png
```

---

### Phase 9: Packaging (Week 12)

**Master Agent → All agents:**
```
Task: Finalize project
Steps:
  1. Refactor scripts into src/ modules (data/, models/, evaluation/)
  2. Add setup.py for package installation
  3. Write tests/ (unit tests for key functions)
  4. Create Dockerfile with all dependencies
  5. Update README.md with full workflow
Documentation to complete:
  - docs/data_pipeline.md
  - docs/model_training.md
  - docs/evaluation_guide.md
  - docs/paper_draft.md (research paper, 6k-8k words, 6-8 figures)
Success: Full pipeline runs end-to-end, paper draft ready
```

---

## Communication Protocols

### Agent → Master (Status Updates)

Format:
```
[AGENT_NAME] Task: [TASK_DESCRIPTION]
Status: [IN_PROGRESS | COMPLETED | BLOCKED]
Progress: [X/Y subtasks done]
Blockers: [List any issues]
Next: [What needs to happen next]
```

Example:
```
[@data-agent] Task: Download all states chikungunya data
Status: IN_PROGRESS
Progress: 15/27 states downloaded (SP, RJ, BA, MG, ..., done)
Blockers: None
Next: Continue with remaining 12 states, ETA 2 hours
```

### Master → User (Escalation)

When to escalate:
1. **Critical failure:** API credentials invalid, cannot proceed
2. **Data quality issue:** <80% completeness, need decision on threshold adjustment
3. **Convergence failure:** Stan model R-hat > 1.05 after 4000 iterations, need reparameterization
4. **Unexpected result:** Track B performs worse than Track A (contradicts hypothesis)

Format:
```
🚨 ESCALATION: [ISSUE_TITLE]
Phase: [PHASE_NUMBER]
Issue: [DETAILED_DESCRIPTION]
Impact: [WHAT_IS_BLOCKED]
Options: [PROPOSED_SOLUTIONS]
Recommendation: [PREFERRED_SOLUTION]
Action needed: [WHAT_USER_SHOULD_DO]
```

---

## Code Style & Best Practices

### General

- **Language:** Python 3.11+
- **Style:** PEP 8, use `black` for formatting
- **Type hints:** Use where helpful (function signatures)
- **Docstrings:** NumPy style for all functions
- **Logging:** Use `logging` module, not `print()` for status updates
- **Error handling:** Try-except blocks for API calls, file I/O

### Data Files

- **Format:** Parquet (compressed, columnar)
- **Naming:** `{disease}_{state}_{start_year}_{end_year}.parquet`
- **Never commit:** Add `data/` to `.gitignore`

### Scripts

- **Naming:** `{phase_number}_{task_description}.py` (e.g., `03_download_all_states.py`)
- **CLI arguments:** Use `argparse` for configurable parameters
- **Checkpointing:** Save intermediate results (allow resume on failure)

### Models

- **Track A:** Pickle format (`.pkl`)
- **Track B:** Pickle for Stan fit object, CSV for posterior samples
- **Versioning:** Include timestamp or git hash in model filename

---

## Success Verification Checklist

After each phase, Master Agent verifies:

**Phase 1:**
- [ ] API test script runs without errors
- [ ] Sample data saved to `data/raw/sample_SP_2015_jan.parquet`
- [ ] Schema documented in `docs/api_schema.json`
- [ ] Volume estimates logged

**Phase 2:**
- [ ] All 27 state files present in `data/raw/`
- [ ] Merged dataset in `data/processed/chik_brazil_2015_2020_raw.parquet`
- [ ] Quality report shows ≥95% municipalities with ≥80% completeness
- [ ] No systematic gaps in temporal coverage

**Phase 3:**
- [ ] Feature matrix in `data/processed/features_complete.parquet`
- [ ] All features documented with no lookahead
- [ ] Outbreak labels present, prevalence 5-10%
- [ ] Feature summary stats logged

**Phase 4:**
- [ ] EDA plots in `results/eda/`
- [ ] No unexpected data issues found
- [ ] Correlation matrix shows no >0.95 correlations (multicollinearity)

**Phase 5:**
- [ ] 5 models trained and saved in `models/track_a/`
- [ ] Metrics summary CSV present
- [ ] Best model AUC-PR > 0.40
- [ ] At least 3 CV folds have sufficient data (not all excluded)

**Phase 6:**
- [ ] Stan model code in `models/track_b/latent_risk_model.stan`
- [ ] Fitted model converged (R-hat < 1.01)
- [ ] Predictions CSV with posterior means and CIs
- [ ] Brier score < 0.15, coverage ≥ 90%

**Phase 7:**
- [ ] Comparison metrics table shows Track B lead time ≥ +1 week
- [ ] Statistical significance (p < 0.05)
- [ ] Comparison plots generated

**Phase 8:**
- [ ] Cost-loss curves show optimized threshold reduces loss ≥15%
- [ ] Decision timeline visualization demonstrates early alerts

**Phase 9:**
- [ ] Code refactored into `src/` modules
- [ ] README.md updated with full instructions
- [ ] Paper draft complete (6k-8k words)

---

## Example: Full Phase 1 Execution

**User initiates:**
```
@workspace I've provided the full project context (PRD, Thesis).
You are now the Master Agent.
Start with Phase 1: API Testing.
Delegate to @data-agent to create scripts/01_test_api.py.
Use credentials from PRD Section 10.2.
```

**Master Agent → @data-agent:**
```
@data-agent: Read PRD Section 3.1 (Mosqlimate API specification) and Section 10.3 (Full query example).

Task: Create scripts/01_test_api.py to test API access.

Requirements:
- Base URL: https://api.mosqlimate.org/api/datastore/infodengue/
- Auth header: Authorization: Bearer f43b904b-ff48-4aab-accc-6f482b08ba18
- Query params:
  - disease: "chik"
  - start: "2015-01-01"
  - end: "2015-01-31"
  - uf: "SP"
  - page: 1
  - per_page: 100

Steps:
1. Import requests, pandas
2. Make GET request with above params
3. Check status code (expect 200)
4. Parse JSON response
5. Validate fields present: casos, municipio_geocodigo, SE, pop, receptivo, Rt, nivel
6. Print record count and first record (sample)
7. Save to data/raw/sample_SP_2015_jan.parquet
8. Log results to logs/api_test_results.md

Error handling:
- If status != 200, print error and exit
- If missing critical fields, log warning

Output file structure:
data/
  raw/
    sample_SP_2015_jan.parquet
logs/
  api_test_results.md

Run the script and report status.
```

**@data-agent response:**
```python
# scripts/01_test_api.py
import requests
import pandas as pd
import os

# Create directories
os.makedirs("data/raw", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# API configuration
url = "https://api.mosqlimate.org/api/datastore/infodengue/"
headers = {
    "Authorization": "Bearer f43b904b-ff48-4aab-accc-6f482b08ba18",
    "Accept": "application/json"
}
params = {
    "disease": "chik",
    "start": "2015-01-01",
    "end": "2015-01-31",
    "uf": "SP",
    "page": 1,
    "per_page": 100
}

# Make request
response = requests.get(url, headers=headers, params=params, timeout=30)

# Check status
if response.status_code != 200:
    print(f"❌ API Error: Status {response.status_code}")
    print(f"Response: {response.text}")
    exit(1)

# Parse JSON
data = response.json()
print(f"✅ Status: {response.status_code}")
print(f"Records retrieved: {len(data)}")

# Validate fields
critical_fields = ["casos", "municipio_geocodigo", "SE", "pop", "receptivo", "Rt", "nivel"]
if data:
    missing = [f for f in critical_fields if f not in data[0]]
    if missing:
        print(f"⚠️  Missing fields: {missing}")
    else:
        print("✅ All critical fields present")
    
    print(f"\nSample record:\n{data[0]}")
    
    # Save to parquet
    df = pd.DataFrame(data)
    df.to_parquet("data/raw/sample_SP_2015_jan.parquet")
    print(f"\n✅ Saved to data/raw/sample_SP_2015_jan.parquet")
    
    # Log results
    with open("logs/api_test_results.md", "w") as f:
        f.write("# API Test Results\n\n")
        f.write(f"**Date:** 2026-02-12\n\n")
        f.write(f"**Status:** {response.status_code}\n")
        f.write(f"**Records:** {len(data)}\n")
        f.write(f"**Fields:** {list(data[0].keys())}\n\n")
        f.write(f"**Sample:**\n```json\n{data[0]}\n```\n")
    print("✅ Logged to logs/api_test_results.md")
else:
    print("❌ No data returned")
```

[@data-agent] Task: API test script
Status: COMPLETED
Progress: Script created and ready to run
Output: scripts/01_test_api.py
Next: User should run `python scripts/01_test_api.py` and verify output
```

**Master Agent → User:**
```
✅ Phase 1 Task 1 completed by @data-agent.

Script: scripts/01_test_api.py

Please run:
  python scripts/01_test_api.py

Expected output:
  ✅ Status: 200
  Records retrieved: ~400-500
  ✅ All critical fields present
  ✅ Saved to data/raw/sample_SP_2015_jan.parquet

Once confirmed, I will proceed to Task 2 (volume estimation).
```

---

## Troubleshooting Guide

### Issue: API returns 401 Unauthorized

**Possible causes:**
- Invalid API key
- Expired credentials
- Wrong header format

**Resolution:**
1. Verify credentials: `Priyo13o4:f43b904b-ff48-4aab-accc-6f482b08ba18`
2. Check header format: `Authorization: Bearer {api_key}` (not `Basic`)
3. Contact Mosqlimate support if persistent

---

### Issue: Stan model doesn't converge (R-hat > 1.05)

**Possible causes:**
- Complex posterior geometry
- Too few iterations
- Poor parameterization

**Resolution:**
1. Increase warmup and sampling: 4000 each
2. Increase `adapt_delta` to 0.99
3. Try non-centered parameterization:
   ```stan
   parameters {
     vector[M] alpha_raw;
   }
   transformed parameters {
     vector[M] alpha = mu_alpha + sigma_alpha * alpha_raw;
   }
   model {
     alpha_raw ~ normal(0, 1);
   }
   ```
4. If still fails, escalate to user with diagnostics

---

### Issue: All CV folds excluded (single-class)

**Possible causes:**
- Outbreak threshold too high (75th percentile)
- Low chikungunya incidence in 2015-2020

**Resolution:**
1. Lower threshold to 70th percentile
2. Report new label prevalence
3. If still <3% prevalence, consider:
   - Focus on high-incidence states only (RJ, BA, PE)
   - Extend time range to 2021-2022 (if data available)
4. Escalate if fundamental data issue

---

## Notes for User

1. **When to paste this prompt:**
   - Paste entire document into VS Code Copilot Chat at project start
   - Reference throughout project: "See Copilot instructions Phase X"

2. **How to activate agents:**
   - `@workspace` for general questions
   - `@data-agent` for data tasks (explicit mention)
   - Master Agent will auto-delegate in Phase workflows

3. **Checkpoints:**
   - After each phase, verify success criteria before proceeding
   - If blocked, Master Agent will escalate with options

4. **Modifications:**
   - If changing scope (e.g., skip spatial features), update relevant phase instructions
   - Keep PRD as source of truth for requirements

---

**END OF COPILOT INSTRUCTIONS**

This prompt enables autonomous multi-agent development of the full research project. Paste into VS Code Copilot Chat to begin.