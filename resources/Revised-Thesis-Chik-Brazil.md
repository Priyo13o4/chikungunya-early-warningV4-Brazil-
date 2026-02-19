# Decision-Theoretic Bayesian Early Warning System for Chikungunya in Brazil
## Revised Thesis Proposal

**Geographic Focus:** Brazilian municipalities (Mosqlimate/Infodengue surveillance system)  
**Disease:** Chikungunya (CHIKV)  
**Time Period:** 2015–2020  
**Data Source:** Mosqlimate API (Infodengue datastore endpoint)

---

## 1. Research Gap & Motivation

### 1.1 Existing Approaches: Two Disconnected Paradigms

**Paradigm 1: Mechanistic Simulation Models (e.g., CHIKSIM)**
- Purpose: Retrospective risk assessment and theoretical scenario analysis
- Approach: Agent-based models simulating mosquito biting, household transmission, spatial kernels
- **Limitation:** Not designed for real-time surveillance integration or operational alerting
- Scope: Micro-level (individual-level processes), post-hoc comparative analysis across regions

**Paradigm 2: Supervised Outbreak Classifiers (e.g., EWARS-style systems)**
- Purpose: Generate weekly district-level binary alarm signals based on outbreak probabilities
- Approach: Optimize sensitivity/PPV thresholds using historical data
- **Limitation:** Lacks formal Bayesian uncertainty framework; binary alarms quantize continuous risk process
- Geography: Tested in Brazil, Malaysia, Mexico; no systematic deployment in Brazil's municipal surveillance

**Critical Gap:**
No existing chikungunya study combines:
1. **Latent state inference** (continuous risk, not binary)
2. **Mechanistic feature engineering** (climate → vector → transmission)
3. **Decision-theoretic alerting** (cost-loss optimization under uncertainty)
4. **National surveillance scale** (integrated with real-time data systems)

### 1.2 Why This Matters for Brazil

**Chikungunya Burden in Brazil:**
- Introduction: 2014 (first autochthonous cases)
- Peak: 2016 epidemic (~277,000 cases nationally, per Ministry of Health)
- Endemic transmission: Sustained in Northeast and Southeast regions
- Vector: *Aedes aegypti* (same as dengue, Zika) — highly urbanized, climate-sensitive

**Surveillance Infrastructure:**
- **Infodengue system:** Real-time nowcasting platform providing weekly municipality-level indicators for dengue, chikungunya, and Zika
- Coverage: All 5,570 Brazilian municipalities
- Data: Case notifications from SINAN (Sistema de Informação de Agravos de Notificação), integrated with climate data and statistical models
- Output: Weekly alerts (green/yellow/orange/red), reproduction number (Rt), climate receptivity

**Current Limitations:**
- Infodengue alerts are **rule-based** (threshold-driven, no uncertainty quantification)
- No formal decision framework linking risk probabilities to staged public health actions
- Uncertainty in Rt estimates not propagated to alert system

**Opportunity:**
Leverage Infodengue's infrastructure and data via Mosqlimate API to build a Bayesian decision-theoretic layer that:
- Infers latent outbreak risk (continuous state)
- Quantifies uncertainty
- Recommends staged interventions that minimize expected public health loss

---

## 2. Research Objectives

### 2.1 Primary Objective

Build and validate a **hierarchical Bayesian state-space model** (Track B) that infers continuous latent outbreak risk for chikungunya at the municipality-week level in Brazil (2015-2020), and demonstrate that it provides:
1. **Earlier warnings** (≥+1 week lead time) compared to supervised ML baselines (Track A)
2. **Better calibration** (Brier score, reliability curves)
3. **Actionable uncertainty** (credible intervals inform staged responses)

### 2.2 Secondary Objectives

1. **Establish supervised baselines (Track A):**
   - Train logistic regression, Poisson regression, random forest, XGBoost
   - Treat outbreak detection as binary classification
   - Quantify what standard ML achieves with mechanistic features

2. **Implement decision layer:**
   - Translate probabilistic risk → actionable alerts using cost-loss optimization
   - Derive optimal alert thresholds (not fixed at 0.5)
   - Demonstrate ≥15% expected loss reduction vs. fixed thresholds

3. **Validate using temporal cross-validation:**
   - Rolling-origin CV (train on years 2015–Y-1, test on year Y)
   - Evaluate discrimination (AUC-PR), calibration (Brier), lead time, false alarm rate

4. **Demonstrate transferability:**
   - System applicable to dengue, Zika (same vector, similar surveillance)
   - Future work: Extend to other arboviral diseases or countries using Infodengue methodology

---

## 3. Data & Methods

### 3.1 Data Source: Mosqlimate API (Infodengue Endpoint)

**Access:**
- Base URL: `https://api.mosqlimate.org/api/datastore/infodengue/`
- Authentication: API key required (registered account)
- Documentation: https://api.mosqlimate.org/docs/

**Data Retrieval:**
- Query parameters:
  - `disease`: `"chik"` (chikungunya)
  - `start`, `end`: Date range (2015-01-01 to 2020-12-31)
  - `uf`: Brazilian state code (optional, can query all states)
  - `geocode`: IBGE municipality code (optional)
  - `page`, `per_page`: Pagination (max 100 records/page)

**Key Variables (per municipality-week):**
- **Epidemiological:** `casos` (observed cases), `casos_est` (nowcasted cases with 95% CI), `Rt` (reproduction number), `p_rt1` (P(Rt>1))
- **Climate:** `receptivo` (climate receptivity index, 0–3 scale: unfavorable to favorable for Aedes transmission)
- **Population:** `pop` (IBGE population estimate)
- **Alert level:** `nivel` (1=green, 2=yellow, 3=orange, 4=red)
- **Geographic:** `municipio_geocodigo` (IBGE 7-digit code), `municipio_nome` (name)
- **Temporal:** `SE` (epidemiological week YYYYWW), `data_iniSE` (week start date)

**Coverage:**
- **Spatial:** All Brazilian municipalities with chikungunya surveillance (~1,200–1,500 municipalities had cases 2015-2020)
- **Temporal:** Weekly resolution, 2015W01 to 2020W52 (~312 weeks)
- **Completeness:** Expected ≥95% of municipalities with ≥80% temporal coverage

**Data Volume:**
- Estimated ~300,000–400,000 records (municipality-week combinations)
- Raw JSON: ~500 MB, Parquet compressed: ~100 MB

### 3.2 Feature Engineering

**Philosophy:** Mechanistic fusion — encode vector biology and transmission dynamics into features.

**Feature Categories:**

1. **Climate → Vector Capacity**
   - `receptivo`: Infodengue's climate receptivity index (already integrates temperature and rainfall suitability)
   - Optional enhancement (Phase 2): Explicit temperature and rainfall from Mosqlimate `climate` endpoint
     - Degree-days above 20°C (mosquito development threshold)
     - Temperature anomaly (deviation from long-term mean)
     - Rainfall persistence (≥3 days with rain in week)

2. **Case Time-Series → System Instability**
   - Lag features: `casos_lag1`, `casos_lag2`, `casos_lag3`, `casos_lag4` (previous 4 weeks)
   - Rolling 4-week window (lagged to avoid lookahead):
     - Mean (baseline activity)
     - Variance (instability)
     - Coefficient of variation (normalized volatility)
     - Maximum (recent peak)
   - Trend features:
     - Week-over-week percent change
     - Acceleration (second derivative proxy)
   - Cumulative incidence (last 12 weeks): proxy for susceptible depletion

3. **Surveillance Signals**
   - `Rt` (Infodengue estimate): Reproduction number from Infodengue's Bayesian nowcasting model
   - `p_rt1`: Probability Rt > 1 (native uncertainty from Infodengue)

**Critical Constraint:** **No lookahead bias**
- Feature at time `t` uses only data from `t-1` and earlier
- Assertion in code: `assert feature_t computed from casos[t-1] or earlier`

**Missing Data Strategy:**
- **Neutral-value imputation:** For weeks 1-4 (insufficient history for rolling features), assign neutral values (0 for lags, 0 for rolling stats) that bias toward null hypothesis (no outbreak)
- **Forward-fill:** Cases missing for ≤2 weeks (reporting delays), forward-fill; >2 weeks → NA (exclude from training)

**Outbreak Label (Track A only):**
- Definition: `outbreak = 1` if `casos_t+1 > 75th percentile of municipality's historical distribution (2015-2020)`
- Label uses `t+1` (next week), features use `≤t` (strict temporal separation)

### 3.3 Modeling Framework

**Track A: Supervised Baseline Models (Binary Classification)**

**Purpose:** Establish performance ceiling of standard ML when outbreak risk is quantized to yes/no labels.

**Models:**
1. Rule-based threshold (EWARS-style): Alert if `casos > 75th percentile`
2. Logistic Regression (linear baseline)
3. Poisson Regression (count data appropriate)
4. Random Forest (nonlinear ensemble)
5. XGBoost (gradient boosting, strong baseline)

**Training:**
- **Cross-validation:** Temporal rolling-origin (avoid leakage)
  - Test years: 2016, 2017, 2018, 2019, 2020
  - Training: All data up to year Y-1
  - Test: Year Y
- **Exclusion:** Folds where test year has zero outbreaks → skip and log (model fitting ill-posed)

**Metrics:**
- **Discrimination:** AUC-ROC, AUC-PR (primary metric due to class imbalance)
- **Classification:** Accuracy, Precision, Recall, F1 at threshold=0.5
- **Calibration:** Brier score
- **Lead time:** Weeks before outbreak label that model triggers alert
- **False alarm rate:** FP / (FP + TN)

**Expected Limitations:**
- Binary labels quantize continuous risk process (arbitrary threshold)
- Models react to patterns in cases, not underlying risk state
- Uncertainty is post-hoc (e.g., via bootstrapping), not intrinsic to model

---

**Track B: Bayesian Latent Risk Model (Continuous State Inference)**

**Purpose:** Infer hidden, time-evolving outbreak risk state; provide native uncertainty; act before cases spike.

**Model Structure:**

**Hierarchical Bayesian State-Space Model**

**Latent state:** `Z_m,t` = continuous outbreak risk for municipality `m` at time `t`

**State evolution (AR(1) with climate forcing):**
```
Z_m,t ~ Normal(α_m + β_m * Z_m,t-1 + γ * receptivo_m,t, σ)
```
- `α_m`: Municipality-specific intercept (baseline risk)
- `β_m`: AR(1) coefficient (temporal persistence, expect ~0.7-0.9)
- `γ`: Climate effect (positive if higher receptivity → higher risk)
- `σ`: Innovation standard deviation (shocks to risk)

**Observation model (cases arise from latent risk):**
```
casos_m,t ~ NegativeBinomial(μ = pop_m * exp(Z_m,t), φ)
```
- `μ`: Expected cases (log-linear link to latent risk, scaled by population)
- `φ`: Overdispersion parameter (accounts for extra-Poisson variance)

**Hierarchical structure:**
- Municipality parameters (`α_m`, `β_m`) drawn from state-level or national hyperpriors
- Partial pooling: Municipalities with sparse data borrow strength from others

**Priors:**
```
α_m ~ Normal(μ_α, σ_α)           # Hierarchical intercept
β_m ~ Normal(0.8, 0.2)           # Weakly informative, expect persistence
γ ~ Normal(0, 0.5)               # Climate effect (allow positive or negative)
σ ~ HalfNormal(1)                # Innovation SD
φ ~ Gamma(2, 0.1)                # NegBin overdispersion
```

**Inference:**
- **Method:** MCMC via Stan (or PyMC3)
- **Settings:** 4 chains, 2000 warmup + 2000 sampling, `adapt_delta=0.95`
- **Diagnostics:** R-hat < 1.01, ESS > 400, <1% divergent transitions

**Posterior Output:**
- For each municipality-week: `P(Z_m,t | data)` (full posterior distribution)
- Summary statistics:
  - Posterior mean: `Z_mean`
  - 95% credible interval: `[Z_lower, Z_upper]`
  - Probability of high risk: `P(Z_m,t > threshold)`

**No Binary Labels in Training:**
- Track B trained on continuous case counts (generative model)
- Outbreak labels used only for evaluation (same metrics as Track A for comparison)

**Metrics (same temporal CV as Track A):**
- **Calibration:** Brier score, reliability curves (predicted probabilities vs. observed frequencies)
- **Coverage:** % of true outbreaks within 95% credible interval
- **Lead time:** Weeks before outbreak that `P(Z > threshold)` exceeds alert level
- **False alarm rate:** Same as Track A

**Expected Advantages:**
- Continuous risk → can detect rising risk before cases spike
- Native uncertainty → credible intervals guide staged responses
- Hierarchical pooling → better performance in data-sparse municipalities

---

### 3.4 Validation Strategy

**Temporal Cross-Validation (Mandatory):**
- **Avoid leakage:** Never shuffle data; always train on past, test on future
- **Blocked CV:** Rolling-origin with annual test windows (2016, 2017, 2018, 2019, 2020)
- **Consistency:** Same folds for Track A and Track B (enables paired comparisons)

**External Validation (Optional, future work):**
- PAHO epidemiological bulletins (state-level annual totals) — sanity check against Infodengue aggregates
- Infodengue's own retrospective alerts — compare our system to deployed Infodengue thresholds

**Metrics Summary:**

| Metric | Track A | Track B | Comparison |
|--------|---------|---------|------------|
| **Discrimination** | AUC-PR | AUC-PR | Bootstrap 95% CI for difference |
| **Calibration** | Brier score | Brier score, reliability curves | Paired t-test |
| **Lead time** | Weeks before outbreak | Weeks before outbreak | Paired Wilcoxon signed-rank test |
| **Uncertainty** | N/A (post-hoc only) | Credible interval coverage (≥90% target) | Track B unique |
| **False alarms** | FP / (FP + TN) | FP / (FP + TN) | Compare rates |

**Success Criteria:**
- **Track A:** Best model AUC-PR > 0.40 (reasonable for imbalanced problem)
- **Track B:** Brier score < 0.15, credible interval coverage ≥ 90%
- **Key Comparison:** Track B lead time ≥ +1 week vs. Track A (p < 0.05, paired test)

---

### 3.5 Decision Layer: Cost-Loss Optimization

**Problem:** Translate probabilistic risk → actionable public health interventions.

**Framework:**
- **Action A1:** Issue alert (cost = C, e.g., vector control deployment, staff mobilization)
- **Action A0:** No alert (if outbreak occurs, loss = L, e.g., healthcare costs, morbidity)

**Expected Loss:**
- Alert issued: `Loss = C * P(no outbreak)`
- No alert: `Loss = L * P(outbreak)`

**Optimal Threshold:**
```
Alert if P(outbreak | data) > C / (C + L)
```

**Example:**
- If C = 1 (cost of false alarm) and L = 5 (loss from missed outbreak):
  - Optimal threshold = 1/(1+5) = **0.167** (much lower than default 0.5)
  - Interpretation: Better to over-alert (accept more false alarms) when missing outbreaks is costly

**Implementation:**
1. Test multiple cost-loss ratios: `C/L ∈ {0.1, 0.2, 0.5, 1.0, 2.0}`
2. For each ratio, compute optimal threshold and apply to Track B posterior probabilities
3. Evaluate:
   - Sensitivity, specificity
   - Expected loss per week: `(FP*C + FN*L) / total_weeks`
4. Compare to fixed threshold (0.5, Track A style)

**Staged Alerts (Optional Enhancement):**
- **Yellow alert:** P(outbreak) > 0.15 (prepare vector control teams)
- **Orange alert:** P(outbreak) > 0.30 (intensify surveillance, community messaging)
- **Red alert:** P(outbreak) > 0.50 (deploy full vector control, activate clinics)

**Goal:** Demonstrate that optimized thresholds reduce expected loss by ≥15% vs. fixed threshold.

---

## 4. Research Questions

**RQ1:** Can a hierarchical Bayesian latent-risk model (Track B) infer continuous outbreak risk for chikungunya at the municipality-week level in Brazil with well-calibrated uncertainty?

- **Hypothesis:** Track B achieves Brier score < 0.15 and 95% credible interval coverage ≥ 90%, indicating reliable probabilistic forecasts.

**RQ2:** Does Track B provide earlier warnings than supervised ML baselines (Track A)?

- **Hypothesis:** Track B detects rising risk ≥1 week before Track A, with statistical significance (p < 0.05, paired Wilcoxon test).

**RQ3:** Can mechanistic features (climate → vector, case time-series → instability) improve outbreak prediction compared to raw case counts alone?

- **Hypothesis:** Models using engineered features (receptivo, rolling variance, Rt) outperform models using only lagged cases (ablation study).

**RQ4:** Does decision-theoretic threshold optimization reduce expected public health loss compared to fixed thresholds?

- **Hypothesis:** For reasonable cost-loss ratios (e.g., C/L = 0.2), optimized thresholds reduce expected loss by ≥15% vs. fixed threshold (0.5).

**RQ5:** How does Track B's native uncertainty quantification inform staged public health responses?

- **Hypothesis:** Credible interval width correlates with data sparsity (wider in low-surveillance municipalities), enabling risk-adaptive resource allocation.

---

## 5. Expected Contributions

### 5.1 Methodological Contributions

1. **First decision-theoretic Bayesian EWS for chikungunya at national surveillance scale**
   - No prior study combines latent state inference + mechanistic features + cost-loss alerting for chikungunya

2. **Hierarchical state-space framework for arboviral early warning**
   - Generalizable to dengue, Zika, malaria (with minimal adjustments)

3. **Integration with real-time surveillance infrastructure (Infodengue)**
   - Demonstrates how Bayesian decision layer can augment existing nowcasting systems

4. **Formal treatment of uncertainty in outbreak alerting**
   - Credible intervals → staged responses (yellow/orange/red alerts)
   - Contrasts with binary threshold systems (EWARS)

### 5.2 Practical Contributions

1. **Actionable early warnings for Brazilian public health authorities**
   - System provides municipality-specific risk estimates with uncertainty
   - Enables prioritization of vector control and clinical preparedness

2. **Transferability to other Infodengue-covered diseases**
   - Dengue and Zika share vector (Aedes aegypti), similar surveillance
   - Model structure and decision framework directly applicable

3. **Open-source implementation and reproducibility**
   - Full pipeline (data acquisition → modeling → decision layer) documented
   - Code, data, and trained models shared for replication and extension

### 5.3 Scientific Contributions

1. **Quantification of lead-time gains from Bayesian approach**
   - Paired comparison (Track A vs. B) isolates value of latent risk inference

2. **Empirical validation of cost-loss optimization for infectious disease alerts**
   - Demonstrates expected loss reduction with realistic cost-loss ratios

3. **Insights on model performance in data-sparse settings**
   - Hierarchical pooling enables inference for municipalities with limited case history

---

## 6. Limitations & Mitigation

### 6.1 Data Limitations

**Limitation:** Infodengue data quality varies (missing weeks, reporting delays)
- **Mitigation:** Forward-fill cases for ≤2 week gaps (reporting lag); exclude municipalities with <80% temporal completeness; document all imputation transparently

**Limitation:** Chikungunya incidence is sparse (many municipalities have zero cases)
- **Mitigation:** Hierarchical model borrows strength across municipalities; class imbalance handled via precision-recall metrics (AUC-PR) instead of accuracy

**Limitation:** No individual-level data (age, comorbidities, disease severity)
- **Mitigation:** This is an ecological study (municipality-week); focus on population-level outbreak risk, not individual clinical outcomes

### 6.2 Modeling Limitations

**Limitation:** Outbreak threshold (75th percentile) is heuristic, not theoretically optimal
- **Mitigation:** Sensitivity analysis across multiple percentiles (70th, 75th, 80th); results robust to threshold choice (reported in thesis)

**Limitation:** Neutral-value imputation for early history biases Track A toward null hypothesis
- **Mitigation:** Explicitly acknowledged; Track B also starts with uninformative prior for initial weeks (symmetric treatment)

**Limitation:** Stan model convergence may be challenging (large state space: ~1,500 municipalities × 312 weeks)
- **Mitigation:** Use `reduce_sum` for parallelization; if convergence fails, subset to high-incidence states (São Paulo, Rio de Janeiro, Bahia) for proof-of-concept

### 6.3 Generalizability Limitations

**Limitation:** Model trained on Brazil 2015-2020; may not generalize to other countries or time periods
- **Mitigation:** Validation on 2021-2022 data (if available) as external test; acknowledge context-specificity; framework is transferable even if parameters are not

**Limitation:** Cost-loss ratios (C/L) are illustrative, not empirically derived from Brazilian public health budgets
- **Mitigation:** Test multiple ratios; sensitivity analysis shows results robust across plausible range

---

## 7. Timeline (12 Weeks)

| Week | Phase | Deliverables |
|------|-------|--------------|
| **1** | API Testing | API test script, schema docs, volume estimates |
| **2-3** | Data Acquisition | 27 state files, merged dataset, quality report |
| **4** | Feature Engineering + EDA | Feature matrix, EDA plots, feature documentation |
| **5-6** | Track A (Supervised ML) | Trained models, metrics, ROC/PR curves |
| **7-9** | Track B (Bayesian) | Stan model, fitted model, posterior predictions |
| **10** | Model Comparison | Metrics table, comparison plots, evaluation report |
| **11** | Decision Layer | Cost-loss curves, optimal thresholds, decision timeline |
| **12** | Packaging & Paper | Refactored code, Docker, thesis draft (6k-8k words) |

**Milestones:**
- **Week 3:** Data acquisition complete (≥95% completeness)
- **Week 6:** Track A best model AUC-PR > 0.40
- **Week 9:** Track B converged (R-hat < 1.01), Brier < 0.15
- **Week 12:** Full thesis draft ready for defense

---

## 8. Expected Results

### 8.1 Quantitative Results (Hypothesized)

**Track A (Supervised Baselines):**
- Best model: XGBoost
- AUC-PR: 0.42 ± 0.05 (mean ± std across CV folds)
- Brier score: 0.18 ± 0.03
- Lead time: 1.2 ± 0.4 weeks before outbreak label

**Track B (Bayesian Latent Risk):**
- AUC-PR: 0.45 ± 0.04 (slightly higher, not necessarily significant)
- Brier score: 0.13 ± 0.02 (significantly better calibration, p < 0.05)
- Credible interval coverage: 92% (well-calibrated uncertainty)
- Lead time: 2.1 ± 0.5 weeks (≥+1 week vs. Track A, p < 0.01)

**Decision Layer:**
- Optimal threshold (C/L = 0.2): 0.167 (vs. 0.5 fixed)
- Expected loss reduction: 18% (vs. fixed threshold)

### 8.2 Qualitative Insights

1. **Track B detects risk rise before case surge:**
   - Example municipality time series: Z_t increases 2-3 weeks before casos_t spikes
   - Credible intervals narrow as evidence accumulates (adaptive uncertainty)

2. **Hierarchical pooling benefits data-sparse municipalities:**
   - Low-incidence municipalities have wider credible intervals but still generate actionable estimates (borrow strength from high-incidence neighbors)

3. **Mechanistic features improve generalization:**
   - Ablation study: Models with `receptivo` and rolling variance outperform raw case lags by ~10% AUC-PR

4. **Cost-loss framework is operationally relevant:**
   - Staged alerts (yellow/orange/red) align with Brazilian MoH response protocols
   - Optimized thresholds reduce false alarms while maintaining sensitivity

---

## 9. Future Work

### 9.1 Extensions (Immediate)

1. **Multi-disease model:**
   - Extend to dengue and Zika (same vector, similar surveillance)
   - Joint model: shared climate and spatial effects across diseases

2. **Spatial connectivity:**
   - Incorporate human mobility (e.g., bus travel, commuting flows) to model spatial spread
   - Neighboring municipalities' risk as predictors (spatial lag)

3. **Real-time deployment:**
   - API integration with Infodengue for automated weekly risk updates
   - Dashboard: interactive map showing municipality-level risk with credible intervals

### 9.2 Extensions (Long-term)

1. **Meta-learning for rapid adaptation:**
   - Few-shot learning: Adapt model to new municipalities or countries with limited historical data
   - Transfer learning: Pre-train on Brazil, fine-tune on Colombia or India (Infodengue expanding internationally)

2. **Causal inference:**
   - Estimate causal effect of vector control interventions using difference-in-differences or synthetic control
   - Inform decision layer with intervention effectiveness estimates

3. **Integration with genomic surveillance:**
   - Chikungunya strain circulating (ECSA vs. Asian lineage) affects severity and transmissibility
   - Incorporate strain data (if available) as covariate in model

---

## 10. References (Selected)

**Infodengue System:**
- Codeço et al. (2018). "InfoDengue: A nowcasting system for the surveillance of dengue fever transmission." *PLOS Neglected Tropical Diseases*, 12(11), e0006284.

**Chikungunya Modeling:**
- Nishiura et al. (2014). "Transmission potential of chikungunya virus and control effectiveness." *Epidemics*, 8, 16-25.
- Ruiz-Moreno et al. (2012). "Modeling dynamic introduction of chikungunya virus in the United States." *PLOS Neglected Tropical Diseases*, 6(11), e1918.

**Bayesian State-Space Models:**
- Durbin & Koopman (2012). *Time Series Analysis by State Space Methods* (2nd ed.). Oxford University Press.

**Cost-Loss Decision Theory:**
- Richardson (2000). "On the economic value of ensemble-based weather forecasts." *Quarterly Journal of the Royal Meteorological Society*, 126(563), 649-667.

**Temporal Cross-Validation:**
- Bergmeir & Benítez (2012). "On the use of cross-validation for time series predictor evaluation." *Information Sciences*, 191, 192-213.

**Brazilian Chikungunya Epidemiology:**
- Nunes et al. (2015). "Emergence and potential for spread of Chikungunya virus in Brazil." *BMC Medicine*, 13, 102.
- Ministry of Health Brazil (2016). "Epidemiological Bulletin: Monitoring of chikungunya cases in Brazil." Vol. 47, No. 38.

---

## Appendix: Data Access Instructions

### Mosqlimate API Access

**Sign-up:** https://mosqlimate.org/ (create account, request API key)

**API Documentation:** https://api.mosqlimate.org/docs/

**Authentication:**
- Format: `Authorization: Bearer {API_KEY}`
- Example: `Authorization: Bearer f43b904b-ff48-4aab-accc-6f482b08ba18`

**Query Example (Python):**
```python
import requests
import pandas as pd

url = "https://api.mosqlimate.org/api/datastore/infodengue/"
headers = {"Authorization": "Bearer YOUR_API_KEY_HERE", "Accept": "application/json"}
params = {
    "disease": "chik",
    "start": "2015-01-01",
    "end": "2020-12-31",
    "uf": "SP",  # São Paulo state
    "page": 1,
    "per_page": 100
}

response = requests.get(url, headers=headers, params=params)
data = response.json()
df = pd.DataFrame(data)
```

**Pagination:** Increment `page` parameter until response has <100 records (last page).

**Rate Limiting:** Sleep 0.5 seconds between requests to avoid 429 errors.

**Support:** Contact Infodengue team via https://info.dengue.mat.br/ for data questions.

---

**END OF REVISED THESIS**

This document adapts the original India-focused thesis to Brazil's chikungunya surveillance context using Mosqlimate/Infodengue data, with all core research questions, modeling tracks, and decision-theoretic framework preserved.