**Project:** Bayesian Decision-Theoretic Early Warning System for Chikungunya (Brazil)  
**Version:** 2.0 (Revised to reuse V3 core pipeline)  
**Target Geography:** Brazilian municipalities  
**Data Source:** Mosqlimate / Infodengue API  
**Owner:** Priyodip Mukhopadhyay

---

## 1. Objective

Reuse the existing **V3 production pipeline** as a **core framework** and extend it to support the Brazil (Infodengue/Mosqlimate) use-case through dataset-specific adapters.

The goal is to minimize re-engineering while preserving the scientific guarantees required by the thesis:

- hierarchical Bayesian latent risk inference (Track B)
- mechanistic feature engineering
- temporal cross-validation
- cost–loss decision layer
- reproducible and auditable runs

---

## 2. Design Principle

The pipeline is split into:

CORE (reused from V3)
+
PROJECT ADAPTER (Brazil-specific)

Only data access, feature logic, labels and CV definitions are replaced.

---

## 3. Architecture

ews-core/
src/
models/
evaluation/
decision_layer/
cv/
diagnostics/
utils/
run_pipeline.py

projects/
brazil_chik/
data_adapter.py
feature_adapter.py
label_adapter.py
cv_adapter.py
config.yaml

The core code MUST remain unchanged.

---

## 4. What is reused from V3

| Component | Status |
|------|------|
Pipeline orchestration | reused
OOF / fold management | reused
Track-A training & metrics | reused
Track-B Bayesian wrapper & diagnostics | reused
Posterior extraction & calibration | reused
Decision layer (cost–loss) | reused
Run manifest & leakage audit | reused

---

## 5. What is implemented only for Brazil

| Component | Implementation |
|------|------|
Data ingestion | `projects/brazil_chik/data_adapter.py`
Feature engineering | `projects/brazil_chik/feature_adapter.py`
Label generation (Track-A) | `projects/brazil_chik/label_adapter.py`
Temporal folds | `projects/brazil_chik/cv_adapter.py`

---

## 6. Data specification (Brazil adapter)

### 6.1 Source

Mosqlimate / Infodengue API  
Endpoint:

https://api.mosqlimate.org/api/datastore/infodengue/

Resolution: municipality × epidemiological week

Primary variables:

- `casos`
- `receptivo`
- `Rt`
- `p_rt1`
- `pop`
- `municipio_geocodigo`
- `SE`

---

### 6.2 Panel construction

The adapter must:

1. build a full panel:

(all municipalities with any data) × (all weeks 2015–2020)

2. left-join API records
3. sort by municipality, week

---

### 6.3 Missing data handling

Rules:

- forward-fill `casos` for ≤2 week gaps
- gaps >2 weeks → keep NA and exclude from training
- `receptivo` → state-level median
- `pop` → forward-fill

Municipality inclusion:

≥ 80% temporal completeness

---

## 7. Feature engineering (Brazil adapter)

All features must satisfy:

feature(t) uses only data ≤ t−1

### 7.1 Raw surveillance features

- `casos_lag1 … casos_lag4`
- `Rt`
- `p_rt1`
- `receptivo`

### 7.2 Rolling instability features (4-week window, lagged)

- rolling mean
- rolling variance
- rolling CV
- rolling max

### 7.3 Trend features

- week-over-week percent change
- acceleration proxy

### 7.4 Cumulative burden

- 12-week cumulative incidence
- attack rate = cumulative / population

---

### 7.5 Neutral-value imputation

For the first 4 weeks of each municipality:

| Feature | Value |
|------|------|
lag features | 0
rolling stats | 0
trend features | 0

This biases Track-A toward the null hypothesis and preserves conservativeness.

---

## 8. Track-A labels (Brazil adapter)

Label is defined only for Track-A.

outbreak(t) = 1
if casos(t+1) > 75th percentile of historical distribution of that municipality

Percentile sensitivity analysis:

- 70
- 75
- 80

Labels are never used by Track-B.

---

## 9. Temporal cross-validation (Brazil adapter)

Rolling-origin yearly folds:

| Fold | Train | Test |
------|------|------
1 | 2015 | 2016
2 | 2015–2016 | 2017
3 | 2015–2017 | 2018
4 | 2015–2018 | 2019
5 | 2015–2019 | 2020

Folds with zero outbreaks in test window are skipped and logged.

---

## 10. Track-A (reused)

Models:

- logistic regression
- random forest
- XGBoost
- rule-based EWARS percentile

Metrics (unchanged):

- AUC-PR (primary)
- AUC-ROC
- Brier score
- F1, precision, recall
- lead time
- false alarm rate

---

## 11. Track-B (reused)

### 11.1 Model class

Hierarchical Bayesian state-space model.

Latent state:

Z_m,t

State evolution:

Z_m,t ~ Normal(
α_m + β_m Z_m,t−1 + γ · receptivo_m,t ,
σ
)

Observation:

casos_m,t ~ NegBinomial( μ = pop_m · exp(Z_m,t), φ )

Municipality parameters are partially pooled.

---

### 11.2 Inference

- Stan or PyMC backend (existing V3 engine)
- chains ≥ 2
- draws and tuning as defined in V3 config
- diagnostics:
  - R-hat
  - ESS
  - divergence count

---

### 11.3 Mandatory constraint

Fallback / simplified model modes must be **disabled** for thesis runs.

force_full_bayesian = true

If convergence fails → run fails.

---

## 12. Posterior risk mapping

For each municipality-week:

- posterior mean of Z
- 95% credible interval
- probability of high risk:

P(Z > z*)

where z* is calibrated using training data only.

---

## 13. Evaluation (reused)

Track-B metrics:

- Brier score
- reliability curve
- credible-interval coverage
- lead time
- false alarm rate

Track-A vs Track-B comparison uses paired folds.

---

## 14. Decision layer (reused)

Cost–loss formulation:

- cost of alert = C
- loss of missed outbreak = L

Decision rule:

alert if P(outbreak) > C / (C + L)

Evaluate multiple ratios:

C/L ∈ {0.1, 0.2, 0.5, 1.0, 2.0}

Expected loss:

(FP · C + FN · L) / total weeks

Staged alerts are optional extensions.

---

## 15. Run artifacts

Each run must produce:

- run_manifest.json
- fold ledger
- leakage audit
- diagnostics summary
- model mode used
- metrics tables
- figures

---

## 16. Scalability constraint

Because the Brazilian panel is large:

- initial development may restrict to high-incidence states
- full-country runs must be attempted for final results
- parallelisation or batching must be enabled in Track-B engine if required

---

## 17. Acceptance criteria

### Data

- ≥95% of municipalities with cases retained
- ≥80% temporal completeness per municipality

### Track-A

- best AUC-PR ≥ 0.40

### Track-B

- Brier score < 0.15
- credible interval coverage ≥ 90%
- R-hat < 1.01

### Comparison

- Track-B lead time ≥ +1 week over Track-A (paired test)

### Decision layer

- expected loss reduction ≥ 15% versus fixed threshold

---

## 18. Implementation plan

Week 1  
→ API adapter + schema validation

Weeks 2–3  
→ full download, panel construction, quality audit

Week 4  
→ feature adapter + EDA

Weeks 5–6  
→ Track-A runs

Weeks 7–9  
→ Track-B runs (full Bayesian mode only)

Week 10  
→ Track comparison

Week 11  
→ decision layer experiments

Week 12  
→ packaging, report, thesis figures

---

## 19. Key guarantee of this revised PRD

This PRD guarantees that:

- the scientific core of the V3 pipeline is reused unchanged
- only data- and domain-specific logic is replaced
- the Brazil system remains fully consistent with the thesis design
- results remain directly comparable across future datasets or countries