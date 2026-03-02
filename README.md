# Chikungunya Early Warning System — Brazil (V4)

> A two-track, probabilistic early warning pipeline for chikungunya outbreaks across Brazilian municipalities. The system combines classical frequentist baselines (Track A) with a fully Bayesian hierarchical model (Track B) and a cost-loss–based decision layer. Every modelling choice is grounded in verifiable epidemiological and statistical methodology described below.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Repository Structure](#2-repository-structure)
3. [Pipeline Stages](#3-pipeline-stages)
   - [3.1 Data Ingestion & Cleaning](#31-data-ingestion--cleaning)
   - [3.2 Climate Imputation](#32-climate-imputation)
   - [3.3 Outbreak Labelling — supervised target](#33-outbreak-labelling--supervised-target)
   - [3.4 Feature Engineering](#34-feature-engineering)
   - [3.5 Temporal Cross-Validation](#35-temporal-cross-validation)
4. [Track A — Frequentist Baseline Models](#4-track-a--frequentist-baseline-models)
5. [Track B — Hierarchical Bayesian Model](#5-track-b--hierarchical-bayesian-model)
   - [5.1 Generative Model Specification](#51-generative-model-specification)
   - [5.2 AR(1) Latent Risk State](#52-ar1-latent-risk-state)
   - [5.3 Posterior Inference & Sampling](#53-posterior-inference--sampling)
   - [5.4 Posterior Predictive Risk Score](#54-posterior-predictive-risk-score)
6. [Decision Layer — Cost-Loss Framework](#6-decision-layer--cost-loss-framework)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Configuration Reference](#8-configuration-reference)
9. [Running the Pipeline](#9-running-the-pipeline)
10. [Mathematical Appendix](#10-mathematical-appendix)

---

## 1. System Overview

The pipeline answers a single operational question: **will the confirmed case count in municipality $m$ during week $t+1$ exceed the historically-derived outbreak threshold?**  

Rather than treating this as pure binary classification, the system outputs a calibrated **exceedance probability** $P(Y_{m,t} \geq \theta_m)$ where $\theta_m$ is the district-specific p75 threshold learned from training data. A cost-loss decision layer then converts probabilities into staged management actions (NO_ACTION / YELLOW / ORANGE / RED).

Two modelling tracks run in parallel:

| Track | Methodology | Output |
|-------|-------------|--------|
| A — Baseline | Rule, Logistic, Poisson, NB-GLM, RF, XGBoost, LightGBM | Binary alert probability |
| B — Bayesian | Hierarchical NB + AR(1) latent temporal state — full MCMC | Calibrated risk with 90 % CI |

Tracks are reconciled at the evaluation layer: metrics are reported side-by-side and a winner table is generated.

---

## 2. Repository Structure

```
chikungunya-early-warningV4/
├── config/
│   ├── cv_config.yaml             # CV fold strategy & gates
│   ├── model_config.yaml          # All model hyper-parameters
│   └── paths.py                   # Canonical I/O path resolution
├── data/
│   ├── raw/                       # InfoDengue + INMET raw pulls
│   ├── processed/                 # Cleaned & merged CSVs
│   └── features/                  # Final feature matrix
├── outputs/
│   ├── models/baselines/          # Serialised baseline estimators
│   ├── models/bayesian/           # InferenceData (ArviZ netCDF)
│   ├── metrics/                   # JSON/CSV score sheets
│   ├── figures/                   # All diagnostic plots
│   └── reports/                   # Fold ledger, audit JSON
├── src/
│   ├── data_preprocessing/        # Cleaning, imputation, labelling, loading
│   ├── feature_engineering/       # Temporal, mechanistic, spatial features
│   ├── models/
│   │   ├── baselines/             # Baseline wrappers, CV splitter, threshold rules
│   │   └── bayesian/              # Hierarchical PyMC model, diagnostics, inference
│   ├── decision_layer/            # Cost-loss threshold & staged alert logic
│   ├── evaluation/                # Metric functions, track comparison
│   ├── pipeline_runtime/          # Phase orchestration, shared state, IO
│   └── visualization/             # All plot functions
├── projects/brazil_chik/          # Brazil-specific data & CV adapter shims
├── scripts/brazil/                # Standalone data download & integrity scripts
├── tests/                         # Pytest suite (≥ 40 tests)
└── run_pipeline.py                # Main orchestration entry-point
```

---

## 3. Pipeline Stages

### 3.1 Data Ingestion & Cleaning

**File:** `src/data_preprocessing/clean_data.py`

Raw tabular data goes through a deterministic cleaning sequence:

1. **Column normalisation** — strip whitespace, lower-case all headers; apply known alias renames (`state_ut → state`, `preci → rainfall`, `longitute → longitude`).
2. **Date synthesis** — if a `date` column is absent but `day`, `mon`, `year` are present, build `date = pd.to_datetime({year, month, day})`.
3. **Schema enforcement** — hard-fail if `district` or `date` are missing; these are mandatory pipeline keys.
4. **Year filtering** — rows outside `[start_year, end_year]` are dropped; default 2009–2019 for the thesis dataset.
5. **Deduplication** — exact-row duplicates removed; configurable key subset.
6. **Numeric coercion** — `cases, rainfall, temperature, temp_kelvin, humidity, population` coerced with `errors="coerce"` (non-numeric → NaN, never fabricated).
7. **Temperature normalisation** — the pipeline auto-detects the unit of any temperature column using a heuristic:

$$
\hat{\text{unit}} = \begin{cases}
\text{Kelvin} & \text{if } q_{0.05} \geq 150 \text{ and } q_{0.95} \leq 360 \\
\text{Celsius} & \text{if } -40 \leq q_{0.05},\, q_{0.95} \leq 80 \\
\text{Kelvin} & \text{if median} > 120 \\
\text{Celsius} & \text{otherwise}
\end{cases}
$$

After detection, both `temperature` (°C) and `temp_kelvin` (K) are always available downstream via:

$$
T_\text{Celsius} = T_\text{Kelvin} - 273.15
$$

---

### 3.2 Climate Imputation

**File:** `src/data_preprocessing/impute_climate.py`

Missing climate values (`rainfall`, `temperature`, `temp_kelvin`, `humidity`) are imputed using a hierarchy:
1. **District + month median** — group by `(district, month)` and fill from median.
2. **Month median** — if district-level is still missing, fall back to global month median.
3. **Global median** — last resort for any remaining NaN.

This is a conservative median-based strategy; no regression imputation is used to avoid inflating out-of-sample predictive performance.

---

### 3.3 Outbreak Labelling — supervised target

**File:** `src/data_preprocessing/label_outbreaks.py`

The binary outcome $y_{m,t} \in \{0, 1\}$ is defined using district-specific percentile thresholds learned exclusively from **training data** to prevent label leakage.

**Step 1 — Compute district thresholds**

For each district $m$ and each percentile $p \in \{70, 75, 80\}$:

$$
\theta_{m,p} = Q_p\bigl(\{c_{m,t} : t \in \mathcal{T}_\text{train}\}\bigr)
$$

where $c_{m,t}$ is weekly confirmed case count and $Q_p$ is the empirical quantile.

**Step 2 — Assign binary label**

$$
y_{m,t} = \mathbf{1}\bigl[c_{m,t} \geq \theta_{m, p^*}\bigr]
$$

The default selected percentile is $p^* = 75$, configurable via `selected_percentile` in `cv_config.yaml`.  
Labels at all three percentiles (70/75/80) are stored as `outbreak_label_p70`, `outbreak_label_p75`, `outbreak_label_p80`; only the canonical alias `outbreak_label` is exposed to models.

> **Why district-specific thresholds?** Outbreak intensity varies enormously across Brazilian municipalities. A single national threshold would systematically over-alert in high-transmission states and under-alert in low-transmission ones. District-level p75 mirrors the WHO EWARS approach and is standard in Brazilian dengue surveillance (SINAN protocols).

---

### 3.4 Feature Engineering

Three complementary feature groups are built in `src/feature_engineering/`.

#### 3.4.1 Temporal Features

**File:** `src/feature_engineering/temporal_features.py`

All temporal features are constructed from **one-week lagged** case counts (lag-1 applied first), so that $c_{m,t-1}$ is used everywhere, avoiding any same-week data leakage.

| Feature | Formula | Purpose |
|---------|---------|---------|
| `case_lag_k` | $c_{m, t-k}$ for $k \in \{1, 2\}$ | Manual AR lags |
| `case_rolling_mean` | $\frac{1}{\min(4, n)}\sum_{s=1}^{4} c_{m, t-s}$ (min\_periods=1) | Smoothed epidemic trend |
| `case_rolling_std` | $\text{std}\bigl(c_{m, t-1}, \ldots, c_{m, t-4}\bigr)$ (min\_periods=3, fill 0) | Epidemic volatility |
| `case_velocity` | $c_{m, t-1} - c_{m, t-2}$ | First difference (epidemic speed) |
| `case_acceleration` | $v_{m, t-1} - v_{m, t-2}$ | Second difference (epidemic curvature) |
| `year`, `month`, `weekofyear` | ISO calendar decomposition | Seasonality proxies |

The rolling computations are grouped **by district** to prevent leakage between municipalities:

$$
\mu_\text{roll}(m, t) = \frac{1}{\min(4, n_{m,t})} \sum_{k=1}^{4} c_{m, t-k}
$$

#### 3.4.2 Mechanistic (Climate-Informed) Features

**File:** `src/feature_engineering/mechanistic_features.py`

These features encode known epidemiological mechanisms of *Aedes aegypti* vector dynamics and chikungunya transmission.

**Temperature conversion** (same unit-detection logic as cleaning):

$$
T_\text{°C} = T_K - 273.15
$$

**Degree-days above 20 °C** — proxy for mosquito gonotrophic cycle acceleration:

$$
\text{DD}_{20}(m, t) = \max\bigl(T_{\text{°C}}(m,t) - 20,\; 0\bigr)
$$

If all values of $\text{DD}_{20}$ are degenerate (near-zero variance), an adaptive baseline $\tau = \text{median}(T_\text{°C})$ replaces the fixed 20 °C threshold:

$$
\text{DD}_{\tau}(m, t) = \max\bigl(T_{\text{°C}}(m,t) - \tau,\; 0\bigr)
$$

**Optimal temperature indicator** — binary flag for the 25–30 °C window of peak *Aedes* biting rate:

$$
\text{opt}(m,t) = \mathbf{1}\bigl[25 \leq T_{\text{°C}}(m,t) \leq 30\bigr]
$$

Adaptive version uses a $\pm 2.5$ °C band around the district median when the fixed window is degenerate.

**Temperature anomaly** — deviation from district × season mean, capturing unusual warmth beyond seasonal baseline:

$$
\Delta T(m,t) = T_{\text{°C}}(m,t) - \bar{T}_{m,s(t)}
$$

where $s(t) \in \{\text{winter, pre\_monsoon, monsoon, post\_monsoon}\}$ is derived from the month number:

| Months | Season |
|--------|--------|
| 12, 1, 2 | winter |
| 3, 4, 5 | pre\_monsoon |
| 6, 7, 8, 9 | monsoon |
| 10, 11 | post\_monsoon |

Fallback hierarchy for degenerate anomaly: district+season → district-only → season-only → global mean.

**4-week cumulative rainfall** — proxy for larval breeding habitat persistence:

$$
R_{4\text{wk}}(m,t) = \sum_{k=0}^{3} r(m, t-k)
$$

Computed with `rolling(window=4, min_periods=1)` grouped by district and ordered by date.

**LAI anomaly** — deviation of leaf area index from district × season mean, encoding vegetation-driven microhabitat effects:

$$
\Delta \text{LAI}(m,t) = \text{LAI}(m,t) - \overline{\text{LAI}}_{m, s(t)}
$$

**Temperature × rainfall interaction** — legacy interaction term capturing joint climate forcing:

$$
\text{TR}(m,t) = T_{\text{°C}}(m,t) \times r(m,t)
$$

#### 3.4.3 Spatial Features

**File:** `src/feature_engineering/spatial_features.py`

Spatial features capture transmission pressure from neighboring municipalities using a k-NN graph built on centroid coordinates.

**Haversine distance** between centroids $(lat_1, lon_1)$ and $(lat_2, lon_2)$:

$$
d = 2R \arcsin\!\left(\sqrt{\sin^2\!\frac{\Delta\phi}{2} + \cos\phi_1 \cos\phi_2 \sin^2\!\frac{\Delta\lambda}{2}}\right)
$$

where $R = 6371$ km.

For each municipality $m$, the $k=5$ nearest neighbors $\mathcal{N}(m)$ are identified. The spatial spillover feature is:

$$
\bar{c}_\text{neighbor}(m, t) = \frac{1}{|\mathcal{N}(m)|} \sum_{m' \in \mathcal{N}(m)} c_{m', t-1}
$$

All neighbor case values are lagged by one week (same anti-leakage rule as temporal features).

---

### 3.5 Temporal Cross-Validation

**File:** `src/models/baselines/cv_splitter.py`  
**Config:** `config/cv_config.yaml`

The pipeline uses a **rolling-origin** (walk-forward) split strategy — also called *time series split* — appropriate for autocorrelated epidemiological time series. No future information ever enters a training fold.

**Split construction (rolling window mode):**

$$
\text{train}_k = \bigl[t_0, \; t_k - 1\bigr], \quad \text{valid}_k = t_k
$$

where $t_k \in \{\text{first\_valid\_year}, \ldots, \text{last\_valid\_year}\}$ and the rolling window limits training to the most recent $W$ years (default $W = 4$):

$$
\text{train\_start}_k = \max\bigl(\text{start\_train\_year},\; t_k - W\bigr)
$$

**Fold quality gates** — folds are skipped (not hard-failed) when:
- Training or validation set is empty.
- Training labels are single-class (no positive examples to learn from).
- Training span $< 1$ year.
- Minimum outbreak count per fold not met (default ≥ 1 in each split).
- Potential temporal leakage detected ($\max(t_\text{train}) \geq \min(t_\text{valid})$).

The minimum number of valid evaluated folds is configurable (`minimum_evaluated_folds: 3`). The pipeline fails explicitly if this gate is not met.

**Default fold schedule** for the Brazil dataset (2015–2020 validation window, 4-year rolling train):

| Fold | Train | Validate |
|------|-------|----------|
| 1 | 2015 | 2016 |
| 2 | 2015–2016 | 2017 |
| 3 | 2015–2017 | 2018 |
| 4 | 2015–2018 | 2019 |
| 5 | 2015–2019 | 2020 |

---

## 4. Track A — Frequentist Baseline Models

**Files:** `src/models/baselines/`

Seven baseline estimators are trained and evaluated in parallel:

### 4.1 Rule-Based EWARS Threshold (`rule_based_threshold`)

**File:** `src/models/baselines/threshold_rules.py`

Mirrors WHO Early Warning, Alert and Response System logic. Learns district-level p75 case thresholds on training data:

$$
\hat{\theta}_{m} = Q_{0.75}\bigl(\{c_{m,t}\}_{t \in \text{train}}\bigr)
$$

At inference:

$$
\hat{p}(m,t) = \mathbf{1}\bigl[c_{m,t} \geq \hat{\theta}_m\bigr] \in \{0, 1\}
$$

Returns a hard binary score; no probabilistic calibration. This serves as the epidemiological rule-of-thumb baseline.

### 4.2 Logistic Regression (`logistic_regression`)

L2-penalised logistic regression with balanced class weights (scikit-learn `saga` solver, $C = 1.0$, max\_iter = 5000):

$$
\log \frac{P(y=1 | \mathbf{x})}{1 - P(y=1 | \mathbf{x})} = \mathbf{w}^\top \mathbf{x} + b - \lambda \|\mathbf{w}\|_2^2
$$

GPU path: cuML `LogisticRegression` when `compute_backend = nvidia_cuda`.

### 4.3 Poisson Regression (`poisson_regression`)

Generalised linear model assuming Poisson count distribution (scikit-learn `PoissonRegressor`, $\alpha = 10^{-3}$):

$$
\log \mathbb{E}[c_{m,t}] = \mathbf{w}^\top \mathbf{x} + b
$$

Predictions are raw expected counts; the pipeline converts them to binary risk via the decision threshold.

### 4.4 Negative Binomial Regression (`negative_binomial_regression`)

Statsmodels GLM with NB family ($\alpha_\text{NB} = 1.0$), addressing the overdispersion in weekly case counts that violates the Poisson equidispersion assumption:

$$
\mathbb{E}[c_{m,t}] = \mu_{m,t} = \exp(\mathbf{w}^\top \mathbf{x}), \quad \text{Var}[c] = \mu + \alpha_\text{NB}^{-1}\mu^2
$$

Falls back to Poisson when statsmodels is unavailable.

### 4.5 Random Forest (`random_forest`)

200 trees, `max_depth=10`, `min_samples_leaf=2`, `class_weight="balanced_subsample"`. Provides non-linear feature interactions and implicit feature selection. GPU path: cuML `RandomForestClassifier`.

### 4.6 XGBoost (`xgboost`)

Gradient boosted trees with `binary:logistic` objective, 300 estimators, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`. GPU path: `device="cuda", tree_method="hist"`.

### 4.7 LightGBM (`lightgbm`)

Leaf-wise growth with 300 estimators, `num_leaves=31`, `max_depth=6`. GPU path: `device_type="gpu"`.

---

## 5. Track B — Hierarchical Bayesian Model

**Files:** `src/models/bayesian/hierarchical_model.py`

### 5.1 Generative Model Specification

The core statistical model is a **hierarchical negative-binomial regression** with partial pooling over districts and a **state-level AR(1) latent temporal process**. The full generative story:

**Likelihood:**

$$
c_{m,t} \,\sim\, \text{NegativeBinomial}\!\left(\mu_{m,t},\; \alpha_\text{NB}\right)
$$

where $\text{Var}[c_{m,t}] = \mu_{m,t} + \mu_{m,t}^2 / \alpha_\text{NB}$. NB is used instead of Poisson to accommodate the **overdispersion** typical in weekly disease count data (variance > mean).

**Linear predictor (log-link):**

$$
\log \mu_{m,t} = \underbrace{\alpha_m}_{\text{district effect}} + \underbrace{\boldsymbol{\beta}^\top \tilde{\mathbf{x}}_{m,t}}_{\text{climate covariates}} + \underbrace{z_{s(m), t}}_{\text{AR(1) temporal state}}
$$

where:
- $m$ is municipality, $s(m)$ is the Brazilian state (2-digit IBGE geocode prefix), $t$ is ISO week.
- $\tilde{\mathbf{x}}_{m,t}$ are standardised climate covariates (z-score normalised using training-fold means and standard deviations).
- $z_{s,t}$ is a state-level latent risk trend (see §5.2).

**Hierarchical district intercepts** (non-centred parameterisation for geometry efficiency):

$$
\mu_\alpha \sim \mathcal{N}(0, 1), \quad \sigma_\alpha \sim \text{HalfNormal}(0.5)
$$
$$
\alpha_\text{raw, m} \sim \mathcal{N}(0, 1)
$$
$$
\alpha_m = \mu_\alpha + \alpha_\text{raw, m} \cdot \sigma_\alpha
$$

Partial pooling shrinks district-specific intercepts toward the global mean, regularising estimates for municipalities with sparse data. The hyperprior $\sigma_\alpha \sim \text{HalfNormal}(0.5)$ weakly regularises the spread of district effects.

**Climate covariate coefficients:**

$$
\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0},\; 0.4^2 \mathbf{I})
$$

The tight $\sigma_\beta = 0.4$ prior reflects that, after z-score normalisation, a one-standard-deviation climate change is not expected to change mu by more than ~50 % (i.e. log-scale change ≲ 0.4).

**Dispersion parameter:**

$$
\alpha_\text{NB} \sim \text{HalfNormal}(1.0)
$$

### 5.2 AR(1) Latent Risk State

The temporal component models **epidemic momentum** at the state level. A non-centred AR(1) process propagates latent risk across weeks:

**Autoregressive structure:**

$$
u_{s, 1} \sim \mathcal{N}(0, 1) \quad \text{(initial state)}
$$
$$
u_{s, t} = \rho \cdot u_{s, t-1} + \varepsilon_{s,t}, \quad \varepsilon_{s,t} \sim \mathcal{N}(0, 1) \quad t \geq 2
$$
$$
z_{s,t} = \sigma_z \cdot u_{s,t}
$$

**Priors on process parameters:**

$$
\rho \sim \text{TruncatedNormal}(\mu=0.85,\; \sigma=0.15,\; \text{lower}=0,\; \text{upper}=0.995)
$$
$$
\sigma_z \sim \text{HalfNormal}(0.25)
$$

The prior mean $\rho = 0.85$ reflects strong weekly autocorrelation in epidemic incidence (empirically validated for arboviral time series in Brazil). The upper bound of 0.995 prevents unit-root non-stationarity. The tight $\sigma_z \sim \text{HalfNormal}(0.25)$ limits the amplitude of the latent trend to be on the order of the log-linear predictor.

**Implementation via `pytensor.scan`:**

```python
def _ar1_step(innov_t, prev_u, rho_param):
    return rho_param * prev_u + innov_t

u_scan_t, _ = pytensor.scan(
    fn=_ar1_step,
    sequences=[z_innov_s.T],       # innovation vectors (states × time)
    outputs_info=[u_init_s],        # initialised at u_init_s ~ N(0,1)
    non_sequences=[rho],
    strict=True,
)
u_t = pm.math.concatenate([u_init_s[:, None], u_scan_t.T], axis=1)
z_state = pm.Deterministic("z_state", sigma_z_state * u_t, dims=("state", "time"))
```

The AR(1) process is **shared across municipalities within the same state**: this is identifiable because Brazilian epidemics propagate through state-level networks of cities, and a single latent trajectory captures this.

**Simplified mode** (fallback): when convergence is poor, the AR(1) component is disabled and the model reduces to:

$$
\log \mu_{m,t} = \alpha_m + \boldsymbol{\beta}^\top \tilde{\mathbf{x}}_{m,t}
$$

### 5.3 Posterior Inference & Sampling

Sampling is performed with NUTS (No-U-Turn Sampler), an adaptive Hamiltonian Monte Carlo variant.

**Default hyperparameters (full inference, `final` profile):**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `draws` | 900 | Posterior samples per chain |
| `tune` | 900 | Warm-up steps per chain |
| `chains` | 3 | Independent Markov chains |
| `target_accept` | 0.95 | Target Metropolis acceptance rate (adaptive step size) |
| `max_treedepth` | 11 | Maximum NUTS tree depth (caps trajectory length) |

**Convergence diagnostics** checked post-sampling (`src/models/bayesian/diagnostics.py`):

| Diagnostic | Threshold | Meaning |
|-----------|---------|---------|
| Divergences | = 0 (warn > 25) | HMC divergence = biased geometry |
| $\hat{R}$ (Gelman-Rubin) | < 1.01 (warn > 1.05) | Between-chain variance ratio; 1.0 = perfect mixing |
| ESS (bulk & tail) | ≥ 400 (warn < 200) | Effective sample size from HMC draws |
| Max tree depth | < 12 | Hitting max depth = trajectory truncated |

If convergence thresholds are violated, the model either retries in simplified mode or raises a hard error depending on `convergence_failure_mode: strict|warn`.

**Sampling backend routing:**

```
compute_backend = auto
 ├── nvidia_cuda  →  jax_numpyro  (vectorised multi-chain on GPU)
 └── cpu / metal  →  pymc         (NUTS on CPU)
```

**Covariate standardisation** (computed on each training fold, applied to both train and test):

$$
\tilde{x}_{k} = \frac{x_k - \bar{x}_k^{\text{train}}}{\hat{\sigma}_k^{\text{train}}}
$$

where $\hat{\sigma}_k = \max(\text{std}(x_k),\; 10^{-6})$ prevents division by zero.

### 5.4 Posterior Predictive Risk Score

**File:** `src/models/bayesian/hierarchical_model.py → predict_with_uncertainty()`

After sampling, the pipeline computes the **posterior predictive exceedance probability** — the probability that the case count in municipality $m$ at time $t$ exceeds the district-specific threshold $\theta_m$:

$$
P_\text{risk}(m,t) = P\bigl(Y_{m,t} \geq \theta_m \,\big|\, \mathbf{x}_{m,t},\; \text{data}\bigr)
$$

Evaluated via Monte Carlo integration over posterior samples:

$$
P_\text{risk}(m,t) \approx \frac{1}{S} \sum_{s=1}^{S} P\bigl(Y \geq \theta_m \,\big|\, \mu_{m,t}^{(s)},\; \alpha_\text{NB}^{(s)}\bigr)
$$

where each draw $(\mu_{m,t}^{(s)}, \alpha_\text{NB}^{(s)})$ comes from the NUTS posterior and the inner probability is the **negative binomial survival function** (CDF complement):

$$
P\bigl(Y \geq k \,\big|\, \mu, \alpha\bigr) = 1 - F_\text{NB}\!\left(k-1;\; n=\alpha,\; p=\frac{\alpha}{\alpha + \mu}\right)
$$

Computed via `scipy.stats.nbinom.sf(k-1, alpha, p)`.

**Uncertainty intervals:** 5th and 95th percentiles across the $S$ posterior risk draws yield a calibrated 90 % credible interval:

$$
\bigl[P_\text{risk}^{q_{0.05}},\; P_\text{risk}^{q_{0.95}}\bigr]
$$

**Point-estimate fallback** (when `idata_` is absent): uses MAP-like posterior mean estimates without sampling uncertainty.

**Posterior sample cap:** to control memory, posterior draws are downsampled to at most `posterior_sample_cap = 500` via uniform striding:

```python
sample_idx = np.linspace(0, n_total - 1, num=500, dtype=int)
```

**Chunked inference:** the inner loop over rows is batched in chunks of `predictive_chunk_rows = 4000` to cap memory at ≈ 4 M floating-point elements per batch.

---

## 6. Decision Layer — Cost-Loss Framework

**File:** `src/decision_layer/cost_loss.py`

Probability scores alone are insufficient for public health operations — they need to be translated into **actionable alerts** accounting for the asymmetric costs of false alarms vs. missed outbreaks.

### 6.1 Optimal Binary Threshold (Analytical)

From decision theory, the optimal action threshold that minimises expected cost under a binary act/no-act decision is:

$$
\theta^* = \frac{C}{C + L} = \frac{C}{L + C}
$$

where:
- $C$ = cost of taking protective action when no outbreak occurs (false alarm cost)
- $L$ = loss from missing a true outbreak (missed detection loss)

```python
def optimal_action_threshold(cost: float, loss: float) -> float:
    return min(max(cost / loss, 0.0), 1.0)
```

For typical public health settings, $C \ll L$ (acting is cheap; missing is catastrophic), so $\theta^* \ll 0.5$.

### 6.2 Empirical Threshold Optimisation

When sufficient OOF samples are available ($n \geq 100$, both classes present), the threshold is further tuned empirically:

$$
\hat{\theta} = \arg\min_{\theta \in [0,1]} \widehat{\mathbb{E}}\bigl[\text{cost}(\theta;\; C, L)\bigr]
$$

where the expected cost objective at threshold $\theta$ is:

$$
\mathcal{L}(\theta) = \frac{1}{n} \sum_{i=1}^{n} \Bigl[ C \cdot \mathbf{1}[\hat{p}_i \geq \theta] + L \cdot \mathbf{1}[y_i = 1,\; \hat{p}_i < \theta] \Bigr]
$$

Grid search over 101 linearly-spaced candidates in $[0, 1]$; falls back to analytical $C/L$ when minimum-sample gate fails.

### 6.3 Staged Alert System

Four alert tiers map to management actions:

$$
\text{alert}(m,t) = \begin{cases}
\text{RED} & P_\text{risk}(m,t) \geq 0.7 \\
\text{ORANGE} & 0.5 \leq P_\text{risk}(m,t) < 0.7 \\
\text{YELLOW} & 0.3 \leq P_\text{risk}(m,t) < 0.5 \\
\text{NO\_ACTION} & P_\text{risk}(m,t) < 0.3
\end{cases}
$$

Thresholds (0.3 / 0.5 / 0.7) are configurable via `AlertThresholds`.

### 6.4 Staged Expected Value

For post-hoc analysis, each alert level is assigned a cost and the total expected value is computed:

$$
\text{EV\_total}(m,t) = \text{cost}\bigl(\text{alert}(m,t)\bigr) + L \cdot \mathbf{1}\bigl[y_{m,t}=1,\; \text{alert}(m,t) = \text{NO\_ACTION}\bigr]
$$

---

## 7. Evaluation Metrics

**Files:** `src/evaluation/metrics_baselines.py`, `src/evaluation/metrics_bayesian.py`

### 7.1 Standard Classification Metrics (Track A)

| Metric | Formula |
|--------|---------|
| Accuracy | $\frac{TP+TN}{N}$ |
| Precision | $\frac{TP}{TP+FP}$ |
| Recall (Sensitivity) | $\frac{TP}{TP+FN}$ |
| F1 | $\frac{2 \cdot P \cdot R}{P + R}$ |
| Cohen's $\kappa$ | $\frac{p_o - p_e}{1 - p_e}$ |
| ROC-AUC | Area under TPR vs FPR curve |
| PR-AUC | Area under Precision-Recall curve |
| False Alarm Rate | $\frac{FP}{FP+TN}$ |

### 7.2 Bayesian-Specific Metrics (Track B)

**Brier Score** — mean squared probability error:

$$
\text{BS} = \frac{1}{n} \sum_{i=1}^{n} (\hat{p}_i - y_i)^2
$$

Lower is better; perfectly calibrated model → $\text{BS} = 0$.

**Calibration / Reliability Diagrams** — binned observed frequency vs. predicted probability. Perfect calibration: $\bar{p}_b = \bar{y}_b$ for each bin $b$.

**CI Coverage** — fraction of true outcomes falling within the 90 % credible interval:

$$
\text{cov} = \frac{1}{n} \sum_i \mathbf{1}\bigl[\hat{p}_i^{q_{0.05}} \leq y_i \leq \hat{p}_i^{q_{0.95}}\bigr]
$$

### 7.3 Epidemiological Lead-Time Metric

This is the most operationally-meaningful metric: **how many weeks before the outbreak does the system raise an alert?**

For each outbreak onset (0→1 transition in `y_true`):

$$
\text{lead}_i = \max\bigl\{j \geq 0 : y_\text{pred}[t_\text{onset} - j] = 1,\; j \leq W\bigr\}
$$

where $W = 8$ weeks is the maximum lookback window. Zero when no prior alert exists.

$$
\overline{\text{lead}} = \frac{1}{|\mathcal{O}|} \sum_{i \in \mathcal{O}} \text{lead}_i
$$

**Lead-time utility** — mean credited lead steps capped at 4 weeks (reflecting that >4 weeks advance warning has diminishing operational value):

$$
U_\text{lead} = \frac{1}{|\mathcal{O}|} \sum_{i \in \mathcal{O}} \min(\text{lead}_i, 4)
$$

---

## 8. Configuration Reference

### `config/model_config.yaml`

```yaml
baseline_models:
  - rule_based_threshold          # EWARS p75 rule
  - logistic_regression           # L2 logit, balanced
  - poisson_regression            # Log-linear count model
  - random_forest                 # 200 trees, balanced
  - xgboost                       # 300 boosted trees
bayesian_model:
  climate_covariates:
    - month_sin                   # sin(2π·month/12) — circular encoding
    - month_cos                   # cos(2π·month/12)
    - rainfall
    - temperature
    - humidity
  draws: 600                      # NUTS samples per chain
  tune: 600                       # Warm-up steps
  chains: 2
  target_accept: 0.95             # Dual-averaging step size target
  max_treedepth: 12
  bayesian_simplified_mode: false # Disable AR(1) if convergence fails
  oof_conditional_rhat_threshold: 1.05
  outbreak_threshold_default_cases: 1.0
  posterior_sample_cap: 500
```

### `config/cv_config.yaml`

```yaml
strategy: time_series_split
n_splits: 5
test_size: 12                     # Weeks per validation hold-out
first_valid_year: 2016
last_valid_year: 2020
start_train_year: 2015
thesis_strict_mode: rolling       # rolling | expanding
train_window_years: 4
skip_single_class_folds: true
minimum_evaluated_folds: 3
```

---

## 9. Running the Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (Track A + Track B, rolling CV + full-fit)
python run_pipeline.py

# With specific config overrides
python run_pipeline.py \
  --model-config config/model_config.yaml \
  --cv-config config/cv_config.yaml \
  --project brazil_chik

# CPU-only, reduce chain count for smoke testing
python run_pipeline.py --bayesian-draws 120 --bayesian-chains 1
```

**Output artifacts:**

| Path | Content |
|------|---------|
| `outputs/metrics/baseline_metrics.json` | Track A OOF metrics |
| `outputs/metrics/bayesian_risk_intervals.csv` | Per-row risk + 90 % CI |
| `outputs/reports/fold_ledger.json` | Full CV fold accounting |
| `outputs/figures/` | Calibration curves, ROC/PR, lead-time boxplots, risk maps |
| `outputs/models/bayesian/*.nc` | ArviZ InferenceData (NetCDF) |

---

## 10. Mathematical Appendix

### A. Full Bayesian Generative Model (compact notation)

$$
\begin{aligned}
&\textbf{Priors:}\\
&\mu_\alpha \sim \mathcal{N}(0, 1), \quad \sigma_\alpha \sim \text{HN}(0.5)\\
&\alpha_\text{raw, m} \sim \mathcal{N}(0, 1), \quad m = 1,\ldots,M\\
&\boldsymbol{\beta} \sim \mathcal{N}(\mathbf{0},\; 0.16\,\mathbf{I}_K)\\
&\rho \sim \text{TruncNorm}(0.85,\; 0.15,\; [0, 0.995])\\
&\sigma_z \sim \text{HN}(0.25)\\
&\alpha_\text{NB} \sim \text{HN}(1.0)\\[6pt]
&\textbf{Deterministic transformations:}\\
&\alpha_m = \mu_\alpha + \sigma_\alpha \cdot \alpha_\text{raw, m}\\
&u_{s,1} \sim \mathcal{N}(0,1); \quad u_{s,t} = \rho\, u_{s,t-1} + \varepsilon_{s,t},\; \varepsilon_{s,t} \sim \mathcal{N}(0,1)\\
&z_{s,t} = \sigma_z\, u_{s,t}\\[6pt]
&\textbf{Likelihood:}\\
&\mu_{m,t} = \exp\!\bigl(\alpha_m + \boldsymbol{\beta}^\top \tilde{\mathbf{x}}_{m,t} + z_{s(m),t}\bigr)\\
&c_{m,t} \sim \text{NB}\!\left(\mu_{m,t},\; \alpha_\text{NB}\right)
\end{aligned}
$$

### B. Negative Binomial Parameterisation

The pipeline uses the mean + dispersion parameterisation:

$$
P(Y = k) = \binom{k + \alpha - 1}{k} \left(\frac{\alpha}{\alpha + \mu}\right)^\alpha \left(\frac{\mu}{\alpha + \mu}\right)^k
$$

$$
\mathbb{E}[Y] = \mu, \quad \text{Var}[Y] = \mu + \frac{\mu^2}{\alpha}
$$

As $\alpha \to \infty$, the NB reduces to Poisson($\mu$). Small $\alpha$ = high overdispersion.

### C. Cost-Loss Decision Theorem

The Bayes-optimal binary decision rule under asymmetric costs is:

$$
a^*(m,t) = \begin{cases} 1 & P(y_{m,t}=1 \mid \mathbf{x}) \geq \frac{C}{C+L} \\ 0 & \text{otherwise} \end{cases}
$$

This follows from minimising the expected loss functional:

$$
\mathbb{E}[\ell(a, y)] = P(y=1)\, L \cdot \mathbf{1}[a=0] + P(y=0)\, C \cdot \mathbf{1}[a=1]
$$

Setting equal: $P(y=1) \cdot L = P(y=0) \cdot C$ yields the indifference probability $\theta^* = C/(C+L)$.

### D. Non-Centred Parameterisation

The non-centred form for district intercepts avoids the funnel geometry that degrades NUTS efficiency when $\sigma_\alpha$ is small:

$$
\alpha_m = \mu_\alpha + \sigma_\alpha \cdot \tilde{\alpha}_m, \quad \tilde{\alpha}_m \sim \mathcal{N}(0,1)
$$

The sampler explores $(\mu_\alpha, \sigma_\alpha, \tilde{\boldsymbol{\alpha}})$ in a better-conditioned geometry, reducing divergences and improving $\hat{R}$ convergence.

### E. Gelman-Rubin $\hat{R}$ Statistic

$$
\hat{R} = \sqrt{\frac{\hat{V}}{W}}
$$

where $\hat{V}$ is the marginal posterior variance estimated from inter- and intra-chain variance, and $W$ is the within-chain variance. Values near 1.0 indicate chain mixing. The pipeline gates at $\hat{R} < 1.01$ (warn $> 1.05$).

---

*Pipeline version: V4 — Brazil Chikungunya EWS — Thesis implementation (2026)*
