# Quick Start Guide: Chikungunya EWS Implementation

**Generated:** February 12, 2026  
**Project:** Bayesian Decision-Theoretic Early Warning System for Chikungunya in Brazil

---

## 📁 Files Generated

You now have **3 comprehensive documents** for your thesis implementation:

### 1. **PRD-Chik-EWS-Brazil.md** (Product Requirements Document)
**Purpose:** Complete technical specification for implementation  
**Length:** ~35,000 words, 70+ pages  
**Contents:**
- Full API specification (Mosqlimate/Infodengue) with your credentials
- 9 implementation phases with detailed tasks
- Success criteria and metrics for each phase
- Data requirements (~300k-400k municipality-week records, 2015-2020)
- Feature engineering specifications (mechanistic + time-series)
- Track A (supervised ML) and Track B (Bayesian) model details
- Evaluation framework and decision layer design
- File structure, tech stack, timeline (12 weeks)

**Use this for:** Day-to-day implementation reference, API integration, code structure

---

### 2. **Copilot-Prompt-Chik.md** (VS Code Copilot Instructions)
**Purpose:** Prompt for VS Code Copilot's agent system  
**Length:** ~12,000 words  
**Contents:**
- Master Agent orchestration instructions
- 7 specialized subagents (Data, Feature, Labeling, ML, Bayesian, Evaluation, Decision)
- Phase-by-phase workflow with delegation patterns
- Communication protocols between agents
- Error handling and escalation procedures
- Code style and best practices
- Example agent interactions

**Use this for:** Paste into Copilot Chat to enable autonomous multi-agent development

---

### 3. **Revised-Thesis-Chik-Brazil.md** (Updated Thesis Document)
**Purpose:** Your research thesis adapted for Brazil context  
**Length:** ~5,000 words  
**Changes from original:**
- Data source: India IDSP → Brazil Infodengue/Mosqlimate API
- Geographic scope: Indian districts → Brazilian municipalities  
- **Focus: Chikungunya only** (as you requested)
- Updated with API details and data access methods
- All research questions and modeling tracks preserved
- Added Brazil-specific chikungunya epidemiology (2016 epidemic, ~277k cases)

**Use this for:** Research proposal, paper introduction, methodology section

---

## 🚀 How to Use These Files

### Step 1: Set Up Project Environment

```bash
# Create project directory
mkdir chikungunya-ews-brazil
cd chikungunya-ews-brazil

# Copy the 3 generated files into docs/
mkdir docs
# (download PRD-Chik-EWS-Brazil.md, Copilot-Prompt-Chik.md, Revised-Thesis-Chik-Brazil.md)

# Initialize Git
git init
echo "data/" > .gitignore
echo "models/" >> .gitignore
echo "*.log" >> .gitignore

# Create Python environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install requests pandas pyarrow cmdstanpy arviz scikit-learn xgboost
pip freeze > requirements.txt
```

---

### Step 2: Test API Access

**Before anything else, verify your credentials work:**

```python
# quick_api_test.py
import requests

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
    "per_page": 10
}

response = requests.get(url, headers=headers, params=params)
print(f"Status: {response.status_code}")
print(f"Records: {len(response.json())}")
print("Sample:", response.json()[0] if response.json() else "Empty")
```

**Expected output:** Status 200, 10 records, JSON with `casos`, `municipio_geocodigo`, `SE` fields.

**If fails:** Check Mosqlimate account activation status, verify credentials.

---

### Step 3: Use Copilot (Option A - Automated)

**In VS Code with Copilot Chat enabled:**

1. **Open Copilot Chat** (Ctrl+Shift+I or Cmd+Shift+I)

2. **Paste the entire Copilot-Prompt-Chik.md** file into the chat

3. **Activate Master Agent:**
   ```
   @workspace I've provided the full project prompt above. 
   You are now the Master Agent.
   
   Read PRD-Chik-EWS-Brazil.md and Revised-Thesis-Chik-Brazil.md for context.
   
   Start with Phase 1: API Testing.
   Delegate to @data-agent to create scripts/01_test_api.py.
   Use credentials: Bearer f43b904b-ff48-4aab-accc-6f482b08ba18
   ```

4. **Follow agent workflow:**
   - Master Agent will decompose tasks
   - Subagents will generate code files
   - You review and approve each phase
   - Agents escalate blockers to you

**Why this works:** Copilot's agent feature allows specialized agents to focus on their domain (data, ML, Bayesian stats) with less context switching.

---

### Step 4: Manual Implementation (Option B - Without Copilot)

**If not using Copilot, follow PRD phases manually:**

**Phase 1: API Testing (Week 1)**
- Read PRD Section 5, Phase 1
- Implement `scripts/01_test_api.py` based on API spec in PRD Section 3
- Test with São Paulo 2015 chikungunya data
- Output: `data/raw/sample_SP_2015.parquet`

**Phase 2: Data Acquisition (Weeks 2-3)**
- Read PRD Section 5, Phase 2
- Implement state-by-state download loop (27 Brazilian states)
- Save raw responses incrementally (checkpoint every state)
- Merge into `data/processed/chik_brazil_2015_2020_raw.parquet`

**Continue through all 9 phases following PRD task lists.**

---

## 📊 Mosqlimate API Quick Reference

### Authentication
```python
headers = {
    "Authorization": "Bearer f43b904b-ff48-4aab-accc-6f482b08ba18",
    "Accept": "application/json"
}
```

### Infodengue Endpoint
```python
url = "https://api.mosqlimate.org/api/datastore/infodengue/"
params = {
    "disease": "chik",  # For chikungunya
    "start": "YYYY-MM-DD",
    "end": "YYYY-MM-DD",
    "uf": "SP" (optional, 2-letter state code),
    "geocode": 3304557 (optional, IBGE code),
    "page": 1,
    "per_page": 100 (max)
}
```

### Key Response Fields
- `casos` — Observed weekly cases (int)
- `municipio_geocodigo` — IBGE 7-digit code
- `SE` — Epiweek YYYYWW (e.g., 201504)
- `Rt` — Reproduction number
- `receptivo` — Climate receptivity (0–3)
- `nivel` — Alert level (1–4)
- `pop` — Population

### Pagination Pattern
```python
all_data = []
page = 1
while True:
    params['page'] = page
    response = requests.get(url, headers=headers, params=params)
    data = response.json()
    if not data or len(data) == 0:
        break
    all_data.extend(data)
    if len(data) < 100:  # Last page
        break
    page += 1
```

---

## 🎯 Critical Success Factors

### Week 1 Checkpoint
✅ API credentials work  
✅ Sample data retrieved (~400 records SP Jan 2015)  
✅ Schema documented  
✅ <5% missing values in core fields

**If failed:** Contact Mosqlimate support, verify account active

---

### Week 3 Checkpoint  
✅ Full panel acquired (~300k-400k records, 2015-2020)  
✅ <2% missing `casos` after imputation  
✅ State totals match PAHO ±10%

**If failed:** Review API logs, check for systematic gaps (missing states/years)

---

### Week 9 Checkpoint (Bayesian Model)
✅ Stan model converges (R-hat < 1.01)  
✅ Brier score < 0.15  
✅ 95% credible interval coverage ≥ 90%

**If failed:** Increase MCMC iterations, adjust priors, try non-centered parameterization

---

### Week 12 Final (Project Complete)
✅ Track A best model AUC-PR > 0.40  
✅ Track B lead-time ≥ +1 week vs Track A (p < 0.05)  
✅ Decision layer: optimized threshold ≠ 0.5, loss reduction ≥15%  
✅ Full pipeline reproducible (Docker)  
✅ Thesis draft ready (5k-6k words, 6-8 figures)

---

## 🔥 Common Pitfalls to Avoid

### 1. **Lookahead Bias in Features**
❌ WRONG: Using `casos[t+1]` to predict `outbreak[t]`  
✅ CORRECT: Only use `casos[t-1]`, `casos[t-2]`, ... (lagged features)

**Check:** For every feature at time `t`, verify it only uses data from `t-1` and earlier.

---

### 2. **Single-Class Training Folds**
**Issue:** Some years have zero outbreaks (all negative labels).  
**Solution:** Exclude these folds transparently, document in `logs/excluded_folds.txt`.  
**Don't:** Try to force-fit models on all-zero labels (will break).

---

### 3. **MCMC Convergence Issues**
**Symptoms:** R-hat > 1.05, divergences, low ESS  
**Solutions:**
- Increase `adapt_delta=0.95` (Stan parameter)
- Run longer chains (4000 iterations instead of 2000)
- Use non-centered parameterization for hierarchical effects
- Tighten priors (use domain knowledge)

---

### 4. **Ignoring API Rate Limits**
**Issue:** Hitting API too fast → 429 errors  
**Solution:** 
- Query state-by-state (not all Brazil at once)
- Add delays between requests (`time.sleep(0.5)`)
- Implement exponential backoff on errors

---

### 5. **Not Saving Raw API Responses**
**Issue:** API schema changes or connection fails mid-download  
**Solution:** Save every API response as Parquet BEFORE processing.  
**Benefit:** Can reprocess without re-downloading if errors found later.

---

## 📚 Additional Resources

### Mosqlimate / Infodengue
- **API Docs:** https://api.mosqlimate.org/docs/
- **Infodengue Portal:** https://info.dengue.mat.br/
- **Support:** Contact via portal if API issues

### Bayesian Modeling
- **Stan User Guide:** https://mc-stan.org/docs/
- **Stan Forums:** https://discourse.mc-stan.org/
- **PyMC Tutorials:** https://www.pymc.io/projects/examples/en/latest/

### Temporal Cross-Validation
- Bergmeir & Benítez (2012): "On the use of cross-validation for time series predictor evaluation"
- Must avoid random shuffling; use rolling-origin or blocked CV

### Cost-Loss Framework
- Richardson (2000): "Skill and relative economic value of ECMWF ensemble prediction system"
- Bayesian decision theory: Alert if P(outbreak) > C/(C+B)

---

## 🤝 Getting Help

### If API Issues
- Check Mosqlimate account status
- Try alternative dates/states
- Review logs in `logs/api_calls.log`
- Contact: Infodengue team via info.dengue.mat.br

### If Statistical Issues
- Stan convergence: Post on Stan Discourse with model code + data summary
- Bayesian modeling: Include prior choices, sample size, diagnostics
- Cross-validation: Ensure temporal ordering preserved

### If Code Issues
- GitHub Issues (once repo created)
- Stack Overflow (tag: python, bayesian, stan)
- VS Code Copilot (paste error message + context)

---

## ✅ Next Steps Checklist

- [ ] Download all 3 generated files (PRD, Copilot Prompt, Revised Thesis)
- [ ] Set up Python 3.11 environment
- [ ] Test API credentials with quick_api_test.py
- [ ] Read PRD Section 3 (API specification) carefully
- [ ] Decide: Use Copilot agents OR manual implementation
- [ ] If Copilot: Paste prompt into VS Code Chat, activate Master Agent
- [ ] If manual: Start with Phase 1 (API Testing), follow PRD task lists
- [ ] Create `data/`, `scripts/`, `models/`, `results/` directories
- [ ] Initialize Git repo, add `.gitignore` for data files
- [ ] Document API test results in `docs/api_test_results.md`

---

## 🎓 Thesis Paper Outline (When Ready)

**Title:** Decision-Theoretic Bayesian Early Warning System for Chikungunya: A Municipality-Level Analysis in Brazil

**Abstract:** 250 words (problem, method, results, impact)

**1. Introduction**
- Chikungunya burden in Brazil (2016 epidemic: 277k cases)
- Limitations of existing EWAS (EWARS, CHIKSIM)
- Research gap and objectives

**2. Methods**
- Data: Infodengue API (2015-2020, ~300k municipality-weeks)
- Feature engineering (mechanistic + temporal)
- Track A: Supervised baselines (Logistic, RF, XGBoost)
- Track B: Bayesian latent risk model (Stan hierarchical state-space)
- Evaluation: Temporal CV, metrics
- Decision layer: Cost-loss optimization

**3. Results**
- Data characteristics (Table 1: summary stats)
- Track A performance (Figure 1: ROC/PR curves)
- Track B calibration (Figure 2: reliability diagram)
- Lead-time comparison (Figure 3: boxplots)
- Decision layer (Figure 4: cost-loss curves)

**4. Discussion**
- Track B advantages (uncertainty, lead-time)
- Operational implications for Brazilian MoH
- Limitations (data quality, threshold arbitrariness)
- Future work (real-time deployment, dengue/Zika extension)

**5. Conclusion**
- Summary of contributions
- Call for integration with Infodengue surveillance

**Figures (6-8):** ROC curves, PR curves, calibration plots, lead-time distributions, cost-loss analysis, spatial risk maps, temporal trajectories, decision simulation

**Target Journal:** PLOS Computational Biology, Epidemics, or Lancet Digital Health

---

## 📝 Key Differences from Original (India) Thesis

| Aspect | Original (India) | Revised (Brazil) |
|--------|------------------|------------------|
| **Disease Focus** | Chikungunya (multi-disease mentioned) | **Chikungunya only** |
| **Geography** | Indian districts | Brazilian municipalities |
| **Data Source** | IDSP (restricted, low quality) | Mosqlimate API (open, high quality) |
| **Spatial Units** | ~700 districts | ~1,500 municipalities (with cases) |
| **Time Period** | 2013-2020 (proposed) | 2015-2020 (chik introduced 2014) |
| **Access Method** | Manual request to MoHFW | REST API with your credentials |
| **Data Quality** | "worst ever" (your words) | High (Infodengue nowcasting system) |
| **Climate Data** | Separate acquisition needed | Integrated (`receptivo` field in API) |
| **Validation** | IDSP outbreak reports | Infodengue retrospective alerts + PAHO |
| **Rt Estimates** | Need to compute manually | Provided by Infodengue model |

**Bottom line:** Brazil/Mosqlimate setup is **significantly easier** than India/IDSP for data acquisition and quality.

---

**END OF QUICK START GUIDE**

**You now have everything needed to implement your thesis. Start with API testing (Week 1), then follow the PRD phases. Good luck! 🚀**

---

## 🔗 File Download Links

1. **PRD-Chik-EWS-Brazil.md** — Click download button above
2. **Copilot-Prompt-Chik.md** — Click download button above  
3. **Revised-Thesis-Chik-Brazil.md** — Click download button above
4. **Quick-Start-Guide.md** (this file) — Click download button above

**Save all 4 files to your `docs/` folder to begin.**