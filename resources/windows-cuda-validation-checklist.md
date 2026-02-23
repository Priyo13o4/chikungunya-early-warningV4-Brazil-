# Windows CUDA Bring-up Validation Checklist

Use this checklist after installing `requirements-cuda-windows.txt`.

## 0) Preflight (Windows host)
- NVIDIA driver installed and visible in terminal:
  - `nvidia-smi`
- Python environment active:
  - `python -V`
  - `python -c "import xgboost, lightgbm, pymc; print('ok')"`

## 1) Configure runtime to request GPU
In `config/model_config.yaml`:
- `compute_backend.mode: nvidia_cuda` (or `auto`)
- `compute_backend.prefer_gpu_for: both`
- Keep `compute_backend.fallback: cpu` for safe degradation.

For Bayesian sampling backend:
- `bayesian_model.sampling_backend: auto`

## 2) Run short smoke test (live output)
Run a short, cheap command first:

```bash
python -u run_pipeline.py \
  --adapter-config projects/brazil_chik/config.yaml \
  --model-config config/model_config.yaml \
  --cv-config config/cv_config.yaml \
  --raw-data data/raw/Epiclim_Final_data.csv \
  --start-year 2020 \
  --end-year 2020 \
  --selected-percentile 75 \
  --skip-visualizations \
  --bayesian-draws 50 \
  --bayesian-tune 50 \
  --bayesian-chains 1
```

## 3) Verify GPU was ACTUALLY used (not just requested)
Check generated metadata files:
- `outputs/reports/run_metadata.json`
- `outputs/metrics/bayesian_risk_metadata.json`

Required fields to inspect:
- `requested_backend`
- `resolved_backend`
- `actual_runtime_backend`
- `backend_implemented`
- `fallback_reason`

Interpretation:
- If `requested_backend=nvidia_cuda` and `actual_runtime_backend=cpu`, GPU was NOT used.
- If fallback reason mentions unavailable CUDA/JAX backend, setup is incomplete for that track.

## 4) Track-by-track expected behavior on native Windows
- Baselines:
  - XGBoost: GPU should work when CUDA is available.
  - LightGBM: may require GPU-capable build; pip wheels may not guarantee GPU build in all setups.
  - sklearn/statsmodels models remain CPU by design.
- Bayesian:
  - Native Windows commonly falls back to CPU for JAX GPU path.

## 5) If you need Bayesian GPU for production
Use WSL2 Ubuntu (or Linux host):
1. Install NVIDIA Windows driver + WSL2 CUDA support.
2. Inside WSL2 venv:
   - `pip install -U "jax[cuda13]" numpyro` (or cuda12 variant as needed)
3. Re-run checklist steps 2 and 3 inside WSL2.

## 6) Final acceptance criteria
- Smoke run exits successfully.
- `run_metadata.json` shows `requested_backend=nvidia_cuda`.
- At least one GPU-capable track reports `actual_runtime_backend` as GPU runtime (not CPU fallback).
- No unexpected strict convergence hard-fail for smoke settings.
