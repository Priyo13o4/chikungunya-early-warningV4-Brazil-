#!/usr/bin/env bash
set -euo pipefail

if [[ "${OSTYPE:-}" == darwin* ]]; then
  echo "This script is intended for Linux GCP CUDA instances, not macOS."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Ensure NVIDIA driver is installed and GPU VM is provisioned."
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON_BIN' not found. Set PYTHON_BIN to a valid Python 3.10+ binary."
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
# Confirmed via JAX docs: "jax[cuda12]" is the correct and current specifier.
python -m pip install --upgrade "jax[cuda12]" numpyro

echo -e "\n=== GPU Driver Check ==="
nvidia-smi

echo -e "\n=== JAX CUDA Check ==="
python - <<'PY'
import jax
print("jax version:", jax.__version__)
print("jax devices:", jax.devices())
print("default backend:", jax.default_backend())
if not any(getattr(device, "platform", "") in {"gpu", "cuda"} for device in jax.devices()):
    raise SystemExit("No CUDA JAX device detected. Check CUDA driver/runtime compatibility.")
PY

echo -e "\nSetup complete. Activate with: source ${VENV_DIR}/bin/activate"
echo "Run smoke test (recommended):"
echo "python -u run_pipeline.py --adapter-config projects/brazil_chik/config.yaml --model-config config/model_config.yaml --cv-config config/cv_config.yaml --raw-data data/raw/Epiclim_Final_data.csv --start-year 2020 --end-year 2020 --selected-percentile 75 --skip-visualizations --bayesian-profile-mode dev"