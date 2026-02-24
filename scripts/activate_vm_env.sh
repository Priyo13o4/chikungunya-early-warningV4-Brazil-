#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${REPO_ROOT}/.venv"

if [[ ! -d "${VENV_PATH}" ]]; then
  echo "Virtual environment not found at ${VENV_PATH}. Create it first with: python3 -m venv .venv"
  return 1 2>/dev/null || exit 1
fi

source "${VENV_PATH}/bin/activate"

NVIDIA_LIB_PATHS="$(python -c "import glob,site; print(':'.join([p for b in site.getsitepackages() for p in glob.glob(b + '/nvidia/*/lib')]))")"
if [[ -n "${NVIDIA_LIB_PATHS}" ]]; then
  export LD_LIBRARY_PATH="${NVIDIA_LIB_PATHS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "Activated ${VENV_PATH}"
echo "PYTHONPATH=${REPO_ROOT}"