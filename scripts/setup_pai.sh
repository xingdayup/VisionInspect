#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUTF8=1 UV_PYTHON_DOWNLOADS=never
export PIP_CACHE_DIR="$PWD/.cache/pip" TMPDIR="$PWD/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" logs
if [[ -L .venv ]]; then
  echo 'Refusing a linked .venv; use a project-owned environment directory' >&2
  exit 1
fi
platform_python=/usr/local/bin/python3
"$platform_python" -B scripts/preflight_pai_app.py 2>&1 | tee logs/pai-app-preflight.log
if [[ ! -e .venv ]]; then
  "$platform_python" -m venv .venv
fi
.venv/bin/python -B scripts/check_project_environment.py 2>&1 | tee logs/pai-app-environment-check.log
.venv/bin/python -m pip install --require-hashes --only-binary=:all: --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-agent-linux.txt 2>&1 | tee logs/pai-app-install.log
.venv/bin/python -m pip check 2>&1 | tee logs/pai-app-pip-check.log
.venv/bin/python -m pip freeze > logs/pai-app-packages.txt
.venv/bin/python -B - <<'PY'
import torch,torchvision,anomalib
from langgraph.graph import StateGraph
assert torch.__version__ == '2.12.0+cpu'
assert torchvision.__version__ == '0.27.0+cpu'
print('CPU application environment ready; platform CUDA/vLLM environment was not modified')
PY
