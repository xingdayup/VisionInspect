#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUTF8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
export VISIONINSPECT_RELOCATED_ASSETS=1
export VISIONINSPECT_GPU_ENABLED="${VISIONINSPECT_GPU_ENABLED:-1}"
export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false
.venv/bin/python -B scripts/check_project_environment.py
.venv/bin/python -B -c 'from visioninspect.model import cuda_device; print(cuda_device())'
test -f frontend/dist/index.html || { echo 'Frontend build missing: restore the migration assets or run npm ci && npm run build in frontend'; exit 1; }
exec .venv/bin/python -B -m uvicorn visioninspect.api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
