#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUTF8=1 UV_PYTHON_DOWNLOADS=never
export PIP_CACHE_DIR="$PWD/.cache/pip" TMPDIR="$PWD/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" logs
if [[ -L .venv ]]; then
  echo 'Refusing a linked .venv' >&2
  exit 1
fi
bootstrap_python="${BOOTSTRAP_PYTHON:-/usr/local/bin/python3}"
index_url="${VISIONINSPECT_PYPI_INDEX:-https://pypi.org/simple}"
[[ "$index_url" == https://* ]] || { echo "Package index must use HTTPS" >&2; exit 1; }
sources=(--index-url "$index_url" --timeout 90 --retries 3)
if [[ -n "${VISIONINSPECT_WHEELHOUSE:-}" ]]; then
  sources+=(--find-links "$VISIONINSPECT_WHEELHOUSE")
  if [[ "${VISIONINSPECT_OFFLINE_INSTALL:-0}" == 1 ]]; then
    sources=(--no-index --find-links "$VISIONINSPECT_WHEELHOUSE")
  fi
fi
"$bootstrap_python" -B scripts/preflight_pai_app.py "$@" --requirements requirements.txt 2>&1 | tee logs/gpu-app-preflight.log
if [[ ! -e .venv ]]; then "$bootstrap_python" -m venv .venv; fi
.venv/bin/python -B scripts/check_project_environment.py 2>&1 | tee logs/gpu-app-environment.log
if ! .venv/bin/python -m pip --version > logs/gpu-app-pip-version.log 2>&1; then
  .venv/bin/python -m ensurepip --default-pip
fi
# FrEIA 0.2 is an approved 34,270-byte pure-Python source dependency of Anomalib.
# Pin the build backend first, then disable isolated/unlocked build dependency installs.
.venv/bin/python -B - <<'BUILD'
from pathlib import Path
import re
text=Path('requirements.txt').read_text()
match=re.search(r'^setuptools==.*?(?=^[\w-]+==|\Z)',text,re.M|re.S)
if match is None:raise RuntimeError('Pinned setuptools build backend missing')
Path('.tmp/build-requirements.txt').write_text(match.group(0))
BUILD
.venv/bin/python -m pip --isolated --cache-dir "$PIP_CACHE_DIR" install "${sources[@]}" --require-hashes --only-binary=:all: -r .tmp/build-requirements.txt
.venv/bin/python -m pip --isolated --cache-dir "$PIP_CACHE_DIR" install "${sources[@]}" --require-hashes --only-binary=:all: --no-binary=freia --no-build-isolation -r requirements.txt 2>&1 | tee logs/gpu-app-install.log
.venv/bin/python -m pip check 2>&1 | tee logs/gpu-app-pip-check.log
.venv/bin/python -m pip freeze > logs/gpu-app-packages.txt
.venv/bin/python -B - <<'CHECK'
import torch,torchvision
from visioninspect.common import configure_environment
from visioninspect.model import cuda_device
from langgraph.graph import StateGraph
from anomalib.models.image.patchcore.torch_model import PatchcoreModel
configure_environment()
if torch.__version__.split('+')[0]!='2.13.0' or torchvision.__version__.split('+')[0]!='0.28.0':
    raise RuntimeError('Unexpected CUDA PyTorch versions')
if not torch.version.cuda or not torch.version.cuda.startswith('13.'):
    raise RuntimeError('Expected a CUDA 13 PyTorch build; CPU builds are forbidden')
device=cuda_device()
tensor=torch.ones((64,64),device=device)
result=tensor@tensor
boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.]],device=device)
kept=torchvision.ops.nms(boxes,torch.tensor([.9,.8],device=device),.5)
torch.cuda.synchronize(device)
if not torch.isfinite(result).all().item() or result[0,0].item()!=64 or kept.tolist()!=[0]:
    raise RuntimeError('CUDA matrix / torchvision operator smoke failed')
print('GPU application environment ready; platform packages were not modified')
CHECK
