#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VISIONINSPECT_GPU_ROOT:?Set the audited GPU asset directory first}"
bootstrap_python="${BOOTSTRAP_PYTHON:-/usr/local/bin/python3}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONUTF8=1
export PIP_CACHE_DIR="$PWD/.cache/pip" TMPDIR="$PWD/.tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" logs
"$bootstrap_python" -B scripts/preflight_pai_app.py "$@" --requirements requirements-vllm.txt 2>&1 | tee logs/gpu-vllm-preflight.log
gpu_root="$("$bootstrap_python" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$VISIONINSPECT_GPU_ROOT")"
if [[ -L "$VISIONINSPECT_GPU_ROOT" || -L "$gpu_root/.venv" ]]; then
  echo 'Refusing a linked GPU environment directory' >&2
  exit 1
fi
mkdir -p "$gpu_root"
if [[ ! -e "$gpu_root/.venv" ]]; then "$bootstrap_python" -m venv "$gpu_root/.venv"; fi
gpu_python="$gpu_root/.venv/bin/python"
"$gpu_python" -B - "$gpu_root" <<'CHECK'
import pathlib,sys
expected=pathlib.Path(sys.argv[1])/'.venv'
if sys.version_info[:2]!=(3,12) or pathlib.Path(sys.prefix).resolve()!=expected or sys.prefix==sys.base_prefix:
    raise RuntimeError('Expected an owned, isolated Python 3.12 GPU environment')
text=(expected/'pyvenv.cfg').read_text().lower()
if 'include-system-site-packages = false' not in text:
    raise RuntimeError('System package sharing is forbidden')
CHECK
index_url="${VISIONINSPECT_PYPI_INDEX:-https://pypi.org/simple}"
[[ "$index_url" == https://* ]] || { echo "Package index must use HTTPS" >&2; exit 1; }
sources=(--index-url "$index_url" --timeout 90 --retries 3)
if [[ -n "${VISIONINSPECT_WHEELHOUSE:-}" ]]; then
  sources+=(--find-links "$VISIONINSPECT_WHEELHOUSE")
  if [[ "${VISIONINSPECT_OFFLINE_INSTALL:-0}" == 1 ]]; then
    sources=(--no-index --find-links "$VISIONINSPECT_WHEELHOUSE")
  fi
fi
"$gpu_python" -m pip --isolated --cache-dir "$PIP_CACHE_DIR" install "${sources[@]}" --require-hashes --only-binary=:all: -r requirements-vllm.txt 2>&1 | tee logs/gpu-vllm-install.log
"$gpu_python" -m pip check
"$gpu_python" -m pip freeze > logs/gpu-vllm-packages.txt
"$gpu_python" -B scripts/pai_vllm.py check --root "$gpu_root"
echo "Use VISIONINSPECT_VLLM_PYTHON=$gpu_python for the application"
