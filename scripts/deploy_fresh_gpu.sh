#!/usr/bin/env bash
# One entry point for a brand-new GPU server.
#
# Order matters and is easy to get wrong by hand: the application environment
# and the vLLM environment must exist before the model can be downloaded, and
# the download refuses to start until the vLLM native check has passed.
#
#   bash scripts/deploy_fresh_gpu.sh
#
# Optional overrides:
#   BOOTSTRAP_PYTHON, APP_DOWNLOAD_BUDGET_BYTES, VLLM_DOWNLOAD_BUDGET_BYTES,
#   MINIMUM_FREE_GIB, SKIP_MODEL_DOWNLOAD=1, SKIP_VERIFY=1
set -euo pipefail
cd "$(dirname "$0")/.."

stage() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
note() { printf '    %s\n' "$*"; }

if [[ -z "${VISIONINSPECT_GPU_ROOT:-}" ]]; then
  if [[ -f scripts/env_autodl.sh ]]; then
    # shellcheck disable=SC1091
    source scripts/env_autodl.sh
    note 'loaded scripts/env_autodl.sh (VISIONINSPECT_GPU_ROOT was unset)'
  else
    echo 'Set VISIONINSPECT_GPU_ROOT or source an environment file first' >&2
    exit 1
  fi
fi

gpu_root="$VISIONINSPECT_GPU_ROOT"
gpu_python="${VISIONINSPECT_VLLM_PYTHON:-$gpu_root/.venv/bin/python}"
bootstrap="${BOOTSTRAP_PYTHON:-$(command -v python3)}"
app_budget="${APP_DOWNLOAD_BUDGET_BYTES:-3200000000}"
vllm_budget="${VLLM_DOWNLOAD_BUDGET_BYTES:-4100000000}"
minimum_free_gib="${MINIMUM_FREE_GIB:-25}"
started=$(date +%s)

stage "0/6  GPU and disk audit"
note "root:    $gpu_root"
note "python:  $gpu_python"
"$bootstrap" -B scripts/audit_gpu_environment.py --output logs/gpu-audit.json

stage "1/6  uploaded assets"
missing=0
for path in \
  data/manifests/bottle.json \
  data/mvtec_ad/bottle/train/good \
  weights/resnet18.a1_in1k/model.safetensors \
  weights/resnet18.a1_in1k/source.json \
  weights/profiles/bottle.json \
  weights/profiles/bottle-references.npz \
  frontend/dist/index.html
do
  if [[ -e "$path" ]]; then note "ok    $path"; else note "MISS  $path"; missing=1; fi
done
if [[ $missing -ne 0 ]]; then
  echo 'Assets are incomplete: upload the asset archive into the project first' >&2
  exit 1
fi

stage "2/6  application environment"
if [[ -x .venv/bin/python ]]; then
  note 'skip: .venv already exists'
else
  bash scripts/setup_gpu.sh --download-budget-bytes "$app_budget"
fi

stage "3/6  vLLM environment"
if [[ -x "$gpu_python" ]]; then
  note "skip: $gpu_python already exists"
else
  bash scripts/setup_vllm.sh --download-budget-bytes "$vllm_budget"
fi
if [[ ! -x "$gpu_python" ]]; then
  echo "vLLM interpreter missing: $gpu_python" >&2
  exit 1
fi

stage "4/6  pinned model assets"
model_dir="$gpu_root/weights/qwen3-vl-8b-instruct"
complete=1
for name in model-00001-of-00004.safetensors model-00002-of-00004.safetensors \
            model-00003-of-00004.safetensors model-00004-of-00004.safetensors source.json
do
  [[ -f "$model_dir/$name" ]] || complete=0
done
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" == 1 ]]; then
  note 'skip: SKIP_MODEL_DOWNLOAD=1'
elif [[ $complete -eq 1 ]]; then
  note 'skip: all four shards and source.json are present'
else
  if ! "$gpu_python" - <<'PROBE'
import urllib.request

with urllib.request.urlopen("https://huggingface.co", timeout=15) as response:
    response.read(1)
PROBE
  then
    echo 'Hugging Face is unreachable from this machine.' >&2
    echo 'On AutoDL run "source /etc/network_turbo" in this shell, then retry.' >&2
    exit 1
  fi
  "$gpu_python" -B scripts/pai_vllm.py preflight \
    --root "$gpu_root" --minimum-free-gib "$minimum_free_gib"
  "$gpu_python" -B scripts/pai_vllm.py download \
    --root "$gpu_root" --minimum-free-gib "$minimum_free_gib"
fi

stage "5/6  acceptance"
if [[ "${SKIP_VERIFY:-0}" == 1 ]]; then
  note 'skip: SKIP_VERIFY=1'
else
  .venv/bin/python -B scripts/verify_gpu.py --ttl-smoke
  .venv/bin/python -B scripts/verify_gpu.py --with-vlm --ttl-smoke
fi

stage "6/6  done"
note "elapsed: $(( $(date +%s) - started ))s"
note 'start the workstation with:'
note "  PORT=6006 bash scripts/start_gpu.sh"
