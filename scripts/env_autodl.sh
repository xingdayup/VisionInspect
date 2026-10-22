# Source this file from an AutoDL SSH/JupyterLab terminal; it installs nothing.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || return 1
export BOOTSTRAP_PYTHON=/root/miniconda3/bin/python
export VISIONINSPECT_GPU_ROOT=/root/autodl-tmp/VisionInspect-GPU
export VISIONINSPECT_VLLM_PYTHON="$VISIONINSPECT_GPU_ROOT/.venv/bin/python"
export VISIONINSPECT_RELOCATED_ASSETS=1
export VISIONINSPECT_PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
export VISIONINSPECT_WHEELHOUSE="$PWD/.cache/wheelhouse"
