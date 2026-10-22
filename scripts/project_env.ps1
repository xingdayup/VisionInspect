$projectRoot = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot '.cache\uv'
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.cache\pip'
$env:HF_HOME = Join-Path $projectRoot '.cache\huggingface'
$env:TORCH_HOME = Join-Path $projectRoot '.cache\torch'
$env:TEMP = Join-Path $projectRoot '.tmp'
$env:TMP = $env:TEMP
$env:UV_PYTHON_DOWNLOADS = 'never'
$env:UV_TORCH_BACKEND = 'cpu'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:NPM_CONFIG_CACHE = Join-Path $projectRoot '.cache\npm'
foreach ($directory in @($env:UV_CACHE_DIR,$env:PIP_CACHE_DIR,$env:HF_HOME,$env:TORCH_HOME,$env:TEMP,(Join-Path $projectRoot 'logs'))) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}
