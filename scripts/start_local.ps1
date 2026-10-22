param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\project_env.ps1"
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not (Test-Path -LiteralPath '.\.venv\Scripts\python.exe')) { throw '项目 .venv 缺失，请先按 docs/ENVIRONMENT.md 建立环境。' }
if (-not (Test-Path -LiteralPath '.\frontend\dist\index.html')) { throw '前端尚未构建，请先在 frontend 执行 npm ci 和 npm run build。' }
Write-Host "VisionInspect: http://127.0.0.1:$Port ，Ctrl+C 停止。"
& '.\.venv\Scripts\python.exe' -B -m uvicorn visioninspect.api:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
