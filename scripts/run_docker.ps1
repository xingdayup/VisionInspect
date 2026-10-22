param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectDirectory = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
docker info *> $null
if ($LASTEXITCODE -ne 0) { throw 'Docker 引擎不可用。先由用户修复 Docker/WSL；本脚本不会修改系统组件。' }
docker build --platform linux/amd64 --tag visioninspect-cpu:0.3.0 $projectDirectory
if ($LASTEXITCODE -ne 0) { throw '镜像构建失败；未启动容器。' }
docker run --rm --name visioninspect-cpu --cpus 4 --memory 12g -p "127.0.0.1:${Port}:8000" `
  --mount "type=bind,source=$projectDirectory\weights,target=/app/weights,readonly" `
  --mount "type=bind,source=$projectDirectory\data,target=/app/data,readonly" `
  --mount "type=bind,source=$projectDirectory\outputs,target=/app/outputs" `
  visioninspect-cpu:0.3.0
exit $LASTEXITCODE
