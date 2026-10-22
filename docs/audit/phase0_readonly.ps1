# Phase 0 audit only. Reads host/project metadata; installs and downloads nothing.
# Invoke from the project root. Python -B suppresses bytecode cache writes.
$projectRoot = 'E:\秋招\项目\VisionInspect'
Write-Output ('AUDIT_TIME=' + (Get-Date -Format o))
Write-Output ('WORKING_DIRECTORY=' + (Get-Location).Path)
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | ConvertTo-Json
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory | ConvertTo-Json
Get-CimInstance Win32_LogicalDisk -Filter 'DriveType=3' | Select-Object DeviceID,Size,FreeSpace | ConvertTo-Json
Get-Command python,python3,py,conda,mamba,uv,pip,nvidia-smi,nvcc,git,node,npm,docker -ErrorAction SilentlyContinue | Select-Object Name,Source | ConvertTo-Json
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free,compute_cap --format=csv
conda --version
conda env list --json
py -0p
uv --version
git --version
node --version
npm --version
Get-FileHash -LiteralPath "$projectRoot\VisionInspect-Agent_交接文档_含前端要求.md" -Algorithm SHA256 | Format-List

$metadataProbe = @'
import sys, importlib.metadata as m
print('INTERPRETER', sys.executable)
print('VERSION', sys.version)
print('PREFIX', sys.prefix)
print('BASE_PREFIX', sys.base_prefix)
d = {x.metadata['Name'].lower(): x.version for x in m.distributions() if x.metadata['Name']}
print({n:d.get(n, 'not installed') for n in ['torch','torchvision','anomalib','lightning','numpy','pillow','pip','timm','transformers']})
'@
$interpreters = @(
    'D:\SOFTWARE\miniconda\python.exe',
    'D:\SOFTWARE\miniconda\envs\test\python.exe',
    'C:\Users\llj\AppData\Local\Programs\Python\Python310\python.exe'
)
foreach ($interpreter in $interpreters) {
    & $interpreter -B -c $metadataProbe
    Write-Output ('METADATA_PROBE_EXIT=' + $LASTEXITCODE)
}

$runtimeProbe = @'
import sys, time, traceback
print('RUNTIME_INTERPRETER', sys.executable)
start = time.perf_counter()
try:
    import torch
    print('torch_version', torch.__version__)
    print('cuda_runtime', torch.version.cuda)
    print('arch_list', torch.cuda.get_arch_list())
    print('cuda_available', torch.cuda.is_available())
    print('cpu_tensor_sum', (torch.ones(4, 4) @ torch.ones(4, 4)).sum().item())
    import torchvision
    print('torchvision_version', torchvision.__version__)
    print('nms_result', torchvision.ops.nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([0.5]), 0.5).tolist())
except Exception:
    traceback.print_exc(limit=5)
    print('RUNTIME_PROBE_FAILED', flush=True)
    print('probe_seconds', round(time.perf_counter()-start, 3), flush=True)
    sys.exit(1)
print('probe_seconds', round(time.perf_counter()-start, 3))
'@
& 'D:\SOFTWARE\miniconda\envs\test\python.exe' -B -c $runtimeProbe
Write-Output ('TEST_RUNTIME_PROBE_EXIT=' + $LASTEXITCODE)
& 'D:\SOFTWARE\miniconda\envs\test\python.exe' -B -m pip check
Write-Output ('TEST_PIP_CHECK_EXIT=' + $LASTEXITCODE)
Write-Output 'AUDIT_COMPLETE. No dependency installation, weight download, or dataset download was performed.'
