$ErrorActionPreference = 'Stop'
. "$PSScriptRoot\project_env.ps1"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$transcriptPath = Join-Path $projectRoot ('logs\phase1-pipeline-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.log')
Start-Transcript -Path $transcriptPath | Out-Null
try {
    function Invoke-BaselineStep {
        param([string[]]$StepArguments)
        Write-Host ('ACTUAL_COMMAND: ' + $projectPython + ' -B -m visioninspect.baseline ' + ($StepArguments -join ' '))
        $stepOutput = & $projectPython -B -m visioninspect.baseline @StepArguments 2>&1
        $stepExit = $LASTEXITCODE
        $stepOutput | ForEach-Object { Write-Host $_ }
        Write-Host "EXIT_CODE=$stepExit"
        if ($stepExit -ne 0) { throw "Baseline step failed: $($StepArguments[0])" }
        $runLine = $stepOutput | Where-Object { "$_" -like 'RUN_DIRECTORY=*' } | Select-Object -First 1
        return "$runLine".Substring('RUN_DIRECTORY='.Length)
    }
    $dataRun = Invoke-BaselineStep -StepArguments @('check-data')
    $smokeRun = Invoke-BaselineStep -StepArguments @('smoke')
    $fitRun = Invoke-BaselineStep -StepArguments @('fit')
    $calibrationRun = Invoke-BaselineStep -StepArguments @('calibrate','--run',$fitRun)
    $evaluationRun = Invoke-BaselineStep -StepArguments @('evaluate','--run',$fitRun)
    $imagePath = Join-Path $projectRoot 'data\mvtec_ad\bottle\test\broken_large\000.png'
    $predictionRun = Invoke-BaselineStep -StepArguments @('predict','--run',$fitRun,'--image',$imagePath)
    $runSummary = @{data=$dataRun;smoke=$smokeRun;fit=$fitRun;calibration=$calibrationRun;evaluation=$evaluationRun;prediction=$predictionRun}
    $runSummary | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $projectRoot 'logs\phase1-runs.json') -Encoding utf8
} finally {
    Stop-Transcript | Out-Null
}
