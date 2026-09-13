$ErrorActionPreference = 'Stop'
$BatchEvaluationPid = 23364
Wait-Process -Id $BatchEvaluationPid -ErrorAction SilentlyContinue
$Run = Join-Path $PSScriptRoot 'run_sensitivity_training.ps1'
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -NoProfile -ExecutionPolicy Bypass -File $Run
if ($LASTEXITCODE -ne 0) { throw 'sensitivity training failed' }
