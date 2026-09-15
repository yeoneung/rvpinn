$ErrorActionPreference = 'Stop'
$TrainingBatchPid = 3892
Wait-Process -Id $TrainingBatchPid -ErrorAction SilentlyContinue
$Extra = Join-Path $PSScriptRoot 'run_additional_central_batches.ps1'
$Run = Join-Path $PSScriptRoot 'run_all_batch_evaluations.ps1'
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -NoProfile -ExecutionPolicy Bypass -File $Extra
if ($LASTEXITCODE -ne 0) { throw 'additional central batches failed' }
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -NoProfile -ExecutionPolicy Bypass -File $Run
if ($LASTEXITCODE -ne 0) { throw 'learned batch evaluation failed' }
