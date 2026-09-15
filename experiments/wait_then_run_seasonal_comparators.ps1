$ErrorActionPreference = 'Stop'

# The winter three-fee comparator batch running when this file was created.
$WinterBatchPid = 7980
Wait-Process -Id $WinterBatchPid -ErrorAction SilentlyContinue

$Seasonal = Join-Path $PSScriptRoot 'run_seasonal_comparators.ps1'
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -NoProfile -ExecutionPolicy Bypass -File $Seasonal
if ($LASTEXITCODE -ne 0) { throw 'seasonal comparator batch failed' }
