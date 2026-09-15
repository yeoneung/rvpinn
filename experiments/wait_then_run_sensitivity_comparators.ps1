$ErrorActionPreference = 'Stop'
$SeasonalComparatorPid = 8264
Wait-Process -Id $SeasonalComparatorPid -ErrorAction SilentlyContinue
$Run = Join-Path $PSScriptRoot 'run_sensitivity_comparators.ps1'
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -NoProfile -ExecutionPolicy Bypass -File $Run
if ($LASTEXITCODE -ne 0) { throw 'sensitivity comparator batch failed' }
