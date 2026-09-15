$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Run = Join-Path $Root 'experiments\run_common_comparators.py'

foreach ($Fee in @(20, 40, 80)) {
    Write-Output "COMPARATORS winter fee=$Fee"
    & $Python $Run --region confirmatory --regimes winter_weekday `
        --fee $Fee --threshold 300 --width 10 --max-days 30 `
        --time-limit 5 --mip-gap 0.01 --scenarios 2 --workers 2
    if ($LASTEXITCODE -ne 0) { throw "comparator run failed: fee $Fee" }
}

Write-Output 'CONFIRMATORY_COMPARATORS_COMPLETE'
