$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Run = Join-Path $Root 'experiments\run_common_comparators.py'

foreach ($Regime in @('spring_weekday', 'summer_weekday', 'autumn_weekday')) {
    Write-Output "COMPARATORS fee=40 regime=$Regime"
    & $Python $Run --region confirmatory --regimes $Regime `
        --fee 40 --threshold 300 --width 10 --max-days 30 `
        --time-limit 5 --mip-gap 0.01 --scenarios 2 --workers 2
    if ($LASTEXITCODE -ne 0) { throw "seasonal comparator failed: $Regime" }
}

Write-Output 'SEASONAL_COMPARATORS_COMPLETE'
