$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Run = Join-Path $Root 'experiments\run_common_comparators.py'
$RunRule = Join-Path $Root 'experiments\run_rule_comparators.py'

foreach ($Fee in @(20, 40, 80)) {
    Write-Output "CONVEX COMPARATOR fee=$Fee regime=winter_weekday"
    & $Python $Run --region confirmatory --regimes winter_weekday `
        --methods convex_envelope_mpc --threshold 300 --fee $Fee --width 10 `
        --max-days 30 --time-limit 5 --mip-gap 0.01 --scenarios 2 --workers 2
    if ($LASTEXITCODE -ne 0) { throw "convex comparator failed: fee=$Fee" }
    & $Python $RunRule --region confirmatory --regime winter_weekday `
        --threshold 300 --fee $Fee --width 10 --max-days 30
    if ($LASTEXITCODE -ne 0) { throw "rule comparator failed: fee=$Fee" }
}

foreach ($Regime in @('spring_weekday', 'summer_weekday', 'autumn_weekday')) {
    Write-Output "CONVEX COMPARATOR fee=40 regime=$Regime"
    & $Python $Run --region confirmatory --regimes $Regime `
        --methods convex_envelope_mpc --threshold 300 --fee 40 --width 10 `
        --max-days 30 --time-limit 5 --mip-gap 0.01 --scenarios 2 --workers 2
    if ($LASTEXITCODE -ne 0) {
        throw "convex comparator failed: regime=$Regime"
    }
    & $Python $RunRule --region confirmatory --regime $Regime `
        --threshold 300 --fee 40 --width 10 --max-days 30
    if ($LASTEXITCODE -ne 0) {
        throw "rule comparator failed: regime=$Regime"
    }
}

Write-Output 'CONVEX_COMPARATORS_COMPLETE'
