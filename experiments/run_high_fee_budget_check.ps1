$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Run = Join-Path $Root 'experiments\run_common_comparators.py'

& $Python $Run --region confirmatory --regimes winter_weekday `
    --methods deterministic_exact_band_miqp,stochastic_two_stage_exact_band_miqp `
    --fee 80 --threshold 300 --width 10 --max-days 30 `
    --time-limit 15 --mip-gap 0.01 --scenarios 2 --workers 2 `
    --output-suffix tl15
if ($LASTEXITCODE -ne 0) { throw 'high-fee solver-budget check failed' }

Write-Output 'HIGH_FEE_BUDGET_CHECK_COMPLETE'
