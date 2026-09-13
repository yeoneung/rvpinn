$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Run = Join-Path $Root 'experiments\run_common_comparators.py'

foreach ($Threshold in @(250, 350)) {
    foreach ($Fee in @(20, 40, 80)) {
        Write-Output "SENSITIVITY COMPARATOR threshold=$Threshold fee=$Fee"
        & $Python $Run --region confirmatory --regimes winter_weekday `
            --methods no_storage,deterministic_exact_band_miqp `
            --threshold $Threshold --fee $Fee --width 10 --max-days 30 `
            --time-limit 5 --mip-gap 0.01 --scenarios 2 --workers 2
        if ($LASTEXITCODE -ne 0) {
            throw "sensitivity comparator failed: threshold=$Threshold fee=$Fee"
        }
    }
}

Write-Output 'SENSITIVITY_COMPARATORS_COMPLETE'
