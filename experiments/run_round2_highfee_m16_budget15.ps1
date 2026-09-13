param(
    [Parameter(Mandatory = $true)]
    [ValidateSet(7301, 7302, 7303)]
    [int]$Stream
)

$ErrorActionPreference = 'Stop'
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Runner = Join-Path $PSScriptRoot 'run_common_comparators.py'
$Suffix = "r2_stream${Stream}_pool16_tl15"

& $Python $Runner `
    --region confirmatory `
    --regimes winter_weekday `
    --methods stochastic_two_stage_exact_band_miqp `
    --threshold 300 --fee 80 --width 10 --max-days 30 `
    --time-limit 15 --mip-gap 0.01 `
    --scenarios 16 --scenario-pool-size 16 `
    --seed $Stream --workers 2 --output-suffix $Suffix
if ($LASTEXITCODE -ne 0) {
    throw "scenario run failed: fee=80 M=16 stream=$Stream budget=15"
}
