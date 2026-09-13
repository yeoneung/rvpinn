$ErrorActionPreference = 'Stop'

$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Runner = Join-Path $PSScriptRoot 'run_common_comparators.py'
$LogDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
$Log = Join-Path $LogDir 'round2_scenario_sensitivity_transcript.log'

Start-Transcript -LiteralPath $Log -Append
try {
    foreach ($Fee in @(40, 80)) {
        $Counts = if ($Fee -eq 40) { @(2, 8, 16) } else { @(16) }
        foreach ($Stream in @(7301, 7302, 7303)) {
            foreach ($Count in $Counts) {
                $Suffix = "r2_stream${Stream}_pool16_tl5"
                & $Python $Runner `
                    --region confirmatory `
                    --regimes winter_weekday `
                    --methods stochastic_two_stage_exact_band_miqp `
                    --threshold 300 --fee $Fee --width 10 --max-days 30 `
                    --time-limit 5 --mip-gap 0.01 `
                    --scenarios $Count --scenario-pool-size 16 `
                    --seed $Stream --workers 2 --output-suffix $Suffix
                if ($LASTEXITCODE -ne 0) {
                    throw "scenario run failed: fee=$Fee M=$Count stream=$Stream"
                }
            }
        }
    }
}
finally {
    Stop-Transcript
}
