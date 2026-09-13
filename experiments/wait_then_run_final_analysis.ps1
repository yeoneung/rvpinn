$ErrorActionPreference = 'Stop'

# These watchers cover the learned-policy batches and factorial training, the
# threshold comparators, and all seasonal/convex/rule comparators.
foreach ($ExperimentPid in @(7428, 26012, 17292, 26224)) {
    Wait-Process -Id $ExperimentPid -ErrorAction SilentlyContinue
}
$Run = Join-Path $PSScriptRoot 'run_final_analysis.ps1'
& 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe' `
    -NoProfile -ExecutionPolicy Bypass -File $Run
if ($LASTEXITCODE -ne 0) { throw 'final analysis failed' }
