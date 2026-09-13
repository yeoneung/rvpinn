$ErrorActionPreference = 'Stop'
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Script = Join-Path $PSScriptRoot 'make_round2_mechanism_figure.py'

foreach ($Controller in @('learned', 'reference', 'm2', 'm16')) {
    & $Python $Script --controller $Controller
    if ($LASTEXITCODE -ne 0) {
        throw "representative-day run failed: $Controller"
    }
}
& $Python $Script --assemble
if ($LASTEXITCODE -ne 0) {
    throw 'representative-day assembly failed'
}
