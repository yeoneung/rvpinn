param(
    [switch]$RequireHigh15
)

$ErrorActionPreference = 'Stop'
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Analyzer = Join-Path $PSScriptRoot 'analyze_round2_scenarios.py'
$MechanismRunner = Join-Path $PSScriptRoot 'run_round2_mechanism.ps1'

if ($RequireHigh15) {
    while ($true) {
        $PreviousPreference = $ErrorActionPreference
        $ErrorActionPreference = 'SilentlyContinue'
        & $Python $Analyzer 1> $null 2> $null
        $AnalyzerExitCode = $LASTEXITCODE
        $ErrorActionPreference = $PreviousPreference
        if ($AnalyzerExitCode -eq 0) {
            $SummaryPath = Join-Path $PSScriptRoot 'results\round2_scenario_summary.json'
            $Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
            if ($Summary.high_fee_15s_complete) { break }
        }
        Start-Sleep -Seconds 30
    }
}
else {
    & $Python $Analyzer
    if ($LASTEXITCODE -ne 0) { throw 'round-2 scenario analysis failed' }
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $MechanismRunner
if ($LASTEXITCODE -ne 0) { throw 'round-2 mechanism run failed' }

& $Python (Join-Path $PSScriptRoot 'analyze_round2_reference.py')
if ($LASTEXITCODE -ne 0) { throw 'round-2 reference analysis failed' }
& $Python (Join-Path $PSScriptRoot 'make_round2_parameter_tables.py')
if ($LASTEXITCODE -ne 0) { throw 'round-2 parameter table generation failed' }

$Manuscript = Join-Path $Root 'manuscript'
Push-Location $Manuscript
try {
    foreach ($Base in @('main', 'supplement')) {
        & pdflatex -interaction=nonstopmode -halt-on-error "$Base.tex"
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed: $Base" }
        & bibtex $Base
        if ($LASTEXITCODE -ne 0) { throw "bibtex failed: $Base" }
        & pdflatex -interaction=nonstopmode -halt-on-error "$Base.tex"
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed: $Base" }
        & pdflatex -interaction=nonstopmode -halt-on-error "$Base.tex"
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed: $Base" }
    }
}
finally {
    Pop-Location
}

Write-Output 'Round-2 generated artifacts and draft PDF builds completed.'
