$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }

& $Python (Join-Path $PSScriptRoot 'audit_fixed_policy_gate.py')
if ($LASTEXITCODE -ne 0) { throw 'fixed-policy gate audit failed' }
& $Python (Join-Path $PSScriptRoot 'audit_selected_derivatives.py')
if ($LASTEXITCODE -ne 0) { throw 'selected derivative audit failed' }
& $Python (Join-Path $PSScriptRoot 'audit_direct_vs_projection.py')
if ($LASTEXITCODE -ne 0) { throw 'direct-versus-projection audit failed' }
& $Python (Join-Path $PSScriptRoot 'audit_deployed_bellman.py')
if ($LASTEXITCODE -ne 0) { throw 'deployment-aligned Bellman audit failed' }
& $Python (Join-Path $PSScriptRoot 'audit_model_loading.py')
if ($LASTEXITCODE -ne 0) { throw 'model-loading audit failed' }
& $Python (Join-Path $PSScriptRoot 'analyze_confirmatory.py')
if ($LASTEXITCODE -ne 0) { throw 'confirmatory analysis failed' }
& $Python (Join-Path $PSScriptRoot 'analyze_seasonal_sensitivity.py')
if ($LASTEXITCODE -ne 0) { throw 'seasonal/sensitivity analysis failed' }
& $Python (Join-Path $PSScriptRoot 'make_v3_figures.py')
if ($LASTEXITCODE -ne 0) { throw 'figure generation failed' }
& $Python (Join-Path $PSScriptRoot 'make_submission_materials.py')
if ($LASTEXITCODE -ne 0) { throw 'submission materials failed' }

$Manuscript = Join-Path $Root 'manuscript'
Push-Location $Manuscript
try {
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'main first LaTeX pass failed' }
    & bibtex main
    if ($LASTEXITCODE -ne 0) { throw 'main BibTeX pass failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'main second LaTeX pass failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw 'main final LaTeX pass failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
    if ($LASTEXITCODE -ne 0) { throw 'supplement first LaTeX pass failed' }
    & bibtex supplement
    if ($LASTEXITCODE -ne 0) { throw 'supplement BibTeX pass failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
    if ($LASTEXITCODE -ne 0) { throw 'supplement second LaTeX pass failed' }
    & pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
    if ($LASTEXITCODE -ne 0) { throw 'supplement final LaTeX pass failed' }
}
finally {
    Pop-Location
}

Write-Output 'FINAL_ANALYSIS_AND_BUILD_COMPLETE'
