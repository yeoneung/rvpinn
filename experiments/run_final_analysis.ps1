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
& $Python (Join-Path $PSScriptRoot 'make_study_figures.py')
if ($LASTEXITCODE -ne 0) { throw 'figure generation failed' }
Write-Output 'EXPERIMENT_ANALYSIS_COMPLETE'
