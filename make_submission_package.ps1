$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$Source = Join-Path $Root 'manuscript'
$Target = Join-Path $Root 'submission_flat'
$Archive = Join-Path $Root 'SMPT_submission.zip'
if (Test-Path -LiteralPath (Join-Path $Source 'alignment_pending.tex')) {
    throw 'Numerical reconciliation is pending. Do not create a warning-free submission package.'
}
if (Test-Path -LiteralPath $Target) {
    throw "Submission target already exists: $Target"
}
if (Test-Path -LiteralPath $Archive) {
    throw "Preserve the previous submission archive before packaging: $Archive"
}
$AuditPython = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
foreach ($AuditScript in @('verify_round2_results.py','verify_alignment_results.py')) {
    & $AuditPython (Join-Path $Root "experiments\$AuditScript")
    if ($LASTEXITCODE -ne 0) { throw "Release audit failed: $AuditScript" }
}
& $AuditPython (Join-Path $Root 'experiments\audit_release_inputs.py') --workspace $Root
if ($LASTEXITCODE -ne 0) { throw 'Release accounting or coverage audit failed' }
& $AuditPython (Join-Path $Root 'experiments\audit_manuscript_format.py') --release
if ($LASTEXITCODE -ne 0) { throw 'Submission format or source/PDF synchronization check failed' }
New-Item -ItemType Directory -Path $Target | Out-Null

$Required = @(
    'main.tex',
    'supplement.tex',
    'references.bib',
    'bibliography_setup.tex',
    'elsarticle.cls',
    'elsarticle-num-names.bst',
    'generated\abstract_result.tex',
    'generated\fixed_policy_audit.tex',
    'generated\derivative_audit.tex',
    'generated\deployment_repair_audit.tex',
    'generated\bellman_audit.tex',
    'generated\confirmatory_results.tex',
    'generated\confirmatory_inference_supplement.tex',
    'generated\mechanism_results.tex',
    'generated\factorial_results.tex',
    'generated\factorial_supplement.tex',
    'generated\additional_results.tex',
    'generated\additional_supplement.tex',
    'generated\supplement_results.tex',
    'generated\round2_parameter_tables.tex',
    'generated\round2_reference.tex',
    'generated\round2_reference_supplement.tex',
    'generated\round2_scenarios.tex',
    'generated\round2_scenario_supplement.tex',
    'generated\round2_mechanism.tex',
    'figures\fig_factorial.pdf',
    'figures\fig_cost_latency.pdf',
    'figures\fig_round2_mechanism.pdf'
)
foreach ($Relative in $Required) {
    $Item = Join-Path $Source $Relative
    if (-not (Test-Path -LiteralPath $Item)) {
        throw "Required submission source is missing: $Item"
    }
}

Copy-Item -LiteralPath (Join-Path $Source 'references.bib') -Destination $Target
foreach ($StyleFile in @('bibliography_setup.tex','elsarticle.cls','elsarticle-num-names.bst')) {
    Copy-Item -LiteralPath (Join-Path $Source $StyleFile) -Destination $Target
}
foreach ($Name in @('abstract_result.tex','fixed_policy_audit.tex','derivative_audit.tex','deployment_repair_audit.tex','bellman_audit.tex',
                     'confirmatory_results.tex','confirmatory_inference_supplement.tex','mechanism_results.tex',
                     'factorial_results.tex','factorial_supplement.tex',
                     'additional_results.tex','additional_supplement.tex',
                     'supplement_results.tex','round2_parameter_tables.tex',
                     'round2_reference.tex','round2_reference_supplement.tex',
                     'round2_scenarios.tex','round2_scenario_supplement.tex',
                     'round2_mechanism.tex')) {
    $GeneratedText = Get-Content -LiteralPath (Join-Path $Source "generated\$Name") -Raw -Encoding utf8
    $GeneratedText = $GeneratedText.Replace('figures/fig_factorial.pdf','fig_factorial.pdf')
    $GeneratedText = $GeneratedText.Replace('figures/fig_cost_latency.pdf','fig_cost_latency.pdf')
    $GeneratedText = $GeneratedText.Replace('figures/fig_round2_mechanism.pdf','fig_round2_mechanism.pdf')
    # Generated bridge files may themselves include other generated tables.
    foreach ($RelativeInclude in $Required) {
        if ($RelativeInclude.StartsWith('generated\')) {
            $IncludeName = Split-Path -Path $RelativeInclude -Leaf
            $IncludeStem = [System.IO.Path]::GetFileNameWithoutExtension($IncludeName)
            $GeneratedText = $GeneratedText.Replace("generated/$IncludeName", $IncludeName)
            $GeneratedText = $GeneratedText.Replace("generated/$IncludeStem", $IncludeStem)
        }
    }
    Set-Content -LiteralPath (Join-Path $Target $Name) -Value $GeneratedText -Encoding utf8
}
Copy-Item -LiteralPath (Join-Path $Source 'figures\fig_factorial.pdf') -Destination $Target
Copy-Item -LiteralPath (Join-Path $Source 'figures\fig_cost_latency.pdf') -Destination $Target
Copy-Item -LiteralPath (Join-Path $Source 'figures\fig_round2_mechanism.pdf') -Destination $Target

foreach ($Name in @('main.tex','supplement.tex')) {
    $Text = Get-Content -LiteralPath (Join-Path $Source $Name) -Raw -Encoding utf8
    $Text = $Text.Replace('generated/abstract_result.tex','abstract_result.tex')
    $Text = $Text.Replace('generated/abstract_result','abstract_result')
    $Text = $Text.Replace('generated/fixed_policy_audit.tex','fixed_policy_audit.tex')
    $Text = $Text.Replace('generated/fixed_policy_audit','fixed_policy_audit')
    $Text = $Text.Replace('generated/derivative_audit.tex','derivative_audit.tex')
    $Text = $Text.Replace('generated/derivative_audit','derivative_audit')
    $Text = $Text.Replace('generated/deployment_repair_audit.tex','deployment_repair_audit.tex')
    $Text = $Text.Replace('generated/deployment_repair_audit','deployment_repair_audit')
    $Text = $Text.Replace('generated/bellman_audit.tex','bellman_audit.tex')
    $Text = $Text.Replace('generated/bellman_audit','bellman_audit')
    $Text = $Text.Replace('generated/confirmatory_results.tex','confirmatory_results.tex')
    $Text = $Text.Replace('generated/confirmatory_results','confirmatory_results')
    $Text = $Text.Replace('generated/confirmatory_inference_supplement.tex','confirmatory_inference_supplement.tex')
    $Text = $Text.Replace('generated/confirmatory_inference_supplement','confirmatory_inference_supplement')
    $Text = $Text.Replace('generated/mechanism_results.tex','mechanism_results.tex')
    $Text = $Text.Replace('generated/mechanism_results','mechanism_results')
    $Text = $Text.Replace('generated/factorial_results.tex','factorial_results.tex')
    $Text = $Text.Replace('generated/factorial_results','factorial_results')
    $Text = $Text.Replace('generated/factorial_supplement.tex','factorial_supplement.tex')
    $Text = $Text.Replace('generated/factorial_supplement','factorial_supplement')
    $Text = $Text.Replace('generated/additional_results.tex','additional_results.tex')
    $Text = $Text.Replace('generated/additional_results','additional_results')
    $Text = $Text.Replace('generated/additional_supplement.tex','additional_supplement.tex')
    $Text = $Text.Replace('generated/additional_supplement','additional_supplement')
    $Text = $Text.Replace('generated/supplement_results.tex','supplement_results.tex')
    $Text = $Text.Replace('generated/supplement_results','supplement_results')
    $Text = $Text.Replace('generated/round2_parameter_tables.tex','round2_parameter_tables.tex')
    $Text = $Text.Replace('generated/round2_parameter_tables','round2_parameter_tables')
    $Text = $Text.Replace('generated/round2_reference_supplement.tex','round2_reference_supplement.tex')
    $Text = $Text.Replace('generated/round2_reference_supplement','round2_reference_supplement')
    $Text = $Text.Replace('generated/round2_reference.tex','round2_reference.tex')
    $Text = $Text.Replace('generated/round2_reference','round2_reference')
    $Text = $Text.Replace('generated/round2_scenario_supplement.tex','round2_scenario_supplement.tex')
    $Text = $Text.Replace('generated/round2_scenario_supplement','round2_scenario_supplement')
    $Text = $Text.Replace('generated/round2_scenarios.tex','round2_scenarios.tex')
    $Text = $Text.Replace('generated/round2_scenarios','round2_scenarios')
    $Text = $Text.Replace('generated/round2_mechanism.tex','round2_mechanism.tex')
    $Text = $Text.Replace('generated/round2_mechanism','round2_mechanism')
    $Text = $Text.Replace('figures/fig_factorial.pdf','fig_factorial.pdf')
    $Text = $Text.Replace('figures/fig_cost_latency.pdf','fig_cost_latency.pdf')
    $Text = $Text.Replace('figures/fig_round2_mechanism.pdf','fig_round2_mechanism.pdf')
    Set-Content -LiteralPath (Join-Path $Target $Name) -Value $Text -Encoding utf8
}

Copy-Item -LiteralPath (Join-Path $Root 'cover_letter.tex') -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root 'highlights.docx') -Destination $Target
Copy-Item -LiteralPath (Join-Path $Root 'highlights.txt') -Destination $Target

Push-Location $Target
try {
    foreach ($Base in @('main','supplement','cover_letter')) {
        & pdflatex -interaction=nonstopmode -halt-on-error "$Base.tex"
        if ($LASTEXITCODE -ne 0) { throw "pdflatex failed: $Base" }
        if ($Base -ne 'cover_letter') {
            & bibtex $Base
            if ($LASTEXITCODE -ne 0) { throw "bibtex failed: $Base" }
            & pdflatex -interaction=nonstopmode -halt-on-error "$Base.tex"
            if ($LASTEXITCODE -ne 0) { throw "pdflatex failed: $Base" }
            & pdflatex -interaction=nonstopmode -halt-on-error "$Base.tex"
            if ($LASTEXITCODE -ne 0) { throw "pdflatex failed: $Base" }
        }
    }
}
finally {
    Pop-Location
}

& $AuditPython (Join-Path $Root 'experiments\audit_manuscript_format.py') --manuscript $Target --materials $Root --release
if ($LASTEXITCODE -ne 0) { throw 'Flattened submission format check failed' }
$ArchiveFiles = @(
    'main.tex','main.pdf','main.bbl','supplement.tex','supplement.pdf',
    'supplement.bbl','references.bib','bibliography_setup.tex','elsarticle.cls','elsarticle-num-names.bst',
    'abstract_result.tex','fixed_policy_audit.tex','derivative_audit.tex','deployment_repair_audit.tex',
    'bellman_audit.tex','confirmatory_results.tex','confirmatory_inference_supplement.tex','mechanism_results.tex',
    'factorial_results.tex','factorial_supplement.tex',
    'additional_results.tex','additional_supplement.tex',
    'supplement_results.tex','round2_parameter_tables.tex',
    'round2_reference.tex','round2_reference_supplement.tex',
    'round2_scenarios.tex','round2_scenario_supplement.tex',
    'round2_mechanism.tex','fig_factorial.pdf','fig_cost_latency.pdf',
    'fig_round2_mechanism.pdf','cover_letter.tex',
    'cover_letter.pdf','highlights.docx','highlights.txt'
) | ForEach-Object { Join-Path $Target $_ }
Compress-Archive -LiteralPath $ArchiveFiles -DestinationPath $Archive
Write-Output $Target
Write-Output $Archive
