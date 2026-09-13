$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Train = Join-Path $Root 'scripts\train_pinn_pi.py'
$Select = Join-Path $Root 'experiments\select_policy_checkpoints.py'
$Evaluate = Join-Path $Root 'experiments\evaluate_selected_policy.py'
$Config = Join-Path $Root 'configs\confirmatory.yaml'

foreach ($Threshold in @(250, 300, 350)) {
    foreach ($Fee in @(20, 40, 80)) {
        foreach ($Width in @(5, 10, 20)) {
            if (($Threshold -eq 300) -and ($Width -eq 10)) {
                # The central-width cells have five or 25 restarts in the
                # confirmatory batch and are analyzed from those outputs.
                continue
            }
            $Tag = "v3_sens_t${Threshold}_f${Fee}_w${Width}"
            Write-Output "SENSITIVITY tag=$Tag"
            & $Python $Train --config $Config --region confirmatory `
                --regimes winter_weekday --seeds 0 --tag $Tag `
                --threshold $Threshold --fee $Fee --width $Width --no-adaptive
            if ($LASTEXITCODE -ne 0) { throw "training failed: $Tag" }
            $RunDir = Join-Path $Root "checkpoints\$Tag\confirmatory\winter_weekday\seed0"
            & $Python $Select $RunDir --region confirmatory `
                --regime winter_weekday --threshold $Threshold --fee $Fee `
                --width $Width --max-days 30 --margin 0.10 --device cuda
            if ($LASTEXITCODE -ne 0) { throw "selection failed: $Tag" }
            & $Python $Evaluate --tag $Tag --region confirmatory `
                --regime winter_weekday --seed 0 --threshold $Threshold `
                --fee $Fee --width $Width --max-days 30 --device cuda
            if ($LASTEXITCODE -ne 0) { throw "evaluation failed: $Tag" }
        }
    }
}

Write-Output 'SENSITIVITY_TRAINING_COMPLETE'
