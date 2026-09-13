$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Train = Join-Path $Root 'scripts\train_pinn_pi.py'
$Select = Join-Path $Root 'experiments\select_policy_checkpoints.py'
$Config = Join-Path $Root 'configs\confirmatory.yaml'
$Tag = 'v3_winter_fee40'
$Regime = 'winter_weekday'

# The first running campaign produced seeds 0--9. Seeds 10--24 complete the
# five independent five-restart batches frozen in revision_v3.yaml.
foreach ($Seed in 10..24) {
    Write-Output "TRAIN CENTRAL EXTRA tag=$Tag seed=$Seed"
    & $Python $Train --config $Config --region confirmatory `
        --regimes $Regime --seeds $Seed --tag $Tag `
        --threshold 300 --fee 40 --width 10 --no-adaptive
    if ($LASTEXITCODE -ne 0) { throw "training failed: $Tag seed $Seed" }

    $RunDir = Join-Path $Root "checkpoints\$Tag\confirmatory\$Regime\seed$Seed"
    & $Python $Select $RunDir --region confirmatory --regime $Regime `
        --threshold 300 --fee 40 --width 10 --max-days 30 `
        --margin 0.10 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "selection failed: $Tag seed $Seed" }
}

Write-Output 'ADDITIONAL_CENTRAL_BATCHES_COMPLETE'
