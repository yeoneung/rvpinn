$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Train = Join-Path $Root 'scripts\train_pinn_pi.py'
$Select = Join-Path $Root 'experiments\select_policy_checkpoints.py'
$Evaluate = Join-Path $Root 'experiments\evaluate_restart_batches.py'
$Config = Join-Path $Root 'configs\confirmatory.yaml'

function Run-Cell {
    param(
        [string]$Tag,
        [string]$Regime,
        [double]$Fee,
        [int[]]$Seeds
    )
    foreach ($Seed in $Seeds) {
        Write-Output "TRAIN tag=$Tag regime=$Regime fee=$Fee seed=$Seed"
        & $Python $Train --config $Config --region confirmatory `
            --regimes $Regime --seeds $Seed --tag $Tag `
            --threshold 300 --fee $Fee --width 10 --no-adaptive
        if ($LASTEXITCODE -ne 0) { throw "training failed: $Tag seed $Seed" }

        $RunDir = Join-Path $Root "checkpoints\$Tag\confirmatory\$Regime\seed$Seed"
        & $Python $Select $RunDir --region confirmatory --regime $Regime `
            --threshold 300 --fee $Fee --width 10 --max-days 30 `
            --margin 0.10 --device cuda
        if ($LASTEXITCODE -ne 0) { throw "selection failed: $Tag seed $Seed" }
    }
    $SeedText = $Seeds -join ','
    & $Python $Evaluate --tag $Tag --region confirmatory --regime $Regime `
        --seeds $SeedText --threshold 300 --fee $Fee --width 10 `
        --max-days 30 --margin 0.10 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "batch evaluation failed: $Tag" }
}

# Primary anchors: five independent five-restart batches at fee 40 and one
# five-restart batch at fees 20 and 80.
Run-Cell -Tag 'v3_winter_fee20' -Regime 'winter_weekday' -Fee 20 -Seeds (0..4)
Run-Cell -Tag 'v3_winter_fee40' -Regime 'winter_weekday' -Fee 40 -Seeds (0..24)
Run-Cell -Tag 'v3_winter_fee80' -Regime 'winter_weekday' -Fee 80 -Seeds (0..4)

# Seasonal expansion at the central fee. Winter is covered by the five batches
# above; the other seasons each receive five independent restarts.
Run-Cell -Tag 'v3_spring_fee40' -Regime 'spring_weekday' -Fee 40 -Seeds (0..4)
Run-Cell -Tag 'v3_summer_fee40' -Regime 'summer_weekday' -Fee 40 -Seeds (0..4)
Run-Cell -Tag 'v3_autumn_fee40' -Regime 'autumn_weekday' -Fee 40 -Seeds (0..4)

Write-Output 'CONFIRMATORY_TRAINING_COMPLETE'
