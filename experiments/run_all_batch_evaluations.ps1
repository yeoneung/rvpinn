$ErrorActionPreference = 'Stop'

$Root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Evaluate = Join-Path $Root 'experiments\evaluate_restart_batches.py'

$Cells = @(
    @{Tag='v3_winter_fee20'; Regime='winter_weekday'; Fee=20; Seeds='0,1,2,3,4'},
    @{Tag='v3_winter_fee40'; Regime='winter_weekday'; Fee=40; Seeds='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24'},
    @{Tag='v3_winter_fee80'; Regime='winter_weekday'; Fee=80; Seeds='0,1,2,3,4'},
    @{Tag='v3_spring_fee40'; Regime='spring_weekday'; Fee=40; Seeds='0,1,2,3,4'},
    @{Tag='v3_summer_fee40'; Regime='summer_weekday'; Fee=40; Seeds='0,1,2,3,4'},
    @{Tag='v3_autumn_fee40'; Regime='autumn_weekday'; Fee=40; Seeds='0,1,2,3,4'}
)

foreach ($Cell in $Cells) {
    Write-Output "EVALUATE tag=$($Cell.Tag)"
    & $Python $Evaluate --tag $Cell.Tag --region confirmatory `
        --regime $Cell.Regime --seeds $Cell.Seeds --threshold 300 `
        --fee $Cell.Fee --width 10 --max-days 30 --margin 0.10 --device cuda
    if ($LASTEXITCODE -ne 0) { throw "evaluation failed: $($Cell.Tag)" }
}

Write-Output 'LEARNED_BATCH_EVALUATION_COMPLETE'
