$ErrorActionPreference = 'Stop'

$Python = if ($env:RVPINN_PYTHON) { $env:RVPINN_PYTHON } else { 'python' }
$Verifier = Join-Path $PSScriptRoot 'verify_round2_results.py'
$Analyzer = Join-Path $PSScriptRoot 'analyze_round2_scenarios.py'
$BudgetRunner = Join-Path $PSScriptRoot 'run_round2_highfee_m16_budget15.ps1'
$Finalizer = Join-Path $PSScriptRoot 'finalize_round2.ps1'
$Results = Join-Path $PSScriptRoot 'results'
$LogDir = Join-Path $PSScriptRoot 'logs'
New-Item -ItemType Directory -Path $LogDir -Force | Out-Null

# The verifier succeeds once all required five-second scenario files contain
# 30 days and 2,880 controller-call records per stream.
while ($true) {
    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = 'SilentlyContinue'
    & $Python $Verifier 1> $null 2> $null
    $VerifierExitCode = $LASTEXITCODE
    $ErrorActionPreference = $PreviousPreference
    if ($VerifierExitCode -eq 0) { break }
    Start-Sleep -Seconds 30
}

& $Python $Analyzer
if ($LASTEXITCODE -ne 0) { throw 'round-2 scenario analysis failed' }
$SummaryPath = Join-Path $Results 'round2_scenario_summary.json'
$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json

if ($Summary.high_fee_15s_triggered -and -not $Summary.high_fee_15s_complete) {
    foreach ($Stream in @(7301, 7302, 7303)) {
        $Stdout = Join-Path $LogDir "round2_highfee_m16_tl15_stream${Stream}.stdout.log"
        $Stderr = Join-Path $LogDir "round2_highfee_m16_tl15_stream${Stream}.stderr.log"
        $Arguments = @(
            '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
            $BudgetRunner, '-Stream', $Stream
        )
        Start-Process `
            -FilePath 'powershell.exe' `
            -ArgumentList $Arguments `
            -WindowStyle Hidden `
            -RedirectStandardOutput $Stdout `
            -RedirectStandardError $Stderr | Out-Null
    }
    $FinalizeStdout = Join-Path $LogDir 'round2_finalize.stdout.log'
    $FinalizeStderr = Join-Path $LogDir 'round2_finalize.stderr.log'
    $FinalizeArguments = @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
        $Finalizer, '-RequireHigh15'
    )
    Start-Process `
        -FilePath 'powershell.exe' `
        -ArgumentList $FinalizeArguments `
        -WindowStyle Hidden `
        -RedirectStandardOutput $FinalizeStdout `
        -RedirectStandardError $FinalizeStderr | Out-Null
    Write-Output '15-second streams and detached finalizer started.'
}
else {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Finalizer
    if ($LASTEXITCODE -ne 0) { throw 'round-2 finalization failed' }
}
