$ErrorActionPreference = 'Stop'
while (Get-Process -Id 7980 -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}
while (Get-Process -Id 8264 -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 20
}
& (Join-Path $PSScriptRoot 'run_convex_comparators.ps1')
if ($LASTEXITCODE -ne 0) { throw 'convex comparator batch failed' }
