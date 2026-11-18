# Starts or resumes training and restarts it after a crash.
# Usage: .\scripts\train.ps1 [-Run runs/main] [-Data data/shards_2026-10] [-Quiet] [extra train.py options]
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Run = "runs/main",
    [string]$Data = "data/shards_2026-10",
    [switch]$Quiet,
    [int]$MaxRestarts = 20,
    [string]$Python = "D:\chess-venv\Scripts\python.exe",
    [Parameter(ValueFromRemainingArguments = $true)] $Extra
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "python"

$trainArgs = @("-m", "chess_ai.train", "--run", $Run, "--data", $Data)
if ($Quiet) { $trainArgs += "--quiet" }
if ($Extra) { $trainArgs += $Extra }

for ($i = 0; $i -le $MaxRestarts; $i++) {
    & $Python @trainArgs
    if ($LASTEXITCODE -eq 0) { break }
    Write-Host "training exited with code $LASTEXITCODE, restarting in 30 seconds (restart $($i + 1) of $MaxRestarts)"
    Start-Sleep -Seconds 30
}
