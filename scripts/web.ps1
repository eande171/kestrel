# Web Launcher
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Run = "runs/rep",
    [int]$Port = 8000,
    [switch]$NoOpen,
    [string]$Python = "D:\chess-venv\Scripts\python.exe"
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "python"
$webArgs = @("-m", "kestrel.web", "--run", $Run, "--port", $Port)
if (-not $NoOpen) { $webArgs += "--open" }
& $Python @webArgs
