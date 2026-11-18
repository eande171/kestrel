# Play Launcher
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Run = "runs/rep",
    [ValidateSet("white", "black", "random")][string]$Color = "random",
    [int]$Depth = 3,
    [string]$Python = "D:\chess-venv\Scripts\python.exe"
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:PYTHONPATH = Join-Path $root "python"
& $Python -m chess_ai.play --run $Run --color $Color --search-depth $Depth
