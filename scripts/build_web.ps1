# Web Build
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Model = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# WebAssembly Engine
Push-Location engine/web
cargo build --release --target wasm32-unknown-unknown --target-dir ../target-web
$built = $LASTEXITCODE
Pop-Location
if ($built -ne 0) { throw "WebAssembly build failed" }
Copy-Item engine/target-web/wasm32-unknown-unknown/release/kestrel_web.wasm docs/play/kestrel_web.wasm -Force

# Model Swap
if ($Model) { Copy-Item $Model docs/play/model.kstn -Force }

Get-ChildItem docs/play/kestrel_web.wasm, docs/play/model.kstn | ForEach-Object { "{0} {1:N2} MB" -f $_.Name, ($_.Length / 1MB) }
