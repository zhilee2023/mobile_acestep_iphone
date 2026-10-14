[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Resolve-Uv {
    $command = Get-Command uv -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $package = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($package) { return $package.FullName }
    throw 'uv is not installed. Install it with: winget install --id astral-sh.uv -e'
}

$uv = Resolve-Uv
& $uv sync --dev --python 3.12
if ($LASTEXITCODE -ne 0) { throw 'Failed to create the project environment.' }

Write-Host 'Bootstrap complete. Run scripts\run-smoke.ps1 next.'

