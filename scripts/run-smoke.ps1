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
    throw 'uv is required. Run scripts\bootstrap.ps1 after installing uv.'
}

$uv = Resolve-Uv

& $uv run --project $root python runtime\smoke_wav.py --output outputs\smoke.wav
if ($LASTEXITCODE -ne 0) { throw 'WAV smoke test failed.' }

& $uv run --project $root pytest
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }

Write-Host 'Smoke checks passed.'

