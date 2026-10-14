[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$command = Get-Command zig -ErrorAction SilentlyContinue
if ($command) {
    $zig = $command.Source
} else {
    $package = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\zig.zig_*\zig-*\zig.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $package) { throw 'Zig is required. Install it with: winget install --id zig.zig -e' }
    $zig = $package.FullName
}

& $zig c++ -std=c++20 -O2 `
    -Iruntime\include `
    runtime\src\WavWriter.cpp `
    runtime\src\AceEngine.cpp `
    runtime\src\smoke_main.cpp `
    -o outputs\ace_runtime_smoke.exe
if ($LASTEXITCODE -ne 0) { throw 'C++ runtime compilation failed.' }

& .\outputs\ace_runtime_smoke.exe .\outputs\smoke-cpp.wav
if ($LASTEXITCODE -ne 0) { throw 'C++ runtime smoke execution failed.' }

Write-Host 'C++ runtime smoke check passed.'

