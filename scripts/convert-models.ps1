[CmdletBinding()]
param(
    [string]$InputDir = '.\models\onnx',
    [string]$OutputDir = '.\models\mnn'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$python = Join-Path $root '.venv\Scripts\python.exe'
$converter = Join-Path $root '.venv\Scripts\mnnconvert.exe'
if (-not (Test-Path $python)) { throw 'Project Python environment is missing. Run scripts\bootstrap.ps1.' }
if (-not (Test-Path $converter)) { throw 'MNN converter is missing from the project environment.' }

& $python quant\convert_mnn.py --input-dir $InputDir --output-dir $OutputDir --converter $converter
if ($LASTEXITCODE -ne 0) { throw 'All-W4 conversion failed.' }
