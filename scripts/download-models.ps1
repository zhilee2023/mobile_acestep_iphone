[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$env:HF_HUB_DOWNLOAD_TIMEOUT = '120'
# Xet transfers stalled after a network interruption on this machine. Keep
# local_dir metadata and completed files, but use the Hub's HTTP fallback.
$env:HF_HUB_DISABLE_XET = '1'
$root = Split-Path -Parent $PSScriptRoot
$upstream = Join-Path $root 'upstream\ACE-Step-1.5'
$destination = Join-Path $upstream 'checkpoints'
$revision = '19671f406d603126926c1b7e2adc169acbcade22'

if (-not (Test-Path (Join-Path $upstream 'pyproject.toml'))) {
    throw "ACE-Step source is missing at $upstream"
}

$command = Get-Command uvx -ErrorAction SilentlyContinue
if ($command) {
    $uvx = $command.Source
} else {
    $package = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_*\uvx.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $package) { throw 'uvx is required.' }
    $uvx = $package.FullName
}

Write-Host 'Downloading only the required Turbo, Text Encoder, and VAE files (about 6.34 GB)...'
& $uvx --from huggingface-hub hf download ACE-Step/Ace-Step1.5 `
    --revision $revision `
    --include 'Qwen3-Embedding-0.6B/*' `
    --include 'acestep-v15-turbo/*' `
    --include 'vae/*' `
    --include 'config.json' `
    --local-dir $destination `
    --max-workers 1
if ($LASTEXITCODE -ne 0) { throw 'Model download failed.' }

Write-Host "Required model files are available at $destination"
