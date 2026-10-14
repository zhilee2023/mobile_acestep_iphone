[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $root 'upstream\ACE-Step-1.5\checkpoints'
$required = @(
    @{ Path = 'acestep-v15-turbo/model.safetensors'; Sha256 = '3f6e0797fad420a39bd33979eb6e840e30989e34a3794e843d23b60ec6e422d7' },
    @{ Path = 'Qwen3-Embedding-0.6B/model.safetensors'; Sha256 = '0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd' }
)

function Get-Sha256Hex([string] $path) {
    $stream = [System.IO.File]::OpenRead($path)
    $hasher = [System.Security.Cryptography.SHA256]::Create()
    try {
        return [System.BitConverter]::ToString($hasher.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
    } finally {
        $hasher.Dispose()
        $stream.Dispose()
    }
}

$command = Get-Command uvx -ErrorAction SilentlyContinue
if ($command) {
    $uvx = $command.Source
} else {
    $package = Get-ChildItem -Path "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_*\uvx.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $package) { throw 'uvx is required.' }
    $uvx = $package.FullName
}

foreach ($item in $required) {
    $target = Join-Path $destination ($item.Path -replace '/', '\')
    if (Test-Path -LiteralPath $target) {
        $actual = Get-Sha256Hex $target
        if ($actual -eq $item.Sha256) {
            Write-Host "Already verified: $($item.Path)"
            continue
        }
        throw "Existing file checksum mismatch; refusing to overwrite: $target"
    }

    Write-Host "Downloading from ModelScope: $($item.Path)"
    & $uvx --from modelscope-hub ms download ACE-Step/Ace-Step1.5 $item.Path --local-dir $destination
    if ($LASTEXITCODE -ne 0) { throw "ModelScope download failed: $($item.Path)" }
    if (-not (Test-Path -LiteralPath $target)) { throw "Downloaded file is missing: $target" }
    $actual = Get-Sha256Hex $target
    if ($actual -ne $item.Sha256) { throw "Checksum mismatch for $($item.Path): $actual" }
    Write-Host "Verified SHA256: $($item.Path)"
}

Write-Host 'DiT and Text Encoder weights are complete and verified.'
