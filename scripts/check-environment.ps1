[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$drive = Get-PSDrive -Name C
$computer = Get-CimInstance Win32_ComputerSystem

function Find-Tool([string]$Name) {
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $fallbacks = @{
        uv = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\astral-sh.uv_*\uv.exe"
        cmake = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Kitware.CMake_*\cmake-*\bin\cmake.exe"
        ninja = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\Ninja-build.Ninja_*\ninja.exe"
        zig = "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\zig.zig_*\zig-*\zig.exe"
    }
    if ($fallbacks.ContainsKey($Name)) {
        $file = Get-ChildItem -Path $fallbacks[$Name] -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($file) { return $file.FullName }
    }
    return $null
}

$report = [ordered]@{
    projectRoot = $root
    ramGB = [math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
    cFreeGB = [math]::Round($drive.Free / 1GB, 1)
    tools = [ordered]@{
        git = Find-Tool 'git'
        uv = Find-Tool 'uv'
        python = Find-Tool 'python'
        cmake = Find-Tool 'cmake'
        ninja = Find-Tool 'ninja'
        zig = Find-Tool 'zig'
        mnnconvert = Find-Tool 'mnnconvert'
        projectMNNConvert = (Join-Path $root '.venv\Scripts\mnnconvert.exe')
    }
    paths = [ordered]@{
        aceStepSource = Test-Path (Join-Path $root 'upstream\ACE-Step-1.5\pyproject.toml')
        pipelineConfig = Test-Path (Join-Path $root 'config\pipeline.json')
        turboWeights = Test-Path (Join-Path $root 'upstream\ACE-Step-1.5\checkpoints\acestep-v15-turbo\model.safetensors')
        textEncoderWeights = Test-Path (Join-Path $root 'upstream\ACE-Step-1.5\checkpoints\Qwen3-Embedding-0.6B\model.safetensors')
        vaeWeights = Test-Path (Join-Path $root 'upstream\ACE-Step-1.5\checkpoints\vae\diffusion_pytorch_model.safetensors')
        textEncoderMNN = Test-Path (Join-Path $root 'models\mnn\text_encoder_w4.mnn')
        ditMNN = Test-Path (Join-Path $root 'models\mnn\dit_w4.mnn')
        vaeMNN = Test-Path (Join-Path $root 'models\mnn\vae_decoder_w4.mnn')
    }
}

$report | ConvertTo-Json -Depth 5

if ($report.ramGB -lt 24) {
    Write-Warning 'Less than 24 GB RAM: full ONNX export is likely to fail.'
}
if ($report.cFreeGB -lt 80) {
    Write-Warning 'Less than 80 GB free on C: model/export intermediates may not fit.'
}

