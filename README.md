# ACE-Step 1.5 → MNN → iPhone 15

这是一个按里程碑推进的实验工程，用于把 ACE-Step 1.5 Turbo 拆成 Text Encoder、DiT 和 VAE，导出 ONNX、转换 MNN，并集成到 iOS/Metal。

> 当前状态：三个 W4 组件已在 iPhone 15 串行运行，并生成可播放的固定条件 30 秒 WAV；v16 真机总耗时 104.45 秒、RTF 3.482。v17 将 VAE 解码由十二段缩为六段，并移除已完成测试的 UI 入口，尚待真机测新 RTF。Encoder 输出仍未接入 DiT；导入的自定义歌词暂不参与生成。详见 `docs/ios-encoder-benchmark.md`。

## 已固定的首版目标

- `acestep-v15-turbo`，关闭 LM
- 10 秒、8 步、batch 1、seed 42
- 48 kHz 双声道 WAV
- 当前实验：Text Encoder、DiT、VAE 分别测试 W4 并保存未量化对照；不预设三者都必须 W4
- 模型顺序加载：Text Encoder → DiT → VAE

所有关键参数都在 [`config/pipeline.json`](config/pipeline.json)。

## Windows 快速开始

```powershell
cd C:\ace-step-mnn-ios
powershell -ExecutionPolicy Bypass -File .\scripts\check-environment.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\bootstrap.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run-smoke.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\build-runtime.ps1
```

`bootstrap.ps1` 使用 `uv` 创建 Python 3.12 隔离环境，只安装轻量开发依赖；它不会自动下载数 GB 模型，也不会安装完整 ACE-Step CUDA 依赖。

`build-runtime.ps1` 使用 Zig 的自带 C++ 工具链，避免强制安装完整 Visual Studio；也保留了标准 `runtime/CMakeLists.txt` 供 CI 或其他平台使用。

首版 Turbo、Qwen3 Text Encoder 和 VAE 权重已下载并校验。需要重新拉取时运行：

```powershell
.\scripts\download-models.ps1
```

脚本固定权重仓库版本并支持断点续传。官方 Windows 项目依赖默认绑定 CUDA 12.8；纯 CPU 导出环境必须单独处理 PyTorch CPU 轮子，不能直接把官方 `uv sync` 当成 CPU 安装命令。

## 当前可执行内容

```powershell
# 生成 1 秒 48 kHz 双声道 WAV，验证音频写入路径
uv run --project . python .\runtime\smoke_wav.py --output .\outputs\smoke.wav

# 对两个 .npy 张量计算相对误差、最大绝对误差和余弦相似度
uv run --project . python .\quant\validate_quant.py golden.npy candidate.npy

# 转换已有 ONNX，输出三个未量化模型及三个 W4 模型
.\scripts\convert-models.ps1 -InputDir .\models\onnx -OutputDir .\models\mnn
```

## 目录

- `upstream/ACE-Step-1.5/`：固定的官方源码快照，本地存在但不提交
- `export/`：捕获契约与后续 ONNX wrappers
- `quant/`：MNN 转换后的误差验证
- `cloud/`：云 GPU 上的未量化音乐参考生成入口
- `runtime/`：跨平台 C++ 接口和独立 WAV 冒烟程序
- `ios/`：SwiftUI 外壳、模型导入、Objective-C++ 桥
- `.github/workflows/build-ios.yml`：云端构建 MNN Metal 与 unsigned IPA
- `models/`：本地模型与 golden tensors（全部忽略）

## 里程碑

1. M0：Windows 环境、权重、官方 CPU 模块前向、golden tensors
2. M1：三个 ONNX 模型与 ONNX Runtime 对齐
3. M2：MNN FP/W8/W4 转换和 CPU 精度验证
4. M3：官方一致的 8 步调度与完整 C++ WAV 输出
5. M4：GitHub Actions 产出 unsigned IPA
6. M5–M7：iPhone Metal、端到端 10/30 秒与 profiling

## 重要限制

`all-w4` 是实验目标，并不表示每一个张量都会变成 4 bit。MNN 的 `weightQuantBits` 只覆盖它支持的 Conv/MatMul/LSTM 浮点权重；模型大小、运行时内存、数值误差和音质都必须实测。尤其需要比较 VAE W4 与 FP16 的音频质量。

三个真实组件的 W4 数值实测与 8 组完整音乐对比记录在 [`docs/all-w4-experiment.md`](docs/all-w4-experiment.md)。`cloud/run_reference.py` 只生成未量化的比较基准，也可用 `--vocal --lyrics-file --duration-seconds` 生成单独的人声参考；它不会把参考音频标记为 W4 结果。

目前没有证据证明 2B DiT W4 在 iPhone 15 的 MNN Metal 路径能在可接受内存和速度下运行。本仓库把“转换成功”“CPU 数值对齐”“Metal 单步可运行”“10 秒端到端生成”设为四个独立关卡，避免把编译成功误当成部署成功。

