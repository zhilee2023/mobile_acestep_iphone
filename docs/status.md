# 实施状态

## 已完成

- 在 C 盘创建独立 Git 工程
- 固定 ACE-Step 与 MNN 上游提交
- 建立 Windows 环境检查、Python 3.12 隔离环境和模型下载入口
- 建立 MNN 转换参数入口和 NumPy 数值误差验证器
- 建立 C++20 运行时边界、WAV 写入器和冒烟程序
- 建立 SwiftUI 模型导入页面及 unsigned IPA 云端构建流程
- 建立 Text Encoder / DiT / VAE 三组件 W4+HQQ 实验配置和转换工具
- 用合成 MatMul 实测 ONNX → MNN W4 转换与 MNN CPU 前向（数据见 `docs/all-w4-experiment.md`）
- 三个官方权重下载完成并逐文件校验 SHA-256
- 本地 CPU 生成 10 秒未量化参考音乐；30 秒中文流行初版未唱出歌词，已重生成人声候选 v2（自动识别到部分歌词，仍待人工试听验收）
- 捕获三组件真实输入输出并分别导出静态 ONNX
- 三组件 MNN FP/W4+HQQ 转换与真实输入单步数值验证完成；VAE 另测 W8
- 同一 seed 的 8 种 FP/W4 组合音频已由 Windows CPU 生成（DiT W4 为真实 MNN 静态模型）
- 30 秒中文流行 v2 已生成 DiT MNN W4+HQQ 对照音频（Text/VAE 保持 FP）；单步相对 L2 0.495，自动识别仅辨认开头歌词，音质尚未验收
- 同一歌词、提示词、seed 的 30 秒中文流行 8 种 FP/W4 组合已由 Windows MNN CPU 生成，并输出波形图、波形数值和辅助语音识别结果；部分 W4 组合不能清晰唱出歌词

## 尚未完成

- 把官方 flow-matching timesteps、condition 构造和 VAE 后处理移植到 C++
- 在 iPhone 15 上验证 MNN Metal 单步、内存峰值和 10 秒端到端生成
- 在云端 GPU 或 iPhone 15 上复验 8 组音频；当前为 Windows MNN CPU 验证
- 为非 10 秒长度导出/验证可用的 MNN 动态或专用静态图

10 秒对比音频是真实官方权重推理；FP 路径使用 PyTorch，W4 组件使用 MNN W4+HQQ。`outputs/smoke.wav` 仍只验证 WAV I/O。

## 权重下载恢复

此前断网后的下载已恢复，Turbo DiT、Text Encoder 和 VAE 权重均完整；ModelScope 镜像下载结果与固定 Hugging Face 版本的 SHA-256 一致。

