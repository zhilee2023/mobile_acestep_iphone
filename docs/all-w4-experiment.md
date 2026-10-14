# ACE-Step 1.5 各组件 W4 敏感度记录

更新于 2026-10-14。目标是按音质和内存选择精度，并非强制所有组件 W4。

## 同一 10 秒音乐样本上的真实模型结果

参考条件：官方 Turbo DiT、Qwen3 Text Encoder 和 VAE 权重；提示词 `Soft piano ambient music`，instrumental，seed 42，8 步，48 kHz 立体声。本地 CPU 生成了未量化 PyTorch 参考 WAV。以下是固定输入下的组件输出误差，不能直接解释为听感分数。

| 组件 | 转换/量化 | 与 PyTorch 输出相对 L2 | 结论 |
| --- | --- | ---: | --- |
| Text Encoder | MNN FP | 0.00000315 | ONNX/MNN 转换准确 |
| Text Encoder | MNN W4+HQQ, block64 | 0.4193 | 偏差大；暂不用于正式音频 |
| Turbo DiT | MNN FP | 0.00000354 | ONNX/MNN 转换准确 |
| Turbo DiT | MNN W4+HQQ, block64 | 0.2426 | 单步偏差明显；需听 8 步完整音乐 |
| VAE Decoder | MNN FP | 0.0000565 | ONNX/MNN 转换准确 |
| VAE Decoder | MNN W4+HQQ, block64 | 5.8514 | 严重失真；不建议 W4 |
| VAE Decoder | MNN W4, block64（无 HQQ） | 11.0041 | 比 HQQ 更差 |
| VAE Decoder | MNN W8+HQQ, block64 | 0.0833 | 候选，仍需听感验收 |

Text Encoder MNN FP 外部权重约 2.385 GB，W4+HQQ 外部权重约 0.898 GB；两者都包含没有被压缩到 4 bit 的权重。VAE FP MNN 约 338 MB，W4+HQQ 约 44 MB，W8+HQQ 约 86 MB。W4 的实际压缩比例和数值误差必须一起看。

VAE W4 的直接输出振幅与误差同量级。已生成 `FP Text + FP DiT + W4 VAE` 的试听 WAV；这**不是**三组件全 W4 生成，且不能因为波形归一化后可以播放就判为合格。

DiT 单步 PyTorch FP32 上的 block64 仿射 W4 权重量化模拟误差为 **0.1891**；真实 MNN W4+HQQ 单步误差为 **0.2426**。MNN DiT W4 模型约 992 MB，FP 约 6.308 GB。关闭缓存的 ONNX 输出与捕获结果完全一致；MNN FP 与 PyTorch 的相对误差仅 0.00000354。这组静态模型只针对 10 秒长度导出。明细见 `outputs/w4-sensitivity/dit_probe.json` 和 `dit_mnn_validation.json`。尚无 iPhone Metal 端到端验收。

## 精度选择（当前）

- VAE：正式部署先保留 FP16/FP32；W8 可继续试听。10 秒实验中 W4 严重失真，但下方 30 秒分块解码实验的偏差明显小得多，需查明差异后才能最终排除或采用 W4。
- Text Encoder：当前 W4 数值偏差大；先保留较高精度。W8 或选择性量化可作为下一步实验。
- DiT：MNN W4 单步误差约 0.243，体积缩至约 16%；完整 8 步对比音频已生成，但音质与 iPhone 内存/速度仍未验收。

## 8 段同种子完整音乐 A/B（本地 CPU）

每段 10 秒、48 kHz 立体声、8 步、seed 42。三个组件的 W4 都是实际 MNN W4+HQQ；DiT 使用针对 10 秒张量静态导出的无缓存 MNN，在每个扩散步替代 PyTorch DiT 输出。八段音频及数值明细在用户输出目录的 `ace-step-ab-mnn-report.json`。较早的 DiT PyTorch W4 模拟版保留在 `ace-step-ab-report.json`，不要混淆。

| Text | DiT | VAE | 与全 FP 波形相关系数 | RMS（都归一化至 -1 dB 峰值） |
| --- | --- | --- | ---: | ---: |
| FP | FP | FP | 1.0000 | 0.112 |
| FP | FP | W4 | -0.4067 | 0.523 |
| W4 | FP | FP | 0.2852 | 0.068 |
| W4 | FP | W4 | 0.1266 | 0.062 |
| FP | MNN W4 | FP | 0.4702 | 0.619 |
| FP | MNN W4 | W4 | -0.4700 | 0.560 |
| W4 | MNN W4 | FP | 0.3141 | 0.039 |
| W4 | MNN W4 | W4 | -0.3077 | 0.109 |

相关系数只描述逐采样波形的相似度，不是听感/音乐质量打分。三组件 W4 版本可播放，但不代表其音质合格。该试验在 Windows MNN CPU 上用静态 10 秒图进行，尚不能充当 iPhone 15 Metal 内存、速度或质量验收。

## 30 秒 DiT 单组件实验

30 秒中文流行人声 v2 的额外实验：同一歌词、提示词、seed 17 和 8 步，专门导出 750 帧 DiT 静态模型并转成 MNN W4+HQQ（约 1.005 GB），Text Encoder 与 VAE 保持 FP。第一扩散步相对 L2 为 **0.4947**；完整音频与 FP 波形相关系数 **0.5813**。辅助语音识别能辨认 W4 版开头的部分歌词，但之后出现大量重复无意义音节，提示歌词清晰度劣化。试听文件 `ace-step-chinese-pop-vocal-30s-dit-mnn-w4.wav` 和报告在用户输出目录。此实验不应作为合格人声或 iPhone Metal 验收。

## 30 秒同歌词八组合与波形

在相同中文歌词、英文风格提示词、seed 17、8 步、48 kHz 双声道下，三组件 FP/W4 全部八种组合均已生成。W4 均为真实 Windows MNN CPU W4+HQQ、block64 权重量化；DiT 使用 750 帧静态图，Text Encoder 使用 107 token 静态图，VAE 使用 192/256/174 帧三种静态分块图。VAE W4 版本对每组相同潜变量重新分块解码，不额外改变随机生成。

| Text | DiT | VAE | 与全 FP 逐样本相关系数 | RMS | 辅助歌词识别观察 |
| --- | --- | --- | ---: | ---: | --- |
| FP | FP | FP | 1.000 | 0.110 | 能辨认部分开头歌词 |
| W4 | FP | FP | 0.340 | 0.110 | 歌词错乱，末段仍有部分词 |
| FP | W4 | FP | 0.515 | 0.114 | 开头部分词，后段重复词较多 |
| FP | FP | W4 | 0.975 | 0.106 | 与 FP 近似，仍有部分词可辨 |
| W4 | W4 | FP | 0.253 | 0.106 | 大量“嗯”等重复音节 |
| W4 | FP | W4 | 0.338 | 0.110 | 大量重复音节 |
| FP | W4 | W4 | 0.507 | 0.100 | 开头部分词，后段重复词较多 |
| W4 | W4 | W4 | 0.245 | 0.106 | 大量重复音节 |

八段 WAV、波形图 `ace-step-vocal30-ab-waveforms.png`、逐样本数值 `ace-step-vocal30-ab-waveform-report.json` 和语音识别原文 `ace-step-vocal30-ab-asr-check.json` 均在用户输出目录。语音识别对音乐可能产生幻觉；这里只作为歌词清晰度的辅助线索，不能替代人工试听。波形相关系数只描述相同时间点的振幅相似度，也不是听感评分。

本次 30 秒 VAE 首块真实输入的 MNN W4 对 PyTorch 相对 L2 为 **0.2292**，与此前 10 秒整段 VAE W4 的 **5.8514** 明显不一致；分块长度/导出图差异是否导致量化表现变化尚未排查。不能据此断言 VAE W4 对所有时长安全。

## 验收边界

这里的 W4 是 MNN 支持的权重算子量化，不保证所有参数或运行时激活都是 int4。固定输入的相对 L2 是筛选指标，最终仍需同 seed 完整生成的听感、波形、峰值内存、耗时和 iPhone 15 Metal 实测。云端 Colab GPU 尚未用于这组验证；目前数据来自本地 CPU。
