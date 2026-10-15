# iPhone 固定歌曲端侧推理范围

首版真机生成只支持捕获条件固定的 30 秒中文流行样例。`ios/Probe` 内的首步噪声、文本条件和 context latents 是开发机上从官方模型捕获的浮点张量；手机并未从当前界面输入重新计算这些条件。因此界面中的歌词/风格提示可以编辑和导入，但**不会改变固定样例的生成结果**。

真正留在 iPhone 上运行的部分是 MNN W4 DiT 八步 Euler 推理，以及三个固定形状 W4 VAE 图的六段解码。采样时间表为 `1, .875, .75, .625, .5, .375, .25, .125`，30 秒音频为 48 kHz 双声道。推理输出保存为 App `Documents/Generated` 中的 PCM16 WAV，播放器读取该文件；RTF 只在完整生成成功时记录。MNN 请求 Metal 后端并允许 CPU 算子回退，因此未经真机 profiler 确认不能声称纯 Metal。

真机 v5 的 JetsamEvent 记录到 `AceStepMobile` 在约 3.27 GB 驻留内存时因 `per-process-limit` 被系统终止。v6 将完整生成（不是单步精度探针）的 MNN Metal 精度从 High 改为 Low，请求更低精度的中间数据；会话创建后调用 `releaseModel()`，并在每步前保留至少 512 MiB 的系统报告进程余量。上述设置只是假设性的减峰方案，不能保证 30 秒推理成功或声音质量不变；必须重新在 iPhone 上验证。每次生成会覆盖 `Documents/Generated/last-inference.log`，逐阶段记录进程 footprint 和剩余可用内存，App 内可直接查看最近阶段并分享日志。即使被 Jetsam 结束，已刷盘的最后阶段仍保留。单步探针继续使用 High 精度，故其 L2 不代表 v6 完整生成的数值误差。

Windows CPU 验证脚本 `scripts/validate-ios-fixed-song.py` 已完成八步 DiT、六段 VAE，生成 1,440,000 帧音频；首步 W4 对官方捕获输出的相对 L2 为 0.4947。该数值只说明图执行，不保证歌声音质或歌词清晰度。iPhone 真机需要先运行单步探针，再运行固定样例，检查是否闪退、峰值内存、WAV 是否可播放、耗时/RTF 与人耳听感。

任意上传歌词要真正参与端侧生成，仍需扩展导出：Qwen tokenizer、歌词 token embedding、DiT 条件编码器、固定形状的长度策略，以及这些组件的 iPhone 数值/内存验证。目前的七个 W4 文件不足以完成这件事。不要把本固定样例称为“任意歌词全 W4 端侧推理”。
