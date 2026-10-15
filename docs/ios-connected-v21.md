# iPhone v21：自定义风格、歌词与 30 秒歌曲

v21 新增真正连接的端侧条件链：用户风格描述经 Qwen 分词、Qwen3-Embedding-0.6B MNN W4、ACE-Step `text_projector`（1024→2048），放入 DiT 条件的最后 107 个位置。用户歌词经同一分词器的词嵌入表、ACE-Step 八层歌词编码器 MNN W4，放入前 73 个位置。中间 1 个 timbre token 仍是无参考音频的固定默认条件。随后 DiT Metal High 八步输出 750×64 latent，VAE CPU Low 六段解码为 30 秒 WAV。Qwen、歌词编码器、DiT 和 VAE 串行创建与释放，前两次阶段切换须连续通过 2400 MB 可用内存门槛；NoTune 整首对照也须通过此门槛。

静态图限制没有消失：风格必须恰好 107 token，歌词必须恰好 73 token。App 对不符长度显示错误，不截断、不补齐。VAE、DiT 仍是固定 30 秒，初始 latent 与无参考音频 timbre 使用原固定样例。当前不支持任意长度、实时 DiT 流式、参考音频 timbre 定制。

额外文件为 `text_projector.f16`，大小 4,194,304 字节。用 `.venv-reference/Scripts/python.exe export/export_text_projector.py` 从 ACE-Step 权重导出并自检，连同原七个基础文件和两个歌词文件导入 App。原始 FP 投影对参考 DiT 风格 slice 的 L2 为 3.43e-7；本地 MNN CPU Qwen W4 加投影对原始 FP 条件 L2 为 0.402557，改 `piano` 为 `guitar` 时两份 W4 投影之间 L2 为 0.168431。此数值说明输入真正影响风格，但也说明 Qwen W4 误差显著；不能据此保证歌曲音质或风格忠实度，必须真机试听。

真机先用预置风格与歌词测试完整链路，再用相同 token 数的改写测试。分享 `Generated/style-encoder.log`、`Generated/last-inference-custom-cpu-vae.log`、WAV 耗时和 RTF；若失败，先看失败阶段及系统 jetsam 日志。固定 CPU VAE 按钮只是旧固定条件基线，不算动态风格测速。

VAE 的 128 latent 帧主体相当于 5.12 秒音频；日志现在报告每段解码完成后的“已解码音频时长”和“从生成开始耗时”。这仅测量潜在首段可播放时间，目前 App 仍等待六段全部完成、统一写 WAV 后才播放。DiT 目前每一步必须计算完整 750 帧，故不具备从开头直接生成首段的真流式能力。未来可尝试在完整 latent 已得后边解码边播放 VAE 段，需处理拼接、播放缓冲、内存和峰值归一化；这不会消除前置 Qwen/DiT 等待。

NoTune 独立 192 帧试验曾测得会话创建 0.131 秒、两次解码 2.815/2.802 秒，均为有限值且非静音。v21 另设完整固定条件 NoTune 按钮，单独验证 DiT 后的整首初始化、六段耗时、峰值内存和听感；它尚非默认后端。
