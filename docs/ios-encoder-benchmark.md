# iPhone 文字编码器测速范围

v13 真机已完成固定 30 秒音频：DiT 八步及 VAE 十二段均输出有限值，WAV 可播放。日志起止相差约 99.56 秒，RTF 约 3.32。DiT High 阶段最高采样 footprint 约 1,966 MB；VAE Low 阶段最高采样 footprint 约 2,481 MB，当时 `os_proc_available_memory()` 约 739 MB。该余量不足以据此宣称另一个模型会话可同时驻留。`text_encoder_w4.mnn.weight` 单独在磁盘上约 899 MB，但磁盘大小不是会话内存峰值；是否能在 DiT 前串行加载，须测 Encoder 本身以及释放后再加载 DiT 的峰值。

现有 `text_encoder_w4.mnn` 是 `Qwen3-Embedding-0.6B` 的固定形状图，输入 `[1,107]` token ID，输出 `[1,107,1024]`。导出代码使用 `use_cache=False`，模型没有 `past_key_values` 输入／输出，因此不具备逐 token KV-cache decode 接口。本图的一次完整前向可以称为编码器全序列推理或 prefill 型测试；不能把 `107 / 耗时` 冒充自回归 decode token/s。当前 DiT 输入还需要 `[1,181,2048]` 的 `encoder_hidden_states` 和 `[1,750,128]` 的 `context_latents`。现有 Encoder 输出不能直接插入 DiT；歌词 tokenizer、投影／歌词条件、长度策略等尚未接通。

v14 增加独立的 Encoder 真机测速按钮。它使用打包在 App 中的捕获 107-token ID，只加载 `text_encoder_w4.mnn`，请求 MNN Metal Low／Memory Low 并允许 CPU 算子回退；分别记录会话创建、首次 `runSession`＋读取输出、三次重传输入后的平均前向耗时、输出有限值和采样 footprint。读取输出会同步 GPU；输入拷贝耗时不包含在前向计时内。日志保存到 `Documents/Generated/encoder-benchmark.log`，不会覆盖完整生成日志。未经过 profiler 验证前不能称为纯 Metal 速度。真机结果尚待用户回传。
