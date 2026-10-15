# iPhone 文字编码器测速范围

v13 真机已完成固定 30 秒音频：DiT 八步及 VAE 十二段均输出有限值，WAV 可播放。日志起止相差约 99.56 秒，RTF 约 3.32。DiT High 阶段最高采样 footprint 约 1,966 MB；VAE Low 阶段最高采样 footprint 约 2,481 MB，当时 `os_proc_available_memory()` 约 739 MB。该余量不足以据此宣称另一个模型会话可同时驻留。`text_encoder_w4.mnn.weight` 单独在磁盘上约 899 MB，但磁盘大小不是会话内存峰值；是否能在 DiT 前串行加载，须测 Encoder 本身以及释放后再加载 DiT 的峰值。

现有 `text_encoder_w4.mnn` 是 `Qwen3-Embedding-0.6B` 的固定形状图，输入 `[1,107]` token ID，输出 `[1,107,1024]`。导出代码使用 `use_cache=False`，模型没有 `past_key_values` 输入／输出，因此不具备逐 token KV-cache decode 接口。本图的一次完整前向可以称为编码器全序列推理或 prefill 型测试；不能把 `107 / 耗时` 冒充自回归 decode token/s。当前 DiT 输入还需要 `[1,181,2048]` 的 `encoder_hidden_states` 和 `[1,750,128]` 的 `context_latents`。现有 Encoder 输出不能直接插入 DiT；歌词 tokenizer、投影／歌词条件、长度策略等尚未接通。

v14 增加独立的 Encoder 真机测速按钮。它使用打包在 App 中的捕获 107-token ID，只加载 `text_encoder_w4.mnn`，请求 MNN Metal Low／Memory Low 并允许 CPU 算子回退；分别记录会话创建、首次 `runSession`＋读取输出、三次重传输入后的平均前向耗时、输出有限值和采样 footprint。读取输出会同步 GPU；输入拷贝耗时不包含在前向计时内。日志保存到 `Documents/Generated/encoder-benchmark.log`，不会覆盖完整生成日志。未经过 profiler 验证前不能称为纯 Metal 速度。真机会话创建 1.698 秒，首次前向 0.195 秒，预热后平均 0.180 秒，采样峰值 2,127 MB。会话释放后立即采样仍为 2,087 MB；该读数发生在测速函数返回前，不足以判断之后能否安全加载 DiT。

v15 增加“Encoder → DiT 串行内存”按钮：完整运行独立 Encoder 测速并离开其自动释放作用域，在立即、1 秒、5 秒三个时点记录 `phys_footprint` 与 `os_proc_available_memory()`。5 秒后若可用内存低于 2,400 MB，则安全停止，不加载 DiT；这个门槛基于此前 DiT High 相对基线约 1.9 GB 的增长，加约 0.5 GB 余量，属于针对当前 iPhone 的保守实验值，不代表系统保证。若通过，则运行原有 DiT Metal High 两步诊断。三个日志分别为 `encoder-dit-serial.log`、`encoder-benchmark.log`、`last-inference.log`。两个模型使用各自固定输入；即使串行测试通过，也**没有**把 Encoder 输出接入 DiT，不能据此宣称任意歌词端侧生成已实现。

v15 真机串行日志：Encoder 函数返回时 footprint 1,621 MB、可用 1,599 MB；1 秒后 676/2,544 MB；5 秒后 50/3,171 MB。内存门槛通过，DiT High 两步返回成功，最终 footprint 82 MB。这证明该次测试中模型先后加载可行；两步 DiT 的峰值和相对 CPU/W4 误差应从 `last-inference.log` 核对，不能仅凭串行总日志推断。

v16 增加完整串行按钮：Encoder 只做一次 107-token 前向并释放，不再重复三次预热；等待 5 秒检查释放情况，然后运行原有的 DiT High 八步与 VAE Low 十二段，写出新的 30 秒 WAV。端到端总耗时和 RTF 包含这 5 秒安全等待；日志分别为 `encoder-full-serial.log`、`encoder-benchmark.log` 和 `last-inference.log`。加载 DiT 前要求可用内存至少 2,400 MB；DiT 释放后加载 VAE 前至少 1,700 MB。这两个门槛均来自既有单机采样峰值和约 0.5 GB 余量，不是 iOS 内存保证。Encoder 与 DiT 仍使用各自固定输入，Encoder 输出**没有**作为 DiT 条件输入；本测试验证三组件串行部署与可播放 WAV，不验证任意歌词。
