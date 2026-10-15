# iPhone 自定义歌词：73-token 试验版

v19 在固定 30 秒中文流行、固定风格和固定 seed 的基础上，新增一条真正把歌词编码结果接入 DiT 的端侧路径。原有的“生成 30 秒样例”仍为固定条件对照；新按钮为“用当前歌词生成”。两条路径不会混淆。

原模型路径是：Qwen 分词 → Qwen 词嵌入表 → ACE-Step 8 层双向歌词编码器 → 将歌词表示与固定 timbre、固定文本风格表示拼成 `[1,181,2048]` → DiT 八步 → VAE 六段。v19 的 `lyric_embedding.f16` 为 151,669 × 1,024 的 fp16 原始嵌入表，`lyric_encoder_73_w4.mnn` 为 8 层歌词编码器 W4 图。因当前 DiT 图是静态 181 token，且上游 decoder 在本图中忽略传入的 attention mask，试验版**只接受恰好 73 个歌词 token**，另有 1 个固定 timbre token 和 107 个固定风格 token。长短不符时拒绝生成，不会悄悄截断或填充。风格输入框目前仍不影响结果。

本地验证：用官方权重重建固定歌曲的完整条件，整体相对 L2 为 `2.95e-8`；导出的歌词 ONNX 子图与捕获条件相对 L2 为 `8.29e-7`；本地 MNN CPU W4 歌词子图（使用 fp16 嵌入表）相对 L2 为 `0.09697`。这证明张量接线，但不能保证任意新歌词的歌声清晰度。手机端会先核对内置固定歌词的分词 ID；固定歌词还会核对歌词编码输出与参考条件的相对 L2，超过 0.3 时安全停止。新歌词无参考音频，必须通过真机生成与试听验收。

文件导入：保留既有七个基础文件，再把 `models/mnn/vocal-30s-v2/lyric_embedding.f16`（310,618,112 字节）和 `models/mnn/vocal-30s-v2/lyric_encoder_73_w4.mnn`（254,171,424 字节）导入同一个 App 文件共享区域。总模型磁盘占用约 2.600 GB。两文件不在 GitHub 仓库或 IPA 内。

Windows 复核命令：

```powershell
.venv-reference\Scripts\python.exe scripts\verify-lyric-condition-contract.py
.venv-reference\Scripts\python.exe export\export_lyric_encoder.py
.venv\Scripts\mnnconvert.exe -f ONNX --modelFile models\onnx\vocal-30s-v2\lyric_encoder_73.onnx --MNNModel models\mnn\vocal-30s-v2\lyric_encoder_73_w4.mnn --weightQuantBits 4 --weightQuantBlock 64 --hqq
.venv\Scripts\python.exe scripts\validate-ios-lyric-encoder.py
```

真机验收先用示例歌词按新按钮，应通过自检并生成与固定条件近似的歌曲；之后改动少数字词，并保持 73-token 长度，再试听有无人声与歌词变化。收集 `last-inference.log`、总耗时/RTF 和崩溃日志。VAE 慢启动另用 v18 新增的五组诊断按钮；如果 NoTune 加载显著变快且输出与 Wide 接近，再决定是否切换默认 VAE 调优策略。当前不能直接把 NoTune 设为生产默认。
