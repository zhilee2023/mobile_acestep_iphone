# 30 秒全组件 W4：iPhone 部署准备

本页针对 `models/mnn/vocal-30s-v2` 的固定形状中文人声样例。七个文件的实测总大小为 **2,035,137,146 字节 = 2.035 GB = 1.895 GiB**。这是模型磁盘占用，不含 App、MNN 框架、配置、运行缓存或推理时的激活/峰值内存。

| 文件 | 字节 |
| --- | ---: |
| `text_encoder_w4.mnn` | 475,208 |
| `text_encoder_w4.mnn.weight` | 898,547,644 |
| `dit_w4.mnn` | 808,424 |
| `dit_w4.mnn.weight` | 1,003,698,366 |
| `vae_decoder_192_w4.mnn` | 43,869,168 |
| `vae_decoder_256_w4.mnn` | 43,869,168 |
| `vae_decoder_174_w4.mnn` | 43,869,168 |

App 的“文件”导入只认这七个名称和字节数，避免漏掉两个 MNN 外部权重或接受中断下载的文件。完整性通过仅表示文件齐全，**不表示推理可运行**。目前 iOS 端还没有接通 MNN 图、分段 VAE 解码和音频播放；iPhone Metal、内存和速度均未实测。

USB 传输版开启 `UIFileSharingEnabled` 和 `LSSupportsOpeningDocumentsInPlace`。Windows 的 Apple Devices「文件」页可将七个文件直接复制到 App 的 `Documents` 根目录；App 点「刷新模型状态」读取并校验这些文件，不会再额外复制一份。旧版通过文件选择器导入到 `Documents/Models` 的文件仍可识别。文件共享会让 `Documents` 中的内容对连接的电脑和「文件」App 可见。

性能页面预留真实生成耗时、RTF、实时倍速、峰值内存和后端。RTF = 生成耗时 / 输出音频时长；例如 30 秒音频用 60 秒生成，RTF 才是 2.0。仅在真实设备成功生成后调用 `GenerationMetricsStore.beginGeneration()` / `finishGeneration(audioSeconds:backend:peakMemoryBytes:)`；失败时调用 `cancelGeneration()`。未接通前显示“未测”，不得填 Windows CPU 或模拟器数据。

这组 MNN 图依赖固定 30 秒和导出时的文本/隐状态形状，不是可输入任意时长或任意长度歌词的通用版。Windows 全 W4 样例的歌词质量也明显下降；正式部署前仍需完成：iOS 真机编译与单图运行、端到端生成、内存/RTF 记录、歌词人工试听，以及通用输入形状策略。导入大文件时当前实现同步复制，后续应增加进度与后台复制，避免界面长时间无响应。
