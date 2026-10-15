# v20：VAE 慢启动独立诊断与 CPU Low 整首对照

## 为什么做

iPhone 15 的五组诊断显示，Metal Low / Wide 的 192 帧 VAE：读取模型约 0.05 秒，显式 `resizeSession` 61.191 秒；常规 `createSession` 61.485 秒。两次解码各约 3.5 秒。CPU Low 会话准备 0.133 秒，第二次解码 4.218 秒。五组诊断的 NoTune 最后一组因可用内存未恢复到 2400 MB 而未运行；因此尚不能归因于自动调优。

## 新入口

1. **单独测 VAE NoTune**：仅用相同的 DiT 后 192 帧固定 latent，运行 Metal Low、禁用调优、两次前向。独立写入 `Generated/vae-notune-diagnostic.log`，不覆盖五组日志，也不生成歌曲。建议关闭并重新打开 App 后先测这一项。主 pipeline 必须为 Metal，输出必须有限且非静音，否则视作失败。
2. **生成 CPU Low VAE 对照歌曲**：运行与原固定样例相同的 Encoder 测试、DiT Metal High 八步、VAE 六段拼接、30 秒 WAV；只把 VAE 改为 CPU Low（四线程）。日志分别写入 `Generated/encoder-cpu-vae-serial.log` 和 `Generated/last-inference-cpu-vae.log`，不覆盖 Metal 对照日志。播放器标注 CPU VAE。

## 测量和验收

- 先记录独立 NoTune 的 `createSession`、两次 run+read/sync、主 backend、峰值和内存；与上一版 Wide 约 61 秒和约 3.5 秒/段比较。NoTune 仅验证数值有限、非静音；没有 Wide 同运行的输出 L2，不能据此断言音质相同。
- 分别生成 Metal 和 CPU Low 30 秒样例，比较整首耗时、RTF、各阶段时间、峰值内存及听感。两次间让手机回到接近相同的温度和可用内存，不把发热差异误判为后端差异。
- 完整对照仍使用**固定预计算 DiT 条件**。Encoder 会运行，但输出尚未接入 DiT；不要把此入口当作自定义歌词验证。自定义歌词仍走 v19 的 Metal VAE 路径。
- 如果独立 NoTune 仍触发安全门槛，日志会说明未执行；不要降低内存门槛强行运行。

v20 仅添加诊断/对照入口，**不更改正式 Metal VAE 生成路径**。结论应等待真机日志和试听。
