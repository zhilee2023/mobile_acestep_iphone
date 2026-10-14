import SwiftUI
import UniformTypeIdentifiers

private enum StudioPalette {
    static let background = Color(red: 0.045, green: 0.06, blue: 0.12)
    static let panel = Color(red: 0.10, green: 0.12, blue: 0.20)
    static let mint = Color(red: 0.43, green: 0.96, blue: 0.75)
    static let violet = Color(red: 0.66, green: 0.52, blue: 1.0)
    static let secondary = Color.white.opacity(0.64)
}

struct ContentView: View {
    @EnvironmentObject private var modelStore: ModelStore
    @EnvironmentObject private var metricsStore: GenerationMetricsStore
    @StateObject private var studio = SongStudioStore()
    @StateObject private var probe = DeviceProbeStore()
    @State private var importingModels = false
    @State private var importingLyrics = false
    @State private var importingAudio = false
    @State private var showsModelDetails = false
    @State private var isGenerating = false
    @State private var generationMessage = "请先完成 DiT 真机单步测试。"

    var body: some View {
        ZStack {
            StudioPalette.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    header
                    songCard
                    lyricsCard
                    modelCard
                    probeCard
                    generationCard
                    playerCard
                    metricsCard
                    Text("ACE-Step 1.5 · MNN W4 · iPhone 实验版")
                        .font(.caption2).foregroundStyle(StudioPalette.secondary)
                        .frame(maxWidth: .infinity).padding(.bottom, 22)
                }
                .padding(.horizontal, 20).padding(.top, 18)
            }
        }
        .preferredColorScheme(.dark)
        .fileImporter(isPresented: $importingModels, allowedContentTypes: [.data], allowsMultipleSelection: true) {
            modelStore.importFiles($0)
        }
        .fileImporter(isPresented: $importingLyrics, allowedContentTypes: [.text]) {
            studio.importLyrics($0.map { [$0] })
        }
        .fileImporter(isPresented: $importingAudio, allowedContentTypes: [.audio]) {
            studio.importAudio($0.map { [$0] })
        }
        .alert("导入结果", isPresented: $modelStore.showingMessage) {
            Button("好", role: .cancel) {}
        } message: { Text(modelStore.message) }
        .alert("工作台提示", isPresented: $studio.showingMessage) {
            Button("好", role: .cancel) {}
        } message: { Text(studio.message) }
        .onReceive(Timer.publish(every: 0.25, on: .main, in: .common).autoconnect()) { _ in
            studio.tick()
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "waveform.path")
                    .font(.title2.weight(.bold)).foregroundStyle(StudioPalette.mint)
                    .frame(width: 46, height: 46)
                    .background(StudioPalette.mint.opacity(0.12), in: RoundedRectangle(cornerRadius: 15))
                Spacer()
                Label("模型 \(modelStore.validCount)/7", systemImage: modelStore.allModelsPresent ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(modelStore.allModelsPresent ? StudioPalette.mint : .orange)
                    .padding(.horizontal, 12).padding(.vertical, 8)
                    .background(StudioPalette.panel, in: Capsule())
            }
            Text("把灵感唱出来")
                .font(.system(size: 34, weight: .bold, design: .rounded))
                .foregroundStyle(.white)
            Text("30 秒中文流行 · 本机 W4 实验")
                .font(.subheadline).foregroundStyle(StudioPalette.secondary)
        }
    }

    private var songCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            heading("歌曲设定", icon: "music.note.list")
            HStack(spacing: 8) {
                tag("中文")
                tag("流行")
                tag("30 秒")
                tag("女声样例")
            }
            Text("风格提示词").font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.secondary)
            TextField("描述歌曲风格", text: $studio.caption, axis: .vertical)
                .lineLimit(2...4)
                .padding(13)
                .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
            Text("当前静态模型只能验证固定样例；修改提示词还不会改变端侧输出。")
                .font(.caption2).foregroundStyle(StudioPalette.secondary)
        }
        .studioCard()
    }

    private var lyricsCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            heading("歌词本", icon: "text.book.closed.fill")
            TextEditor(text: $studio.lyrics)
                .font(.body)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 210)
                .padding(8)
                .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
            HStack {
                Button { importingLyrics = true } label: {
                    Label("导入 .txt 歌词", systemImage: "square.and.arrow.down")
                }
                Spacer()
                Button("恢复示例") { studio.restoreSample() }
                    .foregroundStyle(StudioPalette.secondary)
            }
            .font(.subheadline.weight(.medium))
            .foregroundStyle(StudioPalette.mint)
            Text("可直接粘贴或导入歌词；自定义歌词的端侧条件编码尚未完成。")
                .font(.caption2).foregroundStyle(StudioPalette.secondary)
        }
        .studioCard()
    }

    private var playerCard: some View {
        VStack(alignment: .leading, spacing: 15) {
            heading("试听播放器", icon: "play.rectangle.fill")
            HStack(spacing: 14) {
                Image(systemName: "music.mic")
                    .font(.title2).foregroundStyle(StudioPalette.violet)
                    .frame(width: 54, height: 54)
                    .background(StudioPalette.violet.opacity(0.15), in: RoundedRectangle(cornerRadius: 14))
                VStack(alignment: .leading, spacing: 5) {
                    Text(studio.audioName ?? "还没有可播放的音频")
                        .font(.subheadline.weight(.semibold)).lineLimit(1)
                    Text(studio.audioDescription)
                        .font(.caption).foregroundStyle(StudioPalette.secondary)
                }
                Spacer(minLength: 0)
            }
            Slider(value: Binding(get: { studio.currentTime }, set: { studio.seek(to: $0) }),
                   in: 0...max(studio.duration, 1))
                .tint(StudioPalette.mint)
                .disabled(studio.audioName == nil)
            HStack {
                Text(timeString(studio.currentTime))
                Spacer()
                Button { studio.togglePlayback() } label: {
                    Image(systemName: studio.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3)
                        .frame(width: 58, height: 58)
                        .background(StudioPalette.mint, in: Circle())
                        .foregroundStyle(StudioPalette.background)
                }
                .disabled(studio.audioName == nil)
                .opacity(studio.audioName == nil ? 0.42 : 1)
                Spacer()
                Text(timeString(studio.duration))
            }
            .font(.caption.monospacedDigit()).foregroundStyle(StudioPalette.secondary)
            Button { importingAudio = true } label: {
                Label("导入音频试听", systemImage: "square.and.arrow.down.fill")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
        }
        .studioCard()
    }

    private var modelCard: some View {
        VStack(alignment: .leading, spacing: 13) {
            heading("设备模型", icon: "cpu.fill")
            HStack {
                Text("完整文件")
                Spacer()
                Text("\(modelStore.validCount)/7").fontWeight(.bold)
                    .foregroundStyle(modelStore.allModelsPresent ? StudioPalette.mint : .orange)
            }
            ProgressView(value: Double(modelStore.validCount), total: 7).tint(StudioPalette.mint)
            Text("模型包 2.035 GB · 已识别 \(ByteCountFormatter.string(fromByteCount: modelStore.importedBytes, countStyle: .decimal))")
                .font(.caption).foregroundStyle(StudioPalette.secondary)
            HStack(spacing: 18) {
                Button("刷新状态") { modelStore.refresh() }
                Button(showsModelDetails ? "收起文件" : "查看七个文件") { showsModelDetails.toggle() }
                Spacer()
                Button("导入") { importingModels = true }
            }
            .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            if showsModelDetails {
                ForEach(ModelStore.requiredFiles) { file in
                    HStack(spacing: 8) {
                        Image(systemName: modelStore.fileSizes[file.name] == file.expectedBytes ? "checkmark.circle.fill" : "exclamationmark.circle")
                            .foregroundStyle(modelStore.fileSizes[file.name] == file.expectedBytes ? StudioPalette.mint : .orange)
                        Text(file.name).font(.caption2).lineLimit(1)
                    }
                }
            }
        }
        .studioCard()
    }

    private var probeCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            heading("端侧推理验证", icon: "iphone.radiowaves.left.and.right")
            Text("先在 iPhone 上运行一次真实的 MNN W4 DiT 前向。它是 8 步歌曲生成的第一关，不会输出歌曲。")
                .font(.subheadline).foregroundStyle(StudioPalette.secondary)
            Button { probe.run() } label: {
                HStack {
                    if probe.isRunning { ProgressView().tint(StudioPalette.background) }
                    Text(probe.isRunning ? "正在真机运行…" : "运行 DiT 单步测试")
                    Spacer()
                    Image(systemName: "arrow.up.right")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allModelsPresent || probe.isRunning)
            Text(probe.message).font(.caption).foregroundStyle(StudioPalette.secondary)
            if let seconds = probe.elapsedSeconds, let error = probe.relativeL2 {
                HStack {
                    metric("单步耗时", String(format: "%.2f 秒", seconds))
                    Spacer()
                    metric("相对 L2", String(format: "%.3f", error))
                }
            }
            Label("单步只是第一关；整首生成须独立完成。", systemImage: "info.circle")
                .font(.caption2).foregroundStyle(.orange)
        }
        .studioCard()
    }

    private var generationCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            heading("iPhone 生成", icon: "sparkles.rectangle.stack.fill")
            Text("固定条件样例：手机运行 8 步 DiT 与 6 段 VAE，输出 30 秒 WAV。文本条件来自预计算样例；上方编辑的歌词暂不会参与生成。")
                .font(.subheadline).foregroundStyle(StudioPalette.secondary)
            Button { generateFixedSong() } label: {
                HStack {
                    if isGenerating { ProgressView().tint(StudioPalette.background) }
                    Text(isGenerating ? "手机正在生成…" : "生成固定样例 · 30 秒")
                    Spacer()
                    Image(systemName: "waveform")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allModelsPresent || probe.elapsedSeconds == nil || isGenerating)
            Text(generationMessage).font(.caption).foregroundStyle(StudioPalette.secondary)
        }
        .studioCard()
    }

    private func generateFixedSong() {
        guard !isGenerating else { return }
        isGenerating = true
        generationMessage = "正在手机上执行 MNN 推理，请保持 App 在前台…"
        metricsStore.beginGeneration()
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
        DispatchQueue.global(qos: .userInitiated).async {
            let result = AceBridge.generateFixedSong(atDirectory: directory)
            DispatchQueue.main.async {
                self.isGenerating = false
                self.generationMessage = result["message"] as? String ?? "未收到生成结果。"
                guard (result["success"] as? NSNumber)?.boolValue == true,
                      let path = result["wavPath"] as? String else {
                    self.metricsStore.cancelGeneration()
                    return
                }
                self.metricsStore.finishGeneration(audioSeconds: 30,
                    backend: result["backend"] as? String ?? "MNN Metal 请求 / CPU 回退可能")
                do {
                    try self.studio.useGeneratedAudio(at: URL(fileURLWithPath: path))
                    self.generationMessage += " 已载入播放器。"
                } catch {
                    self.generationMessage += " WAV 已保存，但播放器载入失败：\(error.localizedDescription)"
                }
            }
        }
    }

    private var metricsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            heading("整首生成性能", icon: "speedometer")
            HStack {
                metric("生成耗时", metricsStore.lastRun.map { String(format: "%.2f 秒", $0.elapsedSeconds) } ?? "未测")
                Spacer()
                metric("RTF", metricsStore.lastRun.map { String(format: "%.2f", $0.rtf) } ?? "未测")
                Spacer()
                metric("后端", metricsStore.lastRun?.backend ?? "未测")
            }
            Text("仅在真实 iPhone 完整生成 30 秒音频后记录。")
                .font(.caption2).foregroundStyle(StudioPalette.secondary)
        }
        .studioCard()
    }

    private func heading(_ title: String, icon: String) -> some View {
        Label(title, systemImage: icon)
            .font(.headline.weight(.bold)).foregroundStyle(.white)
            .tint(StudioPalette.mint)
    }

    private func tag(_ text: String) -> some View {
        Text(text).font(.caption2.weight(.semibold))
            .padding(.horizontal, 10).padding(.vertical, 6)
            .background(StudioPalette.violet.opacity(0.14), in: Capsule())
            .foregroundStyle(StudioPalette.violet)
    }

    private func metric(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(StudioPalette.secondary)
            Text(value).font(.subheadline.weight(.semibold)).foregroundStyle(.white)
        }
    }

    private func timeString(_ time: TimeInterval) -> String {
        guard time.isFinite && time >= 0 else { return "0:00" }
        let seconds = Int(time)
        return String(format: "%d:%02d", seconds / 60, seconds % 60)
    }
}

private struct StudioCard: ViewModifier {
    func body(content: Content) -> some View {
        content
            .padding(18)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(StudioPalette.panel, in: RoundedRectangle(cornerRadius: 22))
            .overlay(RoundedRectangle(cornerRadius: 22).stroke(Color.white.opacity(0.08)))
    }
}

private extension View {
    func studioCard() -> some View { modifier(StudioCard()) }
}

private struct StudioButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.subheadline.weight(.bold))
            .padding(.horizontal, 18).padding(.vertical, 14)
            .background(LinearGradient(colors: [StudioPalette.mint, StudioPalette.violet], startPoint: .leading, endPoint: .trailing),
                        in: RoundedRectangle(cornerRadius: 14))
            .foregroundStyle(StudioPalette.background)
            .opacity(configuration.isPressed ? 0.72 : 1)
    }
}
