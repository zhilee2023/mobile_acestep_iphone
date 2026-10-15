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
    @State private var importingModels = false
    @State private var importingLyrics = false
    @State private var importingAudio = false
    @State private var showsModelDetails = false
    @State private var isGenerating = false
    @State private var isDiagnosingVAE = false
    @State private var vaeMessage = "尚未运行 VAE 对照测试。"
    @State private var vaePhase = ""
    @State private var activeVaeTraceURL: URL?
    @State private var generationMessage = "模型文件齐全后即可生成固定样例。"
    @State private var generationTrace = "尚无诊断记录"
    @State private var selectedStylePresetID: String?
    @FocusState private var captionIsFocused: Bool

    private var selectedStylePreset: SongStudioStore.StylePreset? {
        SongStudioStore.stylePresets.first { $0.id == selectedStylePresetID }
    }

    private var traceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/last-inference.log")
    }

    private var optimizedSerialTraceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/encoder-optimized-serial.log")
    }

    private var vaeDiagnosticURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/vae-diagnostic.log")
    }

    private var vaeNoTuneURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/vae-notune-diagnostic.log")
    }

    private var cpuVaeSerialTraceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/encoder-cpu-vae-serial.log")
    }

    private var cpuVaeInferenceTraceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/last-inference-cpu-vae.log")
    }

    private var customInferenceTraceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/last-inference-custom-cpu-vae.log")
    }

    private var styleEncoderTraceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/style-encoder.log")
    }

    private var noTuneSongTraceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/last-inference-notune-vae.log")
    }

    private var repaintTrialTraceURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Generated/repaint-55s-trial.log")
    }

    var body: some View {
        ZStack {
            StudioPalette.background.ignoresSafeArea()
            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
                    header
                    songCard
                    lyricsCard
                    modelCard
                    generationCard
                    playerCard
                    metricsCard
                    Text("ACE-Step 1.5 · MNN W4 · iPhone 实验版 v\(Bundle.main.infoDictionary?["CFBundleVersion"] as? String ?? "?")")
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
        .onReceive(Timer.publish(every: 1, on: .main, in: .common).autoconnect()) { _ in
            refreshGenerationTrace()
            refreshVaePhase()
        }
        .onAppear {
            refreshGenerationTrace()
            refreshVaePhase()
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
                Label("基础模型 \(modelStore.coreValidCount)/7", systemImage: modelStore.allModelsPresent ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
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
            Text("风格预设（均为 107 token）")
                .font(.caption2).foregroundStyle(StudioPalette.secondary)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(SongStudioStore.stylePresets) { preset in
                        Button(preset.title) {
                            captionIsFocused = false
                            selectedStylePresetID = preset.id
                        }
                        .buttonStyle(.bordered)
                        .tint(selectedStylePresetID == preset.id ? StudioPalette.mint : StudioPalette.violet)
                    }
                }
            }
            if let preset = selectedStylePreset {
                Text(preset.caption)
                    .font(.caption)
                    .foregroundStyle(.white)
                    .padding(13)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
                Text("当前模式：\(preset.title)预设（107 token）；生成时直接使用上方只读文本。")
                    .font(.caption2).foregroundStyle(StudioPalette.mint)
                Button("编辑此风格") {
                    studio.caption = preset.caption
                    selectedStylePresetID = nil
                }
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(StudioPalette.violet)
            } else {
                TextField("描述歌曲风格", text: $studio.caption, axis: .vertical)
                    .focused($captionIsFocused)
                    .lineLimit(2...4)
                    .padding(13)
                    .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
                Text("当前模式：自定义风格；需恰好 107 token。")
                    .font(.caption2).foregroundStyle(StudioPalette.secondary)
            }
            Text("预设可直接选择；自定义文字当前仍须恰好 107 token，不能靠增加字数延长歌曲。")
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
            Text("可粘贴或导入歌词。试验版仅接受恰好 73 个模型 token；不符合时会提示，不会截断或假装已使用。")
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
                Text("\(modelStore.coreValidCount)/7").fontWeight(.bold)
                    .foregroundStyle(modelStore.allModelsPresent ? StudioPalette.mint : .orange)
            }
            ProgressView(value: Double(modelStore.coreValidCount), total: 7).tint(StudioPalette.mint)
            Text("基础包 2.035 GB；自定义歌词另需约 565 MB（\(modelStore.lyricValidCount)/2）· 已识别 \(ByteCountFormatter.string(fromByteCount: modelStore.importedBytes, countStyle: .decimal))")
                .font(.caption).foregroundStyle(StudioPalette.secondary)
            HStack(spacing: 18) {
                Button("刷新状态") { modelStore.refresh() }
                Button(showsModelDetails ? "收起文件" : "查看模型文件") { showsModelDetails.toggle() }
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

    private var generationCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            heading("iPhone 生成", icon: "sparkles.rectangle.stack.fill")
            Text("完整生成使用当前风格和歌词；固定条件入口只用于对比速度与声音。")
                .font(.subheadline).foregroundStyle(StudioPalette.secondary)
            Button { generateOptimizedSerialSong(cpuVAE: true) } label: {
                HStack {
                    Text("固定样例 · CPU VAE 基线")
                    Spacer()
                    Image(systemName: "cpu")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allModelsPresent || isGenerating || isDiagnosingVAE)
            Button { generateRepaintTrial() } label: {
                HStack {
                    if isGenerating { ProgressView().tint(StudioPalette.background) }
                    Text("两段续写约 55 秒（真机实验）")
                    Spacer()
                    Image(systemName: "waveform.path")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allModelsPresent || isGenerating || isDiagnosingVAE)
            Text("固定风格与歌词：先生成 30 秒，再保留结尾 5 秒作为第二窗口上下文，新增约 25 秒。第二窗口仍使用同一歌词条件，只验证音频衔接、耗时和内存；不会自动接唱第一段没唱完的字。")
                .font(.caption2).foregroundStyle(StudioPalette.secondary)
            Button { generateNoTuneSong() } label: {
                HStack {
                    Text("生成 Metal NoTune 整首对照（实验）")
                    Spacer()
                    Image(systemName: "bolt")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allModelsPresent || isGenerating || isDiagnosingVAE)
            Text("NoTune 整首仍待真机验证，可能因内存不足提前停止；固定基线和 NoTune 均不使用当前编辑的歌词或风格。")
                .font(.caption2).foregroundStyle(StudioPalette.secondary)
            Text("上方按钮是固定条件对照。下方两个完整生成入口都会使用当前歌词，串行运行 Qwen、歌词编码器、DiT 与 CPU VAE。")
                .font(.caption2).foregroundStyle(.orange)
            Button { generateCustomSong() } label: {
                HStack {
                    if isGenerating { ProgressView().tint(StudioPalette.background) }
                    Text("用当前歌词＋风格生成（试验）")
                    Spacer()
                    Image(systemName: "text.quote")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allCustomModelsPresent || isGenerating || isDiagnosingVAE)
            Button { generateCustomSong(forceBundledCaption: true) } label: {
                HStack {
                    if isGenerating { ProgressView().tint(StudioPalette.background) }
                    Text("直接用内置风格＋当前歌词生成（107 token）")
                    Spacer()
                    Image(systemName: "music.note")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allCustomModelsPresent || isGenerating || isDiagnosingVAE)
            Text("此入口不读取上方风格输入框；它直接使用 App 内置的 304 字符风格，仍会编码你当前的歌词。")
                .font(.caption2).foregroundStyle(StudioPalette.secondary)
            Text(generationMessage).font(.caption).foregroundStyle(StudioPalette.secondary)
            Text("最近阶段：\(generationTrace)")
                .font(.caption.monospacedDigit()).foregroundStyle(StudioPalette.secondary)
            if FileManager.default.fileExists(atPath: traceURL.path) {
                ShareLink(item: traceURL) {
                    Label("分享推理日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: repaintTrialTraceURL.path) {
                ShareLink(item: repaintTrialTraceURL) {
                    Label("分享 55 秒续写实验日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: optimizedSerialTraceURL.path) {
                ShareLink(item: optimizedSerialTraceURL) {
                    Label("分享生成耗时日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: cpuVaeSerialTraceURL.path) {
                ShareLink(item: cpuVaeSerialTraceURL) {
                    Label("分享 CPU VAE 生成耗时", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: cpuVaeInferenceTraceURL.path) {
                ShareLink(item: cpuVaeInferenceTraceURL) {
                    Label("分享 CPU VAE 推理日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: styleEncoderTraceURL.path) {
                ShareLink(item: styleEncoderTraceURL) {
                    Label("分享风格编码日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: customInferenceTraceURL.path) {
                ShareLink(item: customInferenceTraceURL) {
                    Label("分享完整生成日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: noTuneSongTraceURL.path) {
                ShareLink(item: noTuneSongTraceURL) {
                    Label("分享 NoTune 整首日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
        }
        .studioCard()
    }

    private var vaeDiagnosticCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            heading("VAE 慢启动诊断", icon: "stethoscope")
            Text("先单独测 Metal NoTune，避免五组测试后的内存残留挡住最后一组。五组对照保留；两者均不生成歌曲。")
                .font(.subheadline).foregroundStyle(StudioPalette.secondary)
            Button { diagnoseVAE(standaloneNoTune: true) } label: {
                HStack {
                    Text("单独测 VAE NoTune")
                    Spacer()
                    Image(systemName: "timer")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allModelsPresent || isGenerating || isDiagnosingVAE)
            Button { diagnoseVAE() } label: {
                HStack {
                    if isDiagnosingVAE { ProgressView().tint(StudioPalette.background) }
                    Text(isDiagnosingVAE ? "正在诊断 VAE…" : "开始 VAE 五组对照")
                    Spacer()
                    Image(systemName: "timer")
                }
                .frame(maxWidth: .infinity)
            }
            .buttonStyle(StudioButtonStyle())
            .disabled(!modelStore.allModelsPresent || isGenerating || isDiagnosingVAE)
            Text(vaeMessage).font(.caption).foregroundStyle(StudioPalette.secondary)
            if !vaePhase.isEmpty {
                Text("最近阶段：\(vaePhase)")
                    .font(.caption.monospacedDigit()).foregroundStyle(StudioPalette.secondary)
            }
            if FileManager.default.fileExists(atPath: vaeDiagnosticURL.path) {
                ShareLink(item: vaeDiagnosticURL) {
                    Label("分享 VAE 诊断日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
            if FileManager.default.fileExists(atPath: vaeNoTuneURL.path) {
                ShareLink(item: vaeNoTuneURL) {
                    Label("分享 NoTune 独立日志", systemImage: "square.and.arrow.up")
                }
                .font(.caption.weight(.semibold)).foregroundStyle(StudioPalette.mint)
            }
        }
        .studioCard()
    }

    private func refreshVaePhase() {
        guard let content = try? String(contentsOf: activeVaeTraceURL ?? vaeDiagnosticURL, encoding: .utf8),
              let line = content.split(whereSeparator: \.isNewline).last else { return }
        vaePhase = line.split(separator: "\t").dropFirst().joined(separator: " · ")
    }

    private func diagnoseVAE(standaloneNoTune: Bool = false) {
        guard !isGenerating && !isDiagnosingVAE else { return }
        isDiagnosingVAE = true
        activeVaeTraceURL = standaloneNoTune ? vaeNoTuneURL : vaeDiagnosticURL
        vaeMessage = standaloneNoTune ? "正在单独测 Metal NoTune，请保持 App 在前台…"
                                      : "正在逐组运行 CPU 和 Metal，请保持 App 在前台…"
        vaePhase = "正在启动…"
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
        DispatchQueue.global(qos: .userInitiated).async {
            let result = standaloneNoTune
                ? AceBridge.diagnoseVAENoTune(atDirectory: directory)
                : AceBridge.diagnoseVAE(atDirectory: directory)
            DispatchQueue.main.async {
                self.isDiagnosingVAE = false
                self.refreshVaePhase()
                self.vaeMessage = result["message"] as? String ?? "VAE 诊断未返回结果。"
            }
        }
    }

    private func refreshGenerationTrace() {
        let candidates = [repaintTrialTraceURL, customInferenceTraceURL, styleEncoderTraceURL, noTuneSongTraceURL,
                          optimizedSerialTraceURL, cpuVaeSerialTraceURL,
                          cpuVaeInferenceTraceURL, traceURL]
        let latest = candidates.filter { FileManager.default.fileExists(atPath: $0.path) }
            .max { left, right in
                let leftDate = (try? left.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                let rightDate = (try? right.resourceValues(forKeys: [.contentModificationDateKey]))?.contentModificationDate ?? .distantPast
                return leftDate < rightDate
            }
        guard let latest,
              let content = try? String(contentsOf: latest, encoding: .utf8),
              let line = content.split(whereSeparator: \.isNewline).last else { return }
        let fields = line.split(separator: "\t")
        generationTrace = fields.dropFirst().joined(separator: " · ")
    }

    private func generateOptimizedSerialSong(cpuVAE: Bool = false) {
        guard !isGenerating && !isDiagnosingVAE else { return }
        isGenerating = true
        generationMessage = cpuVAE
            ? "正在生成 CPU Low VAE 对照歌曲，请保持 App 在前台…"
            : "正在串行执行 Encoder、DiT 和 VAE，请保持 App 在前台…"
        generationTrace = "正在启动…"
        metricsStore.beginGeneration()
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
        DispatchQueue.global(qos: .userInitiated).async {
            let result = cpuVAE
                ? AceBridge.generateCpuVaeSerialSong(atDirectory: directory)
                : AceBridge.generateOptimizedSerialSong(atDirectory: directory)
            DispatchQueue.main.async {
                self.isGenerating = false
                self.refreshGenerationTrace()
                self.generationMessage = result["message"] as? String ?? "未收到生成结果。"
                guard (result["success"] as? NSNumber)?.boolValue == true,
                      let path = result["wavPath"] as? String else {
                    self.metricsStore.cancelGeneration()
                    return
                }
                self.metricsStore.finishGeneration(audioSeconds: 30,
                    backend: result["backend"] as? String ?? "MNN Metal 请求 / CPU 回退可能")
                do {
                    try self.studio.useGeneratedAudio(at: URL(fileURLWithPath: path),
                        description: cpuVAE ? "iPhone DiT Metal + VAE CPU Low · 固定条件"
                                            : "iPhone DiT Metal + VAE Metal Low · 固定条件")
                    self.generationMessage += " 已载入播放器。"
                } catch {
                    self.generationMessage += " WAV 已保存，但播放器载入失败：\(error.localizedDescription)"
                }
            }
        }
    }

    private func generateNoTuneSong() {
        guard !isGenerating && !isDiagnosingVAE else { return }
        isGenerating = true
        generationMessage = "正在用固定条件测试 Metal NoTune 整首生成，请保持 App 在前台…"
        generationTrace = "正在启动…"
        metricsStore.beginGeneration()
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
        DispatchQueue.global(qos: .userInitiated).async {
            let result = AceBridge.generateNoTuneVAESong(atDirectory: directory)
            DispatchQueue.main.async {
                self.isGenerating = false
                self.refreshGenerationTrace()
                self.generationMessage = result["message"] as? String ?? "NoTune 生成未返回结果。"
                guard (result["success"] as? NSNumber)?.boolValue == true,
                      let path = result["wavPath"] as? String else {
                    self.metricsStore.cancelGeneration()
                    return
                }
                self.metricsStore.finishGeneration(audioSeconds: 30,
                    backend: result["backend"] as? String ?? "DiT Metal + VAE Metal NoTune")
                do {
                    try self.studio.useGeneratedAudio(at: URL(fileURLWithPath: path),
                        description: "iPhone 固定条件 + VAE Metal NoTune · 实验对照")
                    self.generationMessage += " 已载入播放器。"
                } catch {
                    self.generationMessage += " WAV 已保存，但播放器载入失败：\(error.localizedDescription)"
                }
            }
        }
    }

    private func generateRepaintTrial() {
        guard !isGenerating && !isDiagnosingVAE else { return }
        isGenerating = true
        generationMessage = "正在运行固定条件两窗口续写；请保持 App 在前台，过程可能超过一分钟…"
        generationTrace = "正在启动…"
        metricsStore.beginGeneration()
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
        DispatchQueue.global(qos: .userInitiated).async {
            let result = AceBridge.generateRepaintTrial(atDirectory: directory)
            DispatchQueue.main.async {
                self.isGenerating = false
                self.refreshGenerationTrace()
                self.generationMessage = result["message"] as? String ?? "续写实验未返回结果。"
                guard (result["success"] as? NSNumber)?.boolValue == true,
                      let path = result["wavPath"] as? String else {
                    self.metricsStore.cancelGeneration()
                    return
                }
                self.metricsStore.finishGeneration(audioSeconds: 55,
                    backend: result["backend"] as? String ?? "两窗口 DiT Metal + VAE CPU Low")
                do {
                    try self.studio.useGeneratedAudio(at: URL(fileURLWithPath: path),
                        description: "iPhone 固定条件两窗口续写 · 55 秒实验")
                    self.generationMessage += " 已载入播放器；请重点试听 30 秒附近。"
                } catch {
                    self.generationMessage += " WAV 已保存，但播放器载入失败：\(error.localizedDescription)"
                }
            }
        }
    }

    private func generateCustomSong(forceBundledCaption: Bool = false) {
        guard !isGenerating && !isDiagnosingVAE else { return }
        captionIsFocused = false
        let lyrics = studio.lyrics.trimmingCharacters(in: .whitespacesAndNewlines)
        let selectedPreset = selectedStylePreset
        let useBundledCaption = forceBundledCaption
        let caption = (forceBundledCaption ? SongStudioStore.sampleCaption : (selectedPreset?.caption ?? studio.caption))
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard !lyrics.isEmpty else {
            generationMessage = "请先输入歌词。"
            return
        }
        guard !caption.isEmpty else {
            generationMessage = "请先输入风格提示词。"
            return
        }
        isGenerating = true
        let styleName = useBundledCaption ? "内置风格直通" : (selectedPreset?.title ?? "自定义风格")
        generationMessage = "v27 \(styleName)；正在编码当前歌词并运行 Qwen、DiT 与 CPU VAE…"
        generationTrace = "正在启动…"
        metricsStore.beginGeneration()
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
        DispatchQueue.global(qos: .userInitiated).async {
            let result = AceBridge.generateCustomSong(withLyrics: lyrics, caption: caption, atDirectory: directory)
            DispatchQueue.main.async {
                self.isGenerating = false
                self.refreshGenerationTrace()
                self.generationMessage = "v27 \(styleName)：" + (result["message"] as? String ?? "歌词生成未返回结果。")
                guard (result["success"] as? NSNumber)?.boolValue == true,
                      let path = result["wavPath"] as? String else {
                    self.metricsStore.cancelGeneration()
                    return
                }
                self.metricsStore.finishGeneration(audioSeconds: 30,
                    backend: result["backend"] as? String ?? "本机风格和歌词编码 → DiT → CPU VAE")
                do {
                    try self.studio.useGeneratedAudio(at: URL(fileURLWithPath: path),
                        description: "iPhone 新风格 + 新歌词 + DiT Metal + VAE CPU Low")
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
            Text("仅在真实 iPhone 完整生成音频后记录；续写实验按实际 55 秒计算 RTF。")
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
