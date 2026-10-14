import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @EnvironmentObject private var modelStore: ModelStore
    @EnvironmentObject private var metricsStore: GenerationMetricsStore
    @State private var importing = false

    var body: some View {
        NavigationStack {
            List {
                Section("30 秒 W4 模型包") {
                    LabeledContent("完整文件", value: "\(modelStore.validCount)/\(ModelStore.requiredFiles.count)")
                    LabeledContent("完整包大小", value: "2.035 GB / 1.895 GiB")
                    LabeledContent("已导入", value: ByteCountFormatter.string(fromByteCount: modelStore.importedBytes, countStyle: .decimal))
                    ForEach(ModelStore.requiredFiles) { file in
                        HStack {
                            Text(file.name).font(.caption).lineLimit(2)
                            Spacer()
                            let valid = modelStore.fileSizes[file.name] == file.expectedBytes
                            Image(systemName: valid ? "checkmark.circle.fill" : "exclamationmark.circle")
                                .foregroundStyle(valid ? .green : .orange)
                        }
                    }
                    Button("从“文件”导入模型") { importing = true }
                }
                Section("部署状态") {
                    Text(modelStore.allModelsPresent
                         ? "七个模型文件大小校验通过；iPhone 端到端推理尚未接入。"
                         : "请导入七个文件，包括两个 .mnn.weight 外部权重文件。")
                    Text("当前模型为固定形状的 30 秒歌词样例，不支持任意长度的歌词或提示词。")
                        .foregroundStyle(.secondary)
                }
                Section("iPhone 性能实测") {
                    LabeledContent("生成耗时", value: metricsStore.lastRun.map { String(format: "%.2f 秒", $0.elapsedSeconds) } ?? "未测")
                    LabeledContent("RTF（耗时 ÷ 音频时长）", value: metricsStore.lastRun.map { String(format: "%.2f", $0.rtf) } ?? "未测")
                    LabeledContent("实时倍速", value: metricsStore.lastRun.map { String(format: "%.2f×", $0.realtimeSpeed) } ?? "未测")
                    LabeledContent("峰值内存", value: metricsStore.lastRun?.peakMemoryBytes.map { ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .decimal) } ?? "未测")
                    LabeledContent("后端", value: metricsStore.lastRun?.backend ?? "未测")
                    Text("仅在真实 iPhone 推理完成后记录；模型导入不计入 RTF。")
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("ACE-Step Mobile")
            .fileImporter(isPresented: $importing, allowedContentTypes: [.data], allowsMultipleSelection: true) { result in
                modelStore.importFiles(result)
            }
            .alert("导入结果", isPresented: $modelStore.showingMessage) {
                Button("好", role: .cancel) {}
            } message: {
                Text(modelStore.message)
            }
        }
    }
}
