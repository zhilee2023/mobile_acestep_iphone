import Foundation

struct ModelFile: Identifiable {
    let name: String
    let expectedBytes: Int64
    var id: String { name }
}

@MainActor
final class ModelStore: ObservableObject {
    // Fixed-shape 30-second vocal pack; external MNN weights are mandatory.
    static let requiredFiles: [ModelFile] = [
        .init(name: "text_encoder_w4.mnn", expectedBytes: 475_208),
        .init(name: "text_encoder_w4.mnn.weight", expectedBytes: 898_547_644),
        .init(name: "dit_w4.mnn", expectedBytes: 808_424),
        .init(name: "dit_w4.mnn.weight", expectedBytes: 1_003_698_366),
        .init(name: "vae_decoder_192_w4.mnn", expectedBytes: 43_869_168),
        .init(name: "vae_decoder_256_w4.mnn", expectedBytes: 43_869_168),
        .init(name: "vae_decoder_174_w4.mnn", expectedBytes: 43_869_168)
    ]
    static let expectedBytes = requiredFiles.reduce(Int64(0)) { $0 + $1.expectedBytes }

    @Published private(set) var fileSizes: [String: Int64] = [:]
    @Published var showingMessage = false
    @Published var message = ""

    var validCount: Int { Self.requiredFiles.filter { fileSizes[$0.name] == $0.expectedBytes }.count }
    var allModelsPresent: Bool { validCount == Self.requiredFiles.count }
    var importedBytes: Int64 { Self.requiredFiles.reduce(Int64(0)) { $0 + (fileSizes[$1.name] ?? 0) } }

    private let modelDirectory: URL

    init(fileManager: FileManager = .default) {
        let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        modelDirectory = documents.appendingPathComponent("Models", isDirectory: true)
        try? fileManager.createDirectory(at: modelDirectory, withIntermediateDirectories: true)
        refresh(fileManager: fileManager)
    }

    func importFiles(_ result: Result<[URL], Error>, fileManager: FileManager = .default) {
        do {
            let urls = try result.get()
            var imported: [String] = []
            for source in urls {
                guard let spec = Self.requiredFiles.first(where: { $0.name == source.lastPathComponent }) else { continue }
                let scoped = source.startAccessingSecurityScopedResource()
                defer { if scoped { source.stopAccessingSecurityScopedResource() } }
                let sourceSize = try fileManager.attributesOfItem(atPath: source.path)[.size] as? NSNumber
                guard sourceSize?.int64Value == spec.expectedBytes else {
                    throw ImportError.wrongSize(spec.name, spec.expectedBytes, sourceSize?.int64Value ?? -1)
                }
                let destination = modelDirectory.appendingPathComponent(spec.name)
                let temporary = modelDirectory.appendingPathComponent(".import-\(UUID().uuidString)")
                do {
                    try fileManager.copyItem(at: source, to: temporary)
                    let copiedSize = try fileManager.attributesOfItem(atPath: temporary.path)[.size] as? NSNumber
                    guard copiedSize?.int64Value == spec.expectedBytes else {
                        throw ImportError.wrongSize(spec.name, spec.expectedBytes, copiedSize?.int64Value ?? -1)
                    }
                    if fileManager.fileExists(atPath: destination.path) {
                        _ = try fileManager.replaceItemAt(destination, withItemAt: temporary)
                    } else {
                        try fileManager.moveItem(at: temporary, to: destination)
                    }
                    imported.append(spec.name)
                } catch {
                    try? fileManager.removeItem(at: temporary)
                    throw error
                }
            }
            refresh(fileManager: fileManager)
            message = imported.isEmpty ? "没有匹配的模型文件名。" : "已导入 \(imported.count) 个文件；完整性 \(validCount)/7。"
        } catch {
            refresh(fileManager: fileManager)
            message = "导入失败：\(error.localizedDescription)"
        }
        showingMessage = true
    }

    private func refresh(fileManager: FileManager) {
        fileSizes = Dictionary(uniqueKeysWithValues: Self.requiredFiles.compactMap { spec in
            guard let attributes = try? fileManager.attributesOfItem(atPath: modelDirectory.appendingPathComponent(spec.name).path),
                  let size = attributes[.size] as? NSNumber else { return nil }
            return (spec.name, size.int64Value)
        })
    }
}

private enum ImportError: LocalizedError {
    case wrongSize(String, Int64, Int64)

    var errorDescription: String? {
        switch self {
        case .wrongSize(let name, let expected, let actual):
            return "\(name) 大小不符（应为 \(expected) 字节，实际 \(actual) 字节）；请重新下载。"
        }
    }
}
