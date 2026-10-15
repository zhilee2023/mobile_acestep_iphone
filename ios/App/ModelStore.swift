import Foundation

struct ModelFile: Identifiable {
    let name: String
    let expectedBytes: Int64
    var id: String { name }
}

@MainActor
final class ModelStore: ObservableObject {
    // Fixed-shape 30-second vocal pack; external MNN weights are mandatory.
    static let coreFiles: [ModelFile] = [
        .init(name: "text_encoder_w4.mnn", expectedBytes: 475_208),
        .init(name: "text_encoder_w4.mnn.weight", expectedBytes: 898_547_644),
        .init(name: "dit_w4.mnn", expectedBytes: 808_424),
        .init(name: "dit_w4.mnn.weight", expectedBytes: 1_003_698_366),
        .init(name: "vae_decoder_192_w4.mnn", expectedBytes: 43_869_168),
        .init(name: "vae_decoder_256_w4.mnn", expectedBytes: 43_869_168),
        .init(name: "vae_decoder_174_w4.mnn", expectedBytes: 43_869_168)
    ]
    static let lyricFiles: [ModelFile] = [
        .init(name: "lyric_embedding.f16", expectedBytes: 310_618_112),
        .init(name: "lyric_encoder_73_w4.mnn", expectedBytes: 254_171_424)
    ]
    static let styleFiles: [ModelFile] = [
        .init(name: "text_projector.f16", expectedBytes: 4_194_304)
    ]
    static let requiredFiles = coreFiles + lyricFiles + styleFiles
    static let expectedBytes = requiredFiles.reduce(Int64(0)) { $0 + $1.expectedBytes }

    @Published private(set) var fileSizes: [String: Int64] = [:]
    @Published var showingMessage = false
    @Published var message = ""

    var validCount: Int { Self.requiredFiles.filter { fileSizes[$0.name] == $0.expectedBytes }.count }
    var coreValidCount: Int { Self.coreFiles.filter { fileSizes[$0.name] == $0.expectedBytes }.count }
    var lyricValidCount: Int { Self.lyricFiles.filter { fileSizes[$0.name] == $0.expectedBytes }.count }
    var styleValidCount: Int { Self.styleFiles.filter { fileSizes[$0.name] == $0.expectedBytes }.count }
    var allModelsPresent: Bool { coreValidCount == Self.coreFiles.count }
    var allCustomModelsPresent: Bool { allModelsPresent && lyricValidCount == Self.lyricFiles.count && styleValidCount == Self.styleFiles.count }
    var importedBytes: Int64 { Self.requiredFiles.reduce(Int64(0)) { $0 + (fileSizes[$1.name] ?? 0) } }

    private let modelDirectory: URL
    private let legacyModelDirectory: URL

    init(fileManager: FileManager = .default) {
        let documents = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        modelDirectory = documents
        legacyModelDirectory = documents.appendingPathComponent("Models", isDirectory: true)
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
            message = imported.isEmpty ? "没有匹配的模型文件名。" : "已导入 \(imported.count) 个文件；基础模型 \(coreValidCount)/7、歌词模型 \(lyricValidCount)/2、风格投影 \(styleValidCount)/1。"
        } catch {
            refresh(fileManager: fileManager)
            message = "导入失败：\(error.localizedDescription)"
        }
        showingMessage = true
    }

    func refresh(fileManager: FileManager = .default) {
        fileSizes = Dictionary(uniqueKeysWithValues: Self.requiredFiles.compactMap { spec in
            let direct = modelDirectory.appendingPathComponent(spec.name)
            let legacy = legacyModelDirectory.appendingPathComponent(spec.name)
            let path = fileManager.fileExists(atPath: direct.path) ? direct.path : legacy.path
            guard let attributes = try? fileManager.attributesOfItem(atPath: path),
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
