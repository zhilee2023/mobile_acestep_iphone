import AVFoundation
import Foundation

@MainActor
final class SongStudioStore: ObservableObject {
    static let sampleLyrics = """
    [Chorus]
    你听见吗我在唱给你
    把想念写进夜空里
    别让故事停在这里
    陪我唱到天亮起

    [Verse]
    窗外的风轻轻吹过
    你眼里的光还亮着
    现在就跟我一起唱
    让这一刻留在心上
    """

    @Published var caption: String {
        didSet { UserDefaults.standard.set(caption, forKey: "song.caption") }
    }
    @Published var lyrics: String {
        didSet { UserDefaults.standard.set(lyrics, forKey: "song.lyrics") }
    }
    @Published private(set) var audioName: String?
    @Published private(set) var audioDescription = "导入 WAV / M4A / MP3 试听"
    @Published private(set) var isPlaying = false
    @Published private(set) var currentTime: TimeInterval = 0
    @Published private(set) var duration: TimeInterval = 0
    @Published var message = ""
    @Published var showingMessage = false

    private var player: AVAudioPlayer?

    init() {
        caption = UserDefaults.standard.string(forKey: "song.caption") ?? "Chinese Mandarin pop vocal song. Clear solo female singer begins singing the chorus immediately at the first beat, no instrumental intro, no extended instrumental passages. Upfront intelligible Mandarin lead vocal throughout, bright catchy melody, piano, soft synth bass and pop drums, studio production."
        lyrics = UserDefaults.standard.string(forKey: "song.lyrics") ?? Self.sampleLyrics
    }

    func restoreSample() { lyrics = Self.sampleLyrics }

    func importLyrics(_ result: Result<[URL], Error>) {
        do {
            guard let source = try result.get().first else { return }
            let scoped = source.startAccessingSecurityScopedResource()
            defer { if scoped { source.stopAccessingSecurityScopedResource() } }
            let data = try Data(contentsOf: source)
            guard data.count <= 128 * 1024 else { throw StudioError.lyricsTooLarge }
            guard let text = String(data: data, encoding: .utf8)
                    ?? String(data: data, encoding: .utf16),
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw StudioError.invalidLyrics
            }
            lyrics = text
            notify("歌词已导入，可继续在页面中编辑。")
        } catch { notify("导入歌词失败：\(error.localizedDescription)") }
    }

    func importAudio(_ result: Result<[URL], Error>) {
        do {
            guard let source = try result.get().first else { return }
            let scoped = source.startAccessingSecurityScopedResource()
            defer { if scoped { source.stopAccessingSecurityScopedResource() } }
            let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
            let size = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            guard size > 0 && size <= 150_000_000 else { throw StudioError.invalidAudioSize }

            let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let directory = documents.appendingPathComponent("Audio", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent(UUID().uuidString + "-" + source.lastPathComponent)
            try FileManager.default.copyItem(at: source, to: destination)
            do {
                let candidate = try AVAudioPlayer(contentsOf: destination)
                guard candidate.duration > 0 else { throw StudioError.invalidAudio }
                candidate.prepareToPlay()
                player?.stop()
                player = candidate
                audioName = source.lastPathComponent
                audioDescription = "用户导入音频 · 非端侧生成"
                currentTime = 0
                duration = candidate.duration
                isPlaying = false
                notify("音频已导入。点击播放即可试听；这不是手机生成的音频。")
            } catch {
                try? FileManager.default.removeItem(at: destination)
                throw error
            }
        } catch { notify("导入音频失败：\(error.localizedDescription)") }
    }

    func useGeneratedAudio(at url: URL, description: String = "iPhone DiT + VAE 推理 · 固定预计算条件") throws {
        let candidate = try AVAudioPlayer(contentsOf: url)
        guard candidate.duration > 0 else { throw StudioError.invalidAudio }
        candidate.prepareToPlay()
        player?.stop()
        player = candidate
        audioName = url.lastPathComponent
        audioDescription = description
        currentTime = 0
        duration = candidate.duration
        isPlaying = false
    }

    func togglePlayback() {
        guard let player else { return }
        if player.isPlaying {
            player.pause()
            isPlaying = false
        } else {
            try? AVAudioSession.sharedInstance().setCategory(.playback)
            try? AVAudioSession.sharedInstance().setActive(true)
            if player.currentTime >= player.duration { player.currentTime = 0 }
            isPlaying = player.play()
        }
    }

    func seek(to time: TimeInterval) {
        guard let player else { return }
        player.currentTime = min(max(time, 0), player.duration)
        currentTime = player.currentTime
    }

    func tick() {
        guard let player else { return }
        currentTime = player.currentTime
        isPlaying = player.isPlaying
    }

    private func notify(_ text: String) {
        message = text
        showingMessage = true
    }
}

private enum StudioError: LocalizedError {
    case lyricsTooLarge, invalidLyrics, invalidAudioSize, invalidAudio

    var errorDescription: String? {
        switch self {
        case .lyricsTooLarge: return "歌词文件不能超过 128 KB。"
        case .invalidLyrics: return "请选择有内容的 UTF-8 或 UTF-16 文本文件。"
        case .invalidAudioSize: return "音频须为 150 MB 以内的非空文件。"
        case .invalidAudio: return "无法播放此音频，请换一个 WAV、M4A 或 MP3 文件。"
        }
    }
}
