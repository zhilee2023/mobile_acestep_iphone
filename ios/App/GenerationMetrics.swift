import Foundation

struct GenerationMetrics {
    let elapsedSeconds: Double
    let audioSeconds: Double
    let backend: String
    let peakMemoryBytes: UInt64?

    var rtf: Double { elapsedSeconds / audioSeconds }
    var realtimeSpeed: Double { audioSeconds / elapsedSeconds }
}

@MainActor
final class GenerationMetricsStore: ObservableObject {
    @Published private(set) var lastRun: GenerationMetrics?
    private var startedAt: TimeInterval?

    // Call around real on-device inference; never record file import or a simulated run.
    func beginGeneration() { startedAt = ProcessInfo.processInfo.systemUptime }

    func finishGeneration(audioSeconds: Double, backend: String, peakMemoryBytes: UInt64? = nil) {
        guard let startedAt, audioSeconds > 0 else { return }
        let elapsed = ProcessInfo.processInfo.systemUptime - startedAt
        self.startedAt = nil
        guard elapsed > 0 else { return }
        lastRun = GenerationMetrics(elapsedSeconds: elapsed, audioSeconds: audioSeconds,
                                    backend: backend, peakMemoryBytes: peakMemoryBytes)
    }

    func cancelGeneration() { startedAt = nil }
}
