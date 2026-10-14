import Foundation

@MainActor
final class DeviceProbeStore: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var message = "尚未运行真机 MNN 测试"
    @Published private(set) var elapsedSeconds: Double?
    @Published private(set) var relativeL2: Double?

    func run() {
        guard !isRunning else { return }
        isRunning = true
        message = "正在加载约 1 GB 的 DiT 权重并运行单步…"
        elapsedSeconds = nil
        relativeL2 = nil
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].path
        DispatchQueue.global(qos: .userInitiated).async {
            let result = AceBridge.runDiTProbe(atDirectory: directory)
            DispatchQueue.main.async {
                self.isRunning = false
                self.message = result["message"] as? String ?? "真机测试未返回结果。"
                if (result["success"] as? NSNumber)?.boolValue == true {
                    self.elapsedSeconds = (result["elapsedSeconds"] as? NSNumber)?.doubleValue
                    self.relativeL2 = (result["relativeL2"] as? NSNumber)?.doubleValue
                }
            }
        }
    }
}
