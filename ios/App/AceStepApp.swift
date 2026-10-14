import SwiftUI

@main
struct AceStepApp: App {
    @StateObject private var modelStore = ModelStore()
    @StateObject private var metricsStore = GenerationMetricsStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(modelStore)
                .environmentObject(metricsStore)
        }
    }
}

