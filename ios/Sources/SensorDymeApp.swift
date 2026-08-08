import SwiftUI

@main
struct SensorDymeApp: App {
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(state)
                // Applied at the root so every screen, sheet and alert inherits
                // it — a theme applied per-screen leaves modals in the old one.
                .preferredColorScheme(state.appearance.colorScheme)
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var state: AppState
    @StateObject private var watcher = AlertWatcher()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        if state.isSignedIn {
            TabView {
                DevicesView()
                    .tabItem { Label("Devices", systemImage: "video.fill") }
                ShopView()
                    .tabItem { Label("Shop", systemImage: "shippingbox.fill") }
                SettingsView()
                    .tabItem { Label("Settings", systemImage: "gearshape.fill") }
            }
            .tint(SD.Colors.accent)
            // Watch for alerts from every screen, not just the device detail
            // view — a notification you only receive while already looking at
            // the device is not a notification.
            .task { watcher.start(state: state) }
            .onChange(of: scenePhase) { _, phase in
                // Stop when suspended rather than leaving a task that iOS will
                // freeze mid-request and resume against a stale network.
                if phase == .active { watcher.start(state: state) } else { watcher.stop() }
            }
        } else {
            AuthView()
        }
    }
}
