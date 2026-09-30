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
                // At the root, and not per-screen, because losing the backend is
                // not something one screen recovers from: the app has to keep
                // looking whether the user is on the sign-in screen, the device
                // list or Settings.
                .task { state.startBackendMonitoring() }
        }
    }
}

struct RootView: View {
    @EnvironmentObject private var state: AppState
    @StateObject private var watcher = AlertWatcher()
    @Environment(\.scenePhase) private var scenePhase

    var body: some View {
        content
            .onChange(of: scenePhase) { _, phase in
                // Coming back from suspension is the case that used to need a
                // force-quit: iOS freezes the app's tasks, the Mac's address may
                // have moved while it was away, and nothing re-checked on return.
                guard phase == .active else { return }
                switch state.backendStatus {
                case .connected:
                    // Confirm it is still there rather than assuming; a backend
                    // restart while suspended leaves a connection that only looks
                    // live until the next request fails.
                    Task { await state.revalidateBackend() }
                case .searching:
                    // Already looking — leave it alone. This fires on the
                    // inactive→active transition at launch, when the root task's
                    // resolve is only milliseconds old, and `reconnect()` cancels
                    // whatever is in flight before starting again. So asking here
                    // threw away the launch resolve and started a second one,
                    // which is why a pinned address that does not answer got
                    // probed twice before discovery ever ran.
                    break
                case .unreachable, .unknown:
                    state.reconnect()
                }
            }
    }

    @ViewBuilder
    private var content: some View {
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
