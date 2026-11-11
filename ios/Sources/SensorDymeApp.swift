import SwiftUI

@main
struct SensorDymeApp: App {
    @StateObject private var state = AppState()

    init() {
        // The notification-centre delegate has to be in place before launch
        // finishes, or the tap that cold-launched the app is never delivered.
        // AlertNotifier sets it in its initialiser; it used to be created lazily,
        // when the root view's AlertWatcher was first installed — after launch.
        _ = AlertNotifier.shared
        // Before the first window exists, so no bar or tab ever draws in the
        // system theme and then flips.
        OrynAppearance.apply()
    }

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(state)
                // ORYN EDGE is dark by design. Applied at the root so every
                // screen, sheet and alert inherits it — a theme applied
                // per-screen leaves modals in the old one.
                .preferredColorScheme(.dark)
                .tint(SD.Colors.accent)
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
    @State private var tab: Tab = .devices

    enum Tab: String { case devices, shop, settings }

    var body: some View {
        content
            .background(SD.Colors.background.ignoresSafeArea())
            .task {
                if DebugLaunch.signOut, state.isSignedIn { state.signOut() }
                if let session = DebugLaunch.session, !state.isSignedIn {
                    state.signIn(AuthResponse(token: session.token, email: session.email))
                }
                if DebugLaunch.demo && !state.isSignedIn { state.enterDemo() }
                if let requested = DebugLaunch.tab.flatMap(Tab.init(rawValue:)) { tab = requested }
            }
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
                case .reconnecting:
                    // A backoff timer is already armed, but coming back to the
                    // foreground is exactly the moment the network has usually
                    // changed under us -- so try now rather than waiting out a
                    // delay that was sized for a phone sitting in a pocket.
                    state.reconnect()
                case .unreachable, .unknown:
                    state.reconnect()
                }
            }
    }

    @ViewBuilder
    private var content: some View {
        if state.isSignedIn {
            TabView(selection: $tab) {
                DevicesView()
                    .tabItem { Label("Devices", systemImage: "square.grid.2x2") }
                    .tag(Tab.devices)
                ShopView()
                    .tabItem { Label("Shop", systemImage: "bag") }
                    .tag(Tab.shop)
                SettingsView()
                    .tabItem { Label("Settings", systemImage: "slider.horizontal.3") }
                    .tag(Tab.settings)
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
