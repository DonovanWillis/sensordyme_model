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
        } else {
            AuthView()
        }
    }
}
