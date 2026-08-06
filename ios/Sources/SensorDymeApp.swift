import SwiftUI

@main
struct SensorDymeApp: App {
    @StateObject private var state = AppState()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(state)
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
            .tint(Brand.electric)
        } else {
            AuthView()
        }
    }
}
