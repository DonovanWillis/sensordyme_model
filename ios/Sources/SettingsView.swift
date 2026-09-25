import SwiftUI

struct SettingsView: View {
    @State private var showSetup = false
    @EnvironmentObject private var state: AppState
    @StateObject private var discovery = DeviceDiscovery()
    @State private var ntfyTopic = ""
    @State private var savedMessage: String?
    @State private var errorMessage: String?
    @AppStorage("alertPhone") private var alertPhone = ""
    @AppStorage("smsOnLoud") private var smsOnLoud = true
    @State private var phoneSaved = false
    @State private var health: DeviceHealth?
    @State private var healthError: String?
    @State private var findMessage: String?

    var body: some View {
        NavigationStack {
            Form {
                if state.isDemo {
                    Section {
                        HStack(spacing: 12) {
                            LogoMark(size: 30, accented: true)
                            Text("Demo mode — everything you see is a simulated device.")
                                .font(.subheadline)
                        }
                        Button("Exit demo", role: .destructive) { state.signOut() }
                    }
                }

                Section("Appearance") {
                    Picker("Theme", selection: Binding(
                        get: { state.appearance },
                        set: { state.appearance = $0 }
                    )) {
                        ForEach(SD.Appearance.allCases) { option in
                            Label(option.title, systemImage: option.icon).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text("System follows your phone's setting. Light and Dark override it for SensorDyme only.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                }


                Section("Troubleshooting") {
                    NavigationLink {
                        DiagnosticsView()
                    } label: {
                        Label("Connection doctor", systemImage: "stethoscope")
                    }
                    Text("Tests every stage — account, cloud, device, live feed — and points at the first one that fails.")
                        .font(.footnote)
                        .foregroundStyle(SD.Colors.textSecondary)
                }



                Section("Device Wi-Fi setup") {
                    // Same flow as the home screen's Bluetooth button, not a
                    // second one. Two provisioning screens means two state
                    // machines to keep honest, and they drift.
                    Button { showSetup = true } label: {
                        Label("Set up device Wi-Fi over Bluetooth", systemImage: "wave.3.right")
                    }
                    Text("For a brand-new device, or one moved somewhere new. Works whether or not the device currently has Wi-Fi.")
                        .font(.footnote)
                        .foregroundStyle(SD.Colors.textSecondary)
                }

                Section("Device health") {
                    if let health {
                        LabeledContent("Overall") {
                            Label(health.state, systemImage: health.state == "ok" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundStyle(health.state == "ok" ? .green : .orange)
                        }
                        LabeledContent("Uptime", value: Self.uptimeLabel(health.uptime_s))
                        ForEach(health.components.sorted(by: { $0.key < $1.key }), id: \.key) { name, component in
                            if component.state != "ok" || component.critical {
                                LabeledContent(name) {
                                    Text(component.state == "ok" ? "ok" : component.detail)
                                        .font(.caption)
                                        .foregroundStyle(component.state == "ok" ? .green : .orange)
                                        .lineLimit(2)
                                }
                            }
                        }
                    } else if let healthError {
                        Text(healthError).font(.footnote).foregroundStyle(SD.Colors.textSecondary)
                    }
                    Button("Check device health") {
                        Task {
                            do {
                                health = try await DeviceHealth.fetch(
                                    baseURL: state.deviceLocalURL, token: state.deviceToken
                                )
                                healthError = nil
                            } catch {
                                health = nil
                                healthError = error.localizedDescription
                            }
                        }
                    }
                    .disabled(state.deviceLocalURL.isEmpty)
                }

                Section("Text-message alerts") {
                    TextField("+1 555 123 4567", text: $alertPhone)
                        .keyboardType(.phonePad)
                    Toggle("Text me on serious alerts", isOn: $smsOnLoud)
                    Button("Save phone number") {
                        Task {
                            do {
                                try await state.backend.setPhone(
                                    alertPhone.replacingOccurrences(of: " ", with: ""), smsOnLoud: smsOnLoud
                                )
                                phoneSaved = true
                                errorMessage = nil
                            } catch {
                                errorMessage = error.localizedDescription
                                phoneSaved = false
                            }
                        }
                    }
                    .disabled(alertPhone.isEmpty)
                    if phoneSaved {
                        Label("Saved — serious alerts will text this number", systemImage: "checkmark.circle.fill")
                            .font(.footnote).foregroundStyle(SD.Colors.success)
                    }
                    Text("When the sensor detects distracted driving (or a fall, a covered face, an intruder), a text goes to this number within seconds.")
                        .font(.footnote)
                        .foregroundStyle(SD.Colors.textSecondary)
                }

                BackgroundAlertsSection(topic: $ntfyTopic)

                if state.isSignedIn {
                    Section("Account") {
                        LabeledContent("Signed in as", value: state.accountEmail)
                        Button("Log out", role: .destructive) {
                            state.signOut()
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .sheet(isPresented: $showSetup) { SetupFlowView() }
        }
    }

    /// Where the address in use came from, said plainly. The three sources
    /// behave differently — only one of them can be replaced by discovery — so
    /// which one is in play is the first thing worth knowing here.
    private var backendSource: String {
        if !state.configuredBaseURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "entered here"
        }
        if !AppState.productionBaseURL.isEmpty { return "built into this app" }
        return "found on this network"
    }

    /// Search the network on demand, and say what happened either way.
    ///
    /// Ignores whatever is currently saved: the reason to press this button is
    /// that the saved address is wrong, so confirming it is not an answer.
    ///
    /// The result goes to the discovery cache, not to the field above. Pressing
    /// "find my server" is a request to locate one now, not a decision to pin
    /// the app to the address it happens to have today — and writing a found
    /// LAN address into configuration is precisely the conflation that stranded
    /// the app on an IP from a network it had left.
    private func findBackend() async {
        findMessage = nil
        if let found = await BackendLocator.shared.search() {
            state.adoptDiscovered(found)
            findMessage = "Found your server at \(found)."
        } else {
            findMessage = BackendLocator.shared.problem
                ?? "No SensorDyme server answered on this Wi-Fi. Check that it is running and that this phone is on the same network."
        }
    }

    private static func uptimeLabel(_ seconds: Double) -> String {
        let total = Int(seconds)
        if total < 3600 { return "\(total / 60)m" }
        if total < 86400 { return "\(total / 3600)h \(total % 3600 / 60)m" }
        return "\(total / 86400)d \(total % 86400 / 3600)h"
    }
}
