import SwiftUI

struct SettingsView: View {
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

                Section("Cloud backend") {
                    TextField("http://192.168.1.x:8787", text: state.$baseURLString)
                        .keyboardType(.URL)
                        .autocapitalization(.none)
                        .autocorrectionDisabled()
                    Text("Where cloud/app.py runs — your server's public URL, or your Mac's LAN address while developing. (Not \"localhost\": on a phone that means the phone itself.)")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Troubleshooting") {
                    NavigationLink {
                        DiagnosticsView()
                    } label: {
                        Label("Connection doctor", systemImage: "stethoscope")
                    }
                    Text("Tests every stage — account, cloud, device, live feed — and points at the first one that fails.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Live feed (local network)") {
                    TextField("http://192.168.1.20:8899", text: state.$deviceLocalURL)
                        .keyboardType(.URL)
                        .autocapitalization(.none)
                        .autocorrectionDisabled()
                    Button {
                        discovery.start()
                    } label: {
                        if discovery.isSearching {
                            HStack { ProgressView(); Text("Looking for your device…") }
                        } else {
                            Label("Find my device", systemImage: "magnifyingglass")
                        }
                    }
                    ForEach(discovery.found) { device in
                        Button {
                            state.deviceLocalURL = device.url
                            discovery.stop()
                        } label: {
                            Label("\(device.id) — \(device.url)", systemImage: "sensor.tag.radiowaves.forward")
                                .lineLimit(1)
                        }
                    }
                    SecureField("Device token (only if configured)", text: state.$deviceToken)
                        .autocapitalization(.none)
                        .autocorrectionDisabled()
                    Text("The device's dashboard address on your Wi-Fi. Frames stream directly device-to-phone — they never pass through the cloud. “Find my device” discovers it automatically (mDNS).")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Device Wi-Fi setup") {
                    NavigationLink {
                        ProvisioningView()
                    } label: {
                        Label("Set up device Wi-Fi over Bluetooth", systemImage: "wave.3.right")
                    }
                    Text("For a brand-new device, or one taken somewhere without its saved network.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
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
                        Text(healthError).font(.footnote).foregroundStyle(.secondary)
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
                            .font(.footnote).foregroundStyle(.green)
                    }
                    Text("When the sensor detects distracted driving (or a fall, a covered face, an intruder), a text goes to this number within seconds.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Push notifications") {
                    TextField("your-secret-ntfy-topic", text: $ntfyTopic)
                        .autocapitalization(.none)
                        .autocorrectionDisabled()
                    Button("Save topic") {
                        Task {
                            do {
                                try await state.backend.setNtfyTopic(ntfyTopic)
                                savedMessage = "Saved. Subscribe to this topic in the ntfy app to receive alerts."
                                errorMessage = nil
                            } catch {
                                errorMessage = error.localizedDescription
                            }
                        }
                    }
                    .disabled(ntfyTopic.isEmpty)
                    Text("Loud alerts push via ntfy.sh today (install the free ntfy app and subscribe to your topic). Native APNs push replaces this for the App Store build.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                    if let savedMessage {
                        Text(savedMessage).font(.footnote).foregroundStyle(.green)
                    }
                    if let errorMessage {
                        Text(errorMessage).font(.footnote).foregroundStyle(.red)
                    }
                }

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
        }
    }

    private static func uptimeLabel(_ seconds: Double) -> String {
        let total = Int(seconds)
        if total < 3600 { return "\(total / 60)m" }
        if total < 86400 { return "\(total / 3600)h \(total % 3600 / 60)m" }
        return "\(total / 86400)d \(total % 86400 / 3600)h"
    }
}
