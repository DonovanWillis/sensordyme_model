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
    /// Local draft so the field is editable without writing the Keychain on every keystroke.
    @State private var tokenDraft = ""

    var body: some View {
        NavigationStack {
            Form {
            Group {
                if state.isDemo {
                    Section {
                        HStack(spacing: 12) {
                            OrynMark(height: 22)
                            Text("Demo mode — everything you see is a simulated device.")
                                .font(.subheadline)
                        }
                        Button("Exit demo", role: .destructive) { state.signOut() }
                    }
                }

                // Every "can't reach the backend" message in the app tells the
                // user to come here, and for a while there was nothing here to
                // come to: this section had been removed while the messages that
                // point at it stayed, so the app named a fix that did not exist.
                Section("Server") {
                    // A plain row, not LabeledContent: on iOS 26 a LabeledContent
                    // whose value is a Label grew to several hundred points tall,
                    // leaving a blank card. (Same fix as rebuild/oryn-edge-app c3a9c87.)
                    HStack {
                        Text("Status")
                        Spacer(minLength: SD.Space.sm)
                        backendStatusLabel
                    }
                    .accessibilityElement(children: .combine)
                    if let address = backendAddress {
                        LabeledContent("Address", value: address)
                            .font(.footnote)
                            .foregroundStyle(SD.Colors.textSecondary)
                        Text("Found \(backendSource).")
                            .font(.footnote)
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Button {
                        Task { await findBackend() }
                    } label: {
                        Label("Find my server", systemImage: "antenna.radiowaves.left.and.right")
                    }
                    .disabled(state.isLocatingBackend)
                    if let findMessage {
                        Text(findMessage)
                            .font(.footnote)
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Text("ORYN EDGE finds your server on the Wi-Fi by itself and reconnects when it moves, restarts, or the phone rejoins the network. You should not need to type an address.")
                        .font(.footnote)
                        .foregroundStyle(SD.Colors.textSecondary)
                }

                // Deliberately its own collapsed-by-default screen rather than a
                // field in the section above. A typed address outranks discovery,
                // so it is the one setting here that can take the app off the air
                // — it should take a deliberate trip to reach, not sit under the
                // cursor next to the automatic behaviour that normally works.
                Section {
                    NavigationLink {
                        ManualBackendView()
                    } label: {
                        Label("Manual server address", systemImage: "slider.horizontal.3")
                    }
                } footer: {
                    Text("Only needed for a server that Bonjour can't reach — a different subnet, or a deployed URL.")
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

                // `discovery` was already declared here and never used, while
                // DiagnosticsView told people to come to "Find my device" to fix
                // an empty device address. Both halves existed; nothing joined
                // them, so the live feed could not be pointed at a device at all.
                Section("Live feed") {
                    if !state.deviceLocalURL.isEmpty {
                        LabeledContent("Device", value: state.deviceLocalURL)
                            .font(.footnote)
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Button {
                        discovery.start()
                    } label: {
                        Label(
                            discovery.isSearching ? "Looking for your device…" : "Find my device",
                            systemImage: "video.badge.waveform"
                        )
                    }
                    .disabled(discovery.isSearching)
                    ForEach(discovery.found) { device in
                        Button {
                            state.deviceLocalURL = device.url
                            // The id too: the address alone left the stored id
                            // naming a different camera than the one streaming.
                            state.linkedDeviceID = device.id
                            discovery.stop()
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(device.name)
                                    Text(device.url)
                                        .font(.caption)
                                        .foregroundStyle(SD.Colors.textSecondary)
                                }
                                Spacer()
                                if state.deviceLocalURL == device.url {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(SD.Colors.success)
                                }
                            }
                        }
                    }
                    if let problem = discovery.problem {
                        Text(problem)
                            .font(.footnote)
                            .foregroundStyle(SD.Colors.warning)
                    } else if discovery.found.isEmpty && !discovery.isSearching && state.deviceLocalURL.isEmpty {
                        Text("The device publishes itself on the Wi-Fi it is joined to, so the phone has to be on that same network.")
                            .font(.footnote)
                            .foregroundStyle(SD.Colors.textSecondary)
                    }

                    // The feed and /api on the device both require this token, and
                    // AppState.deviceToken had no writer anywhere in the app -- so
                    // the stream's own "add it in Settings -> Device token" message
                    // named a field that did not exist. It does now.
                    SecureField("Device token", text: $tokenDraft)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .onSubmit { state.deviceToken = tokenDraft }
                    if tokenDraft != state.deviceToken {
                        Button("Save token") { state.deviceToken = tokenDraft }
                    }
                    Text(state.deviceToken.isEmpty
                         ? "Set dashboard.token in the device's config.yaml, then paste the same value here. Without it the device refuses the live feed rather than showing its camera to everyone on the Wi-Fi."
                         : "Saved to the keychain. It must match dashboard.token in the device's config.yaml.")
                        .font(.footnote)
                        .foregroundStyle(SD.Colors.textSecondary)
                }

                Section("Device health") {
                    if let health {
                        LabeledContent("Overall") {
                            Label(health.state, systemImage: health.state == "ok" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundStyle(health.state == "ok" ? SD.Colors.success : SD.Colors.warning)
                        }
                        LabeledContent("Uptime", value: Self.uptimeLabel(health.uptime_s))
                        ForEach(health.components.sorted(by: { $0.key < $1.key }), id: \.key) { name, component in
                            if component.state != "ok" || component.critical {
                                LabeledContent(name) {
                                    Text(component.state == "ok" ? "ok" : component.detail)
                                        .font(.caption)
                                        .foregroundStyle(component.state == "ok" ? SD.Colors.success : SD.Colors.warning)
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
            .orynRows()
        }
        .orynFormStyle()
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .toolbarBackground(SD.Colors.background, for: .navigationBar)
            .sheet(isPresented: $showSetup) { SetupFlowView() }
            .onAppear { tokenDraft = state.deviceToken }
        }
    }

    /// The connection state, said in terms of what the app actually verified.
    ///
    /// "Connected" here means a server answered `/v1/health` and identified
    /// itself as the SensorDyme backend — not that a Bonjour record was seen.
    /// The two used to be reported as the same thing, which is how the app
    /// showed a connection to a server that had left the network.
    @ViewBuilder
    private var backendStatusLabel: some View {
        switch state.backendStatus {
        case .connected:
            Label("Connected", systemImage: "checkmark.circle.fill")
                .foregroundStyle(SD.Colors.success)
        case .searching:
            Label("Looking…", systemImage: "arrow.triangle.2.circlepath")
                .foregroundStyle(SD.Colors.textSecondary)
        case .reconnecting:
            Label("Reconnecting…", systemImage: "arrow.triangle.2.circlepath")
                .foregroundStyle(SD.Colors.textSecondary)
        case .unreachable:
            Label("Not connected", systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(SD.Colors.warning)
        case .unknown:
            Label("Not checked yet", systemImage: "minus.circle")
                .foregroundStyle(SD.Colors.textSecondary)
        }
    }

    /// The address in force, or nil when there is nothing to show. Shown
    /// separately from the status so a failing address is still visible — it is
    /// the single most useful thing to see when it is the wrong one.
    private var backendAddress: String? {
        let address = state.effectiveBaseURL
        return address.isEmpty ? nil : address
    }

    /// Where the address in use came from, said plainly. The three sources
    /// behave differently — only one of them can be replaced by discovery — so
    /// which one is in play is the first thing worth knowing here.
    private var backendSource: String {
        if !state.configuredBaseURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "from the address you entered"
        }
        if !AppState.productionBaseURL.isEmpty { return "built into this app" }
        return "on this network"
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
        // Goes through AppState's resolver rather than browsing directly, so this
        // button and the automatic reconnection cannot disagree about what is
        // connected — two code paths writing the same state is what made the
        // status unreliable in the first place. It also clears the confirmed
        // address first, which is the point of pressing it: the reason to look
        // again is that what the app is using is wrong, so re-confirming it is
        // not an answer.
        state.reconnect()
        // Wait for the resolve the line above kicked off, rather than racing it.
        let found = await state.resolveBackend()
        if let found {
            findMessage = "Found your server at \(found)."
        } else if case let .unreachable(reason) = state.backendStatus {
            findMessage = reason
        } else if case let .reconnecting(_, reason) = state.backendStatus {
            findMessage = reason
        } else {
            findMessage = "No SensorDyme server answered on this Wi-Fi. Check that it is running and that this phone is on the same network."
        }
    }

    private static func uptimeLabel(_ seconds: Double) -> String {
        let total = Int(seconds)
        if total < 3600 { return "\(total / 60)m" }
        if total < 86400 { return "\(total / 3600)h \(total % 3600 / 60)m" }
        return "\(total / 86400)d \(total % 86400 / 3600)h"
    }
}

/// The escape hatch: a typed server address.
///
/// Its own screen, and deliberately not the first thing offered. Automatic
/// discovery is what should normally be in play, and a typed address outranks
/// it — so this is the one control here that can leave the app pointed at a
/// server that has ceased to exist. That is exactly what happened when a LAN
/// address was baked into the build: it outranked discovery, the network it
/// named was gone, and there was no way to change it without a rebuild.
///
/// So the field exists (the alternative was error messages naming a screen with
/// nothing on it), it says plainly what it costs, and clearing it is one tap.
struct ManualBackendView: View {
    @EnvironmentObject private var state: AppState
    @State private var draft = ""
    @State private var message: String?
    @State private var checking = false

    var body: some View {
        Form {
            Group {
            Section {
                TextField("http://192.168.1.50:8787", text: $draft)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .keyboardType(.URL)
            } footer: {
                Text("Include the scheme and port, e.g. http://192.168.1.50:8787. This is checked before it is saved — an address that doesn't answer is not saved.")
            }

            Section {
                Button {
                    Task { await save() }
                } label: {
                    if checking { Label("Checking…", systemImage: "arrow.triangle.2.circlepath") }
                    else { Label("Check and save", systemImage: "checkmark.circle") }
                }
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || checking)

                // Also offered when only an identity is pinned: a backend whose id
                // was regenerated (a fresh checkout, cloud/.backend_id deleted) is
                // otherwise refused for ever, and this was the only way out — but
                // it used to appear only after typing an address first.
                if !state.configuredBaseURL.isEmpty || !state.knownBackendID.isEmpty {
                    Button(state.configuredBaseURL.isEmpty
                           ? "Forget remembered server and search again"
                           : "Clear and go back to automatic", role: .destructive) {
                        draft = ""
                        message = "Cleared. ORYN EDGE will find your server on the network again."
                        state.useAutomaticBackend()
                    }
                }
            } footer: {
                if let message {
                    Text(message)
                } else {
                    Text("While this is set, ORYN EDGE uses it in preference to whatever it finds on the network. Clear it to go back to finding your Mac automatically.")
                }
            }
        }
            .orynRows()
        }
        .orynFormStyle()
        .navigationTitle("Manual server address")
        .onAppear { draft = state.configuredBaseURL }
    }

    /// Verify before storing. Saving an address that does not answer is how a
    /// typo becomes "the app is broken" — and because a typed address outranks
    /// discovery, an unverified one silently disables the mechanism that works.
    private func save() async {
        let candidate = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        message = nil
        guard BackendResolution.productionDefault(fromInfoPlist: candidate) == candidate else {
            message = "That isn't a complete address. It needs to start with http:// or https:// and include a host."
            return
        }
        guard BackendResolution.isUsableFromPhone(candidate) else {
            message = "On a phone, localhost means the phone itself. Use your Mac's address on the network instead."
            return
        }
        guard BackendResolution.cleartextPermitted(candidate) else {
            message = BackendResolution.cleartextRefusal(host: BackendResolution.host(of: candidate) ?? "that server")
            return
        }
        checking = true
        defer { checking = false }
        guard await BackendLocator.answers(candidate) else {
            message = "Nothing answered as a SensorDyme server at that address. It was not saved."
            return
        }
        state.pinConfigured(candidate)
        message = "Saved. ORYN EDGE is using that address."
    }
}
