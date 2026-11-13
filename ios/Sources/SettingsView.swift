import SwiftUI

/// Settings, in the order people come here for: who is signed in, their
/// devices, their alerts, then the connection to the server and the tools to
/// fix it. Each row says what it is; anything with more than a line of detail
/// is one level down, so this screen reads as a list, not a manual.
///
/// Every control does what it says, at once or with an explicit Save that
/// says whether it is saved. Device information is shown only as the device
/// itself reported it (Live feed → Device health); nothing here is a
/// placeholder.
struct SettingsView: View {
    @State private var showSetup = false
    @EnvironmentObject private var state: AppState
    @State private var findMessage: String?
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        NavigationStack {
            ScrollViewReader { proxy in
            Form {
            Group {
                if state.isDemo {
                    Section {
                        HStack(spacing: 12) {
                            OrynMark(height: 22)
                                .accessibilityHidden(true)
                            Text("Demo mode — everything you see is a simulated device.")
                                .font(SD.Typography.callout)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Button("Exit demo", role: .destructive) { state.signOut() }
                            .accessibilityIdentifier("settings.exitDemo")
                    }
                } else if state.isSignedIn {
                    Section("Account") {
                        LabeledContent("Signed in as") {
                            Text(state.accountEmail)
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .accessibilityElement(children: .combine)
                    }
                }

                Section {
                    // Same flow as the device list's Bluetooth button, not a
                    // second one. Two provisioning screens means two state
                    // machines to keep honest, and they drift.
                    Button { showSetup = true } label: {
                        Label("Set up device Wi-Fi over Bluetooth", systemImage: "dot.radiowaves.left.and.right")
                    }
                    .accessibilityIdentifier("settings.bluetoothSetup")
                    NavigationLink {
                        LiveFeedSettingsView()
                    } label: {
                        Label("Live feed", systemImage: "video")
                    }
                    .accessibilityIdentifier("settings.liveFeed")
                } header: {
                    Text("Devices")
                } footer: {
                    Text("Bluetooth setup is for a new device, or one moved somewhere new. Live feed is the picture on this Wi-Fi: finding the device, its token and its health.")
                }

                Section("Alerts") {
                    NavigationLink {
                        BackgroundAlertsSettingsView()
                    } label: {
                        Label("When the app is closed", systemImage: "bell.badge")
                    }
                    .accessibilityIdentifier("settings.backgroundAlerts")
                    NavigationLink {
                        TextAlertsSettingsView()
                    } label: {
                        Label("Text messages", systemImage: "message")
                    }
                    .accessibilityIdentifier("settings.textAlerts")
                }

                // Every "can't reach the backend" message in the app tells the
                // user to come here, so it must always be here.
                Section {
                    statusRow
                    if let address = backendAddress {
                        LabeledContent("Address") {
                            Text(address)
                                .font(SD.Typography.callout.monospaced())
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .accessibilityElement(children: .combine)
                    }
                    Button {
                        Task { await findBackend() }
                    } label: {
                        Label(state.isLocatingBackend ? "Looking for your server…" : "Find my server",
                              systemImage: "antenna.radiowaves.left.and.right")
                    }
                    .disabled(state.isLocatingBackend)
                    .accessibilityIdentifier("settings.findServer")
                    if let findMessage {
                        Text(findMessage)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    // Deliberately one level down: a typed address outranks
                    // discovery, so it is the one setting here that can take the
                    // app off the air.
                    NavigationLink {
                        ManualBackendView()
                    } label: {
                        Label("Manual server address", systemImage: "slider.horizontal.3")
                    }
                    NavigationLink {
                        DiagnosticsView()
                    } label: {
                        Label("Connection doctor", systemImage: "stethoscope")
                    }
                } header: {
                    Text("Server")
                } footer: {
                    Text(backendAddress == nil
                         ? "ORYN EDGE finds your server on the Wi-Fi by itself. The connection doctor tests every stage and points at the first one that fails."
                         : "Address \(backendSource). ORYN EDGE finds your server on the Wi-Fi by itself and reconnects when it moves. The connection doctor tests every stage and points at the first one that fails.")
                }

                if state.isSignedIn && !state.isDemo {
                    Section {
                        Button("Log out", role: .destructive) {
                            state.signOut()
                        }
                        .accessibilityIdentifier("settings.logout")
                    } footer: {
                        Text("Signing out also forgets this phone's device token.")
                    }
                    .id("settings.end")
                }
            }
            .orynRows()
        }
        .task {
            guard DebugLaunch.scrollToBottom else { return }
            try? await Task.sleep(for: .seconds(1))
            proxy.scrollTo("settings.end", anchor: .bottom)
        }
            }
        .orynFormStyle()
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.large)
            .toolbarBackground(SD.Colors.background, for: .navigationBar)
            .sheet(isPresented: $showSetup) { SetupFlowView() }
            // DEBUG renders only (DebugLaunch.settingsPage); nil in Release.
            .navigationDestination(isPresented: Binding(get: { debugPage != nil }, set: { if !$0 { debugPage = nil } })) {
                switch debugPage {
                case "live": LiveFeedSettingsView()
                case "text": TextAlertsSettingsView()
                case "alerts": BackgroundAlertsSettingsView()
                case "doctor": DiagnosticsView()
                default: EmptyView()
                }
            }
            .task {
                guard let page = DebugLaunch.settingsPage else { return }
                try? await Task.sleep(for: .seconds(1))
                debugPage = page
            }
        }
    }

    @State private var debugPage: String?

    /// The connection state beside its name, or under it when the reader's
    /// text size will not fit both on one line ("Con / nected" at AX sizes).
    private var statusRow: some View {
        Group {
            if typeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: SD.Space.xxs) {
                    Text("Status")
                    backendStatusLabel
                }
            } else {
                HStack {
                    Text("Status")
                    Spacer(minLength: SD.Space.sm)
                    backendStatusLabel
                }
            }
        }
        // A plain row, not LabeledContent: on iOS 26 a LabeledContent whose
        // value is a Label grew to several hundred points tall.
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("settings.serverStatus")
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
        return "found on this network"
    }

    /// Search the network on demand, and say what happened either way.
    ///
    /// Goes through AppState's resolver rather than browsing directly, so this
    /// button and the automatic reconnection cannot disagree about what is
    /// connected. It also clears the confirmed address first, which is the
    /// point of pressing it: the reason to look again is that what the app is
    /// using is wrong. The result goes to the discovery cache, never into the
    /// typed address (that conflation once stranded the app on an old IP).
    private func findBackend() async {
        findMessage = nil
        state.reconnect()
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
