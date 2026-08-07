import SwiftUI

/// Setting up a SensorDyme, start to finish, from the phone.
///
/// One screen that changes rather than a stack to walk: setup is a single
/// errand, and the only question at any moment is "what is happening now". The
/// checklist stays on screen throughout so the answer is always visible,
/// including during the part where the device vanishes from the network to join
/// another one — the moment that most looks like a failure and is not.
struct SetupFlowView: View {
    @Environment(\.dismiss) private var dismiss
    @StateObject private var flow = SetupFlow()

    @State private var selectedSSID = ""
    @State private var password = ""
    @State private var showPasswordEntry = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.md) {
                    header
                    content
                    SetupDebugPanel(flow: flow, ble: flow.ble)
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.vertical, SD.Space.md)
            }
            .background(SD.Colors.background)
            .navigationTitle("Connect SensorDyme")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { flow.stop(); dismiss() }
                }
            }
            .task { flow.start() }
            .onDisappear { flow.stop() }
        }
        .tint(SD.Colors.accent)
    }

    // MARK: Header

    @ViewBuilder private var header: some View {
        switch flow.phase {
        case .bluetoothUnavailable:
            headline("Bluetooth is off",
                     "SensorDyme sets up your device's Wi-Fi over Bluetooth. Turn it on in Settings, and allow SensorDyme to use it.")
        case .findingDevice where !flow.ble.discovered.isEmpty:
            headline("SensorDyme found", "Tap Connect to set up its Wi-Fi.")
        case .findingDevice:
            headline("Looking for your SensorDyme",
                     "Stand within a few metres of it. Bluetooth doesn't reach much further.")
        case .connecting(let name):
            headline("Connecting to \(name)", nil)
        case .choosingNetwork:
            headline("Choose a Wi-Fi network",
                     "These are the networks your SensorDyme can see from where it is — not the ones your phone can see.")
        case .sending(let ssid), .deviceJoining(let ssid):
            headline("Setting up \(ssid)", nil)
        case .rediscovering(let ssid, _):
            headline("Switching to \(ssid)",
                     "Your SensorDyme is leaving the old network and joining this one. It disappears for a moment — that's normal.")
        case .ready(let ssid, _):
            headline("SensorDyme is ready", "Connected to \(ssid).")
        case .failed:
            headline("Setup didn't finish", nil)
        }
    }

    private func headline(_ title: String, _ detail: String?) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            Text(title)
                .font(SD.Typography.title)
                .foregroundStyle(SD.Colors.textPrimary)
            if let detail {
                Text(detail)
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: Body

    @ViewBuilder private var content: some View {
        switch flow.phase {
        case .bluetoothUnavailable:
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    Label("Control Centre → Bluetooth, or Settings → Bluetooth",
                          systemImage: "1.circle")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Label("Settings → SensorDyme → allow Bluetooth",
                          systemImage: "2.circle")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Button("Open Settings") {
                        if let url = URL(string: UIApplication.openSettingsURLString) {
                            UIApplication.shared.open(url)
                        }
                    }
                    .buttonStyle(SDButtonStyle(kind: .primary))
                }
            }
        case .findingDevice:
            deviceList
        case .connecting:
            SDCard { HStack { ProgressView(); Text("Pairing…").foregroundStyle(SD.Colors.textSecondary) } }
        case .choosingNetwork:
            networkList
        case .sending, .deviceJoining, .rediscovering:
            progressCard
        case .ready(_, let url):
            readySummary(url: url)
        case .failed(let reason):
            failureCard(reason)
        }
    }

    private var deviceList: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                if flow.ble.discovered.isEmpty {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text("Searching…").foregroundStyle(SD.Colors.textSecondary)
                    }
                    // The single most confusing thing about this flow: the
                    // device advertises a custom service, which iOS Settings
                    // does not list. Someone who goes looking there sees
                    // nothing and concludes it is dead.
                    Text("Your SensorDyme won't appear in the iPhone's Bluetooth settings — only here. That's normal.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                    if flow.searchingSeconds > 20 {
                        Divider().overlay(SD.Colors.border)
                        Text("Still nothing after \(flow.searchingSeconds)s")
                            .font(SD.Typography.label)
                            .foregroundStyle(SD.Colors.warning)
                        Text("Check the device is powered on and you are within a few metres of it. If it is already on your Wi-Fi and working, it stays reachable over Bluetooth too — so this should still find it.")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("Search again") { flow.retry() }
                            .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                    }
                } else if flow.ble.discovered.count == 1, let device = flow.ble.discovered.first {
                    // The overwhelmingly common case. A single row that happens
                    // to be tappable asks someone to work out that it is a
                    // button; a labelled button does not.
                    HStack(spacing: SD.Space.sm) {
                        Image(systemName: "checkmark.circle.fill")
                            .foregroundStyle(SD.Colors.success)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(device.name)
                                .font(SD.Typography.body)
                                .foregroundStyle(SD.Colors.textPrimary)
                            Text(device.proximity)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                        }
                        Spacer()
                    }
                    Button("Connect") { flow.connect(to: device) }
                        .buttonStyle(SDButtonStyle(kind: .primary))
                } else {
                    ForEach(flow.ble.discovered) { device in
                        Button { flow.connect(to: device) } label: {
                            HStack(spacing: SD.Space.sm) {
                                Image(systemName: "sensor.tag.radiowaves.forward")
                                    .foregroundStyle(SD.Colors.accent)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(device.name)
                                        .font(SD.Typography.body)
                                        .foregroundStyle(SD.Colors.textPrimary)
                                    Text(device.proximity)
                                        .font(SD.Typography.caption)
                                        .foregroundStyle(SD.Colors.textTertiary)
                                }
                                Spacer()
                                Image(systemName: "chevron.right")
                                    .foregroundStyle(SD.Colors.textTertiary)
                            }
                        }
                        .buttonStyle(.plain)
                        if device.id != flow.ble.discovered.last?.id {
                            Divider().overlay(SD.Colors.border)
                        }
                    }
                }
            }
        }
    }

    private var networkList: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    if flow.ble.networks.isEmpty, flow.ble.scanningNetworks {
                        HStack {
                            ProgressView().controlSize(.small)
                            Text("Scanning…").foregroundStyle(SD.Colors.textSecondary)
                        }
                    } else if flow.ble.networks.isEmpty {
                        // The device looked and found nothing. Saying "scanning"
                        // forever would be a lie and hides the real next step.
                        Text("Your SensorDyme can't see any Wi-Fi from where it is.")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.textPrimary)
                        Text("Move it closer to your router, or type a network name below.")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("Search again") { flow.ble.refreshNetworks() }
                            .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                    } else {
                        ForEach(flow.ble.networks) { network in
                            Button {
                                selectedSSID = network.ssid
                                showPasswordEntry = true
                            } label: {
                                HStack(spacing: SD.Space.sm) {
                                    Image(systemName: network.secured ? "lock.fill" : "lock.open")
                                        .font(.caption)
                                        .foregroundStyle(SD.Colors.textTertiary)
                                    Text(network.ssid)
                                        .font(SD.Typography.body)
                                        .foregroundStyle(SD.Colors.textPrimary)
                                    Spacer()
                                    Text(network.strength)
                                        .font(SD.Typography.caption)
                                        .foregroundStyle(SD.Colors.textTertiary)
                                }
                            }
                            .buttonStyle(.plain)
                            if network.id != flow.ble.networks.last?.id {
                                Divider().overlay(SD.Colors.border)
                            }
                        }
                    }
                }
            }

            HStack {
                Button("Other network…") {
                    selectedSSID = ""
                    showPasswordEntry = true
                }
                .buttonStyle(SDButtonStyle(kind: .tertiary, fullWidth: false))
                Spacer()
                if flow.ble.scanningNetworks {
                    HStack(spacing: SD.Space.xxs) {
                        ProgressView().controlSize(.small)
                        Text("Refreshing").font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                    }
                } else {
                    Button("Refresh") { flow.ble.refreshNetworks() }
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.accent)
                }
            }
        }
        .sheet(isPresented: $showPasswordEntry) {
            PasswordSheet(ssid: $selectedSSID, password: $password) {
                showPasswordEntry = false
                flow.send(ssid: selectedSSID, password: password)
                password = ""   // not kept a moment longer than the write needs it
            }
        }
    }

    private var progressCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.md) {
                SetupProgressView(steps: steps)
                if case .rediscovering(let ssid, let elapsed) = flow.phase, elapsed > 25 {
                    // mDNS does not cross networks. If the phone is still on the
                    // old one the device can be perfectly online and invisible
                    // from here — and this is the case where that bites, because
                    // for many people the old network was their own hotspot.
                    Label(
                        "Still looking. If your phone was on the old network, join \(ssid) on your phone too.",
                        systemImage: "wifi.exclamationmark"
                    )
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func readySummary(url: String?) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDCard { SetupProgressView(steps: steps) }
            DeviceReadinessCard(baseURL: url)
            Button("Done") { flow.stop(); dismiss() }
                .buttonStyle(SDButtonStyle(kind: .primary))
        }
    }

    private func failureCard(_ reason: String) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    SetupProgressView(steps: steps)
                    Text(reason)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            Button("Try again") { flow.retry() }
                .buttonStyle(SDButtonStyle(kind: .primary))
        }
    }

    /// The checklist, derived from the flow rather than from a timer.
    private var steps: [SetupStep] {
        SetupStep.checklist(
            for: flow.ble.state,
            networksFound: flow.ble.networks.count,
            scanning: flow.ble.scanningNetworks,
            stateBeforeFailure: flow.ble.stateBeforeFailure,
            handover: handoverStatus
        )
    }

    /// Progress on the part Bluetooth cannot see: the device joining its new
    /// network and reappearing on Wi-Fi.
    private var handoverStatus: SetupStep.Status {
        switch flow.phase {
        case .rediscovering: .active
        case .ready: .done
        case .failed: .failed
        default: .pending
        }
    }
}

/// Asks for the Wi-Fi password, and nothing else.
///
/// Separate from the network list so the keyboard cannot cover the list behind
/// it, and so the password lives in exactly one place that is easy to audit.
private struct PasswordSheet: View {
    @Binding var ssid: String
    @Binding var password: String
    let onSubmit: () -> Void

    @Environment(\.dismiss) private var dismiss
    @FocusState private var focused: Bool

    var body: some View {
        NavigationStack {
            Form {
                Section("Network") {
                    TextField("Network name", text: $ssid)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                }
                Section("Password") {
                    SecureField("Wi-Fi password", text: $password)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .focused($focused)
                    Text("Sent straight to your SensorDyme over Bluetooth. It isn't stored on this phone or sent anywhere else.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
            }
            .navigationTitle(ssid.isEmpty ? "Other network" : ssid)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { password = ""; dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Connect", action: onSubmit)
                        .disabled(ssid.trimmingCharacters(in: .whitespaces).isEmpty)
                }
            }
            .task { focused = true }
        }
    }
}
