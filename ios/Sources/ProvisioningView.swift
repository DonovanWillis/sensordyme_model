import SwiftUI

/// Bluetooth Wi-Fi setup: find a SensorDyme with no network, hand it credentials.
/// Reached from Settings and from the pairing wizard's Wi-Fi step.
struct ProvisioningView: View {
    @StateObject private var ble = BLEProvisioner()
    @State private var ssid = ""
    @State private var password = ""

    var body: some View {
        Form {
            Section {
                VStack(alignment: .leading, spacing: SD.Space.xs) {
                    Text("If your device has no Wi-Fi yet — new, or you moved it — it waits here for a network. Scan below, then send it yours.")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textPrimary)

                    // The single most confusing thing about this flow. The device
                    // advertises a custom service, which iOS Settings does not
                    // list — it only shows standard profiles like headphones. So
                    // someone checking there sees nothing, concludes the device
                    // is dead, and never gets to this screen, which is the only
                    // place it can appear.
                    Label(
                        "Your SensorDyme won't appear in the iPhone's Bluetooth settings — only here. That's normal.",
                        systemImage: "info.circle"
                    )
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                }
            }

            Section("1. Find your device") {
                switch ble.state {
                case .idle:
                    Button { ble.startScan() } label: {
                        Label("Scan for devices", systemImage: "antenna.radiowaves.left.and.right")
                    }
                case .bluetoothOff:
                    Label("Turn on Bluetooth in Settings to continue", systemImage: "exclamationmark.triangle")
                        .foregroundStyle(SD.Colors.warning)
                case .scanning:
                    VStack(alignment: .leading, spacing: SD.Space.xxs) {
                        HStack { ProgressView(); Text("Scanning…").foregroundStyle(SD.Colors.textSecondary) }
                        Text("Stand within a few metres of your SensorDyme. Bluetooth doesn't reach further.")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                    }
                default:
                    EmptyView()
                }
                ForEach(ble.discovered) { device in
                    Button {
                        ble.connect(id: device.id)
                    } label: {
                        HStack {
                            Label(device.name, systemImage: "sensor.tag.radiowaves.forward")
                            Spacer()
                            Text(device.proximity)
                                .font(SD.Typography.caption)
                                .foregroundStyle(device.rssi > -60
                                                 ? SD.Colors.success : SD.Colors.textTertiary)
                        }
                    }
                }
            }

            if case .ready = ble.state {
                wifiSection
            } else if case .sending = ble.state {
                wifiSection
            } else if case .deviceConnecting = ble.state {
                wifiSection
            }

            statusSection
        }
        .navigationTitle("Bluetooth setup")
        .onDisappear { ble.stop() }
    }

    private var wifiSection: some View {
        Section("2. Choose a network") {
            // The list comes from the DEVICE's own scan, not the phone's. iOS
            // cannot enumerate Wi-Fi networks at all, and the device's list is
            // the one that matters anyway: it is the machine that has to
            // associate, and it may sit behind a wall from the router.
            if ble.networks.isEmpty, ble.scanningNetworks {
                HStack {
                    ProgressView().controlSize(.small)
                    Text("Looking for networks…")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                }
            } else if ble.networks.isEmpty {
                // The device finished and found nothing. Saying "looking"
                // forever would be a lie, and it hides the real next step:
                // the device may be out of range of the router.
                VStack(alignment: .leading, spacing: SD.Space.xs) {
                    Text("No networks found")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Text("The device could not see any Wi-Fi from where it is. Move it closer to the router, or type the name below.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    Button("Search again") { ble.refreshNetworks() }
                        .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                }
            } else {
                ForEach(ble.networks) { network in
                    Button { ssid = network.ssid } label: {
                        HStack(spacing: SD.Space.sm) {
                            Image(systemName: network.secured ? "wifi.square.fill" : "wifi.square")
                                .foregroundStyle(ssid == network.ssid
                                                 ? SD.Colors.accent : SD.Colors.textTertiary)
                            Text(network.ssid)
                                .foregroundStyle(SD.Colors.textPrimary)
                            Spacer()
                            Text(network.strength)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                            if ssid == network.ssid {
                                Image(systemName: "checkmark")
                                    .foregroundStyle(SD.Colors.accent)
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            }

            Button("Scan again") { ble.refreshNetworks() }
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.accent)

            // Hidden networks never appear in a scan, so typing stays possible.
            TextField("Or type a network name", text: $ssid)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()

            SecureField("Password", text: $password)
            Button {
                ble.send(ssid: ssid, password: password)
            } label: {
                if case .ready = ble.state {
                    Label("Send to device", systemImage: "wifi")
                } else {
                    HStack { ProgressView(); Text("Sending…") }
                }
            }
            .disabled(ssid.isEmpty || !isReady)
        }
    }

    private var isReady: Bool {
        if case .ready = ble.state { return true }
        return false
    }

    @ViewBuilder
    private var statusSection: some View {
        switch ble.state {
        case .connecting(let name):
            Section { HStack { ProgressView(); Text("Connecting to \(name)…") } }
        case .ready(let name):
            Section { Label("Connected to \(name)", systemImage: "checkmark.circle").foregroundStyle(SD.Colors.success) }
        case .deviceConnecting:
            Section { HStack { ProgressView(); Text("Device is joining your Wi-Fi…") } }
        case .success(let ip):
            Section {
                Label("Device is online\(ip.map { " at \($0)" } ?? "")", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(SD.Colors.success)
                Text("Bluetooth setup is done — the live feed streams over Wi-Fi. Use “Find my device” in Settings to fill in the feed address.")
                    .font(.footnote)
                    .foregroundStyle(SD.Colors.textSecondary)
            }
        case .failed(let message):
            Section {
                Label(message, systemImage: "xmark.circle").foregroundStyle(SD.Colors.danger)
                Button("Try again") { ble.startScan() }
            }
        default:
            EmptyView()
        }
    }
}
