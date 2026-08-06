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
                Text("If your device has no Wi-Fi yet (new, or you moved it), it broadcasts over Bluetooth as SensorDyme-XXXX. Connect and send it your network.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Section("1. Find your device") {
                switch ble.state {
                case .idle:
                    Button { ble.startScan() } label: {
                        Label("Scan for devices", systemImage: "antenna.radiowaves.left.and.right")
                    }
                case .bluetoothOff:
                    Label("Turn on Bluetooth in Settings to continue", systemImage: "exclamationmark.triangle")
                        .foregroundStyle(.orange)
                case .scanning:
                    HStack { ProgressView(); Text("Scanning…").foregroundStyle(.secondary) }
                default:
                    EmptyView()
                }
                ForEach(ble.discovered, id: \.self) { name in
                    Button {
                        ble.connect(name: name)
                    } label: {
                        Label(name, systemImage: "sensor.tag.radiowaves.forward")
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
        Section("2. Send your Wi-Fi") {
            TextField("Network name (SSID)", text: $ssid)
                .autocapitalization(.none)
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
            Section { Label("Connected to \(name)", systemImage: "checkmark.circle").foregroundStyle(.green) }
        case .deviceConnecting:
            Section { HStack { ProgressView(); Text("Device is joining your Wi-Fi…") } }
        case .success(let ip):
            Section {
                Label("Device is online\(ip.map { " at \($0)" } ?? "")", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                Text("Bluetooth setup is done — the live feed streams over Wi-Fi. Use “Find my device” in Settings to fill in the feed address.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        case .failed(let message):
            Section {
                Label(message, systemImage: "xmark.circle").foregroundStyle(.red)
                Button("Try again") { ble.startScan() }
            }
        default:
            EmptyView()
        }
    }
}
