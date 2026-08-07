import SwiftUI

/// The setup flow's internal state, in plain sight.
///
/// Added after the app spent a whole test run showing "Searching for
/// SensorDyme" while the device sat in `discovered` the entire time. Nothing
/// was broken in a way anyone could see: the Pi advertised, Bluetooth found it,
/// the view simply never redrew. From the outside that is indistinguishable
/// from a device that is not there — and the only way to tell them apart was to
/// go and ask the Pi.
///
/// So the app now says which it is. Collapsed by default: a consumer never
/// needs this, but the difference between "not found" and "found and not shown"
/// is the difference between an afternoon of debugging and a glance.
struct SetupDebugPanel: View {
    @ObservedObject var flow: SetupFlow
    @ObservedObject var ble: BLEProvisioner
    @State private var expanded = false

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            Button {
                withAnimation(SD.Motion.standard) { expanded.toggle() }
            } label: {
                HStack(spacing: SD.Space.xxs) {
                    Image(systemName: expanded ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                    Text("Connection details")
                        .font(SD.Typography.caption)
                    Spacer()
                    Text(summary)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
                .foregroundStyle(SD.Colors.textSecondary)
            }
            .buttonStyle(.plain)

            if expanded {
                VStack(alignment: .leading, spacing: 2) {
                    row("Bluetooth", bluetoothState)
                    row("Devices seen", "\(ble.discovered.count)")
                    if let first = ble.discovered.first {
                        row("Nearest", "\(first.name) (\(first.rssi)dBm)")
                    }
                    row("Setup phase", phaseLabel)
                    row("Wi-Fi scan", ble.scanningNetworks ? "running on device" : "idle")
                    row("Networks received", "\(ble.networks.count)")
                    row("Networks characteristic",
                        ble.networkProblem == nil ? "available" : "MISSING")
                    if let failure = lastError {
                        row("Last error", failure)
                    }
                }
                .padding(.leading, SD.Space.sm)
            }
        }
        .padding(SD.Space.sm)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.sm)
                .fill(SD.Colors.surfaceRaised)
        )
    }

    private func row(_ name: String, _ value: String) -> some View {
        HStack(alignment: .top, spacing: SD.Space.xs) {
            Text(name)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
                .frame(width: 120, alignment: .leading)
            Text(value)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Spacer(minLength: 0)
        }
    }

    /// One line for the collapsed state — enough to tell the two failures apart.
    private var summary: String {
        if ble.discovered.isEmpty { return "no device yet" }
        return "\(ble.discovered.count) found"
    }

    private var bluetoothState: String {
        switch ble.state {
        case .idle: "idle"
        case .bluetoothOff: "off or not permitted"
        case .scanning: "scanning"
        case .connecting(let name): "connecting to \(name)"
        case .ready(let name): "connected to \(name)"
        case .sending: "sending credentials"
        case .deviceConnecting: "device joining Wi-Fi"
        case .success: "device online"
        case .failed: "failed"
        }
    }

    private var phaseLabel: String {
        switch flow.phase {
        case .bluetoothUnavailable: "Bluetooth off or not permitted"
        case .findingDevice: "finding device"
        case .connecting(let name): "connecting to \(name)"
        case .choosingNetwork: "choosing network"
        case .sending(let ssid): "sending \(ssid)"
        case .deviceJoining(let ssid): "device joining \(ssid)"
        case .rediscovering(let ssid, let elapsed): "looking for it on \(ssid) (\(elapsed)s)"
        case .ready(let ssid, let url): "ready on \(ssid) — \(url ?? "no address")"
        case .failed: "failed"
        }
    }

    private var lastError: String? {
        if case let .failed(reason) = flow.phase { return reason }
        if case let .failed(reason) = ble.state { return reason }
        return nil
    }
}
