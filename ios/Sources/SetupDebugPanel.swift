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
                // The dead-gap bug again, in code written to help debug it.
                // Without these the target is the chevron glyph and the two
                // text runs; the wide gap between them — most of the row — is
                // not hit-testable. .frame before .contentShape, so the shape
                // covers the enlarged frame rather than the text's height.
                .frame(minHeight: 44)
                .contentShape(Rectangle())
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
                    row("Device id", ble.deviceID ?? "not yet read")
                    row("Credentials", ble.requiresSetupCode ? "setup code (proof of possession)" : (ble.deviceAuth ?? "unknown"))
                    row("Steps taken", "\(flow.transitions.count)")
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
        case .identifying(let name): "reading device info from \(name)"
        case .ready(let name): "connected to \(name)" + (ble.requiresSetupCode ? " (needs setup code)" : "")
        case .proving: "proving the setup code"
        case .proven: "setup code proven"
        case .sending: "sending credentials"
        case .deviceConnecting: "device joining Wi-Fi"
        case .success: "device online"
        case .joinFailed: "device could not join"
        case .failed: "failed"
        }
    }

    private var phaseLabel: String {
        // The onboarding state, without any associated text a person typed.
        let text = "\(flow.state)"
        return String(text.prefix { $0 != "(" })
    }

    private var lastError: String? {
        if case let .failed(failure) = flow.state { return failure.reason }
        if case let .failed(reason) = ble.state { return reason }
        return ble.proofRejection?.message
    }
}
