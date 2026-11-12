import SwiftUI

/// Says whether the camera is reachable, in words a person can act on.
///
/// Shown only when there is something to say. A banner that is always present
/// becomes furniture, and the one time it matters nobody reads it.
///
/// It never mentions IP addresses, DHCP, subnets or mDNS. The user asked
/// whether their camera works, not how networking does.
struct DeviceLinkBanner: View {
    @ObservedObject var link: DeviceLink
    /// What the backend says about this device, when the device list has loaded.
    ///
    /// Nil means the backend could not be asked — which is itself worth saying,
    /// rather than guessing at the device's state from Bonjour alone.
    var backendDevice: Device?
    var backendReachable: Bool = true
    /// Whether the account has devices at all — with no `backendDevice`, the
    /// difference between "none on the account" and "can't tell which".
    var accountHasDevices: Bool = false

    var body: some View {
        switch link.state {
        case .idle, .connected:
            EmptyView()   // working, or nothing to report yet
        case .searching:
            row(icon: "antenna.radiowaves.left.and.right", tint: SD.Colors.textSecondary,
                title: "Looking for your device…", detail: nil, action: nil)
        case .notFound:
            // Bonjour found nothing, which on its own says only "not reachable
            // from this phone". It used to be reported as "Can't find your
            // device" with advice about Wi-Fi — true of the phone, and the wrong
            // fact for a camera that had been switched off for nine days, which
            // the backend knew all along. `DevicePresence` combines the two.
            notFoundRow
        case .needsProvisioning:
            // Only shown when the device has been SEEN and reports no network —
            // never as a guess. Bluetooth setup needs you within a few metres of
            // it, so suggesting it for a device that is simply elsewhere sends
            // someone hunting for a signal that cannot exist.
            row(icon: "wifi.slash", tint: SD.Colors.warning,
                title: "Your device needs Wi-Fi setup",
                detail: "Stand near your SensorDyme and set it up over Bluetooth.",
                action: "Search again")
        }
    }

    @ViewBuilder
    private var notFoundRow: some View {
        let verdict = DevicePresence.verdict(
            foundOnNetwork: false,
            backend: backendDevice.map {
                DevicePresence.BackendView(
                    lastSeen: $0.lastSeen.map { Date(timeIntervalSince1970: $0) }
                )
            },
            backendReachable: backendReachable,
            accountHasDevices: accountHasDevices,
            now: Date()
        )
        row(
            // A running device that this phone simply cannot see is not a
            // warning about the camera, so it does not get the alarming icon.
            icon: DevicePresence.offersLocalRetry(verdict) ? "wifi.exclamationmark" : "bolt.slash",
            tint: SD.Colors.warning,
            title: DevicePresence.title(verdict),
            detail: DevicePresence.detail(verdict),
            action: DevicePresence.offersLocalRetry(verdict) ? "Search again" : nil
        )
    }

    @ViewBuilder
    private func row(icon: String, tint: Color, title: String,
                     detail: String?, action: String?) -> some View {
        SDCard {
            HStack(alignment: .top, spacing: SD.Space.sm) {
                Image(systemName: icon)
                    .font(SD.Typography.headline)
                    .foregroundStyle(tint)
                VStack(alignment: .leading, spacing: SD.Space.xxs) {
                    // The words as one element; the action stays its own button
                    // (combining the whole card made the button unpressable).
                    VStack(alignment: .leading, spacing: SD.Space.xxs) {
                        Text(title)
                            .font(SD.Typography.label)
                            .foregroundStyle(SD.Colors.textPrimary)
                        if let detail {
                            Text(detail)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    .accessibilityElement(children: .combine)
                    if let action {
                        Button(action) { Task { await link.resolve(force: true) } }
                            .font(SD.Typography.caption.weight(.semibold))
                            .foregroundStyle(SD.Colors.accent)
                            // 44 pt hit area (M8 audit 21); the drawn text is unchanged.
                            .frame(minHeight: 44, alignment: .leading)
                            .contentShape(Rectangle())
                    }
                }
                Spacer(minLength: 0)
            }
        }
        .accessibilityElement(children: .contain)
    }
}
