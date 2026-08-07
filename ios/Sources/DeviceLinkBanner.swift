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

    var body: some View {
        switch link.state {
        case .idle, .connected:
            EmptyView()   // working, or nothing to report yet
        case .searching:
            row(icon: "antenna.radiowaves.left.and.right", tint: SD.Colors.textSecondary,
                title: "Looking for your device…", detail: nil, action: nil)
        case .notFound:
            row(icon: "wifi.exclamationmark", tint: SD.Colors.warning,
                title: "Can't find your device",
                detail: "It may be on a different Wi-Fi network than your phone.",
                action: "Search again")
        case .needsProvisioning:
            row(icon: "wifi.slash", tint: SD.Colors.warning,
                title: "Your device needs Wi-Fi setup",
                detail: "Set it up over Bluetooth — it only takes a moment.",
                action: "Search again")
        }
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
                    Text(title)
                        .font(SD.Typography.label)
                        .foregroundStyle(SD.Colors.textPrimary)
                    if let detail {
                        Text(detail)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if let action {
                        Button(action) { Task { await link.resolve(force: true) } }
                            .font(SD.Typography.caption.weight(.semibold))
                            .foregroundStyle(SD.Colors.accent)
                            .padding(.top, 2)
                    }
                }
                Spacer(minLength: 0)
            }
        }
        .accessibilityElement(children: .combine)
    }
}
