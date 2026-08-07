import SwiftUI

/// A full-width, full-height tappable row.
///
/// Exists because `.buttonStyle(.plain)` around an HStack hit-tests the
/// *rendered content* — the icon, the text, the trailing label — and not the
/// `Spacer()` between them. The gap in the middle of the row, which is most of
/// it, was dead. Tapping there did nothing, so rows felt unreliable and needed
/// a second, better-aimed tap.
///
/// Three things make a row feel solid, and all three were missing:
///   - `contentShape(Rectangle())` so the whole rectangle is the target
///   - a minimum height, because a row of body text is roughly half Apple's
///     44pt guidance and misses are invisible to the person missing
///   - a pressed state, so a tap is acknowledged before whatever it triggers
///     has had time to appear
struct SDRowButtonStyle: ButtonStyle {
    var minHeight: CGFloat = 56

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(maxWidth: .infinity, minHeight: minHeight, alignment: .leading)
            .padding(.horizontal, SD.Space.sm)
            .background(
                RoundedRectangle(cornerRadius: SD.Radius.sm, style: .continuous)
                    .fill(configuration.isPressed
                          ? SD.Colors.accent.opacity(0.14)
                          : Color.clear)
            )
            // The whole rectangle, including empty space, is the target.
            .contentShape(Rectangle())
            // No fade on the press itself: the highlight has to appear at once
            // to read as acknowledgement. The release can settle.
            .animation(configuration.isPressed ? nil : SD.Motion.quick,
                       value: configuration.isPressed)
    }
}

/// One Wi-Fi network, as a row someone can actually hit.
struct NetworkRow: View {
    let network: VisibleNetwork
    let isSelected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: SD.Space.sm) {
                Image(systemName: symbol)
                    .font(.system(size: 18))
                    .foregroundStyle(isSelected ? SD.Colors.accent : SD.Colors.textSecondary)
                    .frame(width: 24)

                VStack(alignment: .leading, spacing: 1) {
                    Text(network.ssid)
                        .font(SD.Typography.body)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                    Text(network.secured ? "\(network.strength) signal" : "\(network.strength) signal · open")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }

                Spacer(minLength: SD.Space.xs)

                if network.secured {
                    Image(systemName: "lock.fill")
                        .font(.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
                Image(systemName: isSelected ? "checkmark.circle.fill" : "chevron.right")
                    .font(.system(size: isSelected ? 18 : 13, weight: .semibold))
                    .foregroundStyle(isSelected ? SD.Colors.accent : SD.Colors.textTertiary)
            }
        }
        .buttonStyle(SDRowButtonStyle())
        .accessibilityLabel(
            "\(network.ssid), \(network.strength) signal, \(network.secured ? "password protected" : "open network")"
        )
    }

    /// Signal shown as bars, the way every other Wi-Fi picker does it.
    private var symbol: String {
        switch network.signal {
        case 60...: "wifi"
        case 30..<60: "wifi.medium"
        default: "wifi.low"
        }
    }
}

/// One discovered SensorDyme.
struct DeviceRow: View {
    let device: DiscoveredDevice
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: SD.Space.sm) {
                Image(systemName: "sensor.tag.radiowaves.forward")
                    .font(.system(size: 18))
                    .foregroundStyle(SD.Colors.accent)
                    .frame(width: 24)
                VStack(alignment: .leading, spacing: 1) {
                    Text(device.name)
                        .font(SD.Typography.body)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Text(device.proximity)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
                Spacer(minLength: SD.Space.xs)
                Image(systemName: "chevron.right")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(SD.Colors.textTertiary)
            }
        }
        .buttonStyle(SDRowButtonStyle())
        .accessibilityLabel("\(device.name), \(device.proximity)")
    }
}
