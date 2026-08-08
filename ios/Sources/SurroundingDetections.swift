import SwiftUI

/// What Jupiter saw around an alert, including everything that never notified.
///
/// The phone interrupts once. This is the rest of the story: the warnings and
/// lower-severity detections either side of the alert, which were recorded all
/// along and had nowhere to be seen. Answering "why did it decide that" needs
/// the run-up, not the single frame that crossed the threshold.
///
/// Nothing here is generated. Every row is an event the device actually
/// reported; if there is no surrounding context, it says so rather than
/// inventing a plausible-looking sequence.
struct SurroundingDetections: View {
    let alert: EventItem
    let all: [EventItem]

    @State private var expanded = false

    /// How far either side of the alert counts as "around" it.
    ///
    /// Wide enough to catch the build-up — a fall has a run-up of a few seconds
    /// and an aftermath of rather longer — without sweeping in an unrelated
    /// incident from earlier in the evening.
    private let windowBefore: TimeInterval = 30
    private let windowAfter: TimeInterval = 60

    var body: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                Button {
                    withAnimation(SD.Motion.standard) { expanded.toggle() }
                } label: {
                    HStack {
                        VStack(alignment: .leading, spacing: 1) {
                            Text("What else was detected")
                                .font(SD.Typography.label)
                                .foregroundStyle(SD.Colors.textPrimary)
                            Text(summary)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                        }
                        Spacer()
                        Image(systemName: expanded ? "chevron.up" : "chevron.down")
                            .font(.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                    }
                }
                .buttonStyle(SDRowButtonStyle(minHeight: 44))

                if expanded {
                    if nearby.isEmpty {
                        Text("Nothing else was recorded around this time.")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                    } else {
                        ForEach(nearby) { item in
                            row(for: item)
                        }
                        Text("Only the highlighted alert sent a notification. "
                             + "The rest were recorded but not interrupted for.")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.top, SD.Space.xxs)
                    }
                }
            }
        }
    }

    private func row(for item: EventItem) -> some View {
        let isTheAlert = item.eventID == alert.eventID
        return HStack(alignment: .top, spacing: SD.Space.sm) {
            Text(item.date.map(Self.clock.string(from:)) ?? "—")
                .font(SD.Typography.caption.monospacedDigit())
                .foregroundStyle(SD.Colors.textTertiary)
                .frame(width: 66, alignment: .leading)

            Circle()
                .fill(colour(for: item))
                .frame(width: 7, height: 7)
                .padding(.top, 5)

            Text(item.title)
                .font(isTheAlert ? SD.Typography.label : SD.Typography.caption)
                .foregroundStyle(isTheAlert ? SD.Colors.textPrimary : SD.Colors.textSecondary)

            Spacer(minLength: 0)
        }
        .padding(.vertical, 1)
    }

    private func colour(for item: EventItem) -> Color {
        switch item.severityStatus {
        case .alert: SD.Colors.danger
        case .warning: SD.Colors.warning
        default: SD.Colors.textTertiary
        }
    }

    /// Events either side of the alert, oldest first.
    private var nearby: [EventItem] {
        guard let anchor = alert.date else { return [] }
        return all
            .filter { item in
                guard let date = item.date else { return false }
                let delta = date.timeIntervalSince(anchor)
                return delta >= -windowBefore && delta <= windowAfter
            }
            .sorted { ($0.date ?? .distantPast) < ($1.date ?? .distantPast) }
    }

    private var summary: String {
        let count = nearby.count
        guard count > 1 else { return "Tap to see the timeline" }
        let others = count - 1
        return others == 1
            ? "1 other detection around this time"
            : "\(others) other detections around this time"
    }

    private static let clock: DateFormatter = {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss"
        return f
    }()
}
