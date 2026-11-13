import SwiftUI

// MARK: - Activity
//
// Events as a chronological flow: a hairline spine, a small marker whose shape
// AND colour say how serious it was, the time in monospaced digits, and the
// device's own words. Critical alerts stand out by weight and marker, not by
// turning the list red or moving.

/// How an event is marked: the severity the notification layer uses
/// (`EventItem.severityStatus`), as a word, a symbol-free shape and a colour.
struct EventSeverityStyle {
    let word: String
    let color: Color
    /// Filled for anything that alerted; a ring for a plain event.
    let filled: Bool
    let size: CGFloat
    let emphasised: Bool

    init(_ event: EventItem) {
        // An experimental observation (M8.4): a quiet ring with its own word,
        // never an alert's colour or weight.
        if event.isExperimental {
            word = "Experimental"; color = SD.Colors.textTertiary; filled = false; size = 8; emphasised = false
            return
        }
        switch event.severityStatus {
        case .alert:
            word = "Urgent"; color = SD.Colors.danger; filled = true; size = 10; emphasised = true
        case .warning:
            word = "Alert"; color = SD.Colors.warning; filled = true; size = 8; emphasised = false
        default:
            word = "Event"; color = SD.Colors.textTertiary; filled = false; size = 8; emphasised = false
        }
    }
}

/// One event on the timeline. Tapping it opens the details in place: the
/// device's signals, what produced it, and the way to "Why did this fire".
struct TimelineRow: View {
    let event: EventItem
    var isLast = false
    var expanded = false
    /// Relative time ("6 min ago") for a short recent list; clock time when the
    /// list is grouped by day.
    var relativeTime = false
    /// `event.row` in Activity; the device screen's short list uses its own,
    /// so a query for the history's rows never finds the screen underneath.
    var identifier = "event.row"
    var onToggle: () -> Void
    var onExplain: () -> Void

    private var style: EventSeverityStyle { EventSeverityStyle(event) }
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        HStack(alignment: .top, spacing: SD.Space.sm) {
            spine
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                Button(action: onToggle) { header }
                    .buttonStyle(SDUndimmedButtonStyle())
                    // The marker's colour and shape are the severity; say it.
                    .accessibilityLabel(EventAccessibility.label(
                        title: event.title, severity: event.isExperimental ? "experimental" : event.severity,
                        signals: event.signals,
                        when: event.date.map { Self.spokenAge.localizedString(for: $0, relativeTo: .now) }))
                    .accessibilityValue(event.isSimulated ? "Simulated" : "")
                    .accessibilityHint(expanded ? "Hides the details" : "Shows the details")
                    .accessibilityAddTraits(expanded ? .isSelected : [])
                    .accessibilityIdentifier(identifier)
                if expanded { details }
            }
            .padding(.bottom, isLast ? 0 : SD.Space.md)
        }
    }

    private var spine: some View {
        VStack(spacing: 0) {
            ZStack {
                if style.filled {
                    Circle().fill(style.color).frame(width: style.size, height: style.size)
                } else {
                    Circle().strokeBorder(style.color, lineWidth: 1.5).frame(width: style.size, height: style.size)
                }
                if style.emphasised {
                    Circle().strokeBorder(style.color.opacity(0.35), lineWidth: 1).frame(width: 18, height: 18)
                }
            }
            .frame(width: 18, height: 22)
            if !isLast {
                Rectangle().fill(SD.Colors.border).frame(width: 1).frame(maxHeight: .infinity)
            }
        }
        .accessibilityHidden(true)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            // The time beside the title, or under it at accessibility sizes,
            // where beside it broke the title mid-word ("Drows-iness", M8.4).
            if typeSize.isAccessibilitySize {
                VStack(alignment: .leading, spacing: 2) {
                    titleText
                    timeText
                }
            } else {
                HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                    titleText
                    Spacer(minLength: SD.Space.xs)
                    timeText
                }
            }
            // The badges on one line (stacked when they don't fit), and the
            // device's own words on their own line under them (M8.4). In one
            // row, a wrapping signals line squeezed "Urgent" until the two
            // overlapped.
            if hasBadges {
                ViewThatFits(in: .horizontal) {
                    HStack(spacing: SD.Space.xs) { badges }
                    VStack(alignment: .leading, spacing: 2) { badges }
                }
            }
            if let signals = event.signals, !signals.isEmpty {
                // Wraps in full: two lines clipped the device's reason (audit).
                Text(signals)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .multilineTextAlignment(.leading)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 44, alignment: .leading)
        .contentShape(Rectangle())
    }

    private var titleText: some View {
        Text(event.title)
            .font(style.emphasised ? SD.Typography.headline : SD.Typography.body)
            .foregroundStyle(SD.Colors.textPrimary)
            .multilineTextAlignment(.leading)
            .fixedSize(horizontal: false, vertical: true)
    }

    @ViewBuilder private var timeText: some View {
        if let date = event.date {
            Group {
                if relativeTime {
                    Text(date, format: .relative(presentation: .numeric, unitsStyle: .abbreviated))
                } else {
                    Text(date, format: .dateTime.hour().minute())
                }
            }
            .font(SD.Typography.caption.monospacedDigit())
            .foregroundStyle(SD.Colors.textTertiary)
            .lineLimit(1)
            .fixedSize()
        }
    }

    private var showsWord: Bool {
        event.isExperimental || style.emphasised || event.severityStatus == .warning
    }
    private var hasBadges: Bool { showsWord || event.isConfigured || event.isSimulated }

    @ViewBuilder private var badges: some View {
        if showsWord {
            Text(style.word)
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(event.isExperimental ? SD.Colors.textSecondary : style.color)
                .fixedSize()
                .accessibilityIdentifier(event.isExperimental ? "event.experimental" : "event.severity")
        }
        if event.isConfigured {
            Text("Your rule")
                .font(SD.Typography.caption.weight(.medium))
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize()
        }
        // Real vs rehearsal, on the row itself (M8.3): a simulated or
        // demo event must never pass for a real detection.
        if event.isSimulated {
            Text("Simulated")
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(SD.Colors.textTertiary)
                .fixedSize()
                .accessibilityIdentifier("event.simulated")
        }
    }

    /// Only what the backend recorded: confidence when the event carries one,
    /// the model that produced it when known. Nothing estimated by the app.
    /// The device-alert line is what the DEVICE reported about its own alert
    /// output (M8.3), never inferred from the event existing.
    private var details: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            if let date = event.date {
                fact(event.timeIsArrival ? "Received" : "When", date.formatted(date: .abbreviated, time: .standard))
            }
            if let delay = event.deliveryDelay {
                fact("Reached the app", "\(Self.duration(delay)) later")
            }
            if event.deviceClockError != nil {
                fact("Device clock", "was wrong; time shown is from the server")
            }
            if event.isExperimental {
                // M8.4: recorded to learn from, never an alert.
                fact("Kind", "Experimental observation (not an alert)")
                    .accessibilityIdentifier("event.alertEvidence")
            } else if event.severityStatus != .online || event.alert != nil {
                if event.alert?.decision != nil {
                    fact("Alert requested", EventEvidence(event).alertRequested)
                }
                fact("Device alert", event.alertEvidence.short)
                    .accessibilityIdentifier("event.alertEvidence")
            }
            if let confidence = event.confidence {
                // The device's confidence that the condition held -- for a
                // measured condition not a detector's score (jupiter/explain.py).
                fact("Recorded confidence", String(format: "%.2f", confidence))
            }
            if let model = event.model?.summary {
                fact("Produced by", model)
            }
            Button(action: onExplain) {
                Label("Why this happened", systemImage: "text.magnifyingglass")
                    .font(SD.Typography.label)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .frame(minHeight: 44)
                    .contentShape(Rectangle())
            }
            .buttonStyle(SDPressStyle())
            .accessibilityIdentifier("event.explain")
        }
        .padding(.top, SD.Space.xxs)
        .transition(.opacity)
    }

    private func fact(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
            Spacer(minLength: SD.Space.xs)
            Text(value)
                .font(SD.Typography.caption.monospacedDigit())
                .foregroundStyle(SD.Colors.textSecondary)
                .multilineTextAlignment(.trailing)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .combine)
    }

    /// "3 min", "2 h", "4 d": how late an event arrived.
    static func duration(_ seconds: TimeInterval) -> String {
        let formatter = DateComponentsFormatter()
        formatter.unitsStyle = .abbreviated
        formatter.maximumUnitCount = 1
        return formatter.string(from: seconds) ?? "\(Int(seconds)) s"
    }

    static let spokenAge: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter
    }()
}

/// Events grouped under day headings ("Today", "Yesterday", "Mon 9 Nov"),
/// newest first, as one continuous spine per day.
struct ActivityTimeline: View {
    let events: [EventItem]
    var onExplain: (EventItem) -> Void

    @State private var expandedID: String?
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var days: [(key: Date, label: String, events: [EventItem])] {
        let calendar = Calendar.current
        var order: [Date] = []
        var grouped: [Date: [EventItem]] = [:]
        for event in events {
            let day = event.date.map { calendar.startOfDay(for: $0) } ?? .distantPast
            if grouped[day] == nil { order.append(day) }
            grouped[day, default: []].append(event)
        }
        return order.map { day in
            let label: String
            if day == .distantPast { label = "Undated" }
            else if calendar.isDateInToday(day) { label = "Today" }
            else if calendar.isDateInYesterday(day) { label = "Yesterday" }
            else { label = day.formatted(.dateTime.weekday(.abbreviated).day().month(.abbreviated)) }
            return (day, label, grouped[day] ?? [])
        }
    }

    var body: some View {
        LazyVStack(alignment: .leading, spacing: 0) {
            ForEach(days, id: \.key) { day in
                Section {
                    ForEach(Array(day.events.enumerated()), id: \.element.id) { index, event in
                        TimelineRow(event: event, isLast: index == day.events.count - 1,
                                    expanded: expandedID == event.id,
                                    onToggle: { toggle(event) },
                                    onExplain: { onExplain(event) })
                    }
                } header: {
                    // Side by side when they fit, the count under the day when not.
                    ViewThatFits(in: .horizontal) {
                        HStack(alignment: .firstTextBaseline) {
                            dayLabel(day.label)
                            Spacer()
                            daySummary(day.events)
                        }
                        VStack(alignment: .leading, spacing: 2) {
                            dayLabel(day.label)
                            daySummary(day.events, wraps: true)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, SD.Space.lg)
                    .padding(.bottom, SD.Space.sm)
                    .accessibilityElement(children: .combine)
                    .accessibilityAddTraits(.isHeader)
                }
            }
        }
        .animation(SD.Motion.resolved(SD.Motion.standard, reduceMotion: reduceMotion), value: expandedID)
    }

    private func dayLabel(_ label: String) -> some View {
        Text(label)
            .font(SD.Typography.headline)
            .foregroundStyle(SD.Colors.textPrimary)
            .fixedSize()
    }

    private func daySummary(_ events: [EventItem], wraps: Bool = false) -> some View {
        Text(summary(events))
            .font(SD.Typography.caption.monospacedDigit())
            .foregroundStyle(SD.Colors.textTertiary)
            .fixedSize(horizontal: !wraps, vertical: true)
    }

    private func toggle(_ event: EventItem) {
        expandedID = expandedID == event.id ? nil : event.id
    }

    /// Counted, not scored: how many events, and how many alerted.
    private func summary(_ events: [EventItem]) -> String {
        let alerts = events.filter { !$0.isExperimental && ($0.isCritical || $0.isConfigured) }.count
        let total = "\(events.count) \(events.count == 1 ? "event" : "events")"
        return alerts == 0 ? total : "\(total) · \(alerts) \(alerts == 1 ? "alert" : "alerts")"
    }
}
