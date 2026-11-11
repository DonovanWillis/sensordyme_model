import SwiftUI

/// Why this alert fired — the answer, not a log line.
///
/// An alert someone cannot interrogate is one they will eventually mute. The
/// content comes from the server, assembled from the stored event and the rule
/// that fired, so nothing here is written by the app or invented by a model.
///
/// Critical events are explained in the device's own plain words rather than a
/// model's rephrasing. That is a deliberate limit set on the backend: given the
/// facts of a real fall, a small model merged "rapid drop" and "detector
/// confidence" into a sentence that was simply untrue. A blunt correct
/// explanation beats a fluent wrong one when someone is on the floor.
struct EventDetailView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    let device: Device
    let event: EventItem
    /// Everything else the device reported, so the alert can be shown in
    /// context. Passed in rather than re-fetched: the caller already has it.
    var allEvents: [EventItem] = []

    @State private var explanation: AlertExplanation?
    @State private var loading = true
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.md) {
                    header
                    // The detections around the alert, including the ones that
                    // never notified. One interruption, the whole story.
                    if !allEvents.isEmpty {
                        SurroundingDetections(alert: event, all: allEvents)
                    }
                    if loading {
                        SDLoadingState(message: "Working out what happened…")
                    } else if let explanation {
                        answerCard(explanation)
                        factsCard(explanation)
                        if !explanation.rule.isEmpty {
                            ruleCard(explanation)
                        }
                    } else if let errorMessage {
                        SDErrorState(
                            title: "Couldn't explain this alert",
                            message: errorMessage,
                            retry: { await load() }
                        )
                    }
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.xl)
            }
            .orynScreen()
            .navigationTitle("Alert")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
            .task { await load() }
        }
        .tint(SD.Colors.accent)
    }

    // MARK: Header

    private var header: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            HStack(spacing: SD.Space.xs) {
                SDStatusPill(
                    status: event.severityStatus,
                    text: event.isCritical ? "Critical" : (event.severity == "soft" ? "Warning" : "Info")
                )
                Spacer()
                if let date = event.date {
                    Text(date, style: .time)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                }
            }
            Text(explanation?.headline ?? event.title)
                .sdScaledFont(30, relativeTo: .title, weight: .semibold)
                .foregroundStyle(SD.Colors.textPrimary)
            Text(device.name)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
        }
        .padding(.top, SD.Space.xs)
    }

    // MARK: The answer

    private func answerCard(_ explanation: AlertExplanation) -> some View {
        SDCard(raised: true) {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                SDSectionHeader(title: "Why you got this")
                Text(explanation.narrative)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)

                // Said out loud rather than hidden: the user should know whether
                // a sentence was written by the device or smoothed by a model.
                if explanation.wasRephrased {
                    Label("Summarised by Jupiter from the facts below",
                          systemImage: "sparkles")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                } else {
                    Label("Reported directly by the device",
                          systemImage: "checkmark.seal")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
            }
        }
    }

    // MARK: Evidence

    private func factsCard(_ explanation: AlertExplanation) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SDSectionHeader(title: "What the camera saw")
                if explanation.facts.isEmpty {
                    Text("The device recorded this event without additional detail.")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                ForEach(explanation.facts) { fact in
                    HStack(alignment: .top, spacing: SD.Space.sm) {
                        Image(systemName: "circle.fill")
                            .font(.system(size: 5))
                            .foregroundStyle(SD.Colors.textTertiary)
                            .padding(.top, 7)
                        Text(fact.label)
                            .font(SD.Typography.body)
                            .foregroundStyle(SD.Colors.textPrimary)
                        Spacer(minLength: SD.Space.xs)
                        if !fact.detail.isEmpty {
                            Text(fact.detail)
                                .font(SD.Typography.caption.monospacedDigit())
                                .foregroundStyle(SD.Colors.textSecondary)
                        }
                    }
                }
            }
        }
    }

    private func ruleCard(_ explanation: AlertExplanation) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                SDSectionHeader(title: "The rule that fired")
                Text(explanation.rule)
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func load() async {
        guard let eventID = event.eventID else {
            // An older backend omits the row id, so this event cannot be
            // addressed. Say that rather than spin forever.
            errorMessage = "This device's backend doesn't support alert explanations yet. Update it to ask why an alert fired."
            loading = false
            return
        }
        do {
            explanation = try await state.backend.explain(deviceID: device.id, eventID: eventID)
            errorMessage = nil
        } catch let error as APIError where error.isAuthFailure {
            state.expireSession(error.message)
        } catch {
            errorMessage = error.localizedDescription
        }
        loading = false
    }
}

// MARK: - Timeline

/// The day at a glance: when things happened and how serious they were.
///
/// Positioned by time of day rather than listed evenly, because the shape of a
/// day — a quiet morning, three events in ten minutes at 2pm — is information a
/// list throws away.
struct EventTimeline: View {
    let events: [EventItem]
    var onSelect: (EventItem) -> Void

    /// Only today's events. A timeline spanning days compresses to meaningless.
    private var todays: [EventItem] {
        let calendar = Calendar.current
        return events.filter { event in
            guard let date = event.date else { return false }
            return calendar.isDateInToday(date)
        }
    }

    private func position(_ event: EventItem) -> Double? {
        guard let date = event.date else { return nil }
        let calendar = Calendar.current
        let start = calendar.startOfDay(for: date)
        return date.timeIntervalSince(start) / 86_400  // 0…1 across the day
    }

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDSectionHeader(title: "Today")

            if todays.isEmpty {
                Text("Nothing today.")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .padding(.vertical, SD.Space.xs)
            } else {
                GeometryReader { geometry in
                    ZStack(alignment: .topLeading) {
                        Capsule()
                            .fill(SD.Colors.border)
                            .frame(height: 3)
                            .offset(y: 14)

                        ForEach(todays) { event in
                            if let fraction = position(event) {
                                Button { onSelect(event) } label: {
                                    Circle()
                                        .fill(event.severityStatus.tint)
                                        .frame(width: event.isCritical ? 14 : 10,
                                               height: event.isCritical ? 14 : 10)
                                        .overlay(
                                            Circle().strokeBorder(SD.Colors.background, lineWidth: 2)
                                        )
                                }
                                .buttonStyle(.plain)
                                // Clamped so a midnight event isn't half off-screen.
                                .offset(
                                    x: min(max(fraction * geometry.size.width - 7, 0),
                                           geometry.size.width - 14),
                                    y: event.isCritical ? 9 : 11
                                )
                                .accessibilityLabel("\(event.title) at \(event.date.map { $0.formatted(date: .omitted, time: .shortened) } ?? "")")
                            }
                        }
                    }
                }
                .frame(height: 32)

                HStack {
                    ForEach(["12a", "6a", "12p", "6p", "12a"], id: \.self) { label in
                        Text(label)
                            .font(.system(size: 10))
                            .foregroundStyle(SD.Colors.textTertiary)
                        if label != "12a" || true { Spacer(minLength: 0) }
                    }
                }
            }
        }
    }
}
