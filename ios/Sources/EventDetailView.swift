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
                VStack(alignment: .leading, spacing: SD.Space.lg) {
                    header
                    deviceAlertCard
                    evidenceSection
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
                            title: "Couldn't explain this event",
                            message: errorMessage,
                            retry: { await load() }
                        )
                    }
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.xl)
            }
            .orynScreen()
            // "Alert" only for what alerted, or was asked for (M8.4): a plain
            // detection or an experimental observation is an event.
            .navigationTitle(event.isCritical || event.isConfigured ? "Alert" : "Event")
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
                    status: event.severityStatus == .online ? .unknown : event.severityStatus,
                    text: event.isExperimental ? "Experimental"
                        : (event.isCritical ? "Critical" : (event.severity == "soft" ? "Warning" : "Info"))
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
            if event.isSimulated {
                Label("Simulated event, not from a real camera", systemImage: "testtube.2")
                    .font(SD.Typography.caption.weight(.semibold))
                    .foregroundStyle(SD.Colors.textSecondary)
                    .accessibilityIdentifier("eventDetail.simulated")
            }
        }
        .padding(.top, SD.Space.xs)
    }

    // MARK: On the device (M8.3)

    /// Detection and warning are different facts. This says what the DEVICE
    /// reported about its own alert output -- "Alert sounded on the device"
    /// only when it reported a real output played -- and how the time shown
    /// was established. Never inferred from the event having been stored.
    private var deviceAlertCard: some View {
        let evidence = event.alertEvidence
        return VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDSectionHeader(title: "On the device")
            if event.isExperimental {
                // M8.4: an experimental signal is recorded to evaluate it, never
                // to warn anyone, and it is no evidence of what anyone was doing.
                Label("Experimental observation (not an alert)", systemImage: "flask")
                    .font(SD.Typography.body.weight(.semibold))
                    .foregroundStyle(SD.Colors.textPrimary)
                    .accessibilityIdentifier("eventDetail.alertEvidence")
                Text("An unvalidated signal the device records so it can be evaluated. It never sounds or "
                     + "notifies, and it isn't evidence of what anyone in view was doing.")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                Label(evidence.headline,
                      systemImage: evidence.confirmedSound ? "speaker.wave.2.fill" : "speaker.slash")
                    .font(SD.Typography.body.weight(.semibold))
                    .foregroundStyle(SD.Colors.textPrimary)
                    .accessibilityIdentifier("eventDetail.alertEvidence")
                Text(evidence.detail)
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let note = timeNote {
                Text(note)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("eventDetail.timeNote")
            }
        }
        .accessibilityElement(children: .combine)
    }

    // MARK: The record (M8.4)

    /// What this event's record supports, from its own fields (EventEvidence):
    /// what the device measured, what it concluded, the detector's score when
    /// one was recorded, whether an alert was asked for, and what is missing.
    private var evidenceSection: some View {
        let evidence = EventEvidence(event)
        return VStack(alignment: .leading, spacing: SD.Space.xs) {
            SDSectionHeader(title: "What the record shows")
            recordRow("Observed", evidence.observed ?? "Not recorded")
            recordRow("Device's conclusion", evidence.inferred)
            recordRow("Confidence the device recorded", evidence.confidence ?? "Not recorded")
            recordRow("Alert requested", evidence.alertRequested)
            if !evidence.gaps.isEmpty {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(evidence.gaps, id: \.self) { gap in
                        Label(gap, systemImage: "info.circle")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .padding(.top, SD.Space.xxs)
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("eventDetail.gaps")
            }
        }
        .accessibilityIdentifier("eventDetail.record")
    }

    /// A label over its value: at every text size the value has the full width.
    private func recordRow(_ label: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(label)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
            Text(value)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 2)
        .accessibilityElement(children: .combine)
    }

    /// How the time shown was established, when that is not simply "the device said so".
    private var timeNote: String? {
        var notes: [String] = []
        if event.timeIsArrival {
            notes.append("The device's clock couldn't be trusted, so the time shown is when the server received this.")
        } else if let error = event.deviceClockError {
            let off = TimelineRow.duration(abs(error))
            notes.append("The device's clock was \(off) \(error > 0 ? "fast" : "slow"); the time shown is from the server.")
        }
        if let delay = event.deliveryDelay, let received = event.receivedDate {
            notes.append("Reached the server \(TimelineRow.duration(delay)) after it happened, at "
                         + received.formatted(date: .omitted, time: .shortened) + ".")
        }
        return notes.isEmpty ? nil : notes.joined(separator: " ")
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
        // Open space, not a box: the evidence reads as a list of facts.
        Group {
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
        Group {
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
                            .fill(SD.Colors.borderStrong)
                            .frame(height: 1)
                            .offset(y: 15.5)

                        ForEach(todays) { event in
                            if let fraction = position(event) {
                                Button { onSelect(event) } label: {
                                    Circle()
                                        .fill(event.severityStatus == .online
                                              ? SD.Colors.textTertiary : event.severityStatus.tint)
                                        .frame(width: event.isCritical ? 12 : 8,
                                               height: event.isCritical ? 12 : 8)
                                        .overlay(
                                            Circle().strokeBorder(SD.Colors.background, lineWidth: 2)
                                        )
                                        // The dot as drawn; a 44 pt target around it
                                        // (audit: hit area too small).
                                        .frame(width: 44, height: 44)
                                        .contentShape(Rectangle())
                                }
                                .buttonStyle(.plain)
                                // Centred on its time, clamped so a midnight event
                                // isn't half off-screen; on the line at y = 16.
                                .offset(
                                    x: min(max(fraction * geometry.size.width, 7), geometry.size.width - 7) - 22,
                                    y: 16 - 22
                                )
                                .accessibilityLabel("\(event.title) at \(event.date.map { $0.formatted(date: .omitted, time: .shortened) } ?? "")")
                            }
                        }
                    }
                }
                .frame(height: 32)

                // A visual scale: each dot already says its time to VoiceOver.
                // A text style, not a fixed 10 pt (audit: Dynamic Type unsupported),
                // and indexed: "12a" twice was a duplicate ForEach id.
                HStack {
                    ForEach(Array(["12a", "6a", "12p", "6p", "12a"].enumerated()), id: \.offset) { index, label in
                        Text(label)
                            .font(.caption2)
                            .foregroundStyle(SD.Colors.textTertiary)
                        if index < 4 { Spacer(minLength: 0) }
                    }
                }
                .accessibilityHidden(true)
            }
        }
    }
}
