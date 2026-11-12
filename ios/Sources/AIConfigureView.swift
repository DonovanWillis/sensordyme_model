import SwiftUI

/// Ask Jupiter: describe what to watch for in plain language; review what
/// Jupiter understood; adjust it; activate it.
///
/// Formerly "Configure with AI". The flow is unchanged — `POST /v1/ai/configure`
/// returns a draft, the person edits and reviews it, and only Activate
/// (`POST /v1/devices/{id}/preset`) changes the device.
///
/// The review step is the point. A user is being asked to trust a camera in
/// their home with a rule they typed one sentence for — showing them the
/// interpretation, in their own words, with the knobs exposed, is what makes
/// that reasonable. It also means a wrong interpretation is caught by the person
/// who knows what they meant, rather than discovered later as a missed alert.
struct AIConfigureView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    let device: Device
    /// A request typed into the device screen's Ask Jupiter bar. When present it
    /// is filled in and sent as the screen opens, so the bar feels like asking,
    /// not like navigating to a form.
    var initialPrompt: String = ""
    let onApplied: () async -> Void

    @State private var prompt = ""
    @State private var didSendInitial = false
    @State private var thinking = false
    @State private var proposal: AIProposal?
    @State private var applied = false
    /// What the backend confirmed on activation (ActivationSummary). Nil until
    /// an activation succeeds.
    @State private var appliedSummary: ActivationSummary?
    @State private var activating = false
    @State private var confirmActivate = false
    /// What the DEVICE reported after activation (ActivationFollowUp). The
    /// server accepting a change is "sent", not "active".
    @State private var followUp: ActivationFollowUp?
    @State private var errorMessage: String?
    /// Not an error: why the screen is showing a fresh proposal (the one being
    /// reviewed was replaced on the server before it could be activated).
    @State private var notice: String?
    @StateObject private var thinkingProgress = ProgressTracker()

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.lg) {
                    promptCard
                    if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.circle.fill")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.danger)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("jupiter.error")
                    }
                    if let notice {
                        Label(notice, systemImage: "arrow.triangle.2.circlepath")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if let proposal {
                        understandingCard(proposal)
                        ForEach(Array(proposal.rules.enumerated()), id: \.offset) { index, _ in
                            ruleEditor(index)
                        }
                        if !proposal.unsupported.isEmpty {
                            limitationsCard(proposal.unsupported)
                        }
                        if !proposal.isEmpty && !applied {
                            previewCard(proposal)
                        }
                        activateSection(proposal)
                    }
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.xl)
            }
            .orynScreen()
            .scrollDismissesKeyboard(.interactively)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    HStack(spacing: 8) {
                        OrynMark(height: 14)
                        Text("Ask Jupiter")
                            .font(.system(.callout).weight(.semibold))  // 16 pt by default, as designed
                            .foregroundStyle(SD.Colors.textPrimary)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityAddTraits(.isHeader)
                    // The navigation bar caps its text size; at large sizes a long
                    // press shows the title in the Large Content Viewer instead.
                    .accessibilityShowsLargeContentViewer {
                        Label("Ask Jupiter", systemImage: "sparkles")
                    }
                    .accessibilityIdentifier("jupiter.title")
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                        .accessibilityIdentifier("jupiter.done")
                }
            }
        }
        .tint(SD.Colors.accent)
        .presentationBackground(SD.Colors.background)
        // Success only when the DEVICE runs it; an error when it refuses.
        .sensoryFeedback(trigger: followUpKind) { _, new in
            switch new {
            case "confirmed": .success
            case "refused": .error
            default: nil
            }
        }
        .onAppear {
            guard !didSendInitial else { return }
            didSendInitial = true
            if let request = JupiterPrompt.normalized(initialPrompt) {
                prompt = request
                configure()
            }
        }
    }

    // MARK: Prompt

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                HStack(spacing: SD.Space.xs) {
                    OrynMark(height: 13, color: SD.Colors.signal)
                    SDEyebrow("Jupiter · \(device.name)")
                }
                Text(proposal == nil && !thinking
                     ? "What should \(device.name) watch for?"
                     : "Your request")
                    .sdScaledFont(28, relativeTo: .title, weight: .semibold)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.top, SD.Space.sm)

            AskJupiterBar(
                text: $prompt,
                busy: thinking,
                expands: true,
                autoFocus: JupiterPrompt.normalized(initialPrompt) == nil
            ) {
                configure()
            }

            if thinking {
                // The request is in flight, and that is all this says: no
                // streamed text, no typing performance.
                HStack(spacing: SD.Space.xs) {
                    ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
                    Text("Jupiter is reading your request…")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                    SDProgress(phase: thinkingProgress.phase == .done ? .idle : thinkingProgress.phase)
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("jupiter.thinking")
                .transition(.opacity)
            } else if proposal == nil {
                suggestionList
            }
        }
        .animation(SD.Motion.standard, value: thinking)
    }

    /// What Jupiter can be asked, from the policy grammar (JupiterSuggestions),
    /// and one request it refuses, labelled as such.
    private var suggestionList: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDEyebrow("Jupiter can watch for")
                .accessibilityAddTraits(.isHeader)
            VStack(spacing: 0) {
                ForEach(JupiterSuggestions.supported) { suggestion in
                    suggestionRow(suggestion, refused: false)
                    OrynRowDivider(inset: 0)
                }
                suggestionRow(JupiterSuggestions.refused, refused: true)
            }
            Text("Jupiter turns your words into rules for this device. Nothing changes until you review and activate them. Requests about who someone is, what they feel or intend, or sounds are refused, with the reason.")
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, SD.Space.xs)
        }
    }

    private func suggestionRow(_ suggestion: JupiterSuggestions.Suggestion, refused: Bool) -> some View {
        Button { prompt = suggestion.text } label: {
            HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
                Image(systemName: refused ? "nosign" : "arrow.turn.down.right")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(refused ? SD.Colors.warning : SD.Colors.textTertiary)
                    .frame(width: 14)
                VStack(alignment: .leading, spacing: 2) {
                    Text(suggestion.text)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(suggestion.kind)
                        .font(SD.Typography.caption)
                        .foregroundStyle(refused ? SD.Colors.warning : SD.Colors.textTertiary)
                }
                Spacer(minLength: 0)
            }
            .padding(.vertical, SD.Space.sm)
            .frame(minHeight: 44)
            .contentShape(Rectangle())
        }
        .buttonStyle(SDPressStyle(scale: 0.99))
        .accessibilityElement(children: .combine)
        .accessibilityHint("Fills in the request")
    }

    // MARK: What was understood

    private func understandingCard(_ proposal: AIProposal) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                HStack {
                    HStack(spacing: 6) {
                        OrynMark(height: 11, color: SD.Colors.signal)
                        SDEyebrow("Jupiter understood")
                    }
                    Spacer()
                    switch followUp {
                    case .confirmed?: SDStatusPill(status: .online, text: "Active")
                    case .refused?: SDStatusPill(status: .alert, text: "Refused")
                    case .waiting?: SDStatusPill(status: .connecting, text: "Sent")
                    case nil: EmptyView()
                    }
                }
                Text(proposal.explanation)
                    .font(SD.Typography.readout.weight(.regular))
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("jupiter.explanation")

                if !proposal.isCustom {
                    Label("Also switches to \(Preset.title(for: proposal.preset))",
                          systemImage: "square.grid.2x2")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                if proposal.source == "unavailable" {
                    Label("The language service isn't configured, so nothing was changed.",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.warning)
                }
                if let confidence = proposal.jupiter?.interpretation_confidence, confidence.worthFlagging {
                    interpretationConfidenceNotice(confidence)
                }
            }
        }
    }

    /// Plain language for everyone; the actual reasons behind a small
    /// disclosure for anyone who wants them. `worthFlagging` already filtered
    /// out the routine case (part of the request being unsupported) — the
    /// `limitationsCard` below says that on its own, and repeating it here
    /// would be the same fact twice. What's left when this shows is something
    /// about the MODEL'S OWN OUTPUT that needed correcting, which nothing else
    /// on this screen surfaces.
    private func interpretationConfidenceNotice(_ confidence: AIInterpretationConfidence) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: SD.Space.xxs) {
                ForEach(confidence.basis, id: \.self) { reason in
                    Text("• \(reason)")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .padding(.top, SD.Space.xxs)
        } label: {
            Label("Double-check this before turning it on", systemImage: "exclamationmark.circle")
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.warning)
        }
    }

    // MARK: Editable rule

    @ViewBuilder
    private func ruleEditor(_ index: Int) -> some View {
        if let binding = ruleBinding(index) {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    Text(binding.wrappedValue.headline)
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Text(binding.wrappedValue.description)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)

                    if let caveat = binding.wrappedValue.caveat {
                        Label(caveat, systemImage: "info.circle")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    if !binding.wrappedValue.zone.isEmpty {
                        Label("Draw the \"\(binding.wrappedValue.zone)\" area on the device to activate this",
                              systemImage: "pencil.and.outline")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("jupiter.zoneRequirement")
                    }

                    Divider().overlay(SD.Colors.border)

                    // Sensitivity is confidence + persistence together. Exposing
                    // two numeric thresholds would be honest and useless; nobody
                    // tunes a home camera by picking a float.
                    settingRow("Sensitivity") {
                        Picker("", selection: sensitivityBinding(binding)) {
                            Text("Low").tag(Sensitivity.low)
                            Text("Medium").tag(Sensitivity.medium)
                            Text("High").tag(Sensitivity.high)
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 220)
                    }

                    settingRow("Alert me") {
                        Picker("", selection: binding.notify) {
                            Text("Push").tag("push")
                            Text("Silent").tag("silent")
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 160)
                    }

                    // Full schema range on a log scale (DurationScale). The linear
                    // 1–60 s / 30–1800 s sliders clamped a valid rule — a 120 s
                    // sleep rule, a 0.5 s phone rule — the moment they were touched.
                    settingRow("Confirm for") {
                        Text(DurationText.short(binding.wrappedValue.persist_s))
                            .font(SD.Typography.caption.monospacedDigit())
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Slider(value: durationBinding(binding.persist_s, scale: .persistence), in: 0...1)

                    settingRow("Then wait") {
                        Text(DurationText.short(binding.wrappedValue.cooldown_s))
                            .font(SD.Typography.caption.monospacedDigit())
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Slider(value: durationBinding(binding.cooldown_s, scale: .cooldown), in: 0...1)
                    Text("How long the condition must hold before alerting, and how long before it can alert again.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
            }
        }
    }

    @ViewBuilder
    private func settingRow<Content: View>(_ label: String, @ViewBuilder content: () -> Content) -> some View {
        HStack {
            Text(label)
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textPrimary)
            Spacer()
            content()
        }
    }

    // MARK: Limitations

    private func limitationsCard(_ items: [AIUnsupported]) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SDSectionHeader(title: "Not yet possible")
                    .accessibilityIdentifier("jupiter.limitations")
                ForEach(items) { item in
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                            Text(item.request)
                                .font(SD.Typography.label)
                                .foregroundStyle(SD.Colors.textPrimary)
                            Spacer(minLength: 0)
                            Text(item.route)
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.warning)
                        }
                        Text(item.reason)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text("To enable: \(item.requires)")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                            .fixedSize(horizontal: false, vertical: true)
                        if let alternatives = item.alternatives, !alternatives.isEmpty {
                            Text("It can watch for: \(alternatives.joined(separator: ", "))")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.accent)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    if item.id != items.last?.id {
                        Divider().overlay(SD.Colors.border)
                    }
                }
            }
        }
    }

    // MARK: Activate

    @ViewBuilder
    private func activateSection(_ proposal: AIProposal) -> some View {
        if proposal.isEmpty {
            SDEmptyState(
                icon: "questionmark.circle",
                title: "Nothing to activate",
                message: "Jupiter couldn't turn that into something this device can watch for. Try describing it differently, or see what it would need above."
            )
        } else if applied {
            VStack(spacing: SD.Space.xxs) {
                // Said from what the device reported, not from the server
                // accepting the request (ActivationFollowUp).
                switch followUp {
                case .confirmed?:
                    Label(appliedSummary?.kind == .modeSwitched
                          ? "Switched to \(Preset.title(for: proposal.preset))"
                          : "Active on \(device.name)", systemImage: "checkmark.circle.fill")
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.success)
                case .refused?:
                    Label("Not running on \(device.name)", systemImage: "xmark.octagon.fill")
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.danger)
                case .waiting?, nil:
                    Label("Sent to \(device.name)", systemImage: "paperplane")
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                }
                if let followUp {
                    HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                        if case .waiting = followUp, waitingForDevice {
                            ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
                        }
                        Text(followUp.text)
                            .font(SD.Typography.caption)
                            .foregroundStyle(followUpColor(followUp))
                            .multilineTextAlignment(.center)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                if let appliedNote = appliedSummary?.note {
                    Text(appliedNote)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .multilineTextAlignment(.center)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            .frame(maxWidth: .infinity)
            .padding(SD.Space.md)
            // The preview becomes the rule's own status surface: what was sent,
            // then what the device reported.
            .background(
                RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
                    .fill(SD.Colors.surface)
            )
            .transition(.opacity)
            .accessibilityElement(children: .combine)
            .accessibilityIdentifier("jupiter.applied")
        } else {
            Button { activateTaps += 1; confirmActivate = true } label: {
                if activating {
                    ProgressView().tint(SD.Colors.textOnAccent)
                } else if !proposal.isCustom {
                    // A named mode is what this turns on; rules never run beside one.
                    Text("Switch to \(Preset.title(for: proposal.preset))")
                } else {
                    Text("Activate \(proposal.rules.count == 1 ? "rule" : "\(proposal.rules.count) rules")")
                }
            }
            .buttonStyle(SDButtonStyle(kind: .primary))
            .disabled(activating)
            .sensoryFeedback(.impact(weight: .medium), trigger: activateTaps)
            .accessibilityIdentifier("jupiter.activate")
            .confirmationDialog(
                proposal.isCustom ? "Turn these rules on?" : "Switch to \(Preset.title(for: proposal.preset))?",
                isPresented: $confirmActivate, titleVisibility: .visible
            ) {
                Button(proposal.isCustom ? "Activate" : "Switch") { apply(proposal) }
                Button("Cancel", role: .cancel) {}
            } message: {
                let preview = PolicyPreview(proposal: proposal, device: device)
                Text("\(preview.will.count) watched, \(preview.wont.count) not watched. "
                     + "Nothing changes until \(device.name) confirms it.")
            }
        }
    }

    // MARK: What will and won't be monitored

    /// The proposal as a before-and-after, from its own fields: the rules (or
    /// the mode) it turns on, everything in `unsupported`, rules that won't be
    /// turned on with a mode switch, and the mode custom rules would replace.
    struct PolicyPreview {
        var will: [String] = []
        var wont: [String] = []
        var notes: [String] = []

        init(proposal: AIProposal, device: Device) {
            let additive = MonitoringStatusPresentation.rulesAdditive(device.status)
            if proposal.isCustom {
                will = proposal.rules.map { rule in
                    rule.zone.isEmpty ? rule.headline
                        : "\(rule.headline) — once the \u{201C}\(rule.zone)\u{201D} area is drawn"
                }
                if additive {
                    notes.append("Runs alongside \(Preset.title(for: device.activePreset)), which keeps running.")
                } else if device.activePreset != nil, device.customRules != .replacing {
                    wont.append("\(Preset.title(for: device.activePreset)) — custom rules replace the current mode")
                }
            } else {
                will = ["Everything \(Preset.title(for: proposal.preset)) watches for"]
                wont += proposal.rules.map { "\($0.headline) — rules aren't turned on with a mode switch" }
            }
            wont += proposal.unsupported.map { "\($0.request) — \($0.reason)" }
            if !device.isLive() {
                notes.append("\(device.name) isn't reporting right now. This applies when it reconnects.")
            }
        }
    }

    private func previewCard(_ proposal: AIProposal) -> some View {
        let preview = PolicyPreview(proposal: proposal, device: device)
        return SDCard(raised: true) {
            VStack(alignment: .leading, spacing: SD.Space.md) {
                SDSectionHeader(title: "If you activate this")
                    .accessibilityIdentifier("jupiter.preview")
                // Bone, not green: nothing is being watched until the device
                // confirms it after Activate.
                previewGroup("Will watch for", icon: "eye", color: SD.Colors.textPrimary, items: preview.will)
                if !preview.wont.isEmpty {
                    previewGroup("Won't watch for", icon: "eye.slash", color: SD.Colors.warning, items: preview.wont)
                }
                ForEach(preview.notes, id: \.self) { note in
                    Text(note)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func previewGroup(_ title: String, icon: String, color: Color, items: [String]) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            Label(title, systemImage: icon)
                .font(SD.Typography.label)
                .foregroundStyle(color)
            ForEach(items, id: \.self) { item in
                HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                    Text("–")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .accessibilityHidden(true)
                    Text(item)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private func followUpColor(_ followUp: ActivationFollowUp) -> Color {
        switch followUp {
        case .confirmed: SD.Colors.success
        case .refused: SD.Colors.danger
        case .waiting: SD.Colors.textSecondary
        }
    }

    @State private var waitingForDevice = false
    @State private var activateTaps = 0

    /// The follow-up as a comparable word, for feedback keyed to it.
    private var followUpKind: String {
        switch followUp {
        case .confirmed?: "confirmed"
        case .refused?: "refused"
        case .waiting?: "waiting"
        case nil: "none"
        }
    }

    /// Ask the device, not the server, whether it runs what was activated:
    /// the device list's status (and the rules list when the server has it)
    /// every 2 s for up to 20 s.
    private func followDevice(proposal: AIProposal, rulesExpected: Int) async {
        let modeTitle = proposal.isCustom ? nil : Preset.title(for: proposal.preset)
        var last = ActivationFollowUp.waiting("Waiting for \(device.name) to confirm…")
        followUp = last
        waitingForDevice = true
        defer { waitingForDevice = false }
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline {
            try? await Task.sleep(for: .seconds(2))
            guard let fresh = try? await state.backend.devices().first(where: { $0.id == device.id }) else { continue }
            var rules: [DeviceRule]?
            if rulesExpected > 0 { rules = try? await state.backend.deviceRules(deviceID: device.id).rules }
            let rejected = fresh.status?.rules?.rejected
            last = ActivationFollowUp.evaluate(
                rulesExpected: rulesExpected, modeTitle: modeTitle,
                modePending: !proposal.isCustom && fresh.desiredPreset != nil,
                deviceLive: fresh.isLive(),
                rejectedReasons: rejected.map { $0.reasons ?? [] },
                runningRuleIDs: fresh.status?.rules?.active, rules: rules)
            followUp = last
            if case .waiting = last { continue }
            await onApplied()
            return
        }
        followUp = ActivationFollowUp.timedOut(last)
    }

    // MARK: Editing plumbing

    private enum Sensitivity: Hashable { case low, medium, high }

    private func ruleBinding(_ index: Int) -> Binding<AIRule>? {
        guard let proposal, proposal.rules.indices.contains(index) else { return nil }
        return Binding(
            get: { self.proposal?.rules[index] ?? proposal.rules[index] },
            set: { self.proposal?.rules[index] = $0 }
        )
    }

    /// Maps the three-way control onto the confidence threshold. High
    /// sensitivity means a lower bar to trigger, which is the opposite of what
    /// the number does — hence the mapping rather than exposing it raw.
    private func sensitivityBinding(_ rule: Binding<AIRule>) -> Binding<Sensitivity> {
        Binding(
            get: {
                switch rule.wrappedValue.confidence {
                case ..<0.5: .high
                case ..<0.75: .medium
                default: .low
                }
            },
            set: { level in
                switch level {
                case .high: rule.wrappedValue.confidence = 0.4
                case .medium: rule.wrappedValue.confidence = 0.6
                case .low: rule.wrappedValue.confidence = 0.85
                }
            }
        )
    }

    /// A slider position bound to a duration on `scale`. Only a touch writes:
    /// an untouched rule keeps its exact value, whatever the scale's steps.
    private func durationBinding(_ seconds: Binding<Double>, scale: DurationScale) -> Binding<Double> {
        Binding(
            get: { scale.position(for: seconds.wrappedValue) },
            set: { seconds.wrappedValue = scale.seconds(at: $0) }
        )
    }

    /// - Parameter notice: shown with the new proposal, to say why it is new.
    private func configure(notice: String? = nil) {
        thinking = true
        errorMessage = nil
        self.notice = notice
        applied = false
        appliedSummary = nil
        followUp = nil
        // The same text the backend will accept: trimmed, and refused rather
        // than truncated when over its limit (the bar already won't send it).
        guard JupiterPrompt.canSubmit(prompt), let request = JupiterPrompt.normalized(prompt) else {
            thinking = false
            if JupiterPrompt.isOverLimit(prompt) {
                errorMessage = "That request is too long for Jupiter (\(JupiterPrompt.maxLength) characters at most). Describe one thing to watch for at a time."
            }
            return
        }
        Task {
            defer { thinking = false }
            do {
                let result = try await thinkingProgress.track {
                    try await state.backend.aiConfigure(deviceID: device.id, prompt: request)
                }
                withAnimation(SD.Motion.standard) { proposal = result }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func apply(_ proposal: AIProposal) {
        activating = true
        errorMessage = nil
        notice = nil
        Task {
            defer { activating = false }
            do {
                // The proposal as reviewed — the edited rules and the draft they
                // came from — not just its preset. See BackendAPI.activate.
                let result = try await state.backend.activate(deviceID: device.id, proposal: proposal)
                let summary = ActivationSummary.summarize(
                    isCustom: proposal.isCustom, rulesSent: proposal.rules.count,
                    rulesActive: result.rulesActive,
                    additive: MonitoringStatusPresentation.rulesAdditive(device.status)
                )
                withAnimation(SD.Motion.standard) {
                    // Every rule the backend could not run goes under "Not yet
                    // possible", with its reason.
                    let known = Set(self.proposal?.unsupported.map(\.id) ?? [])
                    let moved = (result.unsupported ?? []).filter { !known.contains($0.id) }
                    self.proposal?.unsupported += moved
                    // None of the sent rules run: they are not "rules being
                    // edited" any more, they are listed with why.
                    if result.rulesActive == 0 { self.proposal?.rules = [] }
                    appliedSummary = summary
                    applied = summary.kind != .nothingRunning
                    if summary.kind == .nothingRunning { errorMessage = summary.note }
                }
                await onApplied()
                if summary.kind != .nothingRunning {
                    let expected = proposal.isCustom ? (result.rulesActive ?? proposal.rules.count) : 0
                    await followDevice(proposal: proposal, rulesExpected: expected)
                }
            } catch let error as APIError where error.code == "draft_superseded" {
                // The draft this proposal came from was replaced (another
                // preview, from this phone or another). Activating it would
                // run something nobody reviewed, so fetch it again.
                //
                // Only this code. Other refusals share 409 (the device's role
                // forbids the mode, role_forbids_preset), and re-running Jupiter
                // for those hid the reason behind "the proposal changed".
                configure(notice: "The proposal changed — review it again before activating it.")
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
