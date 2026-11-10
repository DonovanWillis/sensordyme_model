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
    @State private var errorMessage: String?
    /// Not an error: why the screen is showing a fresh proposal (the one being
    /// reviewed was replaced on the server before it could be activated).
    @State private var notice: String?
    @StateObject private var thinkingProgress = ProgressTracker()

    /// Deliberately spans what works and what doesn't, so the honest answers are
    /// discovered during setup rather than in front of an investor.
    private let examples = [
        "Watch for someone entering the room after 10 PM",
        "Alert me when someone picks up my laptop",
        "Monitor for signs of driver distraction",
        "Alert me if someone stays on the floor for 30 seconds",
        "Tell me when the room becomes empty",
    ]

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.md) {
                    promptCard
                    if let errorMessage {
                        Label(errorMessage, systemImage: "exclamationmark.circle.fill")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.danger)
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
                            .font(.system(size: 16, weight: .semibold))
                            .foregroundStyle(SD.Colors.textPrimary)
                    }
                    .accessibilityElement(children: .combine)
                    .accessibilityAddTraits(.isHeader)
                }
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .tint(SD.Colors.accent)
        .presentationBackground(SD.Colors.background)
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
            VStack(alignment: .leading, spacing: SD.Space.xxs) {
                SDEyebrow("Jupiter · \(device.name)")
                Text(proposal == nil && !thinking
                     ? "What should \(device.name) watch for?"
                     : "Your request")
                    .font(SD.Typography.title)
                    .foregroundStyle(SD.Colors.textPrimary)
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
                HStack(spacing: SD.Space.xs) {
                    ThinkingDots()
                    Text("Jupiter is working it out…")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                    SDProgress(phase: thinkingProgress.phase)
                }
                .padding(.horizontal, SD.Space.xxs)
                .transition(.opacity)
            } else if proposal == nil {
                VStack(alignment: .leading, spacing: SD.Space.xs) {
                    SDEyebrow("Try")
                        .padding(.horizontal, SD.Space.xxs)
                    VStack(spacing: 0) {
                        ForEach(examples, id: \.self) { example in
                            Button { prompt = example } label: {
                                HStack(spacing: SD.Space.sm) {
                                    Image(systemName: "arrow.turn.down.right")
                                        .font(.system(size: 11, weight: .medium))
                                        .foregroundStyle(SD.Colors.textDisabled)
                                    Text(example)
                                        .font(SD.Typography.callout)
                                        .foregroundStyle(SD.Colors.textSecondary)
                                        .multilineTextAlignment(.leading)
                                    Spacer(minLength: 0)
                                }
                            }
                            .buttonStyle(SDRowButtonStyle(minHeight: 44))
                            if example != examples.last {
                                Rectangle().fill(SD.Colors.border).frame(height: 1)
                                    .padding(.leading, SD.Space.sm + 11 + SD.Space.sm)
                            }
                        }
                    }
                    .padding(SD.Space.xxs)
                    .background(
                        RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                            .fill(SD.Colors.surface)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                            .strokeBorder(SD.Colors.border, lineWidth: 1)
                    )
                    Text("Jupiter turns your words into rules for this device. Nothing changes until you review and activate them.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                        .padding(.horizontal, SD.Space.xxs)
                        .padding(.top, SD.Space.xxs)
                }
            }
        }
        .animation(SD.Motion.standard, value: thinking)
    }

    // MARK: What was understood

    private func understandingCard(_ proposal: AIProposal) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                HStack {
                    HStack(spacing: 6) {
                        OrynMark(height: 11, color: SD.Colors.textTertiary)
                        SDSectionHeader(title: "Jupiter understood")
                    }
                    if applied {
                        SDStatusPill(status: .online, text: "Active")
                    }
                }
                Text(proposal.explanation)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)

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
                if appliedSummary?.kind == .modeSwitched {
                    Label("Switched to \(Preset.title(for: proposal.preset))", systemImage: "checkmark.circle.fill")
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.success)
                } else {
                    Label("Active on \(device.name)", systemImage: "checkmark.circle.fill")
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.success)
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
            .padding(.vertical, SD.Space.xs)
        } else {
            Button { apply(proposal) } label: {
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
        }
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
                    rulesActive: result.rulesActive
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
            } catch let error as APIError where error.status == 409 {
                // The draft this proposal came from was replaced (another
                // preview, from this phone or another). Activating it would
                // run something nobody reviewed, so fetch it again.
                configure(notice: "The proposal changed — review it again before activating it.")
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
