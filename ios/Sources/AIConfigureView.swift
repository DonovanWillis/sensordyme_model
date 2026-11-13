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
///
/// Where the request is (JupiterRequestPhase) is shown once, as a four-step
/// strip under the request: Request → Review → Sent → Running. "Running" is
/// only ever what the DEVICE reported; the server accepting it is "Sent".
struct AIConfigureView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    @Environment(\.dynamicTypeSize) private var typeSize
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
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
    /// The wait for the device ran out without an answer (still "sent").
    @State private var deviceWaitExpired = false
    /// The activation request itself failed (nothing was sent to the device).
    @State private var activationFailed = false
    @State private var errorMessage: String?
    /// Not an error: why the screen is showing a fresh proposal (the one being
    /// reviewed was replaced on the server before it could be activated).
    @State private var notice: String?
    @StateObject private var thinkingProgress = ProgressTracker()

    private var phase: JupiterRequestPhase {
        let outcome: JupiterRequestPhase.DeviceOutcome
        switch followUp {
        case .confirmed?: outcome = .confirmed
        case .refused?: outcome = .refused
        case .waiting?: outcome = .waiting
        case nil: outcome = .none
        }
        return .from(thinking: thinking, hasProposal: proposal != nil, proposalEmpty: proposal?.isEmpty ?? true,
                     activating: activating, applied: applied, outcome: outcome, timedOut: deviceWaitExpired,
                     activationFailed: activationFailed)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    requestSection
                    if let errorMessage, !activationFailed {
                        problem(errorMessage)
                            .padding(.top, SD.Space.md)
                    }
                    if let notice {
                        Label(notice, systemImage: "arrow.triangle.2.circlepath")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                            .padding(.top, SD.Space.md)
                    }
                    if let proposal {
                        understoodSection(proposal)
                            .padding(.top, SD.Space.xl)
                        if !proposal.rules.isEmpty {
                            VStack(alignment: .leading, spacing: SD.Space.sm) {
                                SDSectionHeader(title: proposal.rules.count == 1 ? "The rule" : "The rules")
                                ForEach(Array(proposal.rules.enumerated()), id: \.offset) { index, _ in
                                    ruleEditor(index)
                                }
                            }
                            .padding(.top, SD.Space.xl)
                        }
                        if !proposal.unsupported.isEmpty {
                            limitationsSection(proposal.unsupported)
                                .padding(.top, SD.Space.xl)
                        }
                        if !proposal.isEmpty && !applied {
                            previewCard(proposal)
                                .padding(.top, SD.Space.xl)
                        }
                        activateSection(proposal)
                            .padding(.top, SD.Space.lg)
                    }
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.bottom, SD.Space.xl)
                .animation(SD.Motion.resolved(SD.Motion.standard, reduceMotion: reduceMotion), value: phase)
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

    private func problem(_ message: String) -> some View {
        Label(message, systemImage: "exclamationmark.circle.fill")
            .font(SD.Typography.callout)
            .foregroundStyle(SD.Colors.danger)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("jupiter.error")
    }

    // MARK: Request

    private var requestSection: some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                // Which device this is for, in its own words (sentence case: a
                // name, not a section label).
                HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                    OrynMark(height: 12, color: SD.Colors.signal)
                        .accessibilityHidden(true)
                    Text(device.name)
                        .font(SD.Typography.caption.weight(.medium))
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
                .accessibilityLabel("For \(device.name)")
                Text(phase == .asking ? "What should it watch for?" : "Your request")
                    .sdScaledFont(28, relativeTo: .title, weight: .semibold)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)
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

            if phase != .asking {
                JupiterPhaseStrip(phase: phase)
                    .transition(.opacity)
            }

            if thinking {
                // The request is in flight, and that is all this says: no
                // streamed text, no typing performance.
                HStack(spacing: SD.Space.xs) {
                    ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
                    Text("Jupiter is reading your request…")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                    SDProgress(phase: thinkingProgress.phase == .done ? .idle : thinkingProgress.phase)
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("jupiter.thinking")
                .transition(.opacity)
            } else if proposal == nil {
                suggestionList
            }
        }
    }

    /// What Jupiter can be asked, from the policy grammar (JupiterSuggestions),
    /// and one request it refuses, labelled as such.
    private var suggestionList: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDSectionHeader(title: "Jupiter can watch for")
            VStack(spacing: 0) {
                ForEach(JupiterSuggestions.supported) { suggestion in
                    suggestionRow(suggestion, refused: false)
                    OrynRowDivider(inset: 0)
                }
                suggestionRow(JupiterSuggestions.refused, refused: true)
            }
            Text("Jupiter turns your words into rules for this device. Nothing changes until you review and activate them, and the device confirms it. Requests about who someone is, what they feel or intend, sounds, medical conditions or the experimental driver signals are refused, with the reason.")
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, SD.Space.xs)
        }
        .padding(.top, SD.Space.sm)
    }

    private func suggestionRow(_ suggestion: JupiterSuggestions.Suggestion, refused: Bool) -> some View {
        Button { prompt = suggestion.text } label: {
            HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
                Image(systemName: refused ? "nosign" : "arrow.turn.down.right")
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(refused ? SD.Colors.warning : SD.Colors.textTertiary)
                    .frame(width: 14)
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    Text(suggestion.text)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .multilineTextAlignment(.leading)
                        .fixedSize(horizontal: false, vertical: true)
                    Text(suggestion.kind)
                        .font(SD.Typography.caption)
                        .foregroundStyle(refused ? SD.Colors.warning : SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
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

    /// Open text, not a box: what Jupiter understood is the thing to read.
    private func understoodSection(_ proposal: AIProposal) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            HStack(spacing: 6) {
                OrynMark(height: 11, color: SD.Colors.signal)
                    .accessibilityHidden(true)
                SDEyebrow("Jupiter understood")
                    .accessibilityAddTraits(.isHeader)
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
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let confidence = proposal.jupiter?.interpretation_confidence, confidence.worthFlagging {
                interpretationConfidenceNotice(confidence)
            }
        }
    }

    /// Plain language for everyone; the actual reasons behind a small
    /// disclosure for anyone who wants them. `worthFlagging` already filtered
    /// out the routine case (part of the request being unsupported) — the
    /// limitations section below says that on its own, and repeating it here
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

    /// A rule is a set of controls, so it keeps a surface (SDCard): the one
    /// place on this screen a container means "you can change this".
    @ViewBuilder
    private func ruleEditor(_ index: Int) -> some View {
        if let binding = ruleBinding(index) {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    Text(binding.wrappedValue.headline)
                        .font(SD.Typography.headline)
                        .foregroundStyle(SD.Colors.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
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
                        // From the backend's typed policy when it sent one
                        // (M8.4): "Draw the area" was said even when the area
                        // was already drawn.
                        if proposal?.policy(at: index)?.zone?.drawn == true {
                            Label("Uses your area \u{201C}\(binding.wrappedValue.zone)\u{201D}",
                                  systemImage: "checkmark.circle")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .accessibilityIdentifier("jupiter.zoneRequirement")
                        } else {
                            // Areas are drawn in the app (Detection areas), not on the device.
                            Label("Draw the area named \u{201C}\(binding.wrappedValue.zone)\u{201D} in Detection areas to turn this on",
                                  systemImage: "pencil.and.outline")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                                .accessibilityIdentifier("jupiter.zoneRequirement")
                        }
                    }
                    if let explanation = proposal?.policy(at: index)?.explanation {
                        policyScope(explanation)
                    }

                    Divider().overlay(SD.Colors.border)

                    // Sensitivity is confidence + persistence together. Exposing
                    // two numeric thresholds would be honest and useless; nobody
                    // tunes a home camera by picking a float.
                    settingRow("Sensitivity") {
                        Picker("Sensitivity", selection: sensitivityBinding(binding)) {
                            Text("Low").tag(Sensitivity.low)
                            Text("Medium").tag(Sensitivity.medium)
                            Text("High").tag(Sensitivity.high)
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: typeSize.isAccessibilitySize ? .infinity : 220)
                    }

                    settingRow("Alert me") {
                        Picker("Alert me", selection: binding.notify) {
                            Text("Push").tag("push")
                            Text("Silent").tag("silent")
                        }
                        .pickerStyle(.segmented)
                        .labelsHidden()
                        .frame(maxWidth: typeSize.isAccessibilitySize ? .infinity : 160)
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
                        .accessibilityLabel("Confirm for")
                        .accessibilityValue(DurationText.short(binding.wrappedValue.persist_s))

                    settingRow("Then wait") {
                        Text(DurationText.short(binding.wrappedValue.cooldown_s))
                            .font(SD.Typography.caption.monospacedDigit())
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Slider(value: durationBinding(binding.cooldown_s, scale: .cooldown), in: 0...1)
                        .accessibilityLabel("Then wait")
                        .accessibilityValue(DurationText.short(binding.wrappedValue.cooldown_s))
                    Text("How long the condition must hold before alerting, and how long before it can alert again.")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            // Settled once activated: what was sent is what the device runs.
            .disabled(applied || activating)
        }
    }

    /// What the rule watches and what it does not, in the backend's typed
    /// policy's own words (jupiter/policy.py explain): folded, one tap away.
    @ViewBuilder
    private func policyScope(_ explanation: AIPolicyPreview.Explanation) -> some View {
        let watches = explanation.monitors ?? []
        let ignores = explanation.does_not_monitor ?? []
        if !watches.isEmpty || !ignores.isEmpty {
            DisclosureGroup {
                VStack(alignment: .leading, spacing: SD.Space.xs) {
                    if !watches.isEmpty {
                        scopeList("Watches", watches)
                    }
                    if !ignores.isEmpty {
                        scopeList("Doesn't watch", ignores)
                    }
                }
                .padding(.top, SD.Space.xxs)
            } label: {
                Text("What this rule watches")
                    .font(SD.Typography.caption.weight(.medium))
                    .foregroundStyle(SD.Colors.textSecondary)
            }
            .accessibilityIdentifier("jupiter.policyScope")
        }
    }

    private func scopeList(_ title: String, _ items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(SD.Colors.textTertiary)
            ForEach(items, id: \.self) { item in
                Text("– \(item)")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }

    /// A label and its control side by side, or stacked at accessibility sizes
    /// where a segmented control beside a label would squeeze both.
    @ViewBuilder
    private func settingRow<Content: View>(_ label: String, @ViewBuilder content: () -> Content) -> some View {
        if typeSize.isAccessibilitySize {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                Text(label)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                content()
            }
        } else {
            HStack {
                Text(label)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                Spacer()
                content()
            }
        }
    }

    // MARK: Limitations

    /// What Jupiter could not turn into a rule, each with its reason and what
    /// it would need. Open rows under a section label, not another box.
    private func limitationsSection(_ items: [AIUnsupported]) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            // "Not yet possible" promised a later version for requests that are
            // refused by policy (identity, sound, intent): said as they are.
            SDSectionHeader(title: items.allSatisfy { JupiterRefusal.isPolicy(code: $0.code) }
                            ? "Won't be set up" : "Not possible yet")
                .accessibilityIdentifier("jupiter.limitations")
            ForEach(items) { item in
                HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
                    Image(systemName: "nosign")
                        .font(.system(size: 12, weight: .medium))
                        .foregroundStyle(SD.Colors.warning)
                        .frame(width: 16)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(item.request)
                            .font(SD.Typography.label)
                            .foregroundStyle(SD.Colors.textPrimary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(item.route)
                            .font(SD.Typography.caption.weight(.semibold))
                            .foregroundStyle(SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("jupiter.refusal.\(item.code ?? "untyped")")
                        Text(item.reason)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Text(item.requiresLine)
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
                    Spacer(minLength: 0)
                }
                .accessibilityElement(children: .combine)
                if item.id != items.last?.id {
                    OrynRowDivider(inset: 28)
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
                // A policy refusal is not a phrasing problem (M8.4): rephrasing
                // "a stranger" or "high risk" will never make it possible.
                message: !proposal.unsupported.isEmpty
                    && proposal.unsupported.allSatisfy({ JupiterRefusal.isPolicy(code: $0.code) })
                    ? "That can't be set up, however it's worded. Ask for something the camera can observe, like someone entering or staying in an area you draw."
                    : "Jupiter couldn't turn that into something this device can watch for. Try describing it differently, or see what it would need above."
            )
        } else if applied {
            appliedStatus(proposal)
        } else {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                if activationFailed, let errorMessage {
                    problem(errorMessage)
                }
                Button { activateTaps += 1; confirmActivate = true } label: {
                    if activating {
                        HStack(spacing: SD.Space.xs) {
                            ProgressView().tint(SD.Colors.textOnAccent)
                            Text("Sending…")
                        }
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
                Text("Nothing changes until \(device.name) confirms it. You'll see it here when it does.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    /// After Activate: what was sent, then what the device reported. The
    /// preview's place becomes the rule's own status surface.
    private func appliedStatus(_ proposal: AIProposal) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            // Said from what the device reported, not from the server
            // accepting the request (ActivationFollowUp).
            switch followUp {
            case .confirmed?:
                Label(appliedSummary?.kind == .modeSwitched
                      ? "Switched to \(Preset.title(for: proposal.preset))"
                      : "Running on \(device.name)", systemImage: "checkmark.circle.fill")
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
                        .font(SD.Typography.callout)
                        .foregroundStyle(followUpColor(followUp))
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
            if let appliedNote = appliedSummary?.note {
                Text(appliedNote)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(SD.Space.md)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                .fill(SD.Colors.surface)
        )
        .transition(.opacity)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("jupiter.applied")
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
                will = proposal.rules.enumerated().map { index, rule in
                    if rule.zone.isEmpty || proposal.policy(at: index)?.zone?.drawn == true { return rule.headline }
                    return "\(rule.headline) — once the \u{201C}\(rule.zone)\u{201D} area is drawn"
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


    /// The safety statement before anything changes: will and won't watch for.
    /// Raised, because it is what Activate commits to.
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
        if case .waiting = followUp { deviceWaitExpired = true }
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
        deviceWaitExpired = false
        activationFailed = false
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
        activationFailed = false
        deviceWaitExpired = false
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
                    if summary.kind == .nothingRunning {
                        errorMessage = summary.note
                        activationFailed = true
                    }
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
                activationFailed = true
            }
        }
    }
}

/// Where an Ask Jupiter request is: Request → Review → Sent → Running, with the
/// phase in words under it. A row of four at the default size; a list at
/// accessibility sizes. Green only for Running, which only the device can
/// confirm; waiting is amber, a refusal or failure red, both with a word.
struct JupiterPhaseStrip: View {
    let phase: JupiterRequestPhase
    @Environment(\.dynamicTypeSize) private var typeSize

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            if typeSize.isAccessibilitySize {
                list
            } else {
                // A row when the four fit (they do at the default sizes), a list
                // when the reader's text size makes them too wide.
                ViewThatFits(in: .horizontal) { row; list }
            }
            Text(phase.summary)
                .font(SD.Typography.caption)
                .foregroundStyle(summaryColor)
                .fixedSize(horizontal: false, vertical: true)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(phase.spokenProgress)
        .accessibilityIdentifier("jupiter.phase")
    }

    private var row: some View {
        HStack(spacing: SD.Space.xs) {
            ForEach(JupiterRequestPhase.Step.allCases, id: \.self) { step in
                HStack(spacing: 6) {
                    marker(phase.state(of: step))
                    label(step)
                }
                if step != .running {
                    Rectangle()
                        .fill(phase.state(of: step) == .done ? SD.Colors.textTertiary : SD.Colors.border)
                        .frame(height: 1)
                        .frame(minWidth: 8, maxWidth: .infinity)
                }
            }
        }
    }

    private var list: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            ForEach(JupiterRequestPhase.Step.allCases, id: \.self) { step in
                HStack(spacing: SD.Space.xs) {
                    marker(phase.state(of: step))
                    label(step)
                }
            }
        }
    }

    private func label(_ step: JupiterRequestPhase.Step) -> some View {
        let state = phase.state(of: step)
        return Text(step == .running && state == .failed ? "Not running" : step.title)
            .font(state == .current || state == .failed ? SD.Typography.caption.weight(.semibold) : SD.Typography.caption)
            .foregroundStyle(labelColor(state, step: step))
            .lineLimit(1)
            .fixedSize()
    }

    @ViewBuilder
    private func marker(_ state: JupiterRequestPhase.StepState) -> some View {
        switch state {
        case .done:
            Image(systemName: "checkmark")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(SD.Colors.textOnAccent)
                .frame(width: 14, height: 14)
                .background(Circle().fill(phase == .running ? SD.Colors.success : SD.Colors.textSecondary))
        case .current:
            Circle()
                .strokeBorder(SD.Colors.textPrimary, lineWidth: 2)
                .frame(width: 14, height: 14)
        case .upcoming:
            Circle()
                .strokeBorder(SD.Colors.borderStrong, lineWidth: 1.5)
                .frame(width: 12, height: 12)
                .frame(width: 14, height: 14)
        case .failed:
            Image(systemName: "xmark")
                .font(.system(size: 8, weight: .bold))
                .foregroundStyle(SD.Colors.textOnAccent)
                .frame(width: 14, height: 14)
                .background(Circle().fill(SD.Colors.danger))
        }
    }

    private func labelColor(_ state: JupiterRequestPhase.StepState, step: JupiterRequestPhase.Step) -> Color {
        switch state {
        case .done: phase == .running && step == .running ? SD.Colors.success : SD.Colors.textSecondary
        case .current: SD.Colors.textPrimary
        case .upcoming: SD.Colors.textTertiary
        case .failed: SD.Colors.danger
        }
    }

    private var summaryColor: Color {
        switch phase {
        case .running: SD.Colors.success
        case .failed: SD.Colors.danger
        case .waitingForDevice: SD.Colors.warning
        case .nothingToActivate: SD.Colors.warning
        default: SD.Colors.textTertiary
        }
    }
}
