import SwiftUI

/// Describe what to watch for in plain language; review what SensorDyme
/// understood; adjust it; activate it.
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
    let onApplied: () async -> Void

    @State private var prompt = ""
    @State private var thinking = false
    @State private var proposal: AIProposal?
    @State private var applied = false
    @State private var errorMessage: String?
    @FocusState private var promptFocused: Bool
    @StateObject private var thinkingProgress = ProgressTracker()

    /// Deliberately spans what works and what doesn't, so the honest answers are
    /// discovered during setup rather than in front of an investor.
    private let examples = [
        "Alert me if my dog gets on the couch",
        "Tell me when the room becomes empty",
        "Alert me if someone stays on the floor for 30 seconds",
        "Alert me if more than three people come in",
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
            .background(SD.Colors.background)
            .scrollDismissesKeyboard(.interactively)
            .navigationTitle("Configure with AI")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .tint(SD.Colors.accent)
    }

    // MARK: Prompt

    private var promptCard: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            Text("What should \(device.name) watch for?")
                .font(SD.Typography.title)
                .foregroundStyle(SD.Colors.textPrimary)
                .padding(.top, SD.Space.xs)

            SDCard(padding: SD.Space.xs) {
                TextField("Describe it in your own words…", text: $prompt, axis: .vertical)
                    .font(SD.Typography.body)
                    .lineLimit(3...6)
                    .padding(SD.Space.xs)
                    .focused($promptFocused)
            }

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: SD.Space.xs) {
                    ForEach(examples, id: \.self) { example in
                        Button { prompt = example } label: {
                            Text(example)
                                .font(SD.Typography.caption)
                                .lineLimit(1)
                                .padding(.horizontal, SD.Space.sm)
                                .padding(.vertical, SD.Space.xs)
                                .background(Capsule().fill(SD.Colors.surfaceRaised))
                                .overlay(Capsule().strokeBorder(SD.Colors.border, lineWidth: 1))
                                .foregroundStyle(SD.Colors.textSecondary)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.vertical, 2)
            }

            Button(action: configure) {
                if thinking {
                    HStack(spacing: SD.Space.xs) {
                        ProgressView().tint(SD.Colors.textOnAccent)
                        Text("Working it out…")
                    }
                } else {
                    HStack(spacing: SD.Space.xs) {
                        Text(proposal == nil ? "Continue" : "Try again")
                        SDProgress(phase: thinkingProgress.phase)
                    }
                }
            }
            .buttonStyle(SDButtonStyle(kind: .primary))
            .disabled(prompt.trimmingCharacters(in: .whitespaces).isEmpty || thinking)
        }
    }

    // MARK: What was understood

    private func understandingCard(_ proposal: AIProposal) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                HStack {
                    SDSectionHeader(title: "SensorDyme understands")
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

                    settingRow("Confirm for") {
                        Text("\(Int(binding.wrappedValue.persist_s))s")
                            .font(SD.Typography.caption.monospacedDigit())
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Slider(value: binding.persist_s, in: 1...60, step: 1)

                    settingRow("Then wait") {
                        Text("\(Int(binding.wrappedValue.cooldown_s / 60))m")
                            .font(SD.Typography.caption.monospacedDigit())
                            .foregroundStyle(SD.Colors.textSecondary)
                    }
                    Slider(value: binding.cooldown_s, in: 30...1800, step: 30)
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
                message: "SensorDyme couldn't turn that into something it can watch for. Try describing it differently, or see what it would need above."
            )
        } else if applied {
            Label("Active on \(device.name)", systemImage: "checkmark.circle.fill")
                .font(SD.Typography.headline)
                .foregroundStyle(SD.Colors.success)
                .frame(maxWidth: .infinity)
                .padding(.vertical, SD.Space.xs)
        } else {
            Button { apply(proposal) } label: {
                Text("Activate \(proposal.rules.count == 1 ? "rule" : "\(proposal.rules.count) rules")")
            }
            .buttonStyle(SDButtonStyle(kind: .primary))
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

    private func configure() {
        thinking = true
        errorMessage = nil
        applied = false
        promptFocused = false
        Task {
            defer { thinking = false }
            do {
                let result = try await thinkingProgress.track {
                    try await state.backend.aiConfigure(deviceID: device.id, prompt: prompt)
                }
                withAnimation(SD.Motion.standard) { proposal = result }
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    private func apply(_ proposal: AIProposal) {
        Task {
            do {
                // Rules are already persisted by the configure call, but that
                // alone does not tell the backend to actually turn them on —
                // /v1/devices/{id}/preset is what marks intent
                // (custom_monitoring_wanted on the device row), and nothing
                // else does. This used to be skipped for a genuinely custom
                // proposal (empty preset) on the theory that sending "" was a
                // pointless or even unwanted mode change — but an empty
                // desired_preset is falsy server-side and never produces a
                // set_preset command, so it never actually changes what mode
                // is selected. It only sets the "please deliver and run these
                // rules" flag. Skipping it meant tapping Activate for any
                // open-ended prompt (no named preset matched) saved the rules
                // but never told the device to run them — the button showed
                // "Active on {device}" while nothing was ever delivered. Both
                // branches now call it; for a named-preset proposal this is
                // unchanged from before.
                try await state.backend.setPreset(deviceID: device.id, preset: proposal.preset)
                withAnimation(SD.Motion.standard) { applied = true }
                await onApplied()
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
