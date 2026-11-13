import SwiftUI

/// One preset from the library: what it watches for, what it needs, how far it
/// has been validated, its settings, and the button that activates it.
///
/// Everything here comes from the backend's catalog (GET /v1/presets) and the
/// device's own answer (GET /v1/devices/{id}/presets). Activation is
/// POST /v1/devices/{id}/preset with the settings as edited; nothing reads as
/// applied until the device reports it.
struct PresetDetailView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss

    let device: Device
    let preset: CatalogPreset
    /// Whether the catalog came from the server or is the app's built-in list.
    let catalogSource: PresetCatalog.Source
    /// What the device said about this preset, if the server has that endpoint.
    let deviceState: DevicePresetState?
    /// Why it can't be picked when the per-device endpoint is missing (from
    /// the device's role in `status`), else nil.
    let legacyUnavailableReason: String?
    /// Reload the device screen after a change.
    let onApplied: () async -> Void

    @State private var values: [String: JSONValue] = [:]
    @State private var sending = false
    @State private var errorMessage: String?
    /// What happened after the server accepted the change.
    @State private var outcome: Outcome?
    /// How far each related product capability has been validated
    /// (GET /v1/capability-tiers); nil until loaded or when unsupported.
    @State private var tiers: CapabilityTiersResponse?
    /// The capability lines arrive in sequence once the sheet is up.
    @State private var revealed = false
    @State private var activateTaps = 0
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private enum Outcome: Equatable {
        case waiting(String), confirmed(String, StatusTone), notMonitoring(String, StatusTone),
             refused(String), unconfirmed(String)
    }

    /// The node the device reports for this preset, when it runs one.
    private var node: MonitoringNode? { device.status?.perception?.node(preset.id) }

    /// Each capability this preset needs, judged from what the DEVICE reports.
    private var checks: [CapabilityCheck] {
        CapabilityChecks.make(required: preset.requiredCapabilities, optional: preset.optionalCapabilities,
                              status: device.status, online: device.isLive(), node: node)
    }

    private var availability: PresetAvailability {
        if deviceState != nil { return PresetAvailability.make(deviceState, roleTitle: device.status?.roleTitle) }
        if let legacyUnavailableReason { return PresetAvailability(blocksSelection: true, note: legacyUnavailableReason) }
        return .open
    }

    private var isActive: Bool {
        deviceState?.active ?? (Preset.split(device.activePreset).base == preset.id && device.customRules != .replacing)
    }

    private var validation: ValidationPresentation {
        ValidationPresentation.make(deviceState?.validationStatus ?? preset.validation.status)
    }

    private var problems: [String: String] { PresetSettingsForm.problems(preset.settings, values: values) }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.xl) {
                    VStack(alignment: .leading, spacing: SD.Space.md) {
                        header
                        notices
                        if isActive { monitoringCard }
                    }
                    if !preset.requiredCapabilities.isEmpty || !preset.optionalCapabilities.isEmpty
                        || !preset.hardware.isEmpty {
                        requirementsCard
                    }
                    validationCard
                    if !preset.temporalPolicies.isEmpty { policiesCard }
                    if !preset.limitations.isEmpty { limitationsCard }
                    if !preset.settings.isEmpty { settingsCard }
                    actionSection
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.top, SD.Space.xs)
                .padding(.bottom, SD.Space.xl)
            }
            .orynScreen()
            .navigationTitle(preset.title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Done") { dismiss() } }
            }
        }
        .tint(SD.Colors.accent)
        .presentationBackground(SD.Colors.background)
        .onAppear {
            values = PresetSettingsForm.initialValues(preset.settings, current: deviceState?.settings ?? [:])
        }
        .task { tiers = try? await state.backend.capabilityTiers() }
        // After the zoom has settled: lines that animate while the sheet is
        // still scaling read as scattered text (seen in the recording).
        .task {
            if !reduceMotion { try? await Task.sleep(for: .milliseconds(450)) }
            revealed = true
        }
        // Feedback follows the DEVICE: success only once it confirms, an error
        // when it (or the server) refuses. Never on "sent".
        .sensoryFeedback(trigger: outcome) { _, new in
            switch new {
            case .confirmed?: .success
            case .refused?, .notMonitoring?: .error
            default: nil
            }
        }
    }

    // MARK: Sections

    /// The mode's identity: its symbol, where it is used, its name and what it
    /// watches for, in the device's catalog's own words.
    private var header: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            Image(systemName: Preset.icon(for: preset.id))
                .font(.system(size: 22, weight: .regular))
                .foregroundStyle(SD.Colors.textPrimary)
                .frame(width: 52, height: 52)
                .background(
                    RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                        .fill(SD.Colors.surfaceRaised)
                )
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: SD.Space.xxs) {
                SDEyebrow(isActive ? "Running on \(device.name)" : (ModeEditorial.context(preset.id) ?? "Monitoring mode"))
                Text(preset.title)
                    .sdScaledFont(30, relativeTo: .title, weight: .semibold)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)
                if !preset.description.isEmpty {
                    Text(preset.description)
                        .font(SD.Typography.body)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(.top, SD.Space.sm)
    }

    @ViewBuilder private var notices: some View {
        if case .builtIn(let reason) = catalogSource {
            notice(reason, icon: "info.circle", color: SD.Colors.textTertiary)
        }
        if let note = availability.note {
            notice(note, icon: availability.blocksSelection ? "nosign" : "wifi.slash",
                   color: availability.blocksSelection ? SD.Colors.textSecondary : SD.Colors.warning)
        }
    }

    private func notice(_ text: String, icon: String, color: Color) -> some View {
        Label(text, systemImage: icon)
            .font(SD.Typography.callout)
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
    }

    /// A section on open space: its label, then its lines.
    private func section<Content: View>(_ title: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDEyebrow(title)
                .accessibilityAddTraits(.isHeader)
            content()
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var validationCard: some View {
        section("Validation") {
            Label(validation.chip, systemImage: validation.isValidated ? "checkmark.seal" : "seal")
                .font(SD.Typography.label)
                .foregroundStyle(validation.isValidated ? SD.Colors.success : SD.Colors.warning)
            Text(validation.detail)
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            ForEach(preset.validation.evidence, id: \.self) { line in
                Text("• \(line)")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // What the device itself says about the node running this preset.
            if let node {
                line("On \(device.name)", NodePresentation.badge(node))
                    .accessibilityIdentifier("preset.node.validation")
            }
            // How far each capability it rests on has been checked, from the
            // model registry. Never "validated" short of production approval.
            ForEach(relatedTiers, id: \.capability) { entry in
                line(CapabilityHealth.title(entry.capability), CapabilityTierPresentation.make(entry).chip)
                    .accessibilityIdentifier("preset.tier.\(entry.capability)")
            }
        }
    }

    private var relatedTiers: [CapabilityTierEntry] {
        tiers?.entries(forKinds: preset.requiredCapabilities + preset.optionalCapabilities) ?? []
    }

    /// What the device is doing now with this preset: its own monitoring state,
    /// as the device screen's readout says it.
    private var monitoringCard: some View {
        let presentation = MonitoringStatusPresentation.make(status: device.status, online: device.isLive())
        let impaired = PerceptionPresentation.degradedSummary(
            device.status?.perception, live: device.isLive())
        let label = presentation?.label ?? "Status unknown"
        // The device screen's readout symbol and colour: a stopped device is
        // paused on purpose (steel pause), not a caution.
        let kind = presentation?.phase.instrumentKind ?? .unknown
        return VStack(alignment: .leading, spacing: SD.Space.xs) {
            HStack(spacing: SD.Space.xs) {
                Image(systemName: InstrumentState.symbol(kind))
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(kind.tint)
                Text(label)
                    .font(SD.Typography.readout)
                    .foregroundStyle(SD.Colors.textPrimary)
            }
            .accessibilityElement(children: .ignore)
            .accessibilityLabel(label)
            .accessibilityIdentifier("preset.monitoring")
            Text(presentation?.detail ?? "This server doesn't report the device's monitoring state.")
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if let impaired {
                Label(impaired, systemImage: "exclamationmark.triangle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(SD.Space.md)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: SD.Radius.camera, style: .continuous)
                .fill(SD.Colors.surface)
        )
    }

    /// What it needs, each checked against what THIS device reports: working,
    /// not working (with the device's reason), or unknown. Never assumed. The
    /// lines arrive in sequence (instantly under Reduce Motion).
    private var requirementsCard: some View {
        section("What it needs on \(device.name)") {
            ForEach(Array(checks.enumerated()), id: \.element.id) { index, check in
                capabilityLine(check)
                    .opacity(revealed ? 1 : 0)
                    .animation(reduceMotion ? nil
                               : SD.Motion.reveal.delay(SD.Motion.revealDelay(index, reduceMotion: false)),
                               value: revealed)
            }
            if !preset.hardware.isEmpty {
                line("Hardware", preset.hardware.joined(separator: ", "))
            }
            if let blocking = CapabilityChecks.blockingSummary(checks) {
                Text(blocking)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func capabilityLine(_ check: CapabilityCheck) -> some View {
        let shown: (text: String, color: Color, icon: String) = {
            switch check.verdict {
            case .available: return ("Working", SD.Colors.success, "checkmark.circle.fill")
            case .unavailable(let why): return ("Not working — \(why)", SD.Colors.danger, "xmark.octagon.fill")
            case .unknown(let why): return ("Unknown — \(why)", SD.Colors.textTertiary, "questionmark.circle")
            case .starting(let what): return (what, SD.Colors.warning, "hourglass")
            case .paused(let what): return (what, SD.Colors.textSecondary, "pause.circle")
            }
        }()
        return HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
            Image(systemName: shown.icon)
                .font(.callout)
                .foregroundStyle(shown.color)
                .frame(width: 20)
            VStack(alignment: .leading, spacing: 1) {
                Text(check.title + (check.required ? "" : " (optional)"))
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                Text(shown.text)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 0)
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("preset.capability.\(check.name)")
    }

    private var policiesCard: some View {
        section("When it alerts") {
            ForEach(Array(preset.temporalPolicies.enumerated()), id: \.offset) { _, policy in
                let text = policy.description.isEmpty
                    ? policy.event.replacingOccurrences(of: "_", with: " ").capitalized
                    : policy.description
                line(text, policy.sustainS.map { "held for \(DurationText.short($0))" } ?? "")
            }
        }
    }

    private var limitationsCard: some View {
        section("Limitations") {
            ForEach(preset.limitations, id: \.self) { limitation in
                Text("• \(limitation)")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func line(_ label: String, _ value: String) -> some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textPrimary)
            Spacer(minLength: SD.Space.xs)
            Text(value)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textSecondary)
                .multilineTextAlignment(.trailing)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Settings, from the catalog's schema

    private var settingsCard: some View {
        section("Settings") {
            VStack(alignment: .leading, spacing: SD.Space.md) {
                ForEach(preset.settings) { setting in
                    settingControl(setting)
                    if let problem = problems[setting.key] {
                        Text(problem)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.danger)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func settingControl(_ setting: PresetSetting) -> some View {
        switch setting.kind {
        case .bool:
            Toggle(setting.title, isOn: Binding(
                get: { values[setting.key]?.boolValue ?? false },
                set: { values[setting.key] = .bool($0) }))
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textPrimary)
        case .number:
            let current = values[setting.key]?.doubleValue ?? setting.min ?? 0
            VStack(alignment: .leading, spacing: SD.Space.xxs) {
                HStack {
                    Text(setting.title)
                        .font(SD.Typography.body)
                        .foregroundStyle(SD.Colors.textPrimary)
                    Spacer()
                    Text(PresetSettingsForm.format(current, setting.unit))
                        .font(SD.Typography.caption.monospacedDigit())
                        .foregroundStyle(SD.Colors.textSecondary)
                }
                if let lower = setting.min, let upper = setting.max, upper > lower {
                    Slider(value: Binding(
                        get: { current },
                        set: { values[setting.key] = .number(PresetSettingsForm.normalized($0, for: setting)) }),
                           in: lower...upper)
                } else {
                    Stepper(value: Binding(
                        get: { current },
                        set: { values[setting.key] = .number(PresetSettingsForm.normalized($0, for: setting)) }),
                            step: setting.step ?? 1) { EmptyView() }
                }
            }
        case .choice:
            HStack {
                Text(setting.title)
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                Spacer()
                Picker(setting.title, selection: Binding(
                    get: { values[setting.key]?.stringValue ?? "" },
                    set: { values[setting.key] = .string($0) })) {
                    ForEach(setting.options) { option in Text(option.title).tag(option.value) }
                }
                .pickerStyle(.menu)
            }
        case .unsupported(let type):
            line(setting.title, "Can't be changed in this version of the app (\(type.isEmpty ? "unknown type" : type))")
        }
    }

    // MARK: Activate

    @ViewBuilder
    private var actionSection: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            if let errorMessage {
                Label(errorMessage, systemImage: "exclamationmark.circle.fill")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }
            switch outcome {
            case .waiting(let text):
                // Pending, said as pending: no tick until the device confirms.
                HStack(spacing: SD.Space.xs) {
                    ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
                    Text(text).font(SD.Typography.callout).foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
                .accessibilityIdentifier("preset.outcome.waiting")
            case .confirmed(let text, let tone):
                Label(text, systemImage: tone == .ok ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                    .font(SD.Typography.callout)
                    .foregroundStyle(tone == .ok ? SD.Colors.success : SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("preset.outcome.confirmed")
            case .notMonitoring(let text, let tone):
                Label(text, systemImage: "exclamationmark.triangle.fill")
                    .font(SD.Typography.callout)
                    .foregroundStyle(tone == .critical ? SD.Colors.danger : SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("preset.outcome.notMonitoring")
            case .refused(let text):
                Label(text, systemImage: "nosign")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("preset.outcome.refused")
            case .unconfirmed(let text):
                Label(text, systemImage: "clock")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.warning)
                    .fixedSize(horizontal: false, vertical: true)
            case nil:
                EmptyView()
            }
            Button { activateTaps += 1; activate() } label: {
                if sending {
                    ProgressView().tint(SD.Colors.textOnAccent)
                } else {
                    Text(isActive ? (preset.settings.isEmpty ? "Already running" : "Save settings")
                                  : "Switch to \(preset.title)")
                }
            }
            .buttonStyle(SDButtonStyle(kind: .primary))
            .disabled(sending || availability.blocksSelection || !problems.isEmpty
                      || (isActive && preset.settings.isEmpty))
            .sensoryFeedback(.impact(weight: .medium), trigger: activateTaps)
            .accessibilityIdentifier("preset.activate")
            if availability.blocksSelection {
                Text("This preset can't be selected on \(device.name).")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
            } else {
                Text("Nothing changes until the device confirms it. Alerts from this preset are only as good as its validation above.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private func activate() {
        sending = true
        errorMessage = nil
        outcome = nil
        // Keep the variant the device runs when only settings change.
        let active = Preset.split(device.activePreset)
        let name = isActive && active.base == preset.id ? (device.activePreset ?? preset.id) : preset.id
        let payload = PresetSettingsForm.payload(preset.settings, values: values)
        let requestedAt = Date()
        Task {
            defer { sending = false }
            do {
                _ = try await state.backend.activatePreset(deviceID: device.id, preset: name, settings: payload)
                outcome = .waiting("Sent. Waiting for \(device.name) to confirm…")
                await onApplied()
                await confirm(payload: payload, since: requestedAt)
            } catch let error as APIError where error.isAuthFailure {
                state.expireSession(error.message)
            } catch let error as APIError {
                errorMessage = error.endpointMissing
                    ? ServerSupport.notSupported
                    : PresetActivationError.message(code: error.code, detail: error.message)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }

    /// Follow the DEVICE until it runs this preset (and these settings) and
    /// says whether it is monitoring, or refuses, for up to
    /// ModeSwitchState.switchTimeout. Running but failed is never "confirmed".
    private func confirm(payload: [String: JSONValue], since: Date) async {
        let deadline = Date().addingTimeInterval(ModeSwitchState.switchTimeout)
        while Date() < deadline {
            try? await Task.sleep(for: .seconds(2))
            guard let fresh = try? await state.backend.devices().first(where: { $0.id == device.id }) else { continue }
            var settingsConfirmed = true
            if !payload.isEmpty {
                let mine = (try? await state.backend.devicePresets(deviceID: device.id))?.state(for: preset.id)
                settingsConfirmed = mine.map { PresetSettingsForm.confirmed(sent: payload, reported: $0.settings) } ?? false
            }
            let status = ActivationStatus.evaluate(
                base: preset.id, deviceName: device.name, title: preset.title, reportedPreset: fresh.activePreset,
                customReplacing: fresh.customRules == .replacing, status: fresh.status, online: fresh.isLive(),
                settingsConfirmed: settingsConfirmed, since: since)
            switch status.phase {
            case .waiting:
                outcome = .waiting(status.text)
                continue
            case .confirmed:
                outcome = .confirmed(status.text, status.tone)
            case .notMonitoring:
                outcome = .notMonitoring(status.text, status.tone)
            case .refused:
                outcome = .refused(status.text)
            }
            await onApplied()
            return
        }
        outcome = .unconfirmed("Saved on the server, but \(device.name) hasn't confirmed yet. It applies the change on its next check-in.")
    }
}
