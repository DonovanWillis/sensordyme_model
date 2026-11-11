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

    private enum Outcome: Equatable {
        case waiting(String), confirmed(String, StatusTone), notMonitoring(String, StatusTone),
             refused(String), unconfirmed(String)
    }

    /// The node the device reports for this preset, when it runs one.
    private var node: MonitoringNode? { device.status?.perception?.node(preset.id) }

    /// Each capability this preset needs, judged from what the DEVICE reports.
    private var checks: [CapabilityCheck] {
        CapabilityChecks.make(required: preset.requiredCapabilities, optional: preset.optionalCapabilities,
                              status: device.status, online: device.online, node: node)
    }

    private var availability: PresetAvailability {
        if deviceState != nil { return PresetAvailability.make(deviceState) }
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
                VStack(alignment: .leading, spacing: SD.Space.md) {
                    header
                    if case .builtIn(let reason) = catalogSource {
                        notice(reason, icon: "info.circle", color: SD.Colors.textTertiary)
                    }
                    if let note = availability.note {
                        notice(note, icon: availability.blocksSelection ? "nosign" : "wifi.slash",
                               color: availability.blocksSelection ? SD.Colors.textSecondary : SD.Colors.warning)
                    }
                    if isActive { monitoringCard }
                    validationCard
                    if !preset.requiredCapabilities.isEmpty || !preset.optionalCapabilities.isEmpty
                        || !preset.hardware.isEmpty {
                        requirementsCard
                    }
                    if !preset.temporalPolicies.isEmpty { policiesCard }
                    if !preset.limitations.isEmpty { limitationsCard }
                    if !preset.settings.isEmpty { settingsCard }
                    actionSection
                }
                .padding(.horizontal, SD.Space.md)
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
    }

    // MARK: Sections

    private var header: some View {
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            SDEyebrow(isActive ? "Running on \(device.name)" : "Preset")
            Text(preset.title)
                .font(SD.Typography.title)
                .foregroundStyle(SD.Colors.textPrimary)
            if !preset.description.isEmpty {
                Text(preset.description)
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(.top, SD.Space.sm)
    }

    private func notice(_ text: String, icon: String, color: Color) -> some View {
        Label(text, systemImage: icon)
            .font(SD.Typography.caption)
            .foregroundStyle(color)
            .fixedSize(horizontal: false, vertical: true)
    }

    private var validationCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                HStack {
                    SDSectionHeader(title: "Validation")
                    SDStatusPill(status: validation.isValidated ? .online : .warning, text: validation.chip)
                }
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
    }

    private var relatedTiers: [CapabilityTierEntry] {
        tiers?.entries(forKinds: preset.requiredCapabilities + preset.optionalCapabilities) ?? []
    }

    /// What the device is doing now with this preset: its own monitoring state.
    private var monitoringCard: some View {
        let presentation = MonitoringStatusPresentation.make(status: device.status, online: device.online)
        let impaired = PerceptionPresentation.degradedSummary(
            device.status?.perception, live: device.status?.isLive(online: device.online) ?? false)
        return SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                HStack {
                    SDSectionHeader(title: "Now")
                    SDStatusPill(status: (presentation?.tone ?? .neutral).sdStatus,
                                 text: presentation?.label ?? "Status unknown")
                        .accessibilityIdentifier("preset.monitoring")
                }
                Text(presentation?.detail ?? "This server doesn't report the device's monitoring state.")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
                if let impaired {
                    Text(impaired)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    /// What it needs, each checked against what THIS device reports: working,
    /// not working (with the device's reason), or unknown. Never assumed.
    private var requirementsCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                SDSectionHeader(title: "What it needs on \(device.name)")
                ForEach(checks) { check in
                    capabilityLine(check)
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
    }

    private func capabilityLine(_ check: CapabilityCheck) -> some View {
        let shown: (text: String, color: Color, icon: String) = {
            switch check.verdict {
            case .available: return ("Working", SD.Colors.success, "checkmark.circle.fill")
            case .unavailable(let why): return ("Not working — \(why)", SD.Colors.danger, "xmark.octagon.fill")
            case .unknown(let why): return ("Unknown — \(why)", SD.Colors.textTertiary, "questionmark.circle")
            }
        }()
        return HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
            Image(systemName: shown.icon)
                .font(.caption)
                .foregroundStyle(shown.color)
            VStack(alignment: .leading, spacing: 1) {
                Text(check.title + (check.required ? "" : " (optional)"))
                    .font(SD.Typography.body)
                    .foregroundStyle(SD.Colors.textPrimary)
                Text(shown.text)
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("preset.capability.\(check.name)")
    }

    private var policiesCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                SDSectionHeader(title: "When it alerts")
                ForEach(Array(preset.temporalPolicies.enumerated()), id: \.offset) { _, policy in
                    let text = policy.description.isEmpty
                        ? policy.event.replacingOccurrences(of: "_", with: " ").capitalized
                        : policy.description
                    line(text, policy.sustainS.map { "held for \(DurationText.short($0))" } ?? "")
                }
            }
        }
    }

    private var limitationsCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                SDSectionHeader(title: "Limitations")
                ForEach(preset.limitations, id: \.self) { limitation in
                    Text("• \(limitation)")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
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
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SDSectionHeader(title: "Settings")
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
                HStack(spacing: SD.Space.xs) {
                    ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
                    Text(text).font(SD.Typography.caption).foregroundStyle(SD.Colors.textSecondary)
                }
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
            Button { activate() } label: {
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
                customReplacing: fresh.customRules == .replacing, status: fresh.status, online: fresh.online,
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
