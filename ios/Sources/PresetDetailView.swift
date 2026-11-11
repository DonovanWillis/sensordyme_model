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

    private enum Outcome: Equatable {
        case waiting(String), confirmed(String), unconfirmed(String)
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
            }
        }
    }

    private var requirementsCard: some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                SDSectionHeader(title: "Needs")
                if !preset.requiredCapabilities.isEmpty {
                    line("Required", preset.requiredCapabilities.map(CapabilityHealth.title).joined(separator: ", "))
                }
                if !preset.optionalCapabilities.isEmpty {
                    line("Uses if available", preset.optionalCapabilities.map(CapabilityHealth.title).joined(separator: ", "))
                }
                if !preset.hardware.isEmpty {
                    line("Hardware", preset.hardware.joined(separator: ", "))
                }
            }
        }
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
            case .confirmed(let text):
                Label(text, systemImage: "checkmark.circle.fill")
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.success)
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
        Task {
            defer { sending = false }
            do {
                _ = try await state.backend.activatePreset(deviceID: device.id, preset: name, settings: payload)
                outcome = .waiting("Sent. Waiting for \(device.name) to confirm…")
                await onApplied()
                await confirm(payload: payload)
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

    /// Poll the device's own answer until it reports this preset (and these
    /// settings) as active, for up to ModeSwitchState.switchTimeout.
    private func confirm(payload: [String: JSONValue]) async {
        let deadline = Date().addingTimeInterval(ModeSwitchState.switchTimeout)
        while Date() < deadline {
            try? await Task.sleep(for: .seconds(2))
            let confirmed: Bool
            if let fresh = try? await state.backend.devicePresets(deviceID: device.id),
               let mine = fresh.state(for: preset.id) {
                confirmed = mine.active
                    && (payload.isEmpty || PresetSettingsForm.confirmed(sent: payload, reported: mine.settings))
            } else if payload.isEmpty,
                      let fresh = try? await state.backend.devices().first(where: { $0.id == device.id }) {
                // An older server: the device list's own convergence.
                confirmed = fresh.desiredPreset == nil && Preset.split(fresh.activePreset).base == preset.id
            } else {
                continue
            }
            if confirmed {
                outcome = .confirmed("\(device.name) is running \(preset.title)\(payload.isEmpty ? "" : " with these settings").")
                await onApplied()
                return
            }
        }
        outcome = .unconfirmed("Saved on the server, but \(device.name) hasn't confirmed yet. It applies the change on its next check-in.")
    }
}
