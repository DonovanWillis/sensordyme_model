import SwiftUI

// MARK: - Live feed

/// The picture on this Wi-Fi: which device the app streams from, the token the
/// device requires, and the device's own health report. Everything shown here
/// is what the device (or the network search) actually answered.
struct LiveFeedSettingsView: View {
    @EnvironmentObject private var state: AppState
    @StateObject private var discovery = DeviceDiscovery()
    /// Local draft so the field is editable without writing the Keychain on every keystroke.
    @State private var tokenDraft = ""
    @State private var health: DeviceHealth?
    @State private var healthCheckedAt: Date?
    @State private var healthError: String?
    @State private var checkingHealth = false

    var body: some View {
        Form {
            Group {
                Section {
                    if !state.deviceLocalURL.isEmpty {
                        LabeledContent("Streaming from") {
                            Text(state.deviceLocalURL)
                                .font(SD.Typography.callout.monospaced())
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        .accessibilityElement(children: .combine)
                    }
                    Button {
                        discovery.start()
                    } label: {
                        Label(discovery.isSearching ? "Looking for your device…" : "Find my device",
                              systemImage: "video.badge.waveform")
                    }
                    .disabled(discovery.isSearching)
                    .accessibilityIdentifier("settings.findDevice")
                    ForEach(discovery.found) { device in
                        Button {
                            state.deviceLocalURL = device.url
                            // The id too: the address alone left the stored id
                            // naming a different camera than the one streaming.
                            state.linkedDeviceID = device.id
                            discovery.stop()
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(device.name)
                                    Text(device.url)
                                        .font(SD.Typography.caption)
                                        .foregroundStyle(SD.Colors.textSecondary)
                                }
                                Spacer()
                                if state.deviceLocalURL == device.url {
                                    Image(systemName: "checkmark")
                                        .foregroundStyle(SD.Colors.textPrimary)
                                        .accessibilityLabel("In use")
                                }
                            }
                        }
                        .accessibilityAddTraits(state.deviceLocalURL == device.url ? .isSelected : [])
                    }
                    if let problem = discovery.problem {
                        Label(problem, systemImage: "exclamationmark.triangle.fill")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                } header: {
                    Text("This Wi-Fi")
                } footer: {
                    Text("The device publishes itself on the Wi-Fi it is joined to, so the phone has to be on that same network. Tap a device found to stream from it.")
                }

                // The feed and /api on the device both require this token.
                Section {
                    SecureField("Device token", text: $tokenDraft)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()
                        .onSubmit { state.deviceToken = tokenDraft }
                        .accessibilityIdentifier("settings.deviceToken")
                    if tokenDraft != state.deviceToken {
                        Button("Save token") { state.deviceToken = tokenDraft }
                            .accessibilityIdentifier("settings.saveToken")
                    }
                } header: {
                    Text("Device token")
                } footer: {
                    Text(state.deviceToken.isEmpty
                         ? "Set dashboard.token in the device's config.yaml, then paste the same value here. Without it the device refuses the live feed rather than showing its camera to everyone on the Wi-Fi."
                         : (tokenDraft != state.deviceToken
                            ? "Not saved yet. The saved token is still in use."
                            : "Saved to this phone's keychain. It must match dashboard.token in the device's config.yaml."))
                }

                Section {
                    if let health {
                        LabeledContent("Overall") {
                            Label(health.state == "ok" ? "Working" : health.state.capitalized,
                                  systemImage: health.state == "ok" ? "checkmark.circle.fill" : "exclamationmark.triangle.fill")
                                .foregroundStyle(health.state == "ok" ? SD.Colors.success : SD.Colors.warning)
                        }
                        .accessibilityElement(children: .combine)
                        if let id = health.deviceID {
                            LabeledContent("Device", value: id)
                        }
                        LabeledContent("Up for", value: Self.uptimeLabel(health.uptime_s))
                        ForEach(health.components.sorted(by: { $0.key < $1.key }), id: \.key) { name, component in
                            if component.state != "ok" || component.critical {
                                LabeledContent(name) {
                                    Text(component.state == "ok" ? "Working" : component.detail)
                                        .font(SD.Typography.caption)
                                        .foregroundStyle(component.state == "ok" ? SD.Colors.textSecondary : SD.Colors.warning)
                                        .fixedSize(horizontal: false, vertical: true)
                                }
                                .accessibilityElement(children: .combine)
                            }
                        }
                    } else if let healthError {
                        Label(healthError, systemImage: "exclamationmark.circle")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.danger)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Button {
                        Task { await checkHealth() }
                    } label: {
                        Label(checkingHealth ? "Asking the device…" : (health == nil ? "Check device health" : "Check again"),
                              systemImage: "waveform.path.ecg")
                    }
                    .disabled(state.deviceLocalURL.isEmpty || checkingHealth)
                    .accessibilityIdentifier("settings.checkHealth")
                } header: {
                    Text("Device health")
                } footer: {
                    if state.deviceLocalURL.isEmpty {
                        Text("Find your device on this Wi-Fi first: its health comes from the device itself.")
                    } else if let healthCheckedAt {
                        Text("As the device reported it at \(healthCheckedAt.formatted(date: .omitted, time: .shortened)).")
                    } else {
                        Text("Asks the device directly for its own health report.")
                    }
                }
            }
            .orynRows()
        }
        .orynFormStyle()
        .navigationTitle("Live feed")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { tokenDraft = state.deviceToken }
        .onDisappear { discovery.stop() }
    }

    private func checkHealth() async {
        checkingHealth = true
        defer { checkingHealth = false }
        do {
            health = try await DeviceHealth.fetch(baseURL: state.deviceLocalURL, token: state.deviceToken)
            healthCheckedAt = Date()
            healthError = nil
        } catch {
            health = nil
            healthCheckedAt = nil
            healthError = error.localizedDescription
        }
    }

    static func uptimeLabel(_ seconds: Double) -> String {
        let total = Int(seconds)
        if total < 3600 { return "\(total / 60) min" }
        if total < 86400 { return "\(total / 3600) h \(total % 3600 / 60) min" }
        return "\(total / 86400) d \(total % 86400 / 3600) h"
    }
}

// MARK: - Alerts when the app is closed

struct BackgroundAlertsSettingsView: View {
    @State private var topic = ""

    var body: some View {
        Form {
            BackgroundAlertsSection(topic: $topic)
                .orynRows()
        }
        .orynFormStyle()
        .navigationTitle("When the app is closed")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            // The channel this phone saved last time, not a new one: an empty
            // field used to mean a fresh random channel on every visit,
            // silently replacing the one the person subscribed to.
            if topic.isEmpty { topic = Keychain.get(BackgroundAlertsSection.savedTopicKey) ?? "" }
        }
    }
}

// MARK: - Text messages

/// The account's text-message alerts: one number and one switch, saved
/// together (`POST /v1/account/phone`).
///
/// The switch used to be its own AppStorage toggle that looked live and did
/// nothing until "Save phone number", and a saved number could not be removed.
/// Now the form is a draft against what this phone last saved: Save is offered
/// only when it would change something the server accepts, the screen says
/// when there are unsaved changes, and a saved number can be removed.
struct TextAlertsSettingsView: View {
    @EnvironmentObject private var state: AppState
    /// What this phone last saved to the account (the server can't send it back).
    @AppStorage("alertPhone") private var savedPhone = ""
    @AppStorage("smsOnLoud") private var savedTextOnSerious = true
    @State private var phone = ""
    @State private var textOnSerious = true
    @State private var saving = false
    @State private var message: (text: String, ok: Bool)?

    private var canSave: Bool {
        TextAlertsForm.canSave(phone: phone, textOnSerious: textOnSerious,
                               savedPhone: savedPhone, savedTextOnSerious: savedTextOnSerious)
    }

    private var dirty: Bool {
        TextAlertsForm.normalized(phone) != TextAlertsForm.normalized(savedPhone) || textOnSerious != savedTextOnSerious
    }

    var body: some View {
        Form {
            Group {
                Section {
                    TextField("+1 555 123 4567", text: $phone)
                        .keyboardType(.phonePad)
                        .textContentType(.telephoneNumber)
                        .accessibilityLabel("Phone number")
                        .accessibilityIdentifier("settings.phone")
                    if let problem = TextAlertsForm.problem(phone) {
                        Label(problem, systemImage: "exclamationmark.circle")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    Toggle("Text me on serious alerts", isOn: $textOnSerious)
                        .accessibilityIdentifier("settings.textOnSerious")
                } header: {
                    Text("Phone number")
                } footer: {
                    Text("On a serious alert (distracted driving, a fall, a covered face, an intruder) the server texts this number, if SMS is set up on the server. The number and the switch are saved together.")
                }

                Section {
                    Button {
                        Task { await save(phone: TextAlertsForm.normalized(phone)) }
                    } label: {
                        Label(saving ? "Saving…" : "Save", systemImage: "checkmark.circle")
                            // Dimmed when there is nothing to save: a Form row
                            // otherwise draws a disabled button like a live one.
                            .foregroundStyle(canSave ? SD.Colors.accent : SD.Colors.textDisabled)
                    }
                    .disabled(saving || !canSave)
                    .accessibilityIdentifier("settings.savePhone")
                    if !savedPhone.isEmpty {
                        Button("Remove number", role: .destructive) {
                            Task { await save(phone: "") }
                        }
                        .disabled(saving)
                        .accessibilityIdentifier("settings.removePhone")
                    }
                } footer: {
                    if let message, !dirty || !message.ok {
                        Text(message.text)
                            .foregroundStyle(message.ok ? SD.Colors.textSecondary : SD.Colors.danger)
                    } else if dirty && !(phone.isEmpty && savedPhone.isEmpty) {
                        Text("Not saved yet.")
                            .foregroundStyle(SD.Colors.warning)
                    } else if savedPhone.isEmpty {
                        Text("No number saved from this phone.")
                    } else {
                        // The server can't show saved settings back (no GET).
                        Text("Saved from this phone. The server can't send saved settings back, so another phone won't show them here.")
                    }
                }
            }
            .orynRows()
        }
        .orynFormStyle()
        .navigationTitle("Text messages")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            phone = savedPhone
            textOnSerious = savedTextOnSerious
        }
    }

    private func save(phone number: String) async {
        saving = true
        message = nil
        defer { saving = false }
        do {
            try await state.backend.setPhone(number, smsOnLoud: textOnSerious)
            savedPhone = number
            savedTextOnSerious = textOnSerious
            if number.isEmpty { phone = "" }
            message = (number.isEmpty ? "Number removed from your account." : "Saved to your account.", true)
        } catch {
            message = (error.localizedDescription, false)
        }
    }
}
