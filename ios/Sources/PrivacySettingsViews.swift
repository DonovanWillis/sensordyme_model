import SwiftUI

// MARK: - Delete account (M8.4)

/// Deleting the account: what is deleted, what is released, what is NOT
/// (docs/PRIVACY.md §5), the password again, and a destructive confirmation.
/// On success the phone forgets everything it kept for the account and signs out.
struct AccountDeletionView: View {
    @EnvironmentObject private var state: AppState
    @State private var password = ""
    @State private var deleting = false
    @State private var confirm = false
    @State private var problem: String?
    @FocusState private var passwordFocused: Bool

    var body: some View {
        Form {
            Group {
                Section {
                    bullets(AccountDeletion.deleted, symbol: "trash")
                } header: {
                    Text("Deleted from the server")
                }
                Section {
                    bullets(AccountDeletion.released, symbol: "arrow.uturn.left")
                } header: {
                    Text("Released")
                }
                Section {
                    bullets(AccountDeletion.notDeleted, symbol: "exclamationmark.triangle")
                } header: {
                    Text("Not deleted")
                } footer: {
                    Text("To remove what a device keeps, reset it before you give it away.")
                }
                Section {
                    SecureField("Password", text: $password)
                        .textContentType(.password)
                        .focused($passwordFocused)
                        .accessibilityIdentifier("settings.delete.password")
                    if let problem {
                        Label(problem, systemImage: "exclamationmark.circle")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.danger)
                            .fixedSize(horizontal: false, vertical: true)
                            .accessibilityIdentifier("settings.delete.problem")
                    }
                    Button(role: .destructive) {
                        passwordFocused = false
                        confirm = true
                    } label: {
                        if deleting {
                            HStack { ProgressView(); Text("Deleting…") }
                        } else {
                            Text("Delete account")
                        }
                    }
                    .disabled(!AccountDeletion.canSubmit(password: password, deleting: deleting))
                    .accessibilityIdentifier("settings.delete.submit")
                } header: {
                    Text("Confirm with your password")
                } footer: {
                    Text("Signed in as \(state.accountEmail).")
                }
            }
            .orynRows()
        }
        .orynFormStyle()
        .navigationTitle("Delete account")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog(AccountDeletion.confirmTitle, isPresented: $confirm, titleVisibility: .visible) {
            Button(AccountDeletion.confirmButton, role: .destructive) { Task { await delete() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(AccountDeletion.confirmMessage)
        }
    }

    private func bullets(_ items: [String], symbol: String) -> some View {
        ForEach(items, id: \.self) { item in
            Label {
                Text(item)
                    .font(SD.Typography.callout)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: symbol)
                    .foregroundStyle(SD.Colors.textTertiary)
            }
        }
    }

    private func delete() async {
        guard AccountDeletion.canSubmit(password: password, deleting: deleting) else { return }
        deleting = true
        problem = nil
        defer { deleting = false }
        do {
            let result = try await state.backend.deleteAccount(password: password)
            password = ""
            state.forgetDeletedAccount(notice: AccountDeletion.summary(
                devicesReleased: result.devicesReleased, eventsDeleted: result.eventsDeleted))
        } catch let error as APIError {
            problem = AccountDeletion.message(status: error.status, code: error.code, detail: error.message)
        } catch {
            problem = AccountDeletion.message(status: 0, code: nil, detail: error.localizedDescription)
        }
    }
}

// MARK: - Camera while monitoring is stopped (M8.4, owner Decision C)

/// The account's devices, each with its "Camera only while stopped" switch.
/// Read from the account (GET /v1/devices), so it is the device's own report.
struct CameraModeSection: View {
    @EnvironmentObject private var state: AppState
    @State private var devices: [Device] = []
    @State private var loadProblem: String?
    @State private var loaded = false

    var body: some View {
        Section {
            if !loaded {
                HStack { ProgressView().controlSize(.small); Text("Asking your devices…") }
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
            } else if let loadProblem {
                Label(loadProblem, systemImage: "exclamationmark.circle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
            } else if devices.isEmpty {
                Text("No devices on this account.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
            }
            ForEach(devices) { device in
                CameraModeRow(device: device) { await load() }
            }
        } header: {
            Text("Camera while monitoring is stopped")
        } footer: {
            Text("Off by default: while monitoring is stopped, the app shows no camera picture. Turned on, you can still watch the live camera then; nothing is monitored and no alerts are raised. The switch shows what the device reports.")
        }
        .task { await load() }
    }

    private func load() async {
        do {
            devices = try await state.backend.devices()
            loadProblem = nil
        } catch {
            loadProblem = error.localizedDescription
        }
        loaded = true
    }
}

/// One device's switch. It shows what the DEVICE reports (`status.camera_mode`),
/// never the request; turning it on is confirmed first, and the line under it
/// says when a change is on its way.
struct CameraModeRow: View {
    @EnvironmentObject private var state: AppState
    let device: Device
    /// Re-read the account's devices (the device's report) after a change.
    var reload: () async -> Void

    @State private var sending = false
    @State private var askToConfirm = false
    @State private var note: String?
    @State private var refusal: CameraModeControl.Refusal?

    private var presentation: CameraModePresentation { CameraModePresentation.resolve(device.status?.cameraMode) }

    var body: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            if presentation.canChange && refusal?.disablesSwitch != true {
                Toggle(isOn: Binding(
                    get: { CameraModeControl.switchOn(presentation) },
                    set: { handle(CameraModeControl.toggled(to: $0, presentation: presentation, sending: sending)) }
                )) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("Camera only while stopped")
                            .font(SD.Typography.body)
                        Text(device.name)
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                    }
                }
                .disabled(!CameraModeControl.switchEnabled(presentation, sending: sending))
                .accessibilityIdentifier("settings.cameraMode.\(device.id)")
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Camera only while stopped")
                        .font(SD.Typography.body)
                    Text(device.name)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
            }
            Label(presentation.title, systemImage: presentation.kind == .cameraOnly ? "video" : "video.slash")
                .font(SD.Typography.caption.weight(.semibold))
                .foregroundStyle(presentation.kind == .cameraOnly ? SD.Colors.warning : SD.Colors.textSecondary)
                .accessibilityIdentifier("settings.cameraMode.state")
            Text(note ?? presentation.detail)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if let refusal {
                Label(refusal.message, systemImage: "exclamationmark.circle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityIdentifier("settings.cameraMode.problem")
            }
        }
        .padding(.vertical, SD.Space.xxs)
        .confirmationDialog(CameraModeControl.confirmTitle, isPresented: $askToConfirm, titleVisibility: .visible) {
            Button(CameraModeControl.confirmButton) { handle(CameraModeControl.confirmedOn) }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text(CameraModeControl.confirmMessage)
        }
        .onChange(of: presentation) { _, now in
            // The device's report caught up with what was sent: drop the note.
            if !now.pending { note = nil }
        }
    }

    private func handle(_ action: CameraModeControl.Action) {
        switch action {
        case .none:
            return
        case .askToConfirm:
            askToConfirm = true
        case .send(let cameraOnly, let confirm):
            sending = true
            refusal = nil
            Task {
                defer { sending = false }
                do {
                    try await state.backend.setCameraMode(deviceID: device.id, cameraOnly: cameraOnly, confirm: confirm)
                    note = CameraModeControl.sentNote(cameraOnly: cameraOnly)
                    await reload()
                    // The device answers on its next poll: look again a few times.
                    for _ in 0..<5 {
                        try? await Task.sleep(for: .seconds(3))
                        await reload()
                    }
                } catch let error as APIError {
                    refusal = CameraModeControl.refusal(status: error.status, code: error.code, message: error.message)
                } catch {
                    refusal = CameraModeControl.refusal(status: 0, code: nil, message: error.localizedDescription)
                }
            }
        }
    }
}
