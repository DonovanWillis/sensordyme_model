import SwiftUI

/// Setting up alerts that arrive when SensorDyme is closed.
///
/// The distinction this screen has to make clear, because it is the one people
/// get wrong: the app's own notifications are posted BY the app, so they need
/// it running. Close it and they stop. A push is delivered TO the phone by the
/// OS whether the app is open, backgrounded, force-quit, or the phone locked —
/// which is the only behaviour that makes a safety product a safety product.
///
/// Delivered through ntfy today because it is free and needs no Apple Developer
/// membership. The trade is honest and stated below: the alert is branded ntfy
/// and opens the ntfy app. APNs replaces this transport without changing
/// anything about what gets sent — that decision lives in cloud/notify.py.
struct BackgroundAlertsSection: View {
    @EnvironmentObject private var state: AppState
    @Binding var topic: String

    @State private var saving = false
    @State private var problem: String?
    /// The channel this phone last saved to the server, from the Keychain.
    /// The server has no GET /v1/account, so this is the only record the app
    /// can show; the screen says so.
    @State private var savedTopic = Keychain.get(BackgroundAlertsSection.savedTopicKey) ?? ""
    @StateObject private var progress = ProgressTracker()

    /// Keychain key for the saved channel: it is a credential (see freshTopic).
    static let savedTopicKey = "ntfyTopic"

    private var saved: Bool { !topic.isEmpty && topic == savedTopic }

    var body: some View {
        // Headed by what it is, not by the page's own title said again (M8.4).
        Section("Push alerts through ntfy") {
            VStack(alignment: .leading, spacing: SD.Space.xs) {
                Label(
                    "ORYN EDGE's own notifications only arrive while the app is open. These reach your phone even when it's closed, as long as your SensorDyme is on and online and the ntfy app is subscribed to your channel.",
                    systemImage: "bell.badge"
                )
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
            }
            .padding(.vertical, SD.Space.xxs)

            if topic.isEmpty {
                Button {
                    // Generated, not typed. An ntfy topic is a shared secret in
                    // the URL — anyone who knows it can read your alerts and
                    // publish fake ones — so a memorable topic is an insecure
                    // one, and asking someone to invent it guarantees a weak
                    // choice.
                    topic = Self.freshTopic()
                } label: {
                    Label("Set up background alerts", systemImage: "wand.and.stars")
                }
                .buttonStyle(SDButtonStyle(kind: .primary))
            } else {
                VStack(alignment: .leading, spacing: SD.Space.xs) {
                    Text("Your private alert channel")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                    HStack {
                        Text(topic)
                            .font(SD.Typography.caption.monospaced())
                            .foregroundStyle(SD.Colors.textPrimary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Spacer()
                        Button {
                            UIPasteboard.general.string = topic
                        } label: {
                            Image(systemName: "doc.on.doc")
                        }
                        .buttonStyle(SDRowButtonStyle(minHeight: 44))
                        .fixedSize()
                        .accessibilityLabel("Copy channel name")
                    }

                    steps

                    HStack(spacing: SD.Space.xs) {
                        Button {
                            Task { await save() }
                        } label: {
                            HStack(spacing: SD.Space.xs) {
                                Text(saved ? "Saved" : "Save")
                                SDProgress(phase: progress.phase)
                            }
                        }
                        .buttonStyle(SDButtonStyle(kind: .primary, fullWidth: false))
                        .disabled(saving)

                        Button("New channel") { topic = Self.freshTopic() }
                            .buttonStyle(SDButtonStyle(kind: .tertiary, fullWidth: false))
                    }

                    // There is no way to read the account's settings back from
                    // this server, so say what this line is actually based on.
                    Text(saved
                         ? "Saved to your account from this phone. This server can't send saved settings back, so another phone won't show it here."
                         : (savedTopic.isEmpty
                            ? "Not saved yet. Alerts won't arrive until you tap Save."
                            : "Not saved yet — the server still uses the channel this phone saved before."))
                        .font(SD.Typography.caption)
                        .foregroundStyle(saved ? SD.Colors.textTertiary : SD.Colors.warning)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }

            if let problem {
                Label(problem, systemImage: "exclamationmark.circle")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.danger)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Text("Alerts go out through ntfy.sh, which is free and needs no Apple account. They arrive branded \"ntfy\" — a native ORYN EDGE push needs a paid Apple Developer membership, and swapping to it changes nothing about which alerts are sent.")
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var steps: some View {
        VStack(alignment: .leading, spacing: SD.Space.xxs) {
            stepRow(1, "Install the free ntfy app from the App Store")
            stepRow(2, "In ntfy, tap + and subscribe to the channel above")
            stepRow(3, "Save here — that's it")
        }
        .padding(.vertical, SD.Space.xxs)
    }

    private func stepRow(_ number: Int, _ text: String) -> some View {
        HStack(alignment: .top, spacing: SD.Space.xs) {
            Image(systemName: "\(number).circle.fill")
                .font(.caption)
                .foregroundStyle(SD.Colors.accent)
            Text(text)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private func save() async {
        saving = true
        problem = nil
        defer { saving = false }
        do {
            try await progress.track { try await state.backend.setNtfyTopic(topic) }
            // Kept so the next visit shows this channel instead of generating
            // a new one (which used to happen on every visit to Settings).
            Keychain.set(Self.savedTopicKey, topic)
            savedTopic = topic
        } catch {
            problem = error.localizedDescription
        }
    }

    /// A long random channel name.
    ///
    /// ntfy has no accounts: the topic IS the credential, it travels in a URL,
    /// and anyone who knows it can both read your alerts and publish fake ones.
    /// So it must be unguessable, which means it must not be memorable.
    static func freshTopic() -> String {
        let alphabet = Array("abcdefghijklmnopqrstuvwxyz0123456789")
        let random = (0..<24).map { _ in alphabet[Int.random(in: 0..<alphabet.count)] }
        return "sensordyme-" + String(random)
    }
}
