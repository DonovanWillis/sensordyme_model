import SwiftUI

struct AuthView: View {
    @EnvironmentObject private var state: AppState
    @State private var email = ""
    @State private var password = ""
    @State private var isSignup = false
    @State private var errorMessage: String?
    @State private var busy = false
    @FocusState private var focus: Field?

    private enum Field { case email, password }

    private var canSubmit: Bool {
        !busy && email.contains("@") && password.count >= 8
    }

    var body: some View {
        NavigationStack {
            ZStack {
                SD.Colors.background.ignoresSafeArea()
                AnimatedBrandBackground()

                ScrollView {
                    VStack(spacing: SD.Space.lg) {
                        VStack(spacing: SD.Space.sm) {
                            LogoLockup(markSize: 64)
                            Text("Watchful, wherever it matters.")
                                .font(SD.Typography.callout)
                                .foregroundStyle(SD.Colors.textSecondary)
                        }
                        .padding(.top, SD.Space.xl)

                        // Why the user is suddenly looking at a login screen.
                        if let notice = state.signOutNotice {
                            ConnectionBanner(message: notice, title: "Signed out")
                        }

                        credentialsCard

                        // The address is found on the network, not typed in, so
                        // say so while it happens — otherwise the first seconds
                        // on a new phone look like a frozen screen.
                        if state.isLocatingBackend {
                            Label("Finding your SensorDyme server…", systemImage: "antenna.radiowaves.left.and.right")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        if let errorMessage {
                            Label(errorMessage, systemImage: "exclamationmark.circle.fill")
                                .font(SD.Typography.callout)
                                .foregroundStyle(SD.Colors.danger)
                                .multilineTextAlignment(.leading)
                                .fixedSize(horizontal: false, vertical: true)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }

                        VStack(spacing: SD.Space.sm) {
                            Button(action: submit) {
                                if busy {
                                    ProgressView().tint(SD.Colors.textOnAccent)
                                } else {
                                    Text(isSignup ? "Create account" : "Sign in")
                                }
                            }
                            .buttonStyle(SDButtonStyle(kind: .primary))
                            .disabled(!canSubmit)
                            .opacity(canSubmit ? 1 : 0.55)

                            Button(isSignup ? "I already have an account" : "Create an account") {
                                withAnimation(SD.Motion.standard) {
                                    isSignup.toggle()
                                    errorMessage = nil
                                }
                            }
                            .buttonStyle(SDButtonStyle(kind: .tertiary))
                        }

                        Divider().overlay(SD.Colors.border)

                        VStack(spacing: SD.Space.xs) {
                            Button {
                                state.enterDemo()
                            } label: {
                                Label("Explore the demo device", systemImage: "play.circle")
                            }
                            .buttonStyle(SDButtonStyle(kind: .secondary))

                            Text("A guided walkthrough with simulated footage. Clearly labelled throughout.")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                                .multilineTextAlignment(.center)
                        }

                        NavigationLink("Server settings") { SettingsView() }
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textTertiary)
                            .padding(.top, SD.Space.xxs)
                    }
                    .padding(.horizontal, SD.Space.lg)
                    .padding(.bottom, SD.Space.xl)
                    .frame(maxWidth: 460)
                    .frame(maxWidth: .infinity)
                }
                .scrollDismissesKeyboard(.interactively)
            }
        }
        .tint(SD.Colors.accent)
        // Look for the server while the user is still typing, so signing in is
        // usually instant rather than paying for the search after the tap.
        .task { await state.ensureBackend() }
    }

    private var credentialsCard: some View {
        SDCard(padding: SD.Space.xxs) {
            VStack(spacing: 0) {
                field(
                    "Email", text: $email, field: .email,
                    content: .emailAddress, keyboard: .emailAddress, secure: false
                )
                Divider().overlay(SD.Colors.border).padding(.leading, SD.Space.sm)
                field(
                    "Password", text: $password, field: .password,
                    content: isSignup ? .newPassword : .password, keyboard: .default, secure: true
                )
            }
        }
        .overlay(alignment: .bottomLeading) {
            // Stated up front rather than as a failure after they hit the button.
            if isSignup && !password.isEmpty && password.count < 8 {
                Text("At least 8 characters")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .padding(.leading, SD.Space.xxs)
                    .offset(y: 20)
            }
        }
    }

    @ViewBuilder
    private func field(
        _ label: String, text: Binding<String>, field: Field,
        content: UITextContentType, keyboard: UIKeyboardType, secure: Bool
    ) -> some View {
        HStack(spacing: SD.Space.sm) {
            Image(systemName: secure ? "lock" : "envelope")
                .font(.system(size: 15))
                .foregroundStyle(focus == field ? SD.Colors.accent : SD.Colors.textTertiary)
                .frame(width: 20)
            Group {
                if secure {
                    SecureField(label, text: text)
                } else {
                    TextField(label, text: text)
                }
            }
            .font(SD.Typography.body)
            .textContentType(content)
            .keyboardType(keyboard)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .focused($focus, equals: field)
            .submitLabel(secure ? .go : .next)
            .onSubmit {
                if secure { submit() } else { focus = .password }
            }
        }
        .padding(SD.Space.sm)
        .animation(SD.Motion.gentle, value: focus)
    }

    private func submit() {
        guard canSubmit else { return }
        busy = true
        errorMessage = nil
        focus = nil
        Task {
            defer { busy = false }
            do {
                let auth = isSignup
                    ? try await state.api.signup(email: email, password: password)
                    : try await state.api.login(email: email, password: password)
                state.signIn(auth)
            } catch {
                errorMessage = error.localizedDescription
            }
        }
    }
}
