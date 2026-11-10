import SwiftUI

/// Sign in — the first screen of ORYN EDGE, and the one that establishes the
/// brand: the ORYN lockup on ink, two fields, one action.
///
/// Authentication is unchanged: the same `/v1/login` and `/v1/signup` calls, the
/// same validation (an email, 8+ character password), the same session handling.
/// There is deliberately no "Forgot password" — the backend has no reset
/// endpoint, and a link to nothing is worse than no link. "Remember me" is not
/// offered either: a session already persists in the Keychain until sign-out.
struct AuthView: View {
    @EnvironmentObject private var state: AppState
    @State private var email = ""
    @State private var password = ""
    @State private var isSignup = false
    @State private var revealPassword = false
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
                AnimatedBrandBackground()

                ScrollView {
                    VStack(spacing: 0) {
                        OrynEdgeLockup(markHeight: 54)
                            .padding(.top, 64)
                            .padding(.bottom, 56)

                        // Why the user is suddenly looking at a login screen.
                        if let notice = state.signOutNotice {
                            ConnectionBanner(message: notice, title: "Signed out")
                                .padding(.bottom, SD.Space.lg)
                        }

                        form

                        alternatives
                            .padding(.top, SD.Space.xl)

                        footer
                            .padding(.top, SD.Space.xxl)
                    }
                    .padding(.horizontal, SD.Space.lg)
                    .padding(.bottom, SD.Space.xl)
                    .frame(maxWidth: 440)
                    .frame(maxWidth: .infinity)
                }
                .scrollDismissesKeyboard(.interactively)
                .scrollIndicators(.hidden)
            }
            .toolbar(.hidden, for: .navigationBar)
        }
        .tint(SD.Colors.accent)
        // No search kicked off here any more. The server is already being looked
        // for before this screen appears — `startBackendMonitoring()` runs on the
        // root view and keeps resolving on its own schedule — so asking again from
        // here was a second launch-time resolve that raced the first rather than
        // joining it, and a pinned address that does not answer got probed twice
        // before discovery ever ran.
    }

    // MARK: Form

    private var form: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            HStack(alignment: .firstTextBaseline) {
                Text(isSignup ? "Create your account" : "Sign in")
                    .font(SD.Typography.title)
                    .foregroundStyle(SD.Colors.textPrimary)
                Spacer()
                SDEyebrow(isSignup ? "New" : "Welcome back")
            }
            .padding(.bottom, SD.Space.xs)

            OrynFieldChrome(icon: "envelope", focused: focus == .email) {
                TextField("", text: $email, prompt: orynPrompt("Email"))
                    .textContentType(.emailAddress)
                    .keyboardType(.emailAddress)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .focused($focus, equals: .email)
                    .submitLabel(.next)
                    .onSubmit { focus = .password }
            }

            OrynFieldChrome(icon: "lock", focused: focus == .password) {
                HStack(spacing: SD.Space.xs) {
                    Group {
                        if revealPassword {
                            TextField("", text: $password, prompt: orynPrompt("Password"))
                        } else {
                            SecureField("", text: $password, prompt: orynPrompt("Password"))
                        }
                    }
                    .textContentType(isSignup ? .newPassword : .password)
                    .textInputAutocapitalization(.never)
                    .autocorrectionDisabled()
                    .focused($focus, equals: .password)
                    .submitLabel(.go)
                    .onSubmit(submit)

                    Button {
                        let wasFocused = focus == .password
                        revealPassword.toggle()
                        // Swapping SecureField for TextField drops focus; keep it.
                        if wasFocused { focus = .password }
                    } label: {
                        Image(systemName: revealPassword ? "eye.slash" : "eye")
                            .font(.system(size: 14))
                            .foregroundStyle(SD.Colors.textTertiary)
                            .frame(width: 32, height: 32)
                            .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel(revealPassword ? "Hide password" : "Show password")
                }
            }

            // Stated up front rather than as a failure after they hit the button.
            if isSignup && !password.isEmpty && password.count < 8 {
                Text("At least 8 characters")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .padding(.leading, SD.Space.xxs)
            }

            if let errorMessage {
                HStack(alignment: .firstTextBaseline, spacing: SD.Space.xs) {
                    Image(systemName: "exclamationmark.circle")
                    Text(errorMessage)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .font(SD.Typography.callout)
                .foregroundStyle(SD.Colors.danger)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, SD.Space.xxs)
                .transition(.opacity)
            }

            Button(action: submit) {
                ZStack {
                    if busy {
                        ProgressView().tint(SD.Colors.textOnAccent)
                    } else {
                        Text(isSignup ? "Create account" : "Sign in")
                    }
                }
            }
            .buttonStyle(SDButtonStyle(kind: .primary))
            .disabled(!canSubmit)
            .padding(.top, SD.Space.sm)

            // The address is found on the network, not typed in, so say so while
            // it happens — otherwise the first seconds on a new phone look like
            // a frozen screen.
            if state.isLocatingBackend {
                HStack(spacing: SD.Space.xs) {
                    ProgressView().controlSize(.mini).tint(SD.Colors.textTertiary)
                    Text("Finding your server on this network…")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
                .frame(maxWidth: .infinity)
                .padding(.top, SD.Space.xs)
            }
        }
        .animation(SD.Motion.standard, value: errorMessage)
        .animation(SD.Motion.standard, value: isSignup)
    }

    // MARK: Alternatives

    private var alternatives: some View {
        VStack(spacing: SD.Space.md) {
            Button {
                withAnimation(SD.Motion.standard) {
                    isSignup.toggle()
                    errorMessage = nil
                }
            } label: {
                HStack(spacing: 4) {
                    Text(isSignup ? "Already have an account?" : "New to ORYN EDGE?")
                        .foregroundStyle(SD.Colors.textTertiary)
                    Text(isSignup ? "Sign in" : "Create an account")
                        .foregroundStyle(SD.Colors.textPrimary)
                }
                .font(SD.Typography.callout)
            }
            .buttonStyle(.plain)

            HStack(spacing: SD.Space.sm) {
                Rectangle().fill(SD.Colors.border).frame(height: 1)
                SDEyebrow("or", color: SD.Colors.textDisabled)
                Rectangle().fill(SD.Colors.border).frame(height: 1)
            }

            Button {
                state.enterDemo()
            } label: {
                Label("Explore the demo device", systemImage: "play.circle")
            }
            .buttonStyle(SDButtonStyle(kind: .secondary))

            Text("Simulated footage, clearly labelled throughout.")
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
        }
    }

    private var footer: some View {
        VStack(spacing: SD.Space.sm) {
            NavigationLink {
                SettingsView()
            } label: {
                Label("Server settings", systemImage: "server.rack")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
            }
            Text("ORYN · SENSORDYME")
                .font(SD.Typography.eyebrow)
                .kerning(SD.Typography.eyebrowKerning)
                .foregroundStyle(SD.Colors.textDisabled)
        }
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
