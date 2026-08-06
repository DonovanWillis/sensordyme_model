import SwiftUI

struct AuthView: View {
    @EnvironmentObject private var state: AppState
    @State private var email = ""
    @State private var password = ""
    @State private var isSignup = false
    @State private var errorMessage: String?
    @State private var busy = false

    var body: some View {
        NavigationStack {
            ZStack {
                AnimatedBrandBackground().opacity(0.25)

                VStack(spacing: 22) {
                    LogoLockup(markSize: 76)
                        .padding(.top, 44)

                    Text("One device. Every kind of watchfulness.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    VStack(spacing: 12) {
                        TextField("Email", text: $email)
                            .textContentType(.emailAddress)
                            .keyboardType(.emailAddress)
                            .autocapitalization(.none)
                            .autocorrectionDisabled()
                        SecureField("Password (8+ characters)", text: $password)
                            .textContentType(isSignup ? .newPassword : .password)
                    }
                    .textFieldStyle(.plain)
                    .padding(14)
                    .background(RoundedRectangle(cornerRadius: 16).fill(.background.secondary))
                    .overlay(RoundedRectangle(cornerRadius: 16).strokeBorder(.quaternary))

                    if let errorMessage {
                        Text(errorMessage)
                            .font(.callout)
                            .foregroundStyle(.red)
                            .multilineTextAlignment(.center)
                    }

                    Button(action: submit) {
                        if busy {
                            ProgressView().tint(.white)
                        } else {
                            Text(isSignup ? "Create Account" : "Log In")
                        }
                    }
                    .buttonStyle(BrandButtonStyle())
                    .disabled(busy || email.isEmpty || password.isEmpty)

                    Button(isSignup ? "Have an account? Log in" : "New here? Create an account") {
                        isSignup.toggle()
                        errorMessage = nil
                    }
                    .font(.callout)

                    Spacer()

                    Button {
                        state.enterDemo()
                    } label: {
                        Label("Try the demo device", systemImage: "sparkles")
                    }
                    .buttonStyle(BrandButtonStyle(prominent: false))

                    NavigationLink("Server settings") { SettingsView() }
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 8)
                }
                .padding(24)
            }
        }
    }

    private func submit() {
        busy = true
        errorMessage = nil
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
