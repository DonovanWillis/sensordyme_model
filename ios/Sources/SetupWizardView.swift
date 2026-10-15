import SwiftUI

// MARK: - Pairing wizard: plug in -> join hotspot -> Wi-Fi -> claim. Four taps to a live device.

struct SetupWizardView: View {
    @EnvironmentObject private var state: AppState
    @Environment(\.dismiss) private var dismiss
    let onPaired: () async -> Void

    enum Step: Int, CaseIterable {
        case power, hotspot, wifi, pair, done
    }

    @State private var step: Step = .power
    @State private var claimCode = ""
    @State private var busy = false
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            VStack(spacing: 26) {
                progressBar

                Spacer()

                Group {
                    switch step {
                    case .power: powerStep
                    case .hotspot: hotspotStep
                    case .wifi: wifiStep
                    case .pair: pairStep
                    case .done: doneStep
                    }
                }
                .transition(.asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity),
                                        removal: .move(edge: .leading).combined(with: .opacity)))

                Spacer()

                if step != .done {
                    Button(action: advance) {
                        if busy { ProgressView().tint(.white) } else { Text(buttonTitle) }
                    }
                    .buttonStyle(BrandButtonStyle())
                    .disabled(busy || (step == .pair && claimCode.isEmpty && !state.isDemo))
                } else {
                    Button("Open my device") {
                        dismiss()
                    }
                    .buttonStyle(BrandButtonStyle())
                }

                if let errorMessage {
                    Text(errorMessage).font(.callout).foregroundStyle(SD.Colors.danger)
                }
            }
            .padding(24)
            .animation(.spring(duration: 0.45), value: step)
            .navigationTitle("Set up device")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
            }
        }
    }

    private var buttonTitle: String {
        switch step {
        case .power: "It's glowing"
        case .hotspot: state.isDemo ? "Connected (demo)" : "I'm connected"
        case .wifi: state.isDemo ? "Wi-Fi joined (demo)" : "Device joined my Wi-Fi"
        case .pair: "Pair device"
        case .done: ""
        }
    }

    private var progressBar: some View {
        HStack(spacing: 6) {
            ForEach(0..<4, id: \.self) { index in
                Capsule()
                    .fill(index <= min(step.rawValue, 3) ? AnyShapeStyle(SD.Colors.accent) : AnyShapeStyle(.quaternary))
                    .frame(height: 5)
            }
        }
    }

    private func stepBody(icon: String, title: String, text: String) -> some View {
        VStack(spacing: 16) {
            ZStack {
                Circle().fill(SD.Colors.accent.opacity(0.15)).frame(width: 130, height: 130)
                Image(systemName: icon)
                    .font(.system(size: 52))
                    .foregroundStyle(SD.Colors.accent)
                    .symbolEffect(.pulse)
            }
            Text(title).font(.title2.bold())
            Text(text)
                .font(.subheadline)
                .foregroundStyle(SD.Colors.textSecondary)
                .multilineTextAlignment(.center)
        }
    }

    private var powerStep: some View {
        stepBody(
            icon: "power",
            title: "Power it on",
            text: "Plug your SensorDyme into power (car charger or wall). Give it ~30 seconds to boot — it glows when it's ready to be set up."
        )
    }

    /// The device's own address while it is serving its first-boot setup
    /// hotspot — and ONLY then.
    ///
    /// Isolated behind a named constant because a bare `10.42.0.1` in a view was
    /// indistinguishable from the hardcoded runtime addresses this app has spent
    /// a long time removing. This one is legitimate and is not discovery: it is
    /// the gateway NetworkManager assigns in shared mode, so while the phone is
    /// joined to the device's own AP it is the device, by construction, and
    /// there is no network to discover anything on yet.
    ///
    /// It must never be used as a backend or device address during normal
    /// operation — see `DeviceDiscovery` for that. Belongs to the legacy
    /// `provisioning.method: hotspot` path (scripts/provisioning_hotspot.sh);
    /// the shipped default is BLE, which needs no address at all.
    static let provisioningHotspotSetupURL = URL(string: "http://10.42.0.1:8899/setup")

    /// The setup page, carrying the device token when there is one.
    ///
    /// The page's Wi-Fi form posts to `/setup/wifi`, which on a network bind
    /// requires the token. A page opened in Safari cannot be given a header, so
    /// the device reads `?token=` and sends it on as X-Device-Token. URLComponents
    /// percent-encodes it, so a token with `+ / = &` survives the trip.
    static func setupURL(token: String) -> URL? {
        guard let base = provisioningHotspotSetupURL else { return nil }
        let cleaned = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty,
              var components = URLComponents(url: base, resolvingAgainstBaseURL: false)
        else { return base }
        components.queryItems = [URLQueryItem(name: "token", value: cleaned)]
        // URLQueryItem leaves "+" alone, and a server's form decoding reads it
        // as a space; encode it explicitly.
        components.percentEncodedQuery = components.percentEncodedQuery?
            .replacingOccurrences(of: "+", with: "%2B")
        return components.url ?? base
    }

    private var hotspotStep: some View {
        VStack(spacing: 14) {
            stepBody(
                icon: "wifi.circle.fill",
                title: "Join its setup network",
                text: "In Settings → Wi-Fi on this phone, join the network named SensorDyme-XXXX. Your device broadcasts it on first boot."
            )
            if !state.isDemo, let setup = Self.setupURL(token: state.deviceToken) {
                Link(destination: setup) {
                    Label("Open device setup page", systemImage: "safari")
                        .font(.subheadline.bold())
                }
            }
        }
    }

    private var wifiStep: some View {
        stepBody(
            icon: "house.fill",
            title: "Give it your Wi-Fi",
            text: state.isDemo
                ? "On the device's setup page you pick your home Wi-Fi and enter the password. In this demo, that just happened — instantly."
                : "On the setup page that just opened, choose your home Wi-Fi and enter its password. The device joins it and its setup network disappears — that's your cue."
        )
    }

    private var pairStep: some View {
        VStack(spacing: 16) {
            stepBody(
                icon: "qrcode.viewfinder",
                title: "Pair it to your account",
                text: state.isDemo
                    ? "Enter the claim code from the card in the box. For the demo, just tap Pair."
                    : "Enter the claim code from the card in the box (or under the device)."
            )
            if !state.isDemo {
                TextField("1A2B-3C4D-5E6F", text: $claimCode)
                    .font(.title3.monospaced())
                    .multilineTextAlignment(.center)
                    .autocapitalization(.allCharacters)
                    .autocorrectionDisabled()
                    .padding(14)
                    .background(RoundedRectangle(cornerRadius: 14).fill(.background.secondary))
                    .overlay(RoundedRectangle(cornerRadius: 14).strokeBorder(.quaternary))
            }
        }
    }

    private var doneStep: some View {
        VStack(spacing: 16) {
            ZStack {
                Circle().fill(SD.Colors.success).frame(width: 130, height: 130)
                Image(systemName: "checkmark")
                    .font(.system(size: 56, weight: .heavy))
                    .foregroundStyle(.white)
            }
            .transition(.scale.combined(with: .opacity))
            Text("You're live").font(.title.bold())
            Text("Your SensorDyme is online, watching, and fully controllable from this app — modes, alerts, reports, everything.")
                .font(.subheadline)
                .foregroundStyle(SD.Colors.textSecondary)
                .multilineTextAlignment(.center)
        }
    }

    private func advance() {
        errorMessage = nil
        switch step {
        case .power, .hotspot, .wifi:
            step = Step(rawValue: step.rawValue + 1) ?? .done
        case .pair:
            busy = true
            Task {
                defer { busy = false }
                do {
                    try await state.backend.claim(
                        code: state.isDemo ? "DEMO" : claimCode.trimmingCharacters(in: .whitespaces)
                    )
                    await onPaired()
                    withAnimation(.spring(duration: 0.6)) { step = .done }
                } catch {
                    errorMessage = error.localizedDescription
                }
            }
        case .done:
            break
        }
    }
}
