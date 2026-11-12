import SwiftUI

// MARK: - Adding a device.
//
// The first screen offers Bluetooth setup (SetupFlowView: find, prove the setup
// code, Wi-Fi, claim, mode -- the M8 onboarding) for a new device, and keeps the
// older path for a device that is already on the network or uses the legacy
// setup hotspot: plug in -> join hotspot -> Wi-Fi -> claim.

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
    @State private var showBluetoothSetup = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: SD.Space.lg) {
                progressBar

                Spacer(minLength: SD.Space.md)

                Group {
                    switch step {
                    case .power: powerStep
                    case .hotspot: hotspotStep
                    case .wifi: wifiStep
                    case .pair: pairStep
                    case .done: doneStep
                    }
                }
                .transition(reduceMotion ? .opacity
                            : .asymmetric(insertion: .move(edge: .trailing).combined(with: .opacity),
                                          removal: .move(edge: .leading).combined(with: .opacity)))

                Spacer(minLength: SD.Space.md)

                if step != .done {
                    Button(action: advance) {
                        if busy { ProgressView().tint(SD.Colors.textOnAccent) } else { Text(buttonTitle) }
                    }
                    .accessibilityLabel(busy ? "Pairing" : buttonTitle)
                    .accessibilityIdentifier("wizard.next")
                    .buttonStyle(BrandButtonStyle())
                    .disabled(busy || (step == .pair && claimCode.isEmpty && !state.isDemo))
                } else {
                    Button("Open my device") {
                        dismiss()
                    }
                    .buttonStyle(BrandButtonStyle())
                    .accessibilityIdentifier("wizard.done")
                }

                if let errorMessage {
                    Text(errorMessage).font(SD.Typography.callout).foregroundStyle(SD.Colors.danger)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityLabel("Error: \(errorMessage)")
                        .accessibilityIdentifier("wizard.error")
                }
            }
            .padding(SD.Space.lg)
            .animation(SD.Motion.resolved(SD.Motion.spatial, reduceMotion: reduceMotion), value: step)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .orynScreen()
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

    /// Four hairline segments and the step in words: calm, and never ahead
    /// of what has actually happened.
    private var progressBar: some View {
        VStack(alignment: .leading, spacing: SD.Space.xs) {
            Text(step == .done ? "Paired" : "Step \(min(step.rawValue, 3) + 1) of 4")
                .font(SD.Typography.caption.weight(.medium).monospacedDigit())
                .foregroundStyle(SD.Colors.textTertiary)
            HStack(spacing: 4) {
                ForEach(0..<4, id: \.self) { index in
                    Capsule()
                        .fill(index <= min(step.rawValue, 3) ? SD.Colors.accent : SD.Colors.borderStrong)
                        .frame(height: 2)
                }
            }
            .accessibilityHidden(true)
        }
        .accessibilityElement(children: .combine)
    }

    /// One step: a symbol on a quiet tile, what to do, why. No animated
    /// illustration: the person is doing something with their hands.
    private func stepBody(icon: String, title: String, text: String) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            Image(systemName: icon)
                .font(.system(size: 24, weight: .regular))
                .foregroundStyle(SD.Colors.textPrimary)
                .frame(width: 60, height: 60)
                .background(
                    RoundedRectangle(cornerRadius: SD.Radius.xl, style: .continuous)
                        .fill(SD.Colors.surfaceRaised)
                )
                .accessibilityHidden(true)
            Text(title)
                .sdScaledFont(28, relativeTo: .title, weight: .semibold)
                .foregroundStyle(SD.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)
            Text(text)
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var powerStep: some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            stepBody(
                icon: "power",
                title: "Power it on",
                text: "Plug your SensorDyme into power (car charger or wall). Give it ~30 seconds to boot — it glows when it's ready to be set up."
            )
            if !state.isDemo {
                Button {
                    showBluetoothSetup = true
                } label: {
                    Label("Set up over Bluetooth", systemImage: "dot.radiowaves.left.and.right")
                }
                .buttonStyle(SDButtonStyle(kind: .secondary))
                .accessibilityIdentifier("wizard.bluetooth")
                Text("New SensorDyme? Bluetooth setup finds it, checks the setup code on its label and gives it your Wi-Fi.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .sheet(isPresented: $showBluetoothSetup) {
            SetupFlowView {
                await onPaired()
                dismiss()
            }
        }
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
        VStack(alignment: .leading, spacing: SD.Space.md) {
            stepBody(
                icon: "wifi",
                title: "Join its setup network",
                text: "In Settings → Wi-Fi on this phone, join the network named SensorDyme-XXXX. Your device broadcasts it on first boot."
            )
            if !state.isDemo, let setup = Self.setupURL(token: state.deviceToken) {
                Link(destination: setup) {
                    Label("Open device setup page", systemImage: "safari")
                        .font(SD.Typography.label.weight(.semibold))
                        .frame(minHeight: 44)
                }
            }
        }
    }

    private var wifiStep: some View {
        stepBody(
            icon: "house",
            title: "Give it your Wi-Fi",
            text: state.isDemo
                ? "On the device's setup page you pick your home Wi-Fi and enter the password. In this demo, that just happened — instantly."
                : "On the setup page that just opened, choose your home Wi-Fi and enter its password. The device joins it and its setup network disappears — that's your cue."
        )
    }

    private var pairStep: some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
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
                    .multilineTextAlignment(.leading)
                    .autocapitalization(.allCharacters)
                    .autocorrectionDisabled()
                    .padding(.horizontal, SD.Space.md)
                    .frame(minHeight: 56)
                    .background(RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous).fill(SD.Colors.field))
                    // Its placeholder is an example code, which VoiceOver read as the field's name.
                    .accessibilityLabel("Claim code")
                    .accessibilityIdentifier("wizard.claimCode")
            }
        }
    }

    /// Said as what happened: the claim succeeded. Whether the device is
    /// reporting and watching is the device screen's to say, from its own
    /// telemetry; this screen used to promise "online, watching" on a claim.
    private var doneStep: some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            Image(systemName: "checkmark")
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(SD.Colors.textPrimary)
                .frame(width: 60, height: 60)
                .background(
                    RoundedRectangle(cornerRadius: SD.Radius.xl, style: .continuous)
                        .fill(SD.Colors.surfaceRaised)
                )
                .accessibilityHidden(true)
            Text("Paired to your account")
                .sdScaledFont(28, relativeTo: .title, weight: .semibold)
                .foregroundStyle(SD.Colors.textPrimary)
                .accessibilityAddTraits(.isHeader)
            Text("Open it to see whether it's reporting and what it's watching for. Modes, alerts and reports are all on its screen.")
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .transition(.opacity)
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
