import CoreBluetooth
import SwiftUI

/// Adding a SensorDyme, start to finish, from the phone.
///
/// One screen that changes rather than a stack to walk: setup is a single
/// errand, and the only question at any moment is "what is happening now". The
/// checklist stays on screen throughout so the answer is always visible,
/// including the part where the device vanishes from Bluetooth to join its new
/// network -- the moment that most looks like a failure and is not.
///
/// The journey itself is `Onboarding` (SetupOnboarding.swift), driven by
/// `SetupFlow`; this view only renders the state and hands back what the
/// person typed or chose. Bluetooth problems are named as what they are
/// (SetupBluetoothStatus: permission, restriction, power, no Bluetooth), from
/// what the radio reported. Nothing here simulates a device: the simulated
/// peripheral is reachable only in a DEBUG build launched with
/// `-orynSimulatedBLE` (BLETransports.make).
struct SetupFlowView: View {
    @Environment(\.dismiss) private var dismiss
    @EnvironmentObject private var appState: AppState
    @StateObject private var flow = SetupFlow.app()
    /// Called after setup finished and the person tapped Done (e.g. reload the device list).
    var onFinished: (() async -> Void)?

    @State private var selectedSSID = ""
    @State private var selectedSecured = true
    @State private var showPasswordEntry = false
    @State private var setupCode = ""
    @State private var codeProblem: String?
    @State private var claimCode = ""
    @State private var chosenMode: String?
    /// The clock, for the device's lockout countdown only. It ticks while a
    /// lockout runs (`.task(id: lockedUntil)` below); a 1 Hz timer used to
    /// redraw this whole screen every second of setup for it.
    @State private var now = Date()
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    init(onFinished: (() async -> Void)? = nil) {
        self.onFinished = onFinished
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: SD.Space.lg) {
                    header
                    content
                    SetupDebugPanel(flow: flow, ble: flow.ble)
                        .padding(.top, SD.Space.sm)
                }
                .padding(.horizontal, SD.Space.md)
                .padding(.vertical, SD.Space.md)
                .animation(SD.Motion.resolved(SD.Motion.standard, reduceMotion: reduceMotion), value: flow.state)
            }
            .orynScreen()
            .navigationTitle("Connect SensorDyme")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") { flow.stop(); dismiss() }
                }
            }
            .task {
                flow.attach(account: appState.isSignedIn ? AppStateOnboardingAccount(appState) : nil)
                flow.start()
            }
            .onDisappear {
                flow.stop()
                setupCode = ""
            }
            .task(id: lockedUntil) {
                guard let until = lockedUntil else { return }
                while !Task.isCancelled, Date() < until.addingTimeInterval(1) {
                    now = Date()
                    try? await Task.sleep(for: .seconds(1))
                }
                now = Date()
            }
            .onChange(of: flow.recommendedMode) { _, mode in
                if chosenMode == nil { chosenMode = mode }
            }
        }
        .tint(SD.Colors.accent)
    }

    /// The device's refusal of codes until this time, while it lasts.
    private var lockedUntil: Date? {
        if case .enteringSetupCode(_, let until?) = flow.state { return until }
        return nil
    }

    // MARK: Bluetooth itself

    /// What is wrong with Bluetooth, if anything, while setup is still looking.
    /// From the radio's own report and the app's permission; once a device is
    /// connected the flow handles a radio change itself.
    private var radioProblem: SetupBluetoothStatus.Problem? {
        switch flow.state {
        case .bluetoothUnavailable, .discovering:
            return SetupBluetoothStatus.problem(radio: Self.radio(flow.ble.radio), access: Self.access)
        default:
            return nil
        }
    }

    private static func radio(_ radio: BLERadioState?) -> SetupBluetoothStatus.Radio? {
        switch radio {
        case .poweredOn?: .on
        case .poweredOff?: .off
        case .unauthorized?: .unauthorized
        case .unsupported?: .unsupported
        case .other?: .unknown
        case nil: nil
        }
    }

    /// The app's Bluetooth permission. Reading it never prompts.
    private static var access: SetupBluetoothStatus.Access {
        switch CBManager.authorization {
        case .notDetermined: .notDetermined
        case .denied: .denied
        case .restricted: .restricted
        case .allowedAlways: .allowed
        @unknown default: .notDetermined
        }
    }

    // MARK: Header

    private enum Tone { case neutral, working, attention, failed, done }

    @ViewBuilder private var header: some View {
        if let radioProblem {
            headline(radioProblem.title, nil, symbol: radioProblem.symbol, tone: .attention)
        } else {
            switch flow.state {
            case .bluetoothUnavailable:
                // The radio has not said why yet.
                headline("Waiting for Bluetooth", "ORYN EDGE sets up your SensorDyme's Wi-Fi over Bluetooth.",
                         symbol: "dot.radiowaves.left.and.right", tone: .working)
            case .discovering where !flow.ble.discovered.isEmpty:
                headline(flow.ble.discovered.count == 1 ? "SensorDyme found" : "\(flow.ble.discovered.count) SensorDymes found",
                         "Tap Connect to set it up.", symbol: "dot.radiowaves.left.and.right", tone: .neutral)
            case .discovering:
                headline("Looking for your SensorDyme",
                         "Plug it in and stand within a few metres of it. Bluetooth doesn't reach much further.",
                         symbol: "magnifyingglass", tone: .working)
            case .connecting(let name), .identifying(let name):
                headline("Connecting to \(name)", "Over Bluetooth, to this device only.",
                         symbol: "dot.radiowaves.left.and.right", tone: .working)
            case .enteringSetupCode:
                headline("Prove it's yours",
                         "Being nearby isn't enough: enter the setup code from the label on your SensorDyme, or paste the text of its QR code.",
                         symbol: "lock.shield", tone: .neutral)
            case .proving:
                headline("Checking the setup code", "Your SensorDyme proves it knows the code too, before anything is sent.",
                         symbol: "lock.shield", tone: .working)
            case .choosingNetwork:
                headline("Choose a Wi-Fi network",
                         "These are the networks your SensorDyme can see from where it is — not the ones your phone can see.",
                         symbol: "wifi", tone: .neutral)
            case .sending(let ssid):
                headline("Sending \(ssid)", "Encrypted, to this device only.", symbol: "lock", tone: .working)
            case .joining(let ssid):
                headline("Joining \(ssid)", "Your SensorDyme is connecting. It leaves Bluetooth when it does — that's expected.",
                         symbol: "wifi", tone: .working)
            case .associating:
                headline("Checking your account", nil, symbol: "person.crop.circle", tone: .working)
            case .enteringClaimCode, .claiming:
                headline("Add it to your account",
                         "Enter the claim code printed on the card that came with your SensorDyme.",
                         symbol: "person.crop.circle.badge.plus", tone: .neutral)
            case .waitingForDevice:
                headline("Waiting for your SensorDyme",
                         "It's on \(flow.targetSSID.isEmpty ? "its network" : flow.targetSSID) and connecting to the server.",
                         symbol: "antenna.radiowaves.left.and.right", tone: .working)
            case .choosingMode, .activatingMode:
                headline("What should it watch for?", "You can change this any time from the device's screen.",
                         symbol: "square.grid.2x2", tone: .neutral)
            case .finished(let outcome):
                headline("SensorDyme is set up",
                         outcome.alreadyOwned ? "It's on \(outcome.ssid ?? "its new network") and still on your account."
                             : (outcome.ssid.map { "Connected to \($0)." } ?? nil),
                         symbol: "checkmark", tone: .done)
            case .failed:
                headline("Setup didn't finish", nil, symbol: "exclamationmark.triangle", tone: .failed)
            }
        }
    }

    /// The state as a symbol on a quiet tile, the title and one line of why.
    /// The tile's colour follows the state's meaning (amber attention, red
    /// failure); "done" stays bone — the device screen is where "working" is
    /// said in green, from its own telemetry.
    private func headline(_ title: String, _ detail: String?, symbol: String, tone: Tone) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            Image(systemName: symbol)
                .font(.system(size: 20, weight: .regular))
                .foregroundStyle(toneColor(tone))
                .frame(width: 48, height: 48)
                .background(
                    RoundedRectangle(cornerRadius: SD.Radius.lg, style: .continuous)
                        .fill(tone == .attention ? SD.Colors.warningMuted
                              : tone == .failed ? SD.Colors.dangerMuted : SD.Colors.surfaceRaised)
                )
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: SD.Space.xxs) {
                Text(title)
                    .sdScaledFont(26, relativeTo: .title, weight: .semibold)
                    .foregroundStyle(SD.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)
                    .accessibilityAddTraits(.isHeader)
                    .accessibilityIdentifier("setup.title")
                if let detail {
                    Text(detail)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .padding(.top, SD.Space.xs)
    }

    private func toneColor(_ tone: Tone) -> Color {
        switch tone {
        case .neutral, .done: SD.Colors.textPrimary
        case .working: SD.Colors.textSecondary
        case .attention: SD.Colors.warning
        case .failed: SD.Colors.danger
        }
    }

    // MARK: Body

    @ViewBuilder private var content: some View {
        if let radioProblem {
            radioProblemCard(radioProblem)
        } else {
            switch flow.state {
            case .bluetoothUnavailable:
                busyCard("Waiting for Bluetooth to answer…")
            case .discovering:
                deviceList
            case .connecting:
                busyCard("Pairing…")
            case .identifying:
                busyCard("Asking it what it is…")
            case .enteringSetupCode(let problem, let lockedUntil):
                setupCodeCard(problem: problem, lockedUntil: lockedUntil)
            case .proving:
                busyCard("Checking the setup code. The device proves it back before anything is sent.")
            case .choosingNetwork(let problem):
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    if let problem { warning(problem) }
                    networkList
                }
            case .sending, .joining, .associating, .claiming, .activatingMode:
                progressCard
            case .enteringClaimCode(let problem):
                claimCard(problem: problem)
            case .waitingForDevice(let elapsed):
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    progressCard
                    if elapsed > 30 {
                        warning("Still waiting. If \(flow.targetSSID) needs a browser sign-in (hotel or campus Wi-Fi) the device can't get through it.")
                    }
                }
            case .choosingMode:
                modeCard
            case .finished:
                readySummary
            case .failed(let failure):
                failureCard(failure)
            }
        }
    }

    /// One Bluetooth problem, its fix, and — where the fix is in Settings —
    /// the button that opens ORYN EDGE's page there. Power comes back by
    /// itself (the radio reports it), so that case waits instead of offering a
    /// button iOS gives apps no way to honour.
    private func radioProblemCard(_ problem: SetupBluetoothStatus.Problem) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            Text(problem.detail)
                .font(SD.Typography.body)
                .foregroundStyle(SD.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityIdentifier("setup.bluetooth.problem")
            if problem.opensAppSettings {
                Button {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                } label: {
                    Label("Open Settings", systemImage: "gear")
                }
                .buttonStyle(SDButtonStyle(kind: .primary))
                .accessibilityHint("Opens ORYN EDGE's page in Settings, where Bluetooth can be allowed.")
                .accessibilityIdentifier("setup.bluetooth.settings")
            } else if problem == .poweredOff {
                HStack(spacing: SD.Space.xs) {
                    ProgressView().controlSize(.small).tint(SD.Colors.textSecondary)
                    Text("Waiting for Bluetooth to be turned on")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textTertiary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }

    private func busyCard(_ text: String) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.md) {
                SetupProgressView(steps: steps)
                HStack(spacing: SD.Space.sm) {
                    ProgressView().controlSize(.small)
                    Text(text)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                .accessibilityElement(children: .combine)
            }
        }
    }

    private func warning(_ text: String) -> some View {
        Label(text, systemImage: "exclamationmark.triangle.fill")
            .font(SD.Typography.callout)
            .foregroundStyle(SD.Colors.warning)
            .fixedSize(horizontal: false, vertical: true)
            .accessibilityIdentifier("setup.problem")
    }

    private var deviceList: some View {
        VStack(alignment: .leading, spacing: SD.Space.md) {
            if flow.ble.discovered.isEmpty {
                SDCard {
                    VStack(alignment: .leading, spacing: SD.Space.sm) {
                        HStack(spacing: SD.Space.sm) {
                            ProgressView().controlSize(.small)
                            Text(flow.searchingSeconds > 0 ? "Searching… \(flow.searchingSeconds) s" : "Searching…")
                                .font(SD.Typography.callout.monospacedDigit())
                                .foregroundStyle(SD.Colors.textSecondary)
                        }
                        .accessibilityElement(children: .combine)
                        if flow.searchingSeconds > 20 {
                            Divider().overlay(SD.Colors.border)
                            Text("Still nothing after \(flow.searchingSeconds) s")
                                .font(SD.Typography.label)
                                .foregroundStyle(SD.Colors.warning)
                            Text("Check the device is powered on and you are within a few metres of it.")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                            Button("Search again") { flow.recover(.searchAgain) }
                                .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                        }
                    }
                }
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    // The device advertises a custom service, which iOS Settings
                    // does not list. Someone who looks there sees nothing.
                    hint("antenna.radiowaves.left.and.right",
                         "Your SensorDyme won't appear in the iPhone's Bluetooth settings — only here. That's normal.")
                    // A device already on Wi-Fi accepts Bluetooth setup only for a
                    // few minutes after it starts (services/provisioning.py).
                    hint("arrow.clockwise",
                         "Moving one that's already set up to a new network? Unplug it for a few seconds and plug it back in — it accepts Bluetooth setup for 5 minutes after it starts.")
                }
            } else if flow.ble.discovered.count == 1, let device = flow.ble.discovered.first {
                SDCard {
                    VStack(alignment: .leading, spacing: SD.Space.md) {
                        HStack(spacing: SD.Space.sm) {
                            Image(systemName: "dot.radiowaves.left.and.right")
                                .foregroundStyle(SD.Colors.textPrimary)
                                .accessibilityHidden(true)
                            VStack(alignment: .leading, spacing: 1) {
                                Text(device.name).font(SD.Typography.headline).foregroundStyle(SD.Colors.textPrimary)
                                Text(device.proximity).font(SD.Typography.caption).foregroundStyle(SD.Colors.textTertiary)
                            }
                            Spacer(minLength: 0)
                        }
                        .accessibilityElement(children: .combine)
                        Button("Connect") { flow.connect(to: device) }
                            .buttonStyle(SDButtonStyle(kind: .primary))
                            .accessibilityIdentifier("setup.connect")
                    }
                }
            } else {
                SDCard {
                    VStack(alignment: .leading, spacing: SD.Space.sm) {
                        ForEach(flow.ble.discovered) { device in
                            DeviceRow(device: device) { flow.connect(to: device) }
                            if device.id != flow.ble.discovered.last?.id { Divider().overlay(SD.Colors.border) }
                        }
                    }
                }
            }
        }
    }

    private func hint(_ symbol: String, _ text: String) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
            Image(systemName: symbol)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(SD.Colors.textTertiary)
                .frame(width: 16)
                .accessibilityHidden(true)
            Text(text)
                .font(SD.Typography.caption)
                .foregroundStyle(SD.Colors.textTertiary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    // MARK: Proof of possession

    private func setupCodeCard(problem: String?, lockedUntil: Date?) -> some View {
        let remaining = lockedUntil.map { Int($0.timeIntervalSince(now).rounded(.up)) } ?? 0
        let locked = remaining > 0
        return SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SetupProgressView(steps: steps)
                Divider().overlay(SD.Colors.border)
                TextField("XXXX-XXXX-XXXX", text: $setupCode)
                    .font(SD.Typography.body.monospaced())
                    .textInputAutocapitalization(.characters)
                    .autocorrectionDisabled()
                    .textContentType(.oneTimeCode)
                    .padding(SD.Space.sm)
                    .background(RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous).fill(SD.Colors.field))
                    .accessibilityLabel("Setup code")
                    .accessibilityIdentifier("setup.code")
                Button {
                    if let text = UIPasteboard.general.string { setupCode = text }
                } label: {
                    Label("Paste the QR code's text", systemImage: "doc.on.clipboard")
                        .font(SD.Typography.caption)
                }
                .buttonStyle(SDRowButtonStyle(minHeight: 44))
                .foregroundStyle(SD.Colors.accent)
                .accessibilityIdentifier("setup.code.paste")
                if let message = codeProblem ?? problem {
                    Text(message)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.danger)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("setup.code.problem")
                }
                if locked {
                    // The device's own number, counted down: never a guess.
                    Label("Your SensorDyme will accept a code again in \(remaining) s.", systemImage: "lock.fill")
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.warning)
                        .accessibilityIdentifier("setup.code.locked")
                }
                Button(locked ? "Wait \(remaining) s" : "Continue") {
                    if let entryProblem = flow.prove(entry: setupCode) {
                        codeProblem = SetupCodeEntry.message(entryProblem)
                    } else {
                        codeProblem = nil
                    }
                }
                .buttonStyle(SDButtonStyle(kind: .primary))
                .disabled(setupCode.trimmingCharacters(in: .whitespaces).isEmpty || locked)
                .accessibilityIdentifier("setup.prove")
                Text("Your SensorDyme proves it knows this code too, before your Wi-Fi password is sent — encrypted, to this device only.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    // MARK: Wi-Fi

    private var networkList: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    if flow.ble.networks.isEmpty, flow.ble.scanningNetworks {
                        HStack(spacing: SD.Space.sm) {
                            ProgressView().controlSize(.small)
                            Text("Scanning for Wi-Fi…").font(SD.Typography.callout).foregroundStyle(SD.Colors.textSecondary)
                        }
                        .frame(minHeight: 56)
                    } else if let problem = flow.ble.networkProblem {
                        // "The app never managed to ask", not "the device saw nothing".
                        Label(problem, systemImage: "exclamationmark.triangle.fill")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.warning)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("Try again") { flow.ble.refreshNetworks() }
                            .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                    } else if flow.ble.networks.isEmpty {
                        Text("Your SensorDyme can't see any Wi-Fi from where it is.")
                            .font(SD.Typography.callout)
                            .foregroundStyle(SD.Colors.textPrimary)
                        Text("Move it closer to your router, or type a network name below.")
                            .font(SD.Typography.caption)
                            .foregroundStyle(SD.Colors.textSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                        Button("Search again") { flow.ble.refreshNetworks() }
                            .buttonStyle(SDButtonStyle(kind: .secondary, fullWidth: false))
                    } else {
                        ForEach(flow.ble.networks) { network in
                            NetworkRow(network: network, isSelected: selectedSSID == network.ssid) {
                                selectedSSID = network.ssid
                                selectedSecured = network.secured
                                showPasswordEntry = true
                            }
                            if network.id != flow.ble.networks.last?.id { Divider().overlay(SD.Colors.border) }
                        }
                    }
                }
            }

            HStack(spacing: SD.Space.sm) {
                if !flow.ble.networks.isEmpty {
                    Text(flow.ble.networks.count == 1 ? "1 network" : "\(flow.ble.networks.count) networks")
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.textTertiary)
                }
                Spacer()
                Button("Other network…") {
                    // Hidden networks never appear in a scan.
                    selectedSSID = ""
                    selectedSecured = true
                    showPasswordEntry = true
                }
                .buttonStyle(SDButtonStyle(kind: .tertiary, fullWidth: false))
                if flow.ble.scanningNetworks {
                    HStack(spacing: SD.Space.xxs) {
                        ProgressView().controlSize(.small)
                        Text("Refreshing").font(SD.Typography.caption).foregroundStyle(SD.Colors.textTertiary)
                    }
                    .frame(minHeight: 44)
                } else {
                    Button { flow.ble.refreshNetworks() } label: {
                        Label("Refresh", systemImage: "arrow.clockwise").font(SD.Typography.caption)
                    }
                    .buttonStyle(SDRowButtonStyle(minHeight: 44))
                    .fixedSize()
                    .foregroundStyle(SD.Colors.accent)
                }
            }
        }
        .sheet(isPresented: $showPasswordEntry) {
            // The sheet keeps its own copy of what is typed: this screen
            // re-renders on every scan tick and status push.
            PasswordSheet(initialSSID: selectedSSID, secured: selectedSecured) { ssid, password in
                showPasswordEntry = false
                flow.send(ssid: ssid, password: password)
            }
        }
    }

    // MARK: Account

    private func claimCard(problem: String?) -> some View {
        SDCard {
            VStack(alignment: .leading, spacing: SD.Space.sm) {
                SetupProgressView(steps: steps)
                Divider().overlay(SD.Colors.border)
                TextField("1A2B-3C4D-5E6F", text: $claimCode)
                    .font(SD.Typography.body.monospaced())
                    .textInputAutocapitalization(.characters)
                    .autocorrectionDisabled()
                    .padding(SD.Space.sm)
                    .background(RoundedRectangle(cornerRadius: SD.Radius.md, style: .continuous).fill(SD.Colors.field))
                    .accessibilityLabel("Claim code")
                    .accessibilityIdentifier("setup.claimCode")
                Text("Not the setup code: the claim code links this SensorDyme to your account. It's on the card in the box.")
                    .font(SD.Typography.caption)
                    .foregroundStyle(SD.Colors.textTertiary)
                    .fixedSize(horizontal: false, vertical: true)
                if let problem {
                    Text(problem)
                        .font(SD.Typography.caption)
                        .foregroundStyle(SD.Colors.danger)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("setup.claim.problem")
                }
                Button("Add to my account") {
                    if flow.submitClaim(code: claimCode) { claimCode = "" }
                }
                .buttonStyle(SDButtonStyle(kind: .primary))
                .disabled(Onboarding.normalizeClaimCode(claimCode) == nil)
                .accessibilityIdentifier("setup.claim")
            }
        }
    }

    // MARK: Monitoring mode

    private var modeCard: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    if flow.modes.isEmpty {
                        HStack(spacing: SD.Space.sm) {
                            ProgressView().controlSize(.small)
                            Text("Asking what it can run…").font(SD.Typography.callout).foregroundStyle(SD.Colors.textSecondary)
                        }
                    }
                    ForEach(flow.modes, id: \.id) { mode in
                        Button {
                            if mode.available { chosenMode = mode.id }
                        } label: {
                            HStack(alignment: .firstTextBaseline, spacing: SD.Space.sm) {
                                Image(systemName: chosenMode == mode.id ? "largecircle.fill.circle" : "circle")
                                    .foregroundStyle(mode.available ? SD.Colors.accent : SD.Colors.textTertiary)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(mode.title)
                                        .font(SD.Typography.body)
                                        .foregroundStyle(mode.available ? SD.Colors.textPrimary : SD.Colors.textTertiary)
                                    Text(mode.available ? mode.detail : (mode.reason ?? "Not available on this device"))
                                        .font(SD.Typography.caption)
                                        .foregroundStyle(SD.Colors.textTertiary)
                                        .fixedSize(horizontal: false, vertical: true)
                                        .multilineTextAlignment(.leading)
                                }
                                Spacer(minLength: 0)
                            }
                            .frame(minHeight: 44)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .disabled(!mode.available)
                        .accessibilityIdentifier("setup.mode.\(mode.id)")
                        .accessibilityAddTraits(chosenMode == mode.id ? .isSelected : [])
                    }
                }
            }
            Button("Start watching") { if let chosenMode { flow.chooseMode(chosenMode) } }
                .buttonStyle(SDButtonStyle(kind: .primary))
                .disabled(chosenMode == nil)
                .accessibilityIdentifier("setup.mode.start")
            Button("Not now") { flow.skipMode() }
                .buttonStyle(SDButtonStyle(kind: .tertiary))
                .accessibilityIdentifier("setup.mode.skip")
        }
    }

    // MARK: Progress, done, failure

    private var progressCard: some View {
        SDCard { SetupProgressView(steps: steps) }
    }

    private var readySummary: some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDCard { SetupProgressView(steps: steps) }
            if flow.localURL != nil { DeviceReadinessCard(baseURL: flow.localURL) }
            Button("Done") {
                flow.stop()
                Task {
                    await onFinished?()
                    dismiss()
                }
            }
            .buttonStyle(SDButtonStyle(kind: .primary))
            .accessibilityIdentifier("setup.done")
        }
    }

    private func failureCard(_ failure: Onboarding.Failure) -> some View {
        VStack(alignment: .leading, spacing: SD.Space.sm) {
            SDCard {
                VStack(alignment: .leading, spacing: SD.Space.sm) {
                    SetupProgressView(steps: steps)
                    Text(failure.reason)
                        .font(SD.Typography.callout)
                        .foregroundStyle(SD.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("setup.failure")
                }
            }
            ForEach(Array(failure.recoveries.enumerated()), id: \.element) { index, recovery in
                Button(recovery.title) { flow.recover(recovery) }
                    .buttonStyle(SDButtonStyle(kind: index == 0 ? .primary : .secondary))
                    .accessibilityIdentifier("setup.recover.\(recovery.rawValue)")
            }
        }
    }

    /// The checklist, derived from the flow rather than from a timer.
    private var steps: [SetupStep] {
        SetupStep.checklist(for: flow.state, requiresCode: flow.requiresCode, signedIn: flow.context.signedIn,
                            networksFound: flow.ble.networks.count, scanning: flow.ble.scanningNetworks, now: now)
    }
}

/// Asks for the Wi-Fi password, and nothing else.
///
/// Separate from the network list so the keyboard cannot cover the list behind
/// it, and so the password lives in exactly one place that is easy to audit.
private struct PasswordSheet: View {
    let initialSSID: String
    let secured: Bool
    let onSubmit: (String, String) -> Void

    @Environment(\.dismiss) private var dismiss
    @State private var ssid: String
    @State private var password = ""
    @State private var reveal = false
    @FocusState private var focus: Field?

    private enum Field { case ssid, password }

    init(initialSSID: String, secured: Bool, onSubmit: @escaping (String, String) -> Void) {
        self.initialSSID = initialSSID
        self.secured = secured
        self.onSubmit = onSubmit
        _ssid = State(initialValue: initialSSID)
    }

    var body: some View {
        NavigationStack {
            Form {
                Group {
                    if initialSSID.isEmpty {
                        Section("Network name") {
                            TextField("Exactly as it appears", text: $ssid)
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()
                                .focused($focus, equals: .ssid)
                                .submitLabel(.next)
                                .onSubmit { focus = .password }
                        }
                    }

                    if secured {
                        Section("Password") {
                            HStack {
                                Group {
                                    if reveal {
                                        TextField("Wi-Fi password", text: $password)
                                    } else {
                                        SecureField("Wi-Fi password", text: $password)
                                    }
                                }
                                .textInputAutocapitalization(.never)
                                .autocorrectionDisabled()
                                .textContentType(.password)
                                .focused($focus, equals: .password)
                                .submitLabel(.go)
                                .onSubmit(submit)
                                .accessibilityIdentifier("setup.wifi.password")

                                // Long random passwords typed on a phone: being able
                                // to check what you typed saves a failed join.
                                Button { reveal.toggle() } label: {
                                    Image(systemName: reveal ? "eye.slash" : "eye")
                                        .foregroundStyle(SD.Colors.textSecondary)
                                }
                                .buttonStyle(.plain)
                                .accessibilityLabel(reveal ? "Hide password" : "Show password")
                            }

                            Text("Sent straight to your SensorDyme over Bluetooth. It isn't stored on this phone or sent anywhere else.")
                                .font(SD.Typography.caption)
                                .foregroundStyle(SD.Colors.textTertiary)
                        }
                    } else {
                        Section {
                            Label("This network is open — no password needed.", systemImage: "lock.open")
                                .font(SD.Typography.callout)
                                .foregroundStyle(SD.Colors.textSecondary)
                        }
                    }

                    Section {
                        Button(action: submit) {
                            Text("Connect SensorDyme").frame(maxWidth: .infinity)
                        }
                        .buttonStyle(SDButtonStyle(kind: .primary))
                        .disabled(!canSubmit)
                        .listRowInsets(EdgeInsets())
                        .listRowBackground(Color.clear)
                        .accessibilityIdentifier("setup.wifi.send")
                    }
                }
                .orynRows()
            }
            .orynFormStyle()
            .navigationTitle(initialSSID.isEmpty ? "Other network" : initialSSID)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { password = ""; dismiss() }
                }
            }
            .task { focus = initialSSID.isEmpty ? .ssid : (secured ? .password : nil) }
        }
        .tint(SD.Colors.accent)
    }

    /// WPA keys are at least 8 characters: refusing a shorter one here beats a
    /// rejection thirty seconds later.
    private var canSubmit: Bool {
        guard !ssid.trimmingCharacters(in: .whitespaces).isEmpty else { return false }
        return secured ? password.count >= 8 : true
    }

    private func submit() {
        guard canSubmit else { return }
        onSubmit(networkName, password)
        password = ""
    }

    /// The SSID to send, exactly: spaces are legal at either end of an SSID.
    /// Only a typed name is cleaned, and only of line breaks a paste can bring.
    private var networkName: String {
        initialSSID.isEmpty ? ssid.trimmingCharacters(in: .newlines) : initialSSID
    }
}
