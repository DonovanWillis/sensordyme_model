import Foundation

/// First-time onboarding (and moving an installed device to a new network), as
/// an explicit state machine:
///
///   find -> connect -> prove the setup code -> choose Wi-Fi -> send (encrypted)
///        -> the device joins -> add it to the account (claim) -> it reports in
///        -> choose a monitoring mode -> done
///
/// The same discipline as the device's side (services/ble_pop.py
/// TRANSITIONS): `next` lists every legal move, anything else is refused and
/// leaves the state as it was. No radio, no network here: `SetupFlow` turns
/// what Bluetooth and the backend report into `Event`s and acts on the state
/// that results; ios/Tests/SetupOnboardingTests.swift walks every path, and the
/// cross-language harness (ios/Tests/OnboardingHarness) drives it against the
/// device's real Python state machine.
enum Onboarding {
    /// The checklist rows, in order.
    enum Stage: Int, CaseIterable, Comparable, Sendable {
        case find, connect, prove, network, join, account, online, mode

        static func < (a: Stage, b: Stage) -> Bool { a.rawValue < b.rawValue }

        var title: String {
            switch self {
            case .find: "Find your SensorDyme"
            case .connect: "Connect over Bluetooth"
            case .prove: "Prove it's yours"
            case .network: "Choose its Wi-Fi"
            case .join: "Device joins the network"
            case .account: "Add it to your account"
            case .online: "Device reports in"
            case .mode: "Choose what it watches for"
            }
        }
    }

    /// What the person can do about a failure. Every failure offers at least one.
    enum Recovery: String, Equatable, Sendable, CaseIterable {
        /// Back to the device list (the Bluetooth link is gone).
        case searchAgain
        /// Enter the setup code again (the link is still up).
        case enterCodeAgain
        /// Pick a network / retype the password (the link is still up).
        case chooseNetwork
        /// Look for the device on the account even though Bluetooth lost it:
        /// it may well have joined.
        case checkOnline
        case retryClaim
        case keepWaiting
        /// Pick another monitoring mode.
        case chooseMode
        /// Leave the mode as it is and finish.
        case skipMode

        var title: String {
            switch self {
            case .searchAgain: "Search again"
            case .enterCodeAgain: "Enter the code again"
            case .chooseNetwork: "Choose a network"
            case .checkOnline: "Check if it's online"
            case .retryClaim: "Try the claim code again"
            case .keepWaiting: "Keep waiting"
            case .chooseMode: "Choose another mode"
            case .skipMode: "Finish without a mode"
            }
        }
    }

    struct Failure: Equatable, Sendable {
        let stage: Stage
        let reason: String
        let recoveries: [Recovery]
    }

    /// How setup ended.
    struct Outcome: Equatable, Sendable {
        var ssid: String?
        /// The account's id for the device, when it is on the account.
        var accountDeviceID: Int?
        /// It was on this account already (re-provisioning, or a second run).
        var alreadyOwned = false
        /// The monitoring mode it was left in, if one was chosen here.
        var mode: String?
        /// Whether the account step happened at all (signed out: Wi-Fi only).
        var onAccount = false
    }

    enum State: Equatable, Sendable {
        case bluetoothUnavailable
        case discovering
        case connecting(name: String)
        /// Connected; reading what the device is and how it takes credentials.
        case identifying(name: String)
        /// The device requires its setup code. `problem` is why the last try
        /// did not work, in words; `lockedUntil` is the device's own wait.
        case enteringSetupCode(problem: String?, lockedUntil: Date?)
        case proving
        case choosingNetwork(problem: String?)
        case sending(ssid: String)
        case joining(ssid: String)
        /// Joined. Asking the account whether this device is already on it.
        case associating(ssid: String)
        case enteringClaimCode(problem: String?)
        case claiming
        /// On the account; waiting for it to report in over its new network.
        case waitingForDevice(elapsed: Int)
        case choosingMode(deviceID: Int)
        case activatingMode(deviceID: Int, mode: String)
        case finished(Outcome)
        case failed(Failure)

        var stage: Stage {
            switch self {
            case .bluetoothUnavailable, .discovering: .find
            case .connecting, .identifying: .connect
            case .enteringSetupCode, .proving: .prove
            case .choosingNetwork, .sending: .network
            case .joining: .join
            case .associating, .enteringClaimCode, .claiming: .account
            case .waitingForDevice: .online
            case .choosingMode, .activatingMode, .finished: .mode
            case .failed(let failure): failure.stage
            }
        }
    }

    enum Event: Equatable, Sendable {
        case bluetoothOff
        case bluetoothOn
        case deviceChosen(name: String)
        /// Connected and identified: whether it takes credentials only after the setup code.
        case identified(requiresCode: Bool)
        case proofStarted
        case proven
        /// The device refused the proof; the link is up for another try.
        case proofRefused(message: String, lockedUntil: Date?)
        case networkSent(ssid: String)
        case deviceJoining
        case deviceJoined
        /// The device tried the network and could not join; the link is up.
        case joinRefused(String)
        /// Bluetooth ended or broke (dropped, refused, wrong device, timed out).
        case linkFailed(String)
        /// No account to add the device to (signed out, demo): Wi-Fi only.
        case noAccount(found: Bool)
        case alreadyOnAccount(deviceID: Int)
        case needsClaim
        case claimSubmitted
        case claimRefused(String)
        case claimed
        case waitTick(elapsed: Int)
        case deviceOnline(deviceID: Int)
        case deviceSilent(String)
        case modeChosen(String)
        case modeActivated(String)
        case modeRefused(String)
        case modeSkipped
        case recover(Recovery)
    }

    /// The state after `event`, or nil when `event` is not a legal move from `state`.
    static func next(_ state: State, _ event: Event, context: Context = .init()) -> State? {
        switch (state, event) {
        // Bluetooth off/on matters only before anything was sent.
        case (.discovering, .bluetoothOff), (.connecting, .bluetoothOff), (.identifying, .bluetoothOff):
            return .bluetoothUnavailable
        case (.bluetoothUnavailable, .bluetoothOn):
            return .discovering

        case (.discovering, .deviceChosen(let name)):
            return .connecting(name: name)
        case (.connecting(let name), .identified):
            // Firmware without a device-info characteristic is ready straight away.
            return next(.identifying(name: name), event, context: context)
        case (.identifying, .identified(let requiresCode)):
            return requiresCode ? .enteringSetupCode(problem: nil, lockedUntil: nil) : .choosingNetwork(problem: nil)
        case (.connecting, .linkFailed(let reason)), (.identifying, .linkFailed(let reason)):
            return .failed(.init(stage: .connect, reason: reason, recoveries: [.searchAgain]))

        case (.enteringSetupCode, .proofStarted):
            return .proving
        case (.proving, .proven):
            return .choosingNetwork(problem: nil)
        case (.proving, .proofRefused(let message, let lockedUntil)):
            return .enteringSetupCode(problem: message, lockedUntil: lockedUntil)
        case (.enteringSetupCode, .linkFailed(let reason)), (.proving, .linkFailed(let reason)):
            return .failed(.init(stage: .prove, reason: reason, recoveries: [.searchAgain]))

        case (.choosingNetwork, .networkSent(let ssid)):
            return .sending(ssid: ssid)
        case (.choosingNetwork, .linkFailed(let reason)):
            return .failed(.init(stage: .network, reason: reason, recoveries: [.searchAgain]))
        case (.sending(let ssid), .deviceJoining):
            return .joining(ssid: ssid)
        case (.sending(let ssid), .deviceJoined), (.joining(let ssid), .deviceJoined):
            return .associating(ssid: ssid)
        case (.sending, .joinRefused(let reason)), (.joining, .joinRefused(let reason)):
            return .choosingNetwork(problem: reason)
        case (.sending, .proofRefused(let message, let lockedUntil)):
            // The proof lapsed and could not be renewed: the code again, then the network.
            return .enteringSetupCode(problem: message, lockedUntil: lockedUntil)
        case (.sending, .linkFailed(let reason)), (.joining, .linkFailed(let reason)):
            // The outcome is unknown: the device may well have joined.
            return .failed(.init(stage: .join, reason: reason,
                                 recoveries: context.signedIn ? [.checkOnline, .searchAgain] : [.searchAgain]))

        case (.associating, .noAccount(let found)):
            return found ? .finished(.init(ssid: context.ssid))
                : .failed(.init(stage: .online,
                                reason: "Your SensorDyme said it joined \(context.ssid ?? "the network"), but this phone can't see it yet. "
                                    + "Check your phone is on the same Wi-Fi.",
                                recoveries: [.keepWaiting, .searchAgain]))
        case (.associating, .alreadyOnAccount):
            return .waitingForDevice(elapsed: 0)
        case (.associating, .needsClaim):
            return .enteringClaimCode(problem: nil)
        case (.associating, .claimRefused(let reason)):
            // The account could not be asked (offline, server down).
            return .failed(.init(stage: .account, reason: reason, recoveries: [.retryClaim]))
        case (.enteringClaimCode, .claimSubmitted):
            return .claiming
        case (.claiming, .claimRefused(let reason)):
            return .enteringClaimCode(problem: reason)
        case (.claiming, .claimed):
            return .waitingForDevice(elapsed: 0)
        case (.waitingForDevice, .waitTick(let elapsed)):
            return .waitingForDevice(elapsed: elapsed)
        case (.waitingForDevice, .deviceOnline(let deviceID)):
            return .choosingMode(deviceID: deviceID)
        case (.waitingForDevice, .deviceSilent(let reason)):
            return .failed(.init(stage: .online, reason: reason, recoveries: [.keepWaiting]))
        case (.waitingForDevice, .claimRefused(let reason)):
            // The claim went through, but the device that reported in is another one.
            return .failed(.init(stage: .account, reason: reason, recoveries: [.retryClaim]))

        case (.choosingMode(let deviceID), .modeChosen(let mode)):
            return .activatingMode(deviceID: deviceID, mode: mode)
        case (.choosingMode(let deviceID), .modeSkipped):
            return .finished(.init(ssid: context.ssid, accountDeviceID: deviceID, alreadyOwned: context.alreadyOwned,
                                   onAccount: true))
        case (.activatingMode(let deviceID, _), .modeActivated(let mode)):
            return .finished(.init(ssid: context.ssid, accountDeviceID: deviceID, alreadyOwned: context.alreadyOwned,
                                   mode: mode, onAccount: true))
        case (.activatingMode, .modeRefused(let reason)):
            return .failed(.init(stage: .mode, reason: reason, recoveries: [.chooseMode, .skipMode]))
        case (.failed(let failure), .recover(let recovery)) where failure.recoveries.contains(recovery):
            return recover(recovery, from: failure, context: context)
        case (.failed, .recover(.searchAgain)):
            // Always possible: starting over never needs the old link.
            return .discovering

        // Starting over from anywhere before the device has its network.
        case (.connecting, .recover(.searchAgain)), (.identifying, .recover(.searchAgain)),
             (.enteringSetupCode, .recover(.searchAgain)), (.choosingNetwork, .recover(.searchAgain)),
             (.bluetoothUnavailable, .recover(.searchAgain)), (.discovering, .recover(.searchAgain)):
            return .discovering
        default:
            return nil
        }
    }

    private static func recover(_ recovery: Recovery, from failure: Failure, context: Context) -> State? {
        switch recovery {
        case .searchAgain: return .discovering
        case .enterCodeAgain: return .enteringSetupCode(problem: nil, lockedUntil: nil)
        case .chooseNetwork: return .choosingNetwork(problem: nil)
        case .checkOnline: return .associating(ssid: context.ssid ?? "")
        case .retryClaim: return context.signedIn ? .associating(ssid: context.ssid ?? "") : nil
        case .keepWaiting:
            // On the account already: wait for it to report. Otherwise look again.
            return context.accountDeviceID != nil ? .waitingForDevice(elapsed: 0) : .associating(ssid: context.ssid ?? "")
        case .chooseMode:
            return context.accountDeviceID.map { .choosingMode(deviceID: $0) }
        case .skipMode:
            return .finished(.init(ssid: context.ssid, accountDeviceID: context.accountDeviceID,
                                   alreadyOwned: context.alreadyOwned, onAccount: context.accountDeviceID != nil))
        }
    }

    /// What the machine needs to know beyond the state itself.
    struct Context: Equatable, Sendable {
        var signedIn = false
        var ssid: String?
        var accountDeviceID: Int?
        var alreadyOwned = false

        init(signedIn: Bool = false, ssid: String? = nil, accountDeviceID: Int? = nil, alreadyOwned: Bool = false) {
            self.signedIn = signedIn
            self.ssid = ssid
            self.accountDeviceID = accountDeviceID
            self.alreadyOwned = alreadyOwned
        }
    }

    // MARK: The account side, as pure decisions

    /// A device as the account lists it (GET /v1/devices), reduced to what onboarding needs.
    struct AccountDevice: Equatable, Sendable {
        let id: Int
        let name: String
        /// The SD- id the device reported (its health `device_id`); nil until it has reported.
        let hardwareID: String?
        let online: Bool
        let activePreset: String?
    }

    /// The account's row for the device on the other end of the Bluetooth link.
    ///
    /// Matched by the SD- id the device itself reported over Bluetooth, never
    /// by name or by "the only one": an account with two cameras would
    /// otherwise finish setting up the wrong one.
    static func match(_ devices: [AccountDevice], hardwareID: String?) -> AccountDevice? {
        guard let hardwareID, !hardwareID.isEmpty else { return nil }
        return devices.first { $0.hardwareID?.caseInsensitiveCompare(hardwareID) == .orderedSame }
    }

    /// After a claim: a device the claim added that has reported a DIFFERENT
    /// SD- id than the one set up over Bluetooth -- the claim code was for
    /// another unit. nil while nothing contradicts the claim.
    static func claimedOther(before: [AccountDevice], after: [AccountDevice], hardwareID: String?) -> AccountDevice? {
        let known = Set(before.map(\.id))
        return after.first { device in
            !known.contains(device.id) && device.hardwareID != nil
                && device.hardwareID?.caseInsensitiveCompare(hardwareID ?? "") != .orderedSame
        }
    }

    /// A claim code as printed on the card: case and spaces forgiven, nothing else changed.
    static func normalizeClaimCode(_ text: String) -> String? {
        let cleaned = text.uppercased().filter { !$0.isWhitespace }
        guard !cleaned.isEmpty, cleaned.count <= 64,
              cleaned.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-") }) else { return nil }
        return cleaned
    }

    /// The backend's refusal of a claim, in the person's terms.
    static func claimMessage(status: Int?, code: String?, detail: String?) -> String {
        if code == "device_already_claimed" || status == 409 {
            // M8.4: "its owner has to remove it first" named a step that exists
            // nowhere -- neither the app nor the server can release a device.
            return "This SensorDyme is already on another account, so it can't be added to this one. "
                + "If it's yours, sign in with the account that first added it. Its Wi-Fi is set up "
                + "either way, so you can close setup."
        }
        switch status {
        case 404: return "That claim code isn't one the server knows. Check it against the card that came with your SensorDyme."
        case 429: return "Too many tries. Wait a few minutes, then enter the claim code again."
        case 401: return "You've been signed out. Sign in again, then add the device."
        default: return detail.map { "The server couldn't add the device: \($0)" } ?? "The server couldn't add the device. Try again."
        }
    }

    /// A monitoring mode the device can run, as onboarding offers it.
    struct Mode: Equatable, Sendable {
        let id: String
        let title: String
        let detail: String
        let available: Bool
        /// Why it is unavailable, when it is.
        let reason: String?
    }

    /// The mode to preselect: the one the device already runs, else the first
    /// available. Never one the device says it cannot run.
    static func recommendedMode(_ modes: [Mode], active: String?) -> Mode? {
        let usable = modes.filter(\.available)
        if let active, let current = usable.first(where: { $0.id == active || active.hasPrefix($0.id + ":") }) {
            return current
        }
        return usable.first
    }
}
