import Combine
import Foundation

/// The account side of onboarding: what SetupFlow asks the backend. The app's
/// implementation is `AppStateOnboardingAccount` (SetupAccount.swift, over
/// AppState.backend); the cross-language harness talks HTTP to sim.backend.
@MainActor
protocol OnboardingAccount: AnyObject {
    var isSignedIn: Bool { get }
    func accountDevices() async throws -> [Onboarding.AccountDevice]
    /// POST /v1/devices/claim.
    func claim(code: String) async throws
    /// The modes the device can run, and the one it runs now.
    func modes(deviceID: Int) async throws -> (modes: [Onboarding.Mode], active: String?)
    func activate(mode: String, deviceID: Int) async throws
    /// A refusal in the person's terms (Onboarding.claimMessage for a claim).
    func explain(_ error: Error, claiming: Bool) -> String
}

/// Finding the device on the local network after it joined (mDNS), for the
/// live feed. The app's implementation wraps DeviceLink.
@MainActor
protocol OnboardingLocalLink: AnyObject {
    /// The device's local address, if it answers on this network.
    func find(deviceID: String?) async -> String?
    /// Why discovery cannot work from here, when that is knowable.
    var problem: String? { get }
    func stop()
}

/// Drives onboarding from "where is my device" to "it is watching", across
/// Bluetooth, the device's new network and the account.
///
/// `Onboarding` (SetupOnboarding.swift) holds every legal step; this turns
/// what the radio and the backend report into its events and does what each
/// new state asks for. The interesting part of provisioning is the seams --
/// Bluetooth to Wi-Fi to the account -- and this owns all of them.
@MainActor
final class SetupFlow: ObservableObject {
    struct Transition: Equatable {
        let from: Onboarding.State
        let event: Onboarding.Event
        let to: Onboarding.State
    }

    @Published private(set) var state: Onboarding.State = .discovering
    @Published private(set) var context = Onboarding.Context()
    /// Every accepted move, oldest first (the debug panel; evidence for the harness).
    @Published private(set) var transitions: [Transition] = []
    /// How long discovery has been running, so the UI can offer help rather
    /// than spin indefinitely.
    @Published private(set) var searchingSeconds = 0
    /// The device's address on this network, once it answers there.
    @Published private(set) var localURL: String?
    @Published private(set) var modes: [Onboarding.Mode] = []
    @Published private(set) var recommendedMode: String?
    /// The network the device was told to join, so later screens can name it.
    @Published private(set) var targetSSID = ""
    /// Whether this device asked for its setup code (kept after Bluetooth closes).
    @Published private(set) var requiresCode = false

    let ble: BLEProvisioner
    private let localLink: OnboardingLocalLink?
    private(set) var account: OnboardingAccount?
    /// How long to wait for the device to report in after it joined. Generous:
    /// it has to associate, get an address, reach the server and report.
    var reportBudgetSeconds = 120
    var pollSeconds: Double = 3

    private var observers: [AnyCancellable] = []
    private var work: Task<Void, Never>?
    private var searchClock: Task<Void, Never>?
    /// The account's devices before the claim, to notice a code for another unit.
    private var devicesBeforeClaim: [Onboarding.AccountDevice] = []

    init(ble: BLEProvisioner, localLink: OnboardingLocalLink?, account: OnboardingAccount? = nil) {
        self.ble = ble
        self.localLink = localLink
        self.account = account
    }

    func attach(account: OnboardingAccount?) {
        self.account = account
        context.signedIn = account?.isSignedIn ?? false
    }

    // MARK: Lifecycle

    func start() {
        observe()
        context.signedIn = account?.isSignedIn ?? false
        state = .discovering
        ble.startScan()
        startSearchClock()
    }

    func stop() {
        work?.cancel(); work = nil
        searchClock?.cancel(); searchClock = nil
        ble.stop()
        localLink?.stop()
    }

    // MARK: What the person does

    func connect(to device: DiscoveredDevice) {
        guard apply(.deviceChosen(name: device.name)) else { return }
        ble.connect(id: device.id)
    }

    /// The setup code as typed, or the QR text. Problems with the entry itself
    /// (empty, malformed, another device's label) are answered here without a
    /// word to the device.
    @discardableResult
    func prove(entry: String) -> SetupCodeEntry.Problem? {
        switch SetupCodeEntry.label(from: entry, connectedDeviceID: ble.deviceID) {
        case .failure(let problem):
            return problem
        case .success(let label):
            guard apply(.proofStarted) else { return nil }
            ble.prove(label: label)
            return nil
        }
    }

    func send(ssid: String, password: String) {
        targetSSID = ssid
        context.ssid = ssid
        guard apply(.networkSent(ssid: ssid)) else { return }
        ble.send(ssid: ssid, password: password)
    }

    func submitClaim(code: String) -> Bool {
        guard let normalized = Onboarding.normalizeClaimCode(code), let account else { return false }
        guard apply(.claimSubmitted) else { return false }
        work?.cancel()
        work = Task { [weak self] in
            guard let self else { return }
            do {
                self.devicesBeforeClaim = (try? await account.accountDevices()) ?? []
                try await account.claim(code: normalized)
                guard !Task.isCancelled else { return }
                self.apply(.claimed)
            } catch {
                guard !Task.isCancelled else { return }
                self.apply(.claimRefused(account.explain(error, claiming: true)))
            }
        }
        return true
    }

    func chooseMode(_ mode: String) {
        guard case .choosingMode(let deviceID) = state, let account, apply(.modeChosen(mode)) else { return }
        work?.cancel()
        work = Task { [weak self] in
            guard let self else { return }
            do {
                try await account.activate(mode: mode, deviceID: deviceID)
                self.apply(.modeActivated(mode))
            } catch {
                self.apply(.modeRefused(account.explain(error, claiming: false)))
            }
        }
    }

    func skipMode() { apply(.modeSkipped) }

    func recover(_ recovery: Onboarding.Recovery) {
        guard apply(.recover(recovery)) else { return }
        switch recovery {
        case .searchAgain:
            work?.cancel(); work = nil
            ble.startScan()
            startSearchClock()
        case .enterCodeAgain, .chooseNetwork, .skipMode, .retryClaim, .keepWaiting, .checkOnline, .chooseMode:
            break  // the new state's reaction does the rest
        }
    }

    // MARK: Events

    /// Apply one event; false when it is not a legal move from here.
    @discardableResult
    func apply(_ event: Onboarding.Event) -> Bool {
        guard let next = Onboarding.next(state, event, context: context) else { return false }
        let from = state
        if case .identified(let needsCode) = event { requiresCode = needsCode }
        transitions.append(Transition(from: from, event: event, to: next))
        state = next
        react(to: next, from: from)
        return true
    }

    private func observe() {
        guard observers.isEmpty else { return }
        // SwiftUI does not propagate through nested ObservableObjects: the view
        // observes this flow, so a change to `ble.discovered` reaches nobody
        // unless it is republished here.
        observers.append(ble.objectWillChange.sink { [weak self] in self?.objectWillChange.send() })
        observers.append(ble.$state.removeDuplicates().sink { [weak self] newState in
            // $state publishes before the property changes; read it after.
            Task { @MainActor in self?.absorb(newState) }
        })
    }

    /// The Bluetooth side's state, as onboarding events.
    func absorb(_ bleState: BLEProvisioner.State) {
        switch bleState {
        case .bluetoothOff:
            switch state.stage {
            case .find, .connect: apply(.bluetoothOff)
            default: apply(.linkFailed("Bluetooth was turned off. Turn it on and search again."))
            }
        case .scanning:
            apply(.bluetoothOn)
        case .idle, .connecting, .identifying, .proving, .sending:
            break
        case .ready:
            switch state {
            case .connecting, .identifying:
                apply(.identified(requiresCode: ble.requiresSetupCode))
            case .proving, .sending:
                let rejection = ble.proofRejection
                apply(.proofRefused(message: rejection?.message ?? "The device didn't accept the setup code.",
                                    lockedUntil: ble.lockedUntil))
            default:
                break
            }
        case .proven:
            apply(.proven)
        case .deviceConnecting:
            apply(.deviceJoining)
        case .success:
            apply(.deviceJoined)
        case .joinFailed(let reason):
            apply(.joinRefused(reason))
        case .failed(let reason):
            apply(.linkFailed(reason))
        }
    }

    // MARK: What each state asks for

    private func react(to state: Onboarding.State, from: Onboarding.State) {
        switch (from, state) {
        case (.waitingForDevice, .waitingForDevice):
            break  // a tick of the wait already running
        case (_, .associating):
            beginAssociation()
        case (_, .waitingForDevice):
            beginWaitingForDevice()
        case (_, .choosingMode(let deviceID)):
            loadModes(deviceID: deviceID)
        default:
            break
        }
    }

    /// The device joined. Bluetooth has done its job; holding the link open
    /// keeps the device's radio busy. Now: is this device on the account?
    private func beginAssociation() {
        let hardwareID = ble.deviceID
        work?.cancel()
        work = Task { [weak self] in
            guard let self else { return }
            // After the transition that brought us here, not inside it.
            await Task.yield()
            self.ble.stop()
            guard let account = self.account, account.isSignedIn else {
                // No account to add it to: finish when it answers on this network.
                let url = await self.findLocally(hardwareID, budget: 60)
                self.localURL = url
                self.apply(.noAccount(found: url != nil))
                return
            }
            do {
                let devices = try await account.accountDevices()
                guard !Task.isCancelled else { return }
                if let owned = Onboarding.match(devices, hardwareID: hardwareID) {
                    self.context.accountDeviceID = owned.id
                    self.context.alreadyOwned = true
                    self.apply(.alreadyOnAccount(deviceID: owned.id))
                } else {
                    self.apply(.needsClaim)
                }
            } catch {
                guard !Task.isCancelled else { return }
                self.apply(.claimRefused(account.explain(error, claiming: false)))
            }
        }
    }

    /// On the account; wait for the device itself to report in. The SD- id it
    /// reported over Bluetooth is what identifies it -- never "the newest".
    private func beginWaitingForDevice() {
        let hardwareID = ble.deviceID
        guard let account else { return }
        work?.cancel()
        let budget = reportBudgetSeconds
        let poll = pollSeconds
        work = Task { [weak self] in
            guard let self else { return }
            let started = Date()
            // The live feed is found on the side; it does not hold setup up.
            Task { [weak self] in
                guard let self, self.localURL == nil else { return }
                self.localURL = await self.findLocally(hardwareID, budget: 30)
            }
            while !Task.isCancelled {
                let elapsed = Int(Date().timeIntervalSince(started))
                if let devices = try? await account.accountDevices() {
                    guard !Task.isCancelled else { return }
                    if let mine = Onboarding.match(devices, hardwareID: hardwareID) {
                        self.context.accountDeviceID = mine.id
                        if mine.online {
                            self.apply(.deviceOnline(deviceID: mine.id))
                            return
                        }
                    } else if !self.context.alreadyOwned,
                              let other = Onboarding.claimedOther(before: self.devicesBeforeClaim, after: devices,
                                                                  hardwareID: hardwareID) {
                        self.apply(.claimRefused(
                            "That claim code added \(other.name) (\(other.hardwareID ?? "another device")), not the "
                            + "SensorDyme you just set up (\(hardwareID ?? "unknown")). Check the card that came with this one."))
                        return
                    }
                }
                if elapsed >= budget { break }
                if case .waitingForDevice(let shown) = self.state, shown != elapsed {
                    self.apply(.waitTick(elapsed: elapsed))
                }
                try? await Task.sleep(nanoseconds: UInt64(poll * 1_000_000_000))
            }
            guard !Task.isCancelled else { return }
            self.apply(.deviceSilent(
                "Your SensorDyme joined \(self.context.ssid ?? "the network") but hasn't reported to the server yet. "
                + "Check that network reaches the internet. It will appear in your devices as soon as it does."))
        }
    }

    private func loadModes(deviceID: Int) {
        guard let account else { return }
        work?.cancel()
        work = Task { [weak self] in
            guard let self else { return }
            do {
                let (modes, active) = try await account.modes(deviceID: deviceID)
                self.modes = modes
                self.recommendedMode = Onboarding.recommendedMode(modes, active: active)?.id
            } catch {
                self.modes = []
                self.recommendedMode = nil
            }
        }
    }

    private func findLocally(_ hardwareID: String?, budget: Int) async -> String? {
        guard let localLink else { return nil }
        let started = Date()
        while !Task.isCancelled, Date().timeIntervalSince(started) < Double(budget) {
            if let url = await localLink.find(deviceID: hardwareID) { return url }
            try? await Task.sleep(nanoseconds: 3_000_000_000)
        }
        return nil
    }

    private func startSearchClock() {
        searchClock?.cancel()
        searchingSeconds = 0
        searchClock = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                guard let self else { return }
                guard case .discovering = self.state else { continue }
                self.searchingSeconds += 1
            }
        }
    }
}
