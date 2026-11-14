import Foundation

/// Bluetooth-LE client for provisioning a SensorDyme device's Wi-Fi.
///
/// Counterpart of services/ble_gatt.py on the device: when the device has no
/// Wi-Fi (or for a few minutes after it boots) it advertises
/// "SensorDyme-XXXX"; this finds it, connects, reads who it is and how it
/// takes credentials, proves the setup code when the device asks for one
/// (`auth: pop`, BLEPoPFlow), sends {ssid, psk} and follows progress on the
/// status characteristic. BLE is used ONLY for setup and lightweight commands
/// -- the live camera feed always streams over Wi-Fi.
///
/// Talks to a `BLETransport`, not to CoreBluetooth: the production link is
/// `CoreBluetoothTransport`; DEBUG builds can run against the virtual
/// peripheral (sim/ble_peripheral.py), and the logic tests script the link.

/// One physical device seen over Bluetooth.
///
/// Identified by the peripheral UUID, which is stable for a given device on a
/// given phone. Names are display only -- a device can present more than one,
/// and treating a name as identity is what produced duplicates.
struct DiscoveredDevice: Identifiable, Hashable {
    let id: UUID
    let name: String
    let rssi: Int

    /// How close it is. The device in the room reads "Nearby".
    var proximity: String {
        if rssi >= -60 { return "Nearby" }
        if rssi >= -80 { return "In range" }
        return "Weak"
    }
}

/// One Wi-Fi network the device can see.
struct VisibleNetwork: Identifiable, Hashable {
    let ssid: String
    let signal: Int      // 0-100, as NetworkManager reports it
    let secured: Bool

    var id: String { ssid }

    var strength: String {
        switch signal {
        case 70...: "Strong"
        case 40..<70: "Good"
        default: "Weak"
        }
    }

    /// Coarse strength, for ordering. Deliberately coarse: the point is that
    /// ordinary signal jitter must not be able to move a row.
    var strengthRank: Int {
        switch signal {
        case 70...: 2
        case 40..<70: 1
        default: 0
        }
    }
}

@MainActor
final class BLEProvisioner: ObservableObject {

    enum State: Equatable {
        case idle
        case bluetoothOff
        case scanning
        case connecting(name: String)
        /// Connected; reading the device info (its id and how it takes credentials).
        case identifying(name: String)
        /// Identified. A device that requires the setup code needs `prove` first.
        case ready(name: String)
        /// The proof-of-possession exchange is in flight.
        case proving(name: String)
        /// The device holds the setup code and proved it back: credentials may go.
        case proven(name: String)
        case sending
        case deviceConnecting             // device is trying the network we sent
        case success(ip: String?)
        /// The device tried the network and could not join. Bluetooth is still
        /// up: another network (or the password again) can be sent.
        case joinFailed(String)
        case failed(String)
    }

    /// How long each wait may take. A struct so tests can shorten them.
    struct Timing {
        /// CoreBluetooth never times a connection out by itself, so a device
        /// that walked out of range left "Pairing…" spinning for ever.
        var connect: Double = 20
        /// Reading the device info after connecting.
        var identify: Double = 10
        /// One proof exchange; the device's own challenge lives 30 s.
        var proof: Double = 20
        /// The write plus a join of up to 45 s, generously.
        var send: Double = 90
        var statusPoll: Double = 2
        var statusPolls = 45
        /// A proof older than this is renewed before credentials go (the device keeps one 120 s).
        var provenFresh: Double = 100
    }

    @Published private(set) var state: State = .idle
    /// What the radio last reported (CBManagerState as BLERadioState), nil
    /// until it has. Read-only presentation: `state` folds "unauthorized" and
    /// "powered off" into `.bluetoothOff`, and the person needs to know which,
    /// because the fixes differ (Settings for permission, Control Centre for power).
    @Published private(set) var radio: BLERadioState?
    /// Discovered devices, strongest signal first.
    ///
    /// Signal strength is kept because a name alone can be ambiguous: a device
    /// that has been reflashed or test-configured may advertise under a stale
    /// alias. The nearer, stronger signal is almost always the one in the room.
    @Published private(set) var discovered: [DiscoveredDevice] = []
    /// Networks the DEVICE can see. Scanned on the Pi, not the phone: iOS
    /// cannot enumerate Wi-Fi at all, and the Pi's list is the one that
    /// matters -- it is the machine that has to associate.
    @Published private(set) var networks: [VisibleNetwork] = []
    /// Whether the DEVICE is scanning right now, reported by the device itself,
    /// so an empty list can mean "still looking" or "there are none" truthfully.
    @Published private(set) var scanningNetworks = false
    /// Why the network list is unavailable, when it is: "the device saw nothing"
    /// and "the app never managed to ask" look identical and have different fixes.
    @Published private(set) var networkProblem: String?
    /// The state the flow was in when it failed, so the checklist can mark the
    /// one step that broke and leave the completed ones standing.
    @Published private(set) var stateBeforeFailure: State?
    /// The connected device's own SD- id, from its device-info characteristic
    /// (0002 `id`) or its status (`device_id`). Setup uses it to find THIS
    /// device on Wi-Fi and on the account afterwards.
    @Published private(set) var deviceID: String?
    /// How the connected device accepts Wi-Fi credentials, from its device
    /// info: "pop" needs the setup code; "legacy" or nil is the plain write.
    @Published private(set) var deviceAuth: String?
    /// The device's answer to the last proof that did not succeed (wrong code,
    /// lockout, busy, expired), while the link stays up for another try.
    @Published private(set) var proofRejection: BLEPoPFlow.Rejection?
    /// Lockouts the devices reported, by device id: when each said it would
    /// accept a code again (its `retry_after_s`, counted from when it said so).
    @Published private(set) var lockouts: [String: Date] = [:]

    var timing = Timing()
    /// The clock lockouts and proof freshness are measured on (tests replace it).
    var now: () -> Date = Date.init

    private let transport: BLETransport
    private var linkID: UUID?
    private var linkName = "SensorDyme device"
    private var available: Set<ProvisioningCharacteristic> = []
    /// The proof-of-possession exchange in flight or last proven, if any.
    private var popFlow: BLEPoPFlow?
    /// The session for the label last used, kept so a retry or a renewal does
    /// not derive the (deliberately slow) code key again.
    private var labelSession: BLEProofOfPossession.Session?
    private var provenAt: Date?
    /// Credentials waiting for a renewed proof, and whether that renewal happened.
    private var pendingCredentials: (ssid: String, psk: String?)?
    private var renewedForSend = false

    private var sendTimeout: Task<Void, Never>?
    private var statusPoll: Task<Void, Never>?
    private var connectTimeout: Task<Void, Never>?
    private var identifyTimeout: Task<Void, Never>?
    private var proofTimeout: Task<Void, Never>?
    /// Which status answers the outstanding credentials write. See
    /// `ProvisioningStatus.AttemptTracker`.
    private var attempts = ProvisioningStatus.AttemptTracker()
    /// Whether the end of the device's current scan should be followed by one
    /// more read of the list. Each read starts a scan on the device, so reading
    /// on every "scan finished" status kept the radio rescanning for ever; an
    /// asked-for read earns exactly one follow-up.
    private var followUpReadWanted = false

    init(transport: BLETransport? = nil) {
        self.transport = transport ?? BLETransports.make()
        self.transport.onEvent = { [weak self] event in self?.handle(event) }
    }

    // MARK: Lifecycle

    func startScan() {
        // A retry can come while the previous link is still up (a refused
        // password leaves Bluetooth connected). Drop every reference to it
        // first: a late callback from the old link must not touch the new one.
        if let linkID { transport.disconnect(linkID) }
        clearLink()
        state = .scanning
        discovered = []
        networks = []
        deviceID = nil
        transport.startScan()
    }

    func stop() {
        transport.stopScan()
        if let linkID { transport.disconnect(linkID) }
        clearLink()
        labelSession = nil
        state = .idle
    }

    func connect(id: UUID) {
        let name = discovered.first { $0.id == id }?.name ?? "SensorDyme device"
        guard discovered.contains(where: { $0.id == id }), transport.connect(id) else {
            fail("That SensorDyme is no longer in range. Search again.")
            return
        }
        clearLink()
        linkID = id
        linkName = name
        state = .connecting(name: name)
        connectTimeout = after(timing.connect) { [weak self] in
            guard let self, case .connecting = self.state else { return }
            if let linkID = self.linkID { self.transport.disconnect(linkID) }
            self.clearLink()
            self.fail("Couldn't connect to \(name). Move closer to it and search again.")
        }
    }

    /// Forget everything that belongs to one Bluetooth link.
    private func clearLink() {
        linkID = nil
        available = []
        popFlow = nil
        provenAt = nil
        pendingCredentials = nil
        renewedForSend = false
        deviceAuth = nil
        proofRejection = nil
        for task in [connectTimeout, identifyTimeout, proofTimeout, sendTimeout, statusPoll] { task?.cancel() }
        connectTimeout = nil; identifyTimeout = nil; proofTimeout = nil; sendTimeout = nil; statusPoll = nil
        // Per link: attempt numbers are one device's counter, and the next
        // connection (perhaps to another device) reads its own baseline.
        attempts = ProvisioningStatus.AttemptTracker()
        followUpReadWanted = false
        scanningNetworks = false
    }

    private func isCurrent(_ id: UUID) -> Bool { linkID == id }

    /// Whether a credential write is outstanding. Before this is true the
    /// device's status is a description of itself, not an answer to anything.
    private var awaitingOutcome: Bool {
        switch state {
        case .sending, .deviceConnecting: true
        default: false
        }
    }

    /// Whether the connected device needs the setup code before credentials.
    var requiresSetupCode: Bool { deviceAuth == "pop" || available.contains(.auth) }

    /// When the connected device said it would accept a setup code again, if
    /// it is refusing one now. The device's own number; this only counts down.
    var lockedUntil: Date? {
        guard let deviceID, let until = lockouts[deviceID], until > now() else { return nil }
        return until
    }

    /// Clear everything that points at a link which no longer exists.
    private func handleDisconnect() {
        let was = state
        clearLink()
        networks = []
        switch was {
        case .idle, .success, .failed:
            // We asked for this: SetupFlow drops the link after a successful
            // join, and stop() sets .idle before this fires.
            break
        case .sending, .deviceConnecting:
            // Do not blame the password. The device may well have joined; we
            // just lost the channel it would have told us on.
            fail("Bluetooth to your SensorDyme dropped while it was joining. It may have "
                 + "joined anyway — search again to check before re-entering the password.")
        default:
            fail("Bluetooth to your SensorDyme dropped. Move closer to it and search again.")
        }
    }

    /// Move to `.failed`, remembering what was in progress.
    private func fail(_ reason: String) {
        if case .failed = state {} else { stateBeforeFailure = state }
        attempts.finish()
        for task in [identifyTimeout, proofTimeout, sendTimeout, statusPoll] { task?.cancel() }
        state = .failed(reason)
    }

    /// Order networks so the list does not rearrange itself under a finger:
    /// bucketed strength first, then name, which never changes.
    static func stableOrder(_ networks: [VisibleNetwork]) -> [VisibleNetwork] {
        networks.sorted {
            $0.strengthRank != $1.strengthRank
                ? $0.strengthRank > $1.strengthRank
                : $0.ssid.localizedCaseInsensitiveCompare($1.ssid) == .orderedAscending
        }
    }

    // MARK: Networks

    /// Ask the device for its Wi-Fi list. Reports when it cannot, rather than
    /// returning quietly -- the quiet version left an empty list and a dead button.
    func refreshNetworks() {
        guard let linkID else {
            networkProblem = "Not connected to your SensorDyme."
            return
        }
        guard available.contains(.networks) else {
            networkProblem = "This SensorDyme didn't offer a Wi-Fi list. It may need a software update."
            return
        }
        networkProblem = nil
        followUpReadWanted = true
        transport.read(.networks, on: linkID)
    }

    private func readNetworks() {
        guard let linkID, available.contains(.networks) else { return }
        transport.read(.networks, on: linkID)
    }

    // MARK: Proof of possession

    /// Prove the setup code to the connected device (docs/BLE_PROOF_OF_POSSESSION.md).
    ///
    /// Ends in `.proven`, or back in `.ready` with `proofRejection` saying why
    /// (wrong code, locked with the device's wait, busy, expired) while the link
    /// stays up -- or in `.failed` when the device is not the one on the label
    /// or cannot prove the code back, which ends the link.
    func prove(label: BLEProofOfPossession.SetupLabel) {
        guard let linkID, available.contains(.auth) else {
            fail("This SensorDyme didn't offer secure setup over Bluetooth. It may need a software update.")
            return
        }
        switch state {
        case .ready, .proven, .joinFailed: break
        default: return
        }
        state = .proving(name: linkName)
        proofRejection = nil
        popFlow = nil
        provenAt = nil
        proofTimeout = after(timing.proof) { [weak self] in
            guard let self, case .proving = self.state else { return }
            self.popFlow = nil
            self.reject(.init(kind: .other, message: "The device didn't answer the setup code in time. Move closer to it and try again.",
                              retryAfterSeconds: nil))
        }
        Task { @MainActor in
            let session: BLEProofOfPossession.Session
            if let cached = self.labelSession, cached.label == label {
                session = cached.renewed()
            } else {
                do {
                    session = try await Task.detached(priority: .userInitiated) {
                        try BLEProofOfPossession.Session(label: label)
                    }.value
                } catch let failure as BLEProofOfPossession.Failure {
                    self.reject(BLEPoPFlow.rejection(failure))
                    return
                } catch {
                    self.reject(.init(kind: .other, message: "This phone couldn't prepare the secure setup. Try again.",
                                      retryAfterSeconds: nil))
                    return
                }
            }
            guard self.isCurrent(linkID), case .proving = self.state else { return }
            self.labelSession = session
            let flow = BLEPoPFlow(session: session)
            self.popFlow = flow
            self.perform(flow.start(), flow: flow)
        }
    }

    private func perform(_ action: BLEPoPFlow.Action, flow: BLEPoPFlow) {
        guard let linkID else { return }
        switch action {
        case .write(let data):
            transport.write(data, to: .auth, on: linkID)
        case .proven:
            proofTimeout?.cancel()
            provenAt = now()
            proofRejection = nil
            if let deviceID { lockouts[deviceID] = nil }
            state = .proven(name: linkName)
        case .accepted:
            // The device records the attempt before it answers, so from here on
            // a status describes this write.
            pendingCredentials = nil
            attempts.acknowledgeWrite()
            startStatusPolling()
        case .failed:
            let rejection = flow.rejection ?? .init(kind: .other, message: "The setup failed.", retryAfterSeconds: nil)
            if case .sending = state, rejection.kind == .expired, !renewedForSend, let pending = pendingCredentials,
               let session = labelSession {
                // The proof lapsed (or the device moved on after a failed join):
                // one fresh handshake, then the same credentials.
                renewedForSend = true
                let renewed = BLEPoPFlow(session: session.renewed(), ssid: pending.ssid, psk: pending.psk)
                popFlow = renewed
                perform(renewed.start(), flow: renewed)
                return
            }
            popFlow = nil
            reject(rejection)
        }
    }

    /// A proof (or a secure send) that did not go through.
    private func reject(_ rejection: BLEPoPFlow.Rejection) {
        proofTimeout?.cancel()
        if rejection.endsLink {
            // Not the device on the label, or it cannot prove the code: nothing
            // more is said to it.
            if let linkID { transport.disconnect(linkID) }
            clearLink()
            fail(rejection.message)
            return
        }
        if let seconds = rejection.retryAfterSeconds, let deviceID {
            lockouts[deviceID] = now().addingTimeInterval(TimeInterval(seconds))
        }
        if awaitingOutcome {
            attempts.finish()
            sendTimeout?.cancel()
            statusPoll?.cancel()
        }
        pendingCredentials = nil
        proofRejection = rejection
        state = .ready(name: linkName)
    }

    // MARK: Credentials

    /// Send Wi-Fi credentials to the connected device. A device that requires
    /// the setup code must have been proven first (`prove`); its credentials go
    /// encrypted on the auth characteristic, renewing the proof when it lapsed.
    func send(ssid: String, password: String) {
        if requiresSetupCode {
            sendSecure(ssid: ssid, psk: password.isEmpty ? nil : password)
            return
        }
        guard let linkID, available.contains(.credentials),
              let payload = try? JSONSerialization.data(withJSONObject: ["ssid": ssid, "psk": password])
        else {
            fail("not connected to a device")
            return
        }
        state = .sending
        attempts.beginWrite(ssid: ssid)
        transport.write(payload, to: .credentials, on: linkID)
        startSendTimeout()
        startStatusPolling()
    }

    private func sendSecure(ssid: String, psk: String?) {
        guard linkID != nil, available.contains(.auth), let session = labelSession else {
            fail("This SensorDyme needs its setup code before it will take a Wi-Fi password.")
            return
        }
        state = .sending
        attempts.beginWrite(ssid: ssid)
        startSendTimeout()
        pendingCredentials = (ssid, psk)
        if let flow = popFlow, flow.step == .proven, let provenAt,
           now().timeIntervalSince(provenAt) < timing.provenFresh {
            renewedForSend = false
            perform(flow.sendCredentials(ssid: ssid, psk: psk), flow: flow)
        } else {
            // After a failed join, or a proof that is about to lapse: the device
            // wants a new handshake. Same label, same code key, new nonces.
            renewedForSend = true
            let flow = BLEPoPFlow(session: session.renewed(), ssid: ssid, psk: psk)
            popFlow = flow
            perform(flow.start(), flow: flow)
        }
    }

    /// Read the status characteristic on a timer while a join is in flight.
    /// Notifications fail silently; a read cannot be missed the same way.
    private func startStatusPolling() {
        statusPoll?.cancel()
        let interval = timing.statusPoll
        let polls = timing.statusPolls
        statusPoll = Task { [weak self] in
            for _ in 0..<polls {
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                guard !Task.isCancelled, let self else { return }
                guard self.awaitingOutcome else { return }
                if let linkID = self.linkID, self.available.contains(.status) {
                    self.transport.read(.status, on: linkID)
                }
            }
        }
    }

    /// Give up waiting, with a reason. Generous on purpose: firing before the
    /// device has finished trying would report a failure that did not happen.
    private func startSendTimeout() {
        sendTimeout?.cancel()
        sendTimeout = after(timing.send) { [weak self] in
            guard let self, self.awaitingOutcome else { return }
            self.fail(
                "The device didn't confirm within a minute. It's usually the Wi-Fi password — "
                + "check it and try again. Other causes: the network needs a browser sign-in "
                + "(hotel or campus Wi-Fi), or it's 5GHz-only. A phone hotspot is the most "
                + "reliable way to get the device online."
            )
        }
    }

    private func after(_ seconds: Double, _ body: @escaping @MainActor () -> Void) -> Task<Void, Never> {
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
            guard !Task.isCancelled else { return }
            body()
        }
    }

    // MARK: Link events

    func handle(_ event: BLETransportEvent) {
        switch event {
        case .radio(let radio):
            radioChanged(radio)
        case .discovered(let advert):
            sighted(advert)
        case .connected:
            break  // the characteristics follow
        case .connectFailed(let id, let reason):
            guard isCurrent(id) else { return }
            clearLink()
            fail(reason ?? "could not connect")
        case .disconnected(let id):
            // Only the link in use: a late callback from one already let go of
            // must not tear down the search or connection that replaced it.
            guard isCurrent(id) else { return }
            handleDisconnect()
        case .serviceMissing(let id):
            guard isCurrent(id) else { return }
            transport.disconnect(id)
            clearLink()
            fail("device is missing the provisioning service")
        case .characteristics(let id, let found):
            guard isCurrent(id) else { return }
            characteristicsFound(found, on: id)
        case .value(let id, let characteristic, let data):
            guard isCurrent(id) else { return }
            received(data, on: characteristic)
        case .readFailed(let id, let characteristic, let reason):
            guard isCurrent(id) else { return }
            if characteristic == .networks {
                networkProblem = "Couldn't read the device's Wi-Fi list: \(reason)"
            } else if characteristic == .deviceInfo, case .identifying = state {
                identified()
            }
        case .wrote(let id, let characteristic, let error):
            guard isCurrent(id) else { return }
            if let error {
                fail(error)
            } else if characteristic == .credentials {
                // The device records the new attempt before it answers the
                // write, so from here on a status describes this write.
                attempts.acknowledgeWrite()
            }
        }
    }

    private func radioChanged(_ radio: BLERadioState) {
        // Recorded as reported, for the setup screen to say WHICH problem it is
        // (permission, off, unsupported); nothing below depends on it.
        self.radio = radio
        switch radio {
        case .poweredOn:
            // Back from off: say so, rather than staying on "Bluetooth is off".
            if case .bluetoothOff = state { state = .scanning }
        case .unauthorized, .poweredOff:
            // Off mid-setup ends the link without a disconnect callback.
            let wasJoining = awaitingOutcome
            clearLink()
            if wasJoining {
                fail("Bluetooth was turned off while your SensorDyme was joining. It may have "
                     + "joined anyway — search again to check before re-entering the password.")
            } else {
                state = .bluetoothOff
            }
        case .unsupported:
            fail("This phone can't use Bluetooth, which setup needs.")
        case .other:
            break
        }
    }

    private func sighted(_ advert: BLEAdvertisement) {
        let name = advert.advertisedName ?? advert.cachedName ?? "SensorDyme device"
        if let index = discovered.firstIndex(where: { $0.id == advert.id }) {
            // Same device, fresher reading. Prefer the advertised name: it is
            // what the device says right now, not what iOS remembers.
            discovered[index] = DiscoveredDevice(id: advert.id, name: advert.advertisedName ?? discovered[index].name,
                                                 rssi: advert.rssi)
        } else {
            discovered.append(DiscoveredDevice(id: advert.id, name: name, rssi: advert.rssi))
        }
        // Strongest first: the device you are standing next to is offered first.
        discovered.sort { $0.rssi > $1.rssi }
    }

    private func characteristicsFound(_ found: Set<ProvisioningCharacteristic>, on id: UUID) {
        connectTimeout?.cancel()
        available = found
        if found.contains(.status) {
            transport.subscribe(.status, on: id)
            // Read once now: the attempt number it carries is the baseline a
            // later outcome has to beat.
            transport.read(.status, on: id)
        }
        if found.contains(.auth) {
            // Replies to each proof-of-possession write are notified back.
            transport.subscribe(.auth, on: id)
        }
        guard found.contains(.credentials) else {
            // Without it nothing can be sent, and the flow used to sit on
            // "Pairing…" for ever.
            fail("This SensorDyme didn't offer Wi-Fi setup over Bluetooth. It may need a software update.")
            return
        }
        // Ask immediately: the device scans on read, and the answer is wanted
        // the moment the network screen appears.
        refreshNetworks()
        guard found.contains(.deviceInfo) else {
            state = .ready(name: linkName)
            return
        }
        state = .identifying(name: linkName)
        transport.read(.deviceInfo, on: id)
        identifyTimeout = after(timing.identify) { [weak self] in
            guard let self, case .identifying = self.state else { return }
            if let linkID = self.linkID { self.transport.disconnect(linkID) }
            self.clearLink()
            self.fail("Your SensorDyme didn't say what it is. Move closer to it and search again.")
        }
    }

    private func identified() {
        identifyTimeout?.cancel()
        if case .identifying = state { state = .ready(name: linkName) }
    }

    private func received(_ data: Data, on characteristic: ProvisioningCharacteristic) {
        switch characteristic {
        case .deviceInfo:
            let info = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            if let id = (info?["id"] as? String).flatMap({ $0.uppercased().hasPrefix("SD-") ? $0 : nil }) {
                deviceID = id
            }
            deviceAuth = info?["auth"] as? String
            identified()
        case .auth:
            guard let flow = popFlow else { return }
            perform(flow.handle(reply: data), flow: flow)
        case .networks:
            let parsed = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] ?? []
            let scanned = parsed.compactMap { entry -> VisibleNetwork? in
                guard let ssid = entry["ssid"] as? String, !ssid.isEmpty else { return nil }
                return VisibleNetwork(ssid: ssid, signal: entry["signal"] as? Int ?? 0,
                                      secured: entry["secured"] as? Bool ?? true)
            }
            networks = Self.stableOrder(scanned)
        case .status:
            guard let status = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return }
            statusReceived(status)
        case .credentials, .command:
            break
        }
    }

    private func statusReceived(_ status: [String: Any]) {
        if let id = ProvisioningStatus.deviceID(in: status), deviceID == nil {
            deviceID = id
        }
        let wasScanning = scanningNetworks
        scanningNetworks = (status["scanning"] as? Bool) ?? false
        // A scan that just FINISHED has a list waiting -- read it once per
        // asked-for read, never on every "not scanning" status.
        if wasScanning, !scanningNetworks, followUpReadWanted {
            followUpReadWanted = false
            readNetworks()
        }
        // Only an answer to THIS write is an outcome (AttemptTracker). A device
        // already on Wi-Fi reports "online" continuously and truthfully; read as
        // an outcome, setup "finished" before a network was ever chosen.
        guard awaitingOutcome else {
            attempts.observeIdle(status)
            return
        }
        let outcome = attempts.outcome(for: status)
        // Only a settled answer stops the clock: cancelling on "connecting"
        // disarmed the deadline exactly when the part that can hang began.
        if ProvisioningStatus.isTerminal(outcome) {
            sendTimeout?.cancel()
            statusPoll?.cancel()
        }
        switch outcome {
        case .none:
            break
        case .joining:
            state = .deviceConnecting
        case .joined(let ip):
            attempts.finish()
            state = .success(ip: ip)
        case .failed(let reason):
            // The link is still up: another network, or the password again.
            stateBeforeFailure = state
            attempts.finish()
            popFlow = nil
            state = .joinFailed(reason)
        }
    }
}
