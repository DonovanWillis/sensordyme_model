import CoreBluetooth
import Foundation

/// Bluetooth-LE client for provisioning a SensorDyme device's Wi-Fi.
///
/// Counterpart of services/ble_gatt.py on the device: when the device has no
/// Wi-Fi it advertises "SensorDyme-XXXX"; this scans for that service, connects,
/// writes {ssid, psk} to the credentials characteristic, and follows progress
/// on the status characteristic. BLE is used ONLY for setup and lightweight
/// commands — the live camera feed always streams over Wi-Fi.
// Must match services/ble_gatt.py exactly. File-scope (not actor-isolated):
// CoreBluetooth delegate callbacks arrive nonisolated and need these too.
private let bleServiceUUID = CBUUID(string: "5d1e0001-5344-4d45-b000-536e447a6d65")
private let bleDeviceInfoUUID = CBUUID(string: "5d1e0002-5344-4d45-b000-536e447a6d65")
private let bleWifiCredsUUID = CBUUID(string: "5d1e0003-5344-4d45-b000-536e447a6d65")
private let bleStatusUUID = CBUUID(string: "5d1e0004-5344-4d45-b000-536e447a6d65")
private let bleNetworksUUID = CBUUID(string: "5d1e0006-5344-4d45-b000-536e447a6d65")

/// One physical device seen over Bluetooth.
///
/// Identified by CoreBluetooth's peripheral UUID, which is stable for a given
/// device on a given phone. Names are display only — a device can present more
/// than one, and treating a name as identity is what produced duplicates.
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
final class BLEProvisioner: NSObject, ObservableObject {

    enum State: Equatable {
        case idle
        case bluetoothOff
        case scanning
        case connecting(name: String)
        case ready(name: String)          // connected; awaiting credentials
        case sending
        case deviceConnecting             // device is trying the network we sent
        case success(ip: String?)
        case failed(String)
    }

    @Published var state: State = .idle
    /// Discovered devices, strongest signal first.
    ///
    /// Signal strength is kept because a name alone can be ambiguous: a device
    /// that has been reflashed or test-configured may advertise under a stale
    /// alias, so two entries can appear for what is really one unit — or one
    /// real device alongside a leftover advertisement. The nearer, stronger
    /// signal is almost always the one in the room with you.
    @Published var discovered: [DiscoveredDevice] = []

    /// Networks the DEVICE can see. Scanned on the Pi, not the phone: iOS
    /// cannot enumerate Wi-Fi at all, and the Pi's list is the one that
    /// matters — it is the machine that has to associate.
    @Published private(set) var networks: [VisibleNetwork] = []
    /// Whether the DEVICE is scanning right now, reported by the device itself.
    ///
    /// Without this an empty list has two meanings — "still looking" and "there
    /// are none" — and the screen has to guess. It used to guess "still
    /// looking", so a device that found nothing showed a spinner forever.
    @Published private(set) var scanningNetworks = false
    /// Why the network list is unavailable, when it is.
    ///
    /// Distinguishes "the device looked and saw nothing" from "the app never
    /// managed to ask" — which look identical on screen and have completely
    /// different fixes.
    @Published private(set) var networkProblem: String?
    /// The state the flow was in when it failed.
    ///
    /// "Setup failed" is nearly useless on its own: failing to find the device
    /// and failing to join the network are different problems with different
    /// fixes. Keeping the last good state means the checklist can mark the one
    /// step that broke and leave the completed ones standing.
    @Published private(set) var stateBeforeFailure: State?
    /// The connected device's own SD- id, from its device-info characteristic
    /// (0002 `id`) or its status (`device_id`). Setup uses it to find THIS
    /// device on Wi-Fi afterwards, rather than whichever camera was linked
    /// before or happened to answer first.
    @Published private(set) var deviceID: String?
    private var networksCharacteristic: CBCharacteristic?
    private var deviceInfoCharacteristic: CBCharacteristic?
    private var sendTimeout: Task<Void, Never>?
    private var statusPoll: Task<Void, Never>?
    private var connectTimeout: Task<Void, Never>?
    /// Which status answers the outstanding credentials write. See
    /// `ProvisioningStatus.AttemptTracker`.
    private var attempts = ProvisioningStatus.AttemptTracker()
    /// Whether the end of the device's current scan should be followed by one
    /// more read of the list.
    ///
    /// The device starts a fresh scan on every read of the list, and the app
    /// read the list on every status that said the scan had finished — so each
    /// read started the scan whose end triggered the next read, and the device's
    /// radio rescanned without pause for as long as Bluetooth stayed connected,
    /// through the network join itself. A read asked for (on connect, or a tap)
    /// earns exactly one follow-up when the scan it started finishes.
    private var followUpReadWanted = false

    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var credsCharacteristic: CBCharacteristic?
    private var statusCharacteristic: CBCharacteristic?
    private var peripheralsByID: [UUID: CBPeripheral] = [:]

    func startScan() {
        // A retry can come while the previous link is still up (a refused
        // password leaves Bluetooth connected). Close it and drop every
        // reference to it first: a late callback from the old link would
        // otherwise clear the NEW connection's characteristics.
        central?.stopScan()
        if let peripheral { central?.cancelPeripheralConnection(peripheral) }
        clearLink()
        state = .scanning
        discovered = []
        networks = []
        peripheralsByID = [:]
        deviceID = nil
        central = CBCentralManager(delegate: self, queue: .main)
        // Scanning begins in centralManagerDidUpdateState once Bluetooth is powered on.
    }

    func stop() {
        central?.stopScan()
        if let peripheral { central?.cancelPeripheralConnection(peripheral) }
        clearLink()
        state = .idle
    }

    /// How long a connection attempt may take. CoreBluetooth never times one
    /// out by itself, so a device that walked out of range left "Pairing…"
    /// spinning for ever.
    private static let connectTimeoutSeconds: Double = 20

    func connect(id: UUID) {
        guard let target = peripheralsByID[id] else {
            fail("That SensorDyme is no longer in range. Search again.")
            return
        }
        let name = discovered.first { $0.id == id }?.name ?? "SensorDyme device"
        state = .connecting(name: name)
        central?.stopScan()
        peripheral = target
        central?.connect(target)
        connectTimeout?.cancel()
        connectTimeout = Task { [weak self] in
            try? await Task.sleep(for: .seconds(Self.connectTimeoutSeconds))
            guard !Task.isCancelled, let self else { return }
            guard case .connecting = self.state else { return }
            if let peripheral = self.peripheral { self.central?.cancelPeripheralConnection(peripheral) }
            self.clearLink()
            self.fail("Couldn't connect to \(name). Move closer to it and search again.")
        }
    }

    /// Forget everything that belongs to one Bluetooth link.
    private func clearLink() {
        peripheral = nil
        credsCharacteristic = nil
        statusCharacteristic = nil
        networksCharacteristic = nil
        deviceInfoCharacteristic = nil
        connectTimeout?.cancel(); connectTimeout = nil
        sendTimeout?.cancel(); sendTimeout = nil
        statusPoll?.cancel(); statusPoll = nil
        // Per link: attempt numbers are one device's counter, and the next
        // connection (perhaps to another device) reads its own baseline.
        attempts = ProvisioningStatus.AttemptTracker()
        followUpReadWanted = false
        scanningNetworks = false
    }

    /// Whether a callback is about the link this provisioner is using now.
    private func isCurrent(_ candidate: CBPeripheral) -> Bool {
        peripheral?.identifier == candidate.identifier
    }

    /// Send Wi-Fi credentials to the connected device.
    /// Re-scan on the device. The list is a moment in time, and the network
    /// someone is looking for may only appear on the second sweep.
    /// Whether a credential write is outstanding.
    ///
    /// Before this is true the device's status is just a description of itself,
    /// not an answer to anything the app asked.
    private var awaitingOutcome: Bool {
        switch state {
        case .sending, .deviceConnecting: true
        default: false
        }
    }

    /// Clear everything that points at a link which no longer exists.
    ///
    /// This is what makes the guards elsewhere mean what they say: while these
    /// stayed set, `refreshNetworks()` and `send()` both passed their checks and
    /// then quietly did nothing.
    private func handleDisconnect() {
        clearLink()
        networks = []

        switch state {
        case .idle, .success, .failed:
            // We asked for this. SetupFlow drops the link deliberately after a
            // successful join, and stop() sets .idle before this fires — without
            // this case a completed setup would flip to failed.
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
        state = .failed(reason)
    }

    /// Order networks so the list does not rearrange itself under a finger.
    ///
    /// The device sorts by signal, and signal moves by a few points between
    /// scans — so sorting on the raw number reshuffles rows on every refresh.
    /// A refresh landing between touch-down and touch-up then sends the tap to
    /// a different network, or to nothing. That is invisible from the outside
    /// and reads as "the row didn't respond".
    ///
    /// Bucketing first means only a real change in strength can move a row,
    /// and ties break on name, which never changes.
    static func stableOrder(_ networks: [VisibleNetwork]) -> [VisibleNetwork] {
        networks.sorted {
            $0.strengthRank != $1.strengthRank
                ? $0.strengthRank > $1.strengthRank
                : $0.ssid.localizedCaseInsensitiveCompare($1.ssid) == .orderedAscending
        }
    }

    /// Ask the device for its Wi-Fi list.
    ///
    /// Reports when it cannot rather than returning quietly. The quiet version
    /// of this guard is how a missing characteristic turned into a permanently
    /// empty network list and a button that did nothing: every layer behaved
    /// exactly as written, and none of them said so.
    func refreshNetworks() {
        guard peripheral != nil else {
            networkProblem = "Not connected to your SensorDyme."
            return
        }
        guard networksCharacteristic != nil else {
            networkProblem = "This SensorDyme didn't offer a Wi-Fi list. It may need a software update."
            return
        }
        networkProblem = nil
        // An asked-for read: the scan it starts on the device earns one
        // follow-up read when it finishes (see `followUpReadWanted`).
        followUpReadWanted = true
        readNetworks()
    }

    private func readNetworks() {
        guard let peripheral, let characteristic = networksCharacteristic else { return }
        peripheral.readValue(for: characteristic)
    }

    func send(ssid: String, password: String) {
        guard let peripheral, let credsCharacteristic,
              let payload = try? JSONSerialization.data(withJSONObject: ["ssid": ssid, "psk": password])
        else {
            fail("not connected to a device")
            return
        }
        state = .sending
        attempts.beginWrite(ssid: ssid)
        peripheral.writeValue(payload, for: credsCharacteristic, type: .withResponse)
        startSendTimeout()
        startStatusPolling()
    }

    /// Read the status characteristic on a timer while a join is in flight.
    ///
    /// Notifications are the primary channel, but they fail silently: if the
    /// subscription did not take, or the device could not flush it because its
    /// event loop was blocked, the app waits forever for a message that will
    /// never arrive. A read cannot be missed the same way — it asks, and the
    /// device answers with whatever it has right now.
    ///
    /// Belt and braces on purpose. This is the exact path where silence is
    /// indistinguishable from failure, and where the user is standing there
    /// watching.
    private func startStatusPolling() {
        statusPoll?.cancel()
        statusPoll = Task { [weak self] in
            for _ in 0..<45 {   // ~90s: the write window plus a join of up to 45s
                try? await Task.sleep(for: .seconds(2))
                guard !Task.isCancelled, let self else { return }
                guard self.awaitingOutcome else {
                    // Resolved, or we are no longer waiting on anything.
                    return
                }
                if let peripheral = self.peripheral, let characteristic = self.statusCharacteristic {
                    peripheral.readValue(for: characteristic)
                }
            }
        }
    }

    /// Give up waiting, with a reason.
    ///
    /// The device has up to 45 seconds to join a network, and older firmware
    /// blocks its own event loop for that whole time — so a slow join and a
    /// dead one look identical from here. Without a deadline the screen sits on
    /// "Sending…" indefinitely, which teaches the user nothing and looks like a
    /// hung app.
    ///
    /// Generous on purpose: firing before the device has finished trying would
    /// report a failure that did not happen.
    private func startSendTimeout() {
        sendTimeout?.cancel()
        sendTimeout = Task { [weak self] in
            try? await Task.sleep(for: .seconds(90))
            guard !Task.isCancelled, let self else { return }
            // Covers the join too, not just the write.
            //
            // The deadline used to test only `.sending`, and the device moves to
            // "connecting" within a fraction of a second — so by the time the
            // timer fired there was nothing left to time out. The one state that
            // can genuinely hang, the join itself, was the one state with no
            // deadline at all.
            if self.awaitingOutcome {
                self.fail(
                    "The device didn't confirm within a minute. It's usually the Wi-Fi password — "
                    + "check it and try again. Other causes: the network needs a browser sign-in "
                    + "(hotel or campus Wi-Fi), or it's 5GHz-only. A phone hotspot is the most "
                    + "reliable way to get the device online."
                )
            }
        }
    }
}

extension BLEProvisioner: CBCentralManagerDelegate, CBPeripheralDelegate {
    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        Task { @MainActor in
            guard central === self.central else { return }  // a manager startScan() replaced
            switch central.state {
            case .poweredOn:
                // Back from off: say so. Leaving `.bluetoothOff` in place kept
                // setup on "Bluetooth is off" while it was discovering devices.
                if case .bluetoothOff = self.state { self.state = .scanning }
                central.scanForPeripherals(withServices: [bleServiceUUID])
            case .unauthorized, .poweredOff:
                // Off mid-setup ends the link without a disconnect callback.
                let wasJoining = self.awaitingOutcome
                self.clearLink()
                if wasJoining {
                    // Not "Bluetooth is off" with the setup screen still saying
                    // "Setting up…": the outcome is unknown, and the device may
                    // well have joined.
                    self.fail("Bluetooth was turned off while your SensorDyme was joining. It may have "
                              + "joined anyway — search again to check before re-entering the password.")
                } else {
                    self.state = .bluetoothOff
                }
            case .unsupported:
                self.fail("This phone can't use Bluetooth, which setup needs.")
            default:
                break
            }
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager,
        didDiscover peripheral: CBPeripheral,
        advertisementData: [String: Any],
        rssi RSSI: NSNumber
    ) {
        // Identity is the peripheral's UUID, never its name.
        //
        // CoreBluetooth calls this repeatedly for one device, and the name can
        // differ between calls: CBAdvertisementDataLocalNameKey comes from the
        // advertisement packet while peripheral.name is the cached GAP name
        // (the adapter alias). A Pi advertising "SensorDyme" that iOS has
        // cached as "SensorDyme-TEST" produced TWO entries for ONE device —
        // which is exactly the duplicate that has been showing up. Keying on
        // identifier makes that impossible rather than filtered.
        let advertised = advertisementData[CBAdvertisementDataLocalNameKey] as? String
        let name = advertised ?? peripheral.name ?? "SensorDyme device"
        let id = peripheral.identifier
        Task { @MainActor in
            self.peripheralsByID[id] = peripheral
            let device = DiscoveredDevice(id: id, name: name, rssi: RSSI.intValue)
            if let index = self.discovered.firstIndex(where: { $0.id == id }) {
                // Same device, fresher reading. Prefer the advertised name: it
                // is what the device is saying right now, rather than what iOS
                // remembers it being called.
                self.discovered[index] = DiscoveredDevice(
                    id: id,
                    name: advertised ?? self.discovered[index].name,
                    rssi: RSSI.intValue
                )
            } else {
                self.discovered.append(device)
            }
            // Strongest first, so the device you are standing next to is offered
            // first rather than whichever answered soonest.
            self.discovered.sort { $0.rssi > $1.rssi }
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        Task { @MainActor in
            guard self.isCurrent(peripheral) else { return }
            peripheral.delegate = self
            peripheral.discoverServices([bleServiceUUID])
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?
    ) {
        Task { @MainActor in
            guard self.isCurrent(peripheral) else { return }
            self.clearLink()
            self.fail(error?.localizedDescription ?? "could not connect")
        }
    }

    /// An established link dropped.
    ///
    /// This delegate was simply absent. `didFailToConnect` only fires for a
    /// connection that never came up, so a link that dies afterwards — the
    /// device rebooting after it joins Wi-Fi, someone walking out of range,
    /// BlueZ dropping it — called a method that did not exist, and nothing ran.
    /// Every stored reference stayed non-nil pointing at a dead connection, so
    /// every later `guard let` passed, reads and writes were discarded silently
    /// by CoreBluetooth, and a minute later the app told the user their Wi-Fi
    /// password was probably wrong. It was blaming the password for a Bluetooth
    /// problem.
    nonisolated func centralManager(
        _ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?
    ) {
        Task { @MainActor in
            // Only the link in use. One this provisioner already let go of — a
            // retry's startScan(), a connect timeout, stop() — has nothing left
            // to clear, and acting on its late callback failed (or tore down)
            // the search or connection that replaced it.
            guard central === self.central, self.isCurrent(peripheral) else { return }
            self.handleDisconnect()
        }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        Task { @MainActor in
            guard self.isCurrent(peripheral) else { return }
            guard let service = peripheral.services?.first(where: { $0.uuid == bleServiceUUID }) else {
                self.connectTimeout?.cancel()
                self.fail("device is missing the provisioning service")
                return
            }
            // Discover everything the service offers, rather than a list kept
            // by hand.
            //
            // That list omitted the networks characteristic. CoreBluetooth only
            // delivers what you ask for, so it was never discovered, the
            // property stayed nil, and refreshNetworks() hit its guard and did
            // nothing — for ever, without a word. The device saw twelve
            // networks and the app showed none and a dead "Search again".
            //
            // A filter here has to be kept in step with services/ble_gatt.py by
            // memory, and the penalty for forgetting is invisible. Passing nil
            // costs one extra round trip and removes the whole failure mode.
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?
    ) {
        Task { @MainActor in
            guard self.isCurrent(peripheral) else { return }
            self.connectTimeout?.cancel()
            for characteristic in service.characteristics ?? [] {
                switch characteristic.uuid {
                case bleWifiCredsUUID:
                    self.credsCharacteristic = characteristic
                case bleStatusUUID:
                    self.statusCharacteristic = characteristic
                    peripheral.setNotifyValue(true, for: characteristic)
                    // Read once now: the attempt number it carries is the
                    // baseline a later outcome has to beat.
                    peripheral.readValue(for: characteristic)
                case bleNetworksUUID:
                    self.networksCharacteristic = characteristic
                case bleDeviceInfoUUID:
                    self.deviceInfoCharacteristic = characteristic
                    peripheral.readValue(for: characteristic)
                default:
                    break
                }
            }
            guard self.credsCharacteristic != nil else {
                // Without it nothing can be sent, and the flow used to sit on
                // "Pairing…" for ever.
                self.fail("This SensorDyme didn't offer Wi-Fi setup over Bluetooth. It may need a software update.")
                return
            }
            // Ask immediately: the device scans on read, and the answer is
            // wanted the moment the setup form appears.
            self.refreshNetworks()
            self.state = .ready(name: peripheral.name ?? "SensorDyme device")
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?
    ) {
        // A failed read leaves `characteristic.value` holding the PREVIOUS
        // value. Parsing it again replayed an old status (or an old list) as if
        // it had just arrived. The next poll or notification will bring a real one.
        if let error {
            let failedNetworksRead = characteristic.uuid == bleNetworksUUID
            let reason = error.localizedDescription
            Task { @MainActor in
                guard self.isCurrent(peripheral), failedNetworksRead else { return }
                self.networkProblem = "Couldn't read the device's Wi-Fi list: \(reason)"
            }
            return
        }
        guard let data = characteristic.value else { return }

        if characteristic.uuid == bleDeviceInfoUUID {
            let info = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
            let id = (info?["id"] as? String).flatMap { $0.uppercased().hasPrefix("SD-") ? $0 : nil }
            Task { @MainActor in
                guard self.isCurrent(peripheral), let id else { return }
                self.deviceID = id
            }
            return
        }

        if characteristic.uuid == bleNetworksUUID {
            let parsed = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] ?? []
            let scanned = parsed.compactMap { entry -> VisibleNetwork? in
                guard let ssid = entry["ssid"] as? String, !ssid.isEmpty else { return nil }
                return VisibleNetwork(
                    ssid: ssid,
                    signal: entry["signal"] as? Int ?? 0,
                    secured: entry["secured"] as? Bool ?? true
                )
            }
            Task { @MainActor in
                guard self.isCurrent(peripheral) else { return }
                self.networks = Self.stableOrder(scanned)
            }
            return
        }

        guard characteristic.uuid == bleStatusUUID,
              let status = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }
        Task { @MainActor in
            guard self.isCurrent(peripheral) else { return }
            if let id = ProvisioningStatus.deviceID(in: status), self.deviceID == nil {
                self.deviceID = id
            }
            let wasScanning = self.scanningNetworks
            self.scanningNetworks = (status["scanning"] as? Bool) ?? false
            // A scan that just FINISHED has a list waiting to be read — but only
            // once per asked-for read. Reading on every "not scanning" status
            // restarted the device's scan each time, for ever.
            if wasScanning, !self.scanningNetworks, self.followUpReadWanted {
                self.followUpReadWanted = false
                self.readNetworks()
            }

            // Only read this as an outcome once one has been asked for, and only
            // a status that answers THIS write (see AttemptTracker).
            //
            // A device that is already on Wi-Fi reports "online" continuously
            // and truthfully. Treating that as the result of a credential write
            // meant the app declared setup finished the moment it connected
            // over Bluetooth — jumping from "Connect SensorDyme" straight to
            // "SensorDyme is ready" without ever showing a network or asking
            // for a password.
            guard self.awaitingOutcome else {
                self.attempts.observeIdle(status)
                return
            }
            let outcome = self.attempts.outcome(for: status)
            // Only a settled answer stops the clock. Cancelling on "connecting"
            // disarmed the deadline at the exact moment the device began the
            // part that can hang, leaving notifications as the only channel —
            // and a missed notification then meant waiting for ever.
            if ProvisioningStatus.isTerminal(outcome) {
                self.sendTimeout?.cancel()
                self.statusPoll?.cancel()
            }

            switch outcome {
            case .none:
                break
            case .joining:
                self.state = .deviceConnecting
            case .joined(let ip):
                self.attempts.finish()
                self.state = .success(ip: ip)
            case .failed(let reason):
                self.fail(reason)
            }
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?
    ) {
        let isCredentials = characteristic.uuid == bleWifiCredsUUID
        Task { @MainActor in
            guard self.isCurrent(peripheral) else { return }
            if let error {
                self.fail(error.localizedDescription)
            } else if isCredentials {
                // The device records the new attempt before it answers the
                // write, so from here on a status describes this write.
                self.attempts.acknowledgeWrite()
            }
        }
    }
}
