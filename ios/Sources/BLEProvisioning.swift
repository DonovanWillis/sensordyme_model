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
    private var networksCharacteristic: CBCharacteristic?
    private var sendTimeout: Task<Void, Never>?
    private var statusPoll: Task<Void, Never>?

    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var credsCharacteristic: CBCharacteristic?
    private var statusCharacteristic: CBCharacteristic?
    private var peripheralsByID: [UUID: CBPeripheral] = [:]

    func startScan() {
        state = .scanning
        discovered = []
        networks = []
        peripheralsByID = [:]
        central = CBCentralManager(delegate: self, queue: .main)
        // Scanning begins in centralManagerDidUpdateState once Bluetooth is powered on.
    }

    func stop() {
        central?.stopScan()
        if let peripheral { central?.cancelPeripheralConnection(peripheral) }
        peripheral = nil
        credsCharacteristic = nil
        statusCharacteristic = nil
        state = .idle
    }

    func connect(id: UUID) {
        guard let target = peripheralsByID[id] else { return }
        let name = discovered.first { $0.id == id }?.name ?? "SensorDyme device"
        state = .connecting(name: name)
        central?.stopScan()
        peripheral = target
        central?.connect(target)
    }

    /// Send Wi-Fi credentials to the connected device.
    /// Re-scan on the device. The list is a moment in time, and the network
    /// someone is looking for may only appear on the second sweep.
    func refreshNetworks() {
        guard let peripheral, let characteristic = networksCharacteristic else { return }
        peripheral.readValue(for: characteristic)
    }

    func send(ssid: String, password: String) {
        guard let peripheral, let credsCharacteristic,
              let payload = try? JSONSerialization.data(withJSONObject: ["ssid": ssid, "psk": password])
        else {
            state = .failed("not connected to a device")
            return
        }
        state = .sending
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
            for _ in 0..<30 {   // ~60s, matching the send deadline
                try? await Task.sleep(for: .seconds(2))
                guard !Task.isCancelled, let self else { return }
                guard case .sending = self.state else {
                    // Something already resolved it; stop asking.
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
            try? await Task.sleep(for: .seconds(60))
            guard !Task.isCancelled, let self else { return }
            if case .sending = self.state {
                self.state = .failed(
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
            switch central.state {
            case .poweredOn:
                central.scanForPeripherals(withServices: [bleServiceUUID])
            case .unauthorized, .poweredOff:
                self.state = .bluetoothOff
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
            peripheral.delegate = self
            peripheral.discoverServices([bleServiceUUID])
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?
    ) {
        Task { @MainActor in
            self.state = .failed(error?.localizedDescription ?? "could not connect")
        }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        Task { @MainActor in
            guard let service = peripheral.services?.first(where: { $0.uuid == bleServiceUUID }) else {
                self.state = .failed("device is missing the provisioning service")
                return
            }
            peripheral.discoverCharacteristics(
                [bleWifiCredsUUID, bleStatusUUID, bleDeviceInfoUUID], for: service
            )
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?
    ) {
        Task { @MainActor in
            for characteristic in service.characteristics ?? [] {
                switch characteristic.uuid {
                case bleWifiCredsUUID:
                    self.credsCharacteristic = characteristic
                case bleStatusUUID:
                    self.statusCharacteristic = characteristic
                    peripheral.setNotifyValue(true, for: characteristic)
                case bleNetworksUUID:
                    self.networksCharacteristic = characteristic
                    // Ask immediately: the device scans on read, and the answer
                    // is wanted the moment the setup form appears.
                    peripheral.readValue(for: characteristic)
                default:
                    break
                }
            }
            if self.credsCharacteristic != nil {
                self.state = .ready(name: peripheral.name ?? "SensorDyme device")
            }
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?
    ) {
        guard let data = characteristic.value else { return }

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
            Task { @MainActor in self.networks = scanned }
            return
        }

        guard characteristic.uuid == bleStatusUUID,
              let status = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }
        Task { @MainActor in
            self.scanningNetworks = (status["scanning"] as? Bool) ?? false
            // A scan that just finished has a list waiting to be read.
            if !self.scanningNetworks { self.refreshNetworks() }
            // The device answered, so the deadline has done its job.
            if let state = status["state"] as? String,
               ["connecting", "online", "error"].contains(state) {
                self.sendTimeout?.cancel()
                // "connecting" means the write landed; keep polling for the
                // outcome. Anything terminal ends the polling too.
                if state != "connecting" { self.statusPoll?.cancel() }
            }
            switch status["state"] as? String {
            case "connecting":
                self.state = .deviceConnecting
            case "online":
                self.state = .success(ip: status["ip"] as? String)
            case "error":
                self.state = .failed((status["error"] as? String) ?? "device could not join that network")
            default:
                break
            }
        }
    }

    nonisolated func peripheral(
        _ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?
    ) {
        if let error {
            Task { @MainActor in self.state = .failed(error.localizedDescription) }
        }
    }
}
