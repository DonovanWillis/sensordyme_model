import CoreBluetooth
import Foundation

/// What the provisioning logic needs from a Bluetooth link, and nothing more.
///
/// `BLEProvisioner` used to talk to CoreBluetooth directly, so every line of
/// its state machine could only run against a real peripheral -- and the iOS
/// Simulator has no Bluetooth at all. It now talks to this seam:
///
///   * `CoreBluetoothTransport` (below) is the production link, the same
///     CoreBluetooth calls the provisioner made before;
///   * `BLESimulatedTransport` (DEBUG builds only) carries the same
///     operations to the virtual peripheral (sim/ble_peripheral.py), which runs
///     the device's real provisioning and proof-of-possession code;
///   * tests drive the provisioner with a scripted transport.
///
/// Events arrive on the main actor, in the order the link produced them.

/// The SensorDyme provisioning service and its characteristics. Must match
/// services/ble_gatt.py exactly.
enum ProvisioningCharacteristic: String, CaseIterable, Sendable {
    case deviceInfo = "5d1e0002-5344-4d45-b000-536e447a6d65"
    case credentials = "5d1e0003-5344-4d45-b000-536e447a6d65"
    case status = "5d1e0004-5344-4d45-b000-536e447a6d65"
    case command = "5d1e0005-5344-4d45-b000-536e447a6d65"
    case networks = "5d1e0006-5344-4d45-b000-536e447a6d65"
    /// Proof of possession: present only on a device configured `auth: pop`.
    case auth = "5d1e0007-5344-4d45-b000-536e447a6d65"

    static let serviceUUID = "5d1e0001-5344-4d45-b000-536e447a6d65"

    init?(uuidString: String) {
        self.init(rawValue: uuidString.lowercased())
    }
}

enum BLERadioState: Equatable, Sendable {
    case poweredOn, poweredOff, unauthorized, unsupported, other
}

/// One sighting of an advertising device.
struct BLEAdvertisement: Equatable, Sendable {
    let id: UUID
    /// What the advertisement packet says right now (the device's own name).
    let advertisedName: String?
    /// What the phone remembers the peripheral being called, if anything.
    let cachedName: String?
    let rssi: Int
}

enum BLETransportEvent: Equatable, Sendable {
    case radio(BLERadioState)
    case discovered(BLEAdvertisement)
    case connected(UUID)
    case connectFailed(UUID, String?)
    /// An established link dropped (or the device closed its window).
    case disconnected(UUID)
    /// Connected, but the device does not offer the provisioning service.
    case serviceMissing(UUID)
    /// The provisioning service's characteristics, discovered.
    case characteristics(UUID, Set<ProvisioningCharacteristic>)
    /// A read answered or a notification arrived.
    case value(UUID, ProvisioningCharacteristic, Data)
    case readFailed(UUID, ProvisioningCharacteristic, String)
    /// A write with response finished; `error` is the link's refusal, if any.
    case wrote(UUID, ProvisioningCharacteristic, error: String?)
}

@MainActor
protocol BLETransport: AnyObject {
    var onEvent: ((BLETransportEvent) -> Void)? { get set }
    /// A fresh session: forget every earlier peripheral and its callbacks, and
    /// look for advertising devices once the radio is on.
    func startScan()
    func stopScan()
    /// Connect to a device seen in this session. False when it is unknown.
    @discardableResult func connect(_ id: UUID) -> Bool
    func disconnect(_ id: UUID)
    func read(_ characteristic: ProvisioningCharacteristic, on id: UUID)
    func write(_ data: Data, to characteristic: ProvisioningCharacteristic, on id: UUID)
    func subscribe(_ characteristic: ProvisioningCharacteristic, on id: UUID)
}

enum BLETransports {
    /// The link the app uses: CoreBluetooth, or -- in a DEBUG build launched
    /// with `-orynSimulatedBLE 127.0.0.1:<port>` -- the virtual peripheral.
    /// A Release build has no simulated transport compiled in.
    @MainActor static func make() -> BLETransport {
        #if DEBUG
        if let endpoints = BLESimulatedTransport.launchEndpoints() {
            return BLESimulatedTransport(endpoints: endpoints)
        }
        #endif
        return CoreBluetoothTransport()
    }
}

// MARK: - CoreBluetooth

/// The production link. The CoreBluetooth calls `BLEProvisioner` made before
/// the seam existed, unchanged in substance.
@MainActor
final class CoreBluetoothTransport: NSObject, BLETransport {
    var onEvent: ((BLETransportEvent) -> Void)?

    private var central: CBCentralManager?
    private var peripheralsByID: [UUID: CBPeripheral] = [:]
    private var characteristicsByID: [UUID: [ProvisioningCharacteristic: CBCharacteristic]] = [:]
    private var wantsScan = false
    private static let service = CBUUID(string: ProvisioningCharacteristic.serviceUUID)

    func startScan() {
        central?.stopScan()
        for peripheral in peripheralsByID.values where peripheral.state != .disconnected {
            central?.cancelPeripheralConnection(peripheral)
        }
        peripheralsByID = [:]
        characteristicsByID = [:]
        wantsScan = true
        // A new manager per session: a late callback from the previous one is
        // recognised (central !== self.central) and dropped.
        central = CBCentralManager(delegate: self, queue: .main)
        // Scanning begins in centralManagerDidUpdateState once Bluetooth is on.
    }

    func stopScan() {
        wantsScan = false
        central?.stopScan()
    }

    func connect(_ id: UUID) -> Bool {
        guard let target = peripheralsByID[id] else { return false }
        stopScan()
        central?.connect(target)
        return true
    }

    func disconnect(_ id: UUID) {
        if let peripheral = peripheralsByID[id] { central?.cancelPeripheralConnection(peripheral) }
        characteristicsByID[id] = nil
    }

    func read(_ characteristic: ProvisioningCharacteristic, on id: UUID) {
        guard let peripheral = peripheralsByID[id], let chrc = characteristicsByID[id]?[characteristic] else { return }
        peripheral.readValue(for: chrc)
    }

    func write(_ data: Data, to characteristic: ProvisioningCharacteristic, on id: UUID) {
        guard let peripheral = peripheralsByID[id], let chrc = characteristicsByID[id]?[characteristic] else {
            onEvent?(.wrote(id, characteristic, error: "not connected"))
            return
        }
        peripheral.writeValue(data, for: chrc, type: .withResponse)
    }

    func subscribe(_ characteristic: ProvisioningCharacteristic, on id: UUID) {
        guard let peripheral = peripheralsByID[id], let chrc = characteristicsByID[id]?[characteristic] else { return }
        peripheral.setNotifyValue(true, for: chrc)
    }

    private func emit(_ event: BLETransportEvent) { onEvent?(event) }
}

extension CoreBluetoothTransport: CBCentralManagerDelegate, CBPeripheralDelegate {
    nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
        let state = central.state
        Task { @MainActor in
            guard central === self.central else { return }  // a manager startScan() replaced
            switch state {
            case .poweredOn:
                if self.wantsScan { central.scanForPeripherals(withServices: [Self.service]) }
                self.emit(.radio(.poweredOn))
            case .poweredOff: self.emit(.radio(.poweredOff))
            case .unauthorized: self.emit(.radio(.unauthorized))
            case .unsupported: self.emit(.radio(.unsupported))
            default: self.emit(.radio(.other))
            }
        }
    }

    nonisolated func centralManager(
        _ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
        advertisementData: [String: Any], rssi RSSI: NSNumber
    ) {
        // Identity is the peripheral's UUID, never its name: CoreBluetooth calls
        // this repeatedly for one device and the name can differ between calls
        // (advertised local name vs the cached GAP name).
        let advert = BLEAdvertisement(
            id: peripheral.identifier,
            advertisedName: advertisementData[CBAdvertisementDataLocalNameKey] as? String,
            cachedName: peripheral.name, rssi: RSSI.intValue)
        Task { @MainActor in
            guard central === self.central else { return }
            self.peripheralsByID[advert.id] = peripheral
            self.emit(.discovered(advert))
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
        Task { @MainActor in
            guard central === self.central else { return }
            peripheral.delegate = self
            self.emit(.connected(peripheral.identifier))
            peripheral.discoverServices([Self.service])
        }
    }

    nonisolated func centralManager(_ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?) {
        let reason = error?.localizedDescription
        Task { @MainActor in
            guard central === self.central else { return }
            self.emit(.connectFailed(peripheral.identifier, reason))
        }
    }

    /// An established link dropped: the device rebooting after it joins Wi-Fi,
    /// someone walking out of range, BlueZ dropping it.
    nonisolated func centralManager(_ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?) {
        Task { @MainActor in
            guard central === self.central else { return }
            self.characteristicsByID[peripheral.identifier] = nil
            self.emit(.disconnected(peripheral.identifier))
        }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
        Task { @MainActor in
            guard let service = peripheral.services?.first(where: { $0.uuid == Self.service }) else {
                self.emit(.serviceMissing(peripheral.identifier))
                return
            }
            // Everything the service offers, not a hand-kept list: CoreBluetooth
            // delivers only what is asked for, and a list that forgot the
            // networks characteristic once left the network screen empty for ever.
            peripheral.discoverCharacteristics(nil, for: service)
        }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?) {
        Task { @MainActor in
            var found: [ProvisioningCharacteristic: CBCharacteristic] = [:]
            for chrc in service.characteristics ?? [] {
                if let known = ProvisioningCharacteristic(uuidString: chrc.uuid.uuidString) { found[known] = chrc }
            }
            self.characteristicsByID[peripheral.identifier] = found
            self.emit(.characteristics(peripheral.identifier, Set(found.keys)))
        }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didUpdateValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let known = ProvisioningCharacteristic(uuidString: characteristic.uuid.uuidString) else { return }
        // A failed read leaves `characteristic.value` holding the PREVIOUS value;
        // parsing it again would replay an old status (or list) as new.
        if let error {
            let reason = error.localizedDescription
            Task { @MainActor in self.emit(.readFailed(peripheral.identifier, known, reason)) }
            return
        }
        guard let data = characteristic.value else { return }
        Task { @MainActor in self.emit(.value(peripheral.identifier, known, data)) }
    }

    nonisolated func peripheral(_ peripheral: CBPeripheral, didWriteValueFor characteristic: CBCharacteristic, error: Error?) {
        guard let known = ProvisioningCharacteristic(uuidString: characteristic.uuid.uuidString) else { return }
        let reason = error?.localizedDescription
        Task { @MainActor in self.emit(.wrote(peripheral.identifier, known, error: reason)) }
    }
}
