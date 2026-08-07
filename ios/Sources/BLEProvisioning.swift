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
    @Published var discovered: [String] = []
    @Published private(set) var signalByName: [String: Int] = [:]

    private var central: CBCentralManager?
    private var peripheral: CBPeripheral?
    private var credsCharacteristic: CBCharacteristic?
    private var statusCharacteristic: CBCharacteristic?
    private var peripheralsByName: [String: CBPeripheral] = [:]

    func startScan() {
        state = .scanning
        discovered = []
        signalByName = [:]
        peripheralsByName = [:]
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

    func connect(name: String) {
        guard let target = peripheralsByName[name] else { return }
        state = .connecting(name: name)
        central?.stopScan()
        peripheral = target
        central?.connect(target)
    }

    /// Send Wi-Fi credentials to the connected device.
    func send(ssid: String, password: String) {
        guard let peripheral, let credsCharacteristic,
              let payload = try? JSONSerialization.data(withJSONObject: ["ssid": ssid, "psk": password])
        else {
            state = .failed("not connected to a device")
            return
        }
        state = .sending
        peripheral.writeValue(payload, for: credsCharacteristic, type: .withResponse)
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
        let name = (advertisementData[CBAdvertisementDataLocalNameKey] as? String)
            ?? peripheral.name ?? "SensorDyme device"
        Task { @MainActor in
            self.peripheralsByName[name] = peripheral
            self.signalByName[name] = RSSI.intValue
            if !self.discovered.contains(name) { self.discovered.append(name) }
            // Strongest first, so the device you are standing next to is the one
            // offered first rather than whichever happened to answer soonest.
            self.discovered.sort { (self.signalByName[$0] ?? -200) > (self.signalByName[$1] ?? -200) }
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
        guard characteristic.uuid == bleStatusUUID, let data = characteristic.value,
              let status = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return }
        Task { @MainActor in
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
