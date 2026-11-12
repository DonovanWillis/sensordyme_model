#if DEBUG
import Foundation
import Network

/// A Bluetooth link to the virtual peripheral (sim/ble_peripheral.py), over
/// localhost TCP. DEBUG BUILDS ONLY: in a Release build this file compiles to
/// nothing, and `BLETransports.make()` can only return CoreBluetooth.
///
/// The iOS Simulator has no Bluetooth, so without this the setup screens could
/// not run at all before a phone and a Pi are in the same room. The peripheral
/// runs the device's real provisioning and proof-of-possession code; this
/// carries the same operations CoreBluetoothTransport performs (scan, connect,
/// discover, read, write with response, notify) as one JSON object per line.
///
/// Opt in per launch with `-orynSimulatedBLE 127.0.0.1:8790` (several devices:
/// comma-separated). Only loopback addresses are accepted, so not even a debug
/// build can be pointed at a "device" elsewhere on the network.
@MainActor
final class BLESimulatedTransport: BLETransport {
    struct Endpoint: Hashable, CustomStringConvertible {
        let host: String
        let port: UInt16
        var description: String { "\(host):\(port)" }
    }

    var onEvent: ((BLETransportEvent) -> Void)?

    /// The endpoints named by `-orynSimulatedBLE`, if the launch asked for any.
    static func launchEndpoints(arguments: [String] = ProcessInfo.processInfo.arguments) -> [Endpoint]? {
        guard let index = arguments.firstIndex(of: "-orynSimulatedBLE"), arguments.indices.contains(index + 1) else {
            return nil
        }
        let endpoints = arguments[index + 1].split(separator: ",").compactMap { parse(String($0)) }
        return endpoints.isEmpty ? nil : endpoints
    }

    /// "host:port", loopback only.
    static func parse(_ text: String) -> Endpoint? {
        guard let colon = text.lastIndex(of: ":"), let port = UInt16(text[text.index(after: colon)...]), port > 0 else {
            return nil
        }
        let host = String(text[..<colon]).trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        guard ["127.0.0.1", "localhost", "::1"].contains(host.lowercased()) else { return nil }
        return Endpoint(host: host, port: port)
    }

    private let endpoints: [Endpoint]
    private var connections: [Endpoint: Connection] = [:]
    private var endpointByPeripheral: [UUID: Endpoint] = [:]
    private var scanTask: Task<Void, Never>?
    private var nextSeq = 0
    /// What each outstanding read or write was for, by sequence number.
    private var pending: [Int: (UUID, ProvisioningCharacteristic)] = [:]

    init(endpoints: [Endpoint]) {
        self.endpoints = endpoints
    }

    // MARK: BLETransport

    func startScan() {
        // A new session, like a new CBCentralManager: every old link is closed
        // and nothing it says afterwards reaches the provisioner.
        stopScan()
        for connection in connections.values { connection.close() }
        connections = [:]
        endpointByPeripheral = [:]
        pending = [:]
        // The simulated radio is always on.
        onEvent?(.radio(.poweredOn))
        scanTask = Task { [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                for endpoint in self.endpoints {
                    self.connection(for: endpoint).send(["op": "scan"])
                }
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    func stopScan() {
        scanTask?.cancel()
        scanTask = nil
    }

    func connect(_ id: UUID) -> Bool {
        guard let endpoint = endpointByPeripheral[id] else { return false }
        stopScan()
        connection(for: endpoint).send(["op": "connect"])
        return true
    }

    func disconnect(_ id: UUID) {
        guard let endpoint = endpointByPeripheral[id] else { return }
        connections[endpoint]?.send(["op": "disconnect"])
    }

    func read(_ characteristic: ProvisioningCharacteristic, on id: UUID) {
        request("read", characteristic, on: id)
    }

    func write(_ data: Data, to characteristic: ProvisioningCharacteristic, on id: UUID) {
        request("write", characteristic, on: id, value: data)
    }

    func subscribe(_ characteristic: ProvisioningCharacteristic, on id: UUID) {
        request("subscribe", characteristic, on: id)
    }

    // MARK: The wire

    private func request(_ op: String, _ characteristic: ProvisioningCharacteristic, on id: UUID, value: Data? = nil) {
        guard let endpoint = endpointByPeripheral[id], let connection = connections[endpoint] else {
            if op == "write" { onEvent?(.wrote(id, characteristic, error: "not connected")) }
            return
        }
        nextSeq += 1
        pending[nextSeq] = (id, characteristic)
        var message: [String: Any] = ["op": op, "char": characteristic.rawValue, "seq": nextSeq]
        if let value { message["value"] = value.base64EncodedString() }
        connection.send(message)
    }

    private func connection(for endpoint: Endpoint) -> Connection {
        if let existing = connections[endpoint], !existing.closed { return existing }
        let connection = Connection(endpoint: endpoint)
        connection.onMessage = { [weak self, weak connection] message in
            guard let self, let connection, self.connections[endpoint] === connection else { return }
            self.received(message, from: endpoint)
        }
        connection.onClose = { [weak self, weak connection] in
            guard let self, let connection, self.connections[endpoint] === connection else { return }
            self.connections[endpoint] = nil
            for (id, at) in self.endpointByPeripheral where at == endpoint {
                self.onEvent?(.disconnected(id))
            }
        }
        connections[endpoint] = connection
        connection.start()
        return connection
    }

    private func received(_ message: [String: Any], from endpoint: Endpoint) {
        let ev = message["ev"] as? String
        if ev == "adv" {
            guard message["advertising"] as? Bool == true, let raw = message["id"] as? String,
                  let id = UUID(uuidString: raw) else { return }
            endpointByPeripheral[id] = endpoint
            guard scanTask != nil else { return }
            onEvent?(.discovered(BLEAdvertisement(id: id, advertisedName: message["name"] as? String, cachedName: nil,
                                                  rssi: message["rssi"] as? Int ?? -60)))
            return
        }
        guard let id = endpointByPeripheral.first(where: { $0.value == endpoint })?.key else { return }
        let characteristic = (message["char"] as? String).flatMap { ProvisioningCharacteristic(uuidString: $0) }
        let seq = message["seq"] as? Int
        let error = message["error"] as? String
        if let seq { pending[seq] = nil }
        switch ev {
        case "connected":
            onEvent?(.connected(id))
        case "connect_failed":
            onEvent?(.connectFailed(id, message["reason"] as? String))
        case "services":
            let found = (message["characteristics"] as? [[String: Any]] ?? []).compactMap {
                ($0["uuid"] as? String).flatMap { ProvisioningCharacteristic(uuidString: $0) }
            }
            onEvent?(.characteristics(id, Set(found)))
        case "read":
            guard let characteristic else { return }
            if let error {
                onEvent?(.readFailed(id, characteristic, error))
            } else if let value = (message["value"] as? String).flatMap({ Data(base64Encoded: $0) }) {
                onEvent?(.value(id, characteristic, value))
            }
        case "notify":
            guard let characteristic, let value = (message["value"] as? String).flatMap({ Data(base64Encoded: $0) }) else {
                return
            }
            onEvent?(.value(id, characteristic, value))
        case "wrote":
            guard let characteristic else { return }
            onEvent?(.wrote(id, characteristic, error: error))
        case "disconnected":
            onEvent?(.disconnected(id))
        default:
            break
        }
    }

    /// One TCP connection to one virtual peripheral, newline-delimited JSON.
    @MainActor
    final class Connection {
        let endpoint: Endpoint
        var onMessage: (([String: Any]) -> Void)?
        var onClose: (() -> Void)?
        private(set) var closed = false
        private let connection: NWConnection
        private var buffer = Data()

        init(endpoint: Endpoint) {
            self.endpoint = endpoint
            connection = NWConnection(host: NWEndpoint.Host(endpoint.host),
                                      port: NWEndpoint.Port(rawValue: endpoint.port) ?? 8790, using: .tcp)
        }

        func start() {
            connection.stateUpdateHandler = { [weak self] state in
                switch state {
                case .failed, .cancelled:
                    MainActor.assumeIsolated { self?.finish() }
                default:
                    break
                }
            }
            connection.start(queue: .main)
            receive()
        }

        func send(_ message: [String: Any]) {
            guard !closed, var line = try? JSONSerialization.data(withJSONObject: message) else { return }
            line.append(0x0A)
            connection.send(content: line, completion: .contentProcessed { _ in })
        }

        func close() {
            closed = true
            connection.cancel()
        }

        private func receive() {
            connection.receive(minimumIncompleteLength: 1, maximumLength: 65_536) { [weak self] data, _, complete, error in
                MainActor.assumeIsolated {
                    guard let self else { return }
                    if let data { self.consume(data) }
                    if complete || error != nil {
                        self.finish()
                    } else if !self.closed {
                        self.receive()
                    }
                }
            }
        }

        private func consume(_ data: Data) {
            buffer.append(data)
            while let newline = buffer.firstIndex(of: 0x0A) {
                let line = buffer[buffer.startIndex..<newline]
                buffer = Data(buffer[buffer.index(after: newline)...])
                if let message = (try? JSONSerialization.jsonObject(with: line)) as? [String: Any] {
                    onMessage?(message)
                }
            }
        }

        private func finish() {
            guard !closed else { return }
            closed = true
            connection.cancel()
            onClose?()
        }
    }
}
#endif
