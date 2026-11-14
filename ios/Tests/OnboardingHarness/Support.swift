import Foundation

// Support for the cross-language onboarding harness (main.swift). Built by
// sim/ble_harness.py with swiftc from the REAL app sources (SetupFlow,
// SetupOnboarding, BLEProvisioning, BLEPoPFlow, BLEProofOfPossession,
// ProvisioningStatus, BLETransport, BLESimulatedTransport) plus these two
// files; it never runs inside the app.

/// Evidence, one JSON object per line. Never the setup code, the passphrase,
/// the claim code, a token, or protocol key material.
final class Evidence {
    private let handle: FileHandle?
    private let started = Date()
    private(set) var failures: [String] = []

    init(path: String?) {
        if let path {
            FileManager.default.createFile(atPath: path, contents: nil)
            handle = FileHandle(forWritingAtPath: path)
        } else {
            handle = nil
        }
    }

    func record(_ kind: String, _ fields: [String: Any] = [:]) {
        var line = fields
        line["kind"] = kind
        line["t"] = NSDecimalNumber(string: String(format: "%.3f", Date().timeIntervalSince(started)))
        guard let data = try? JSONSerialization.data(withJSONObject: line, options: [.sortedKeys]) else { return }
        handle?.write(data + Data([0x0A]))
        if let text = String(data: data, encoding: .utf8) { print(text) }
    }

    func check(_ condition: Bool, _ name: String, _ detail: String = "") {
        record("check", ["name": name, "ok": condition, "detail": detail])
        if !condition { failures.append(name) }
    }
}

struct HarnessError: Error, CustomStringConvertible {
    let description: String
}

/// The operation an auth-characteristic message carries ("hello", "proof", ...).
func authOp(_ data: Data) -> String? {
    ((try? JSONSerialization.jsonObject(with: data)) as? [String: Any])?["op"] as? String
}

/// A man in the middle of the simulated link: records what the app writes and
/// can delay or alter messages in either direction. This is how the harness
/// plays an attacker on the air (or a slow phone) without touching the device.
@MainActor
final class InterceptingTransport: BLETransport {
    let inner: BLETransport
    var onEvent: ((BLETransportEvent) -> Void)?
    /// (characteristic, auth op or nil) of every write the app made, in order.
    private(set) var writes: [(ProvisioningCharacteristic, String?)] = []
    var alterIncoming: ((ProvisioningCharacteristic, Data) -> Data?)?
    var alterOutgoing: ((ProvisioningCharacteristic, Data) -> Data?)?
    var delayOutgoing: ((ProvisioningCharacteristic, Data) -> Double)?
    var onTamper: ((String) -> Void)?

    init(inner: BLETransport) {
        self.inner = inner
        inner.onEvent = { [weak self] event in self?.forward(event) }
    }

    private func forward(_ event: BLETransportEvent) {
        if case .value(let id, let characteristic, let data) = event, let altered = alterIncoming?(characteristic, data) {
            onTamper?("device->app \(characteristic) altered")
            onEvent?(.value(id, characteristic, altered))
            return
        }
        onEvent?(event)
    }

    func startScan() { inner.startScan() }
    func stopScan() { inner.stopScan() }
    func connect(_ id: UUID) -> Bool { inner.connect(id) }
    func disconnect(_ id: UUID) { inner.disconnect(id) }
    func read(_ characteristic: ProvisioningCharacteristic, on id: UUID) { inner.read(characteristic, on: id) }
    func subscribe(_ characteristic: ProvisioningCharacteristic, on id: UUID) { inner.subscribe(characteristic, on: id) }

    func write(_ data: Data, to characteristic: ProvisioningCharacteristic, on id: UUID) {
        writes.append((characteristic, characteristic == .auth ? authOp(data) : nil))
        var out = data
        if let altered = alterOutgoing?(characteristic, data) {
            onTamper?("app->device \(characteristic) altered")
            out = altered
        }
        let delay = delayOutgoing?(characteristic, data) ?? 0
        guard delay > 0 else {
            inner.write(out, to: characteristic, on: id)
            return
        }
        onTamper?("app->device \(characteristic) delayed \(delay)s")
        let payload = out
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            self.inner.write(payload, to: characteristic, on: id)
        }
    }

    func authWrites() -> [String] { writes.filter { $0.0 == .auth }.compactMap(\.1) }
    func credentialWrites() -> Int { writes.filter { $0.0 == .credentials || ($0.0 == .auth && $0.1 == "creds") }.count }
}

/// Rewrite one JSON field of a message (base64 values get one byte flipped).
func tamper(_ data: Data, field: String, replaceWith value: String? = nil) -> Data? {
    guard var object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
          let current = object[field] as? String else { return nil }
    if let value {
        object[field] = value
    } else if var bytes = Data(base64Encoded: current), !bytes.isEmpty {
        bytes[bytes.startIndex] ^= 0x01
        object[field] = bytes.base64EncodedString()
    } else {
        return nil
    }
    return try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
}

/// The backend the app talks to (cloud/app.py, run by sim.backend), over HTTP.
/// The harness's stand-in for AppState.backend; same endpoints, same bodies.
@MainActor
final class HTTPAccount: OnboardingAccount {
    struct HTTPFailure: Error {
        let status: Int
        let code: String?
        let detail: String?
    }

    let base: URL
    private var token: String?
    private let evidence: Evidence
    var isSignedIn: Bool { token != nil }

    init(base: URL, evidence: Evidence) {
        self.base = base
        self.evidence = evidence
    }

    func signIn(email: String, password: String) async throws {
        let reply = try await call("POST", "v1/login", body: ["email": email, "password": password], log: "{credentials}")
        token = (reply as? [String: Any])?["token"] as? String
    }

    func accountDevices() async throws -> [Onboarding.AccountDevice] {
        let reply = try await call("GET", "v1/devices")
        let rows = ((reply as? [String: Any])?["devices"] as? [[String: Any]]) ?? []
        return rows.compactMap { row in
            guard let id = row["id"] as? Int else { return nil }
            return Onboarding.AccountDevice(id: id, name: row["name"] as? String ?? "", hardwareID: row["hardware_id"] as? String,
                                            online: row["online"] as? Bool ?? false,
                                            activePreset: row["active_preset"] as? String)
        }
    }

    func claim(code: String) async throws {
        _ = try await call("POST", "v1/devices/claim", body: ["claim_code": code], log: "{claim_code: <\(code.count) chars>}")
    }

    func modes(deviceID: Int) async throws -> (modes: [Onboarding.Mode], active: String?) {
        let catalog = (try await call("GET", "v1/presets") as? [String: Any])?["presets"] as? [[String: Any]] ?? []
        let device = try await call("GET", "v1/devices/\(deviceID)/presets") as? [String: Any]
        let states = Dictionary(uniqueKeysWithValues: ((device?["presets"] as? [[String: Any]]) ?? []).compactMap { row in
            (row["id"] as? String).map { ($0, row) }
        })
        let modes = catalog.compactMap { entry -> Onboarding.Mode? in
            guard let id = entry["id"] as? String else { return nil }
            let state = states[id]
            return Onboarding.Mode(id: id, title: entry["title"] as? String ?? id,
                                   detail: entry["description"] as? String ?? "",
                                   available: state?["available"] as? Bool ?? true, reason: state?["reason"] as? String)
        }
        return (modes, device?["active_preset"] as? String)
    }

    func activate(mode: String, deviceID: Int) async throws {
        _ = try await call("POST", "v1/devices/\(deviceID)/preset", body: ["preset": mode])
    }

    func explain(_ error: Error, claiming: Bool) -> String {
        guard let failure = error as? HTTPFailure else { return error.localizedDescription }
        if claiming { return Onboarding.claimMessage(status: failure.status, code: failure.code, detail: failure.detail) }
        return failure.detail ?? "HTTP \(failure.status)"
    }

    private func call(_ method: String, _ path: String, body: [String: Any]? = nil, log: String? = nil) async throws -> Any? {
        var request = URLRequest(url: base.appendingPathComponent(path))
        request.httpMethod = method
        request.timeoutInterval = 15
        if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
        if let body {
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        let object = try? JSONSerialization.jsonObject(with: data)
        var fields: [String: Any] = ["method": method, "path": "/" + path, "status": status]
        if let log { fields["body"] = log } else if let body { fields["body"] = body }
        evidence.record("api", fields)
        guard (200..<300).contains(status) else {
            let detail = (object as? [String: Any])?["detail"]
            let code = (detail as? [String: Any])?["code"] as? String ?? (object as? [String: Any])?["code"] as? String
            let message = (detail as? String) ?? ((detail as? [String: Any])?["message"] as? String)
            throw HTTPFailure(status: status, code: code, detail: message)
        }
        return object
    }
}
