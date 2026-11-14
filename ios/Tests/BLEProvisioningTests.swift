import CryptoKit
import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: BLETransport BLEPoPFlow BLEProofOfPossession ProvisioningStatus
//
// BLEProvisioner over a scripted link: the logic CoreBluetooth used to hide
// (identify, prove, lockout, send, outcome, drops) without a radio.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

@MainActor
func settle(_ what: String, timeout: Double = 5, _ condition: () -> Bool) async {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() && Date() < deadline { try? await Task.sleep(nanoseconds: 10_000_000) }
    check(condition(), what)
}

/// A SensorDyme on the other end: proof of possession (v1) and the status
/// characteristic, answering the way services/ble_pop.py and
/// services/provisioning.py do.
@MainActor
final class ScriptedDevice: BLETransport {
    var onEvent: ((BLETransportEvent) -> Void)?
    let peripheral = UUID()
    var deviceID = "SD-0A1B2C3D"
    var code = "ABCD-EFGH-JKMN"
    var auth: String? = "pop"
    var joinAccepts = true
    var wrongCodesBeforeLock = 5
    var lockSeconds = 30
    private(set) var failuresSoFar = 0
    private(set) var writes: [(ProvisioningCharacteristic, String)] = []
    private(set) var disconnects = 0
    private(set) var joined: [String] = []
    private var keys: BLEProofOfPossession.SessionKeys?
    private var proven = false
    private var attempt = 0
    private var characteristics: Set<ProvisioningCharacteristic> {
        auth == "pop" ? Set(ProvisioningCharacteristic.allCases) : Set(ProvisioningCharacteristic.allCases).subtracting([.auth])
    }

    func later(_ event: BLETransportEvent) {
        Task { @MainActor in self.onEvent?(event) }
    }

    func startScan() {
        later(.radio(.poweredOn))
        later(.discovered(BLEAdvertisement(id: peripheral, advertisedName: nil, cachedName: "SensorDyme-TEST", rssi: -70)))
        later(.discovered(BLEAdvertisement(id: peripheral, advertisedName: "SensorDyme-3C4D", cachedName: nil, rssi: -48)))
    }
    func stopScan() {}
    func connect(_ id: UUID) -> Bool {
        guard id == peripheral else { return false }
        later(.connected(id))
        later(.characteristics(id, characteristics))
        return true
    }
    func disconnect(_ id: UUID) { disconnects += 1 }
    func subscribe(_ c: ProvisioningCharacteristic, on id: UUID) {}
    func read(_ c: ProvisioningCharacteristic, on id: UUID) {
        switch c {
        case .deviceInfo:
            var info: [String: Any] = ["id": deviceID, "name": "SensorDyme-3C4D"]
            if let auth { info["auth"] = auth }
            later(.value(id, c, json(info)))
        case .status:
            later(.value(id, c, json(["state": "provisioning", "device_id": deviceID, "provision": NSNull()])))
        case .networks:
            later(.value(id, c, json([["ssid": "Home", "signal": 80, "secured": true]])))
        default: break
        }
    }
    func write(_ data: Data, to c: ProvisioningCharacteristic, on id: UUID) {
        let msg = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        writes.append((c, msg["op"] as? String ?? "plain"))
        later(.wrote(id, c, error: nil))
        if c == .credentials {
            guard auth != "pop" else { return }  // refused under proof of possession, as on the device
            join(ssid: msg["ssid"] as? String ?? "")
            return
        }
        guard c == .auth else { return }
        later(.value(id, .auth, reply(msg)))
    }

    func json(_ object: Any) -> Data { try! JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]) }

    private func reply(_ msg: [String: Any]) -> Data {
        switch msg["op"] as? String {
        case "hello":
            if failuresSoFar >= wrongCodesBeforeLock {
                return json(["op": "error", "error": "locked", "retry_after_s": lockSeconds])
            }
            let na = Data(base64Encoded: msg["na"] as! String)!
            let nd = Data((0..<16).map { _ in UInt8.random(in: 0...255) })
            keys = BLEProofOfPossession.sessionKeys(
                codeKey: try! BLEProofOfPossession.codeKey(code: code, deviceID: deviceID), deviceID: deviceID,
                nonceApp: na, nonceDevice: nd)
            proven = false
            return json(["op": "challenge", "v": 1, "id": deviceID, "nd": nd.base64EncodedString()])
        case "proof":
            guard let keys, Data(base64Encoded: msg["tag"] as! String) == BLEProofOfPossession.appProof(keys) else {
                failuresSoFar += 1
                var error: [String: Any] = ["op": "error", "error": "wrong_code"]
                if failuresSoFar >= wrongCodesBeforeLock { error["retry_after_s"] = lockSeconds }
                return json(error)
            }
            proven = true
            return json(["op": "proven", "tag": BLEProofOfPossession.deviceProof(keys).base64EncodedString()])
        case "creds":
            guard proven, let keys else { return json(["op": "error", "error": "unexpected"]) }
            let iv = Data(base64Encoded: msg["iv"] as! String)!
            let ct = Data(base64Encoded: msg["ct"] as! String)!
            let box = try! AES.GCM.SealedBox(nonce: AES.GCM.Nonce(data: iv), ciphertext: ct.dropLast(16), tag: ct.suffix(16))
            let plain = try! AES.GCM.open(box, using: keys.enc, authenticating: Data("wifi-credentials".utf8) + keys.transcript)
            let creds = try! JSONSerialization.jsonObject(with: plain) as! [String: Any]
            proven = false
            join(ssid: creds["ssid"] as! String)
            return json(["op": "accepted"])
        default:
            return json(["op": "error", "error": "malformed"])
        }
    }

    private func join(ssid: String) {
        attempt += 1
        let n = attempt
        later(.value(peripheral, .status, json(["state": "provisioning", "provision": ["state": "connecting", "attempt": n]])))
        let outcome: [String: Any] = joinAccepts
            ? ["state": "ok", "ip": "192.0.2.10", "attempt": n]
            : ["state": "error", "error": "That password was not accepted by '\(ssid)'.", "attempt": n]
        if joinAccepts { joined.append(ssid) }
        Task { @MainActor in
            try? await Task.sleep(nanoseconds: 50_000_000)
            self.onEvent?(.value(self.peripheral, .status, self.json(["state": "online", "provision": outcome])))
        }
    }

    func dropLink() { later(.disconnected(peripheral)) }
}

let label = BLEProofOfPossession.SetupLabel(deviceID: "SD-0A1B2C3D", code: "ABCDEFGHJKMN")
let wrongLabel = BLEProofOfPossession.SetupLabel(deviceID: "SD-0A1B2C3D", code: "ZZZZZZZZZZZZ")

@MainActor
func connected(_ device: ScriptedDevice) async -> BLEProvisioner {
    let ble = BLEProvisioner(transport: device)
    ble.startScan()
    await settle("one entry per device, the advertised name, the fresher signal") {
        ble.discovered.count == 1 && ble.discovered[0].name == "SensorDyme-3C4D" && ble.discovered[0].rssi == -48
    }
    ble.connect(id: device.peripheral)
    await settle("identified and ready") { if case .ready = ble.state { return true }; return false }
    return ble
}

@MainActor
func main() async {
    print("connecting reads who the device is and how it takes credentials:")
    let device = ScriptedDevice()
    let ble = await connected(device)
    check(ble.deviceID == "SD-0A1B2C3D" && ble.deviceAuth == "pop" && ble.requiresSetupCode, "id and auth from device info")
    check(!ble.networks.isEmpty, "the device's networks were asked for straight away")

    print("\na wrong code: refused, the link stays, nothing secret sent:")
    ble.prove(label: wrongLabel)
    await settle("back to ready with the device's refusal") {
        if case .ready = ble.state { return ble.proofRejection?.kind == .wrongCode }
        return false
    }
    check(device.writes.map(\.1) == ["hello", "proof"], "hello and proof only")
    check(device.disconnects == 0, "still connected")

    print("\nthe lockout carries the device's wait, from when it said so:")
    var clock = Date(timeIntervalSince1970: 10_000)
    ble.now = { clock }
    device.wrongCodesBeforeLock = 2
    ble.prove(label: wrongLabel)
    await settle("refused again") { ble.proofRejection?.retryAfterSeconds == 30 }
    check(ble.lockedUntil == clock.addingTimeInterval(30), "locked until the device's 30 s from now")
    clock = clock.addingTimeInterval(31)
    check(ble.lockedUntil == nil, "and no longer once that has passed")
    device.wrongCodesBeforeLock = 99

    print("\nthe right code proves; credentials then go encrypted on the auth characteristic:")
    ble.prove(label: label)
    await settle("proven") { if case .proven = ble.state { return true }; return false }
    ble.send(ssid: "Home", password: "s3cret-pass")
    await settle("joined", timeout: 5) { if case .success(let ip) = ble.state { return ip == "192.0.2.10" }; return false }
    check(device.writes.suffix(1).map(\.1) == ["creds"] && device.writes.last?.0 == .auth, "creds on 5d1e0007")
    check(!device.writes.contains { $0.0 == .credentials }, "never the plain characteristic")

    print("\na refused password keeps the link; the next try renews the handshake:")
    let second = ScriptedDevice()
    let ble2 = await connected(second)
    ble2.prove(label: label)
    await settle("proven") { if case .proven = ble2.state { return true }; return false }
    second.joinAccepts = false
    ble2.send(ssid: "Home", password: "wrong-pass")
    await settle("join refused, link up") {
        if case .joinFailed(let reason) = ble2.state { return reason.contains("not accepted") }
        return false
    }
    check(second.disconnects == 0, "not disconnected")
    second.joinAccepts = true
    ble2.send(ssid: "Home", password: "right-pass")
    await settle("joined on the second try") { if case .success = ble2.state { return true }; return false }
    check(second.writes.map(\.1) == ["hello", "proof", "creds", "hello", "proof", "creds"], "a fresh hello carried the retry")

    print("\nBluetooth dropping mid-join does not blame the password:")
    let third = ScriptedDevice()
    third.auth = nil
    let ble3 = await connected(third)
    check(!ble3.requiresSetupCode, "a legacy device needs no code")
    ble3.send(ssid: "Home", password: "x")
    third.dropLink()
    await settle("failed with 'may have joined anyway'") {
        if case .failed(let reason) = ble3.state { return reason.contains("may have joined anyway") }
        return false
    }

    print("\na device that answers as someone else is dropped:")
    let fourth = ScriptedDevice()
    let ble4 = await connected(fourth)
    fourth.deviceID = "SD-FFFFFFFF"  // its challenge now names another device
    ble4.prove(label: label)
    await settle("failed, naming the other device") {
        if case .failed(let reason) = ble4.state { return reason.contains("SD-FFFFFFFF") }
        return false
    }
    check(fourth.disconnects >= 1, "disconnected")

    print("\nlate callbacks from another link change nothing:")
    let fifth = ScriptedDevice()
    let ble5 = await connected(fifth)
    ble5.handle(.disconnected(UUID()))
    ble5.handle(.value(UUID(), .status, fifth.json(["provision": ["state": "ok", "attempt": 9]])))
    check({ if case .ready = ble5.state { return true }; return false }(), "still ready")

    print("\nBluetooth off before sending is its own state:")
    ble5.handle(.radio(.poweredOff))
    check(ble5.state == .bluetoothOff, "bluetoothOff")
}

await main()
print("\n\(failures == 0 ? "all passed" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
