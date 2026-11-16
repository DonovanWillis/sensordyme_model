import CryptoKit
import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: LocalAccess BLEProofOfPossession LocalEvents EventModel AlertSound
//
// LocalDeviceClient against a simulated sd-local-v1 device written from
// docs/LOCAL_ACCESS.md (the device's real Python dashboard is exercised by
// tests/test_m12_ios_local_access.py). Scenarios: a normal session, the
// session renewed before it ends, the device power-cycled (sessions lost), the
// camera's address changed, a revoked phone (403), a look-alike that cannot
// sign as the cached key, a lockout, the role floor's 409, and a stop that is
// confirmed only from /api/state.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}
func unhex(_ text: String) -> Data {
    var out = Data()
    var index = text.startIndex
    while index < text.endIndex {
        let next = text.index(index, offsetBy: 2)
        out.append(UInt8(text[index..<next], radix: 16)!)
        index = next
    }
    return out
}

/// TEST KEYS ONLY (tests/local_access_vectors.json).
let phoneSigner = SoftwareSigner(scalar: unhex(String(repeating: "11", count: 32)))!
let deviceKey = try! P256.Signing.PrivateKey(rawRepresentation: unhex(String(repeating: "22", count: 32)))
let deviceID = "SD-1A2B3C4D"

final class Clock: @unchecked Sendable {
    var now = Date(timeIntervalSince1970: 1_800_000_000)
}

/// A device that speaks sd-local-v1, answering at one host.
final class FakeDevice: LocalTransport, @unchecked Sendable {
    var host = "192.168.1.20"
    var signingKey = deviceKey
    var trusted: Set<String> = []          // client ids
    var keys: [String: Data] = [:]
    var challenges: Set<String> = []
    var sessions: [String: (client: String, expires: Date)] = [:]
    var protection = "protecting"
    var active = "driver_safety"
    var roleForbids: Set<String> = ["general"]
    /// False: the device answers 202 to a mode switch and never applies it.
    var appliesPreset = true
    /// What `policy` says beyond protection (role, mode), as the device sends it.
    var policyExtra: [String: Any] = [:]
    var lockedOut = false
    var applyStopAfterReads = 1            // the runtime applies a stop a little later
    private var readsSinceStop = -1
    var pendingStop: String?
    var log: [String] = []
    var events: [[String: Any]] = []
    let clock: Clock

    init(clock: Clock) { self.clock = clock }

    func enrol(_ publicKey: Data) {
        let id = LocalAccess.clientID(publicKey: publicKey)
        trusted.insert(id)
        keys[id] = publicKey
    }

    func reboot() { sessions.removeAll(); challenges.removeAll() }

    private func reply(_ object: Any, _ status: Int = 200) -> (Data, Int) {
        (try! JSONSerialization.data(withJSONObject: object), status)
    }

    func send(_ request: URLRequest) async throws -> (Data, Int) {
        guard request.url?.host == host else { throw URLError(.cannotConnectToHost) }
        let path = request.url!.path
        let query = request.url!.query ?? ""
        log.append("\(request.httpMethod ?? "?") \(path)")
        if lockedOut { return reply(["error": "locked", "code": "locked_out", "retry_after_s": 60], 429) }
        if path == "/api/local/hello" {
            let challenge = LocalAccess.randomHex16()
            challenges.insert(challenge)
            return reply(["contract": "sd-local-v1", "device_id": deviceID,
                          "device_key_fp": LocalAccess.fingerprint(publicKey: deviceKey.publicKey.x963Representation),
                          "challenge": challenge, "challenge_ttl_s": 60, "token_enrollment": false])
        }
        if request.httpMethod == "POST" {
            let body = request.httpBody ?? Data()
            let header = request.value(forHTTPHeaderField: LocalAccess.signatureHeader) ?? ""
            guard let client = verify(header: header, method: "POST", path: path, body: body) else {
                return reply(["error": "refused", "code": lastCode], lastCode == "unknown_client" ? 403 : 401)
            }
            switch path {
            case "/api/local/session":
                let nonce = (try! JSONSerialization.jsonObject(with: body) as! [String: Any])["nonce"] as! String
                let session = LocalAccess.b64u(Data((0..<32).map { _ in UInt8.random(in: 0...255) }))
                sessions[session] = (client.id, clock.now.addingTimeInterval(900))
                let message = LocalAccess.sessionMessage(deviceID: deviceID, clientID: client.id,
                                                         challenge: client.challenge, nonce: nonce,
                                                         session: session, expiresIn: 900)
                let sig = try! signingKey.signature(for: message).rawRepresentation
                return reply(["contract": "sd-local-v1", "session": session, "expires_in_s": 900,
                              "client_id": client.id, "device_id": deviceID,
                              "device_key_fp": "x", "device_sig": LocalAccess.b64u(sig)])
            case "/api/protection":
                let payload = try! JSONSerialization.jsonObject(with: body) as! [String: Any]
                let action = payload["action"] as! String
                if action == "stop", payload["confirm"] as? Bool != true { return reply(["error": "confirm"], 400) }
                pendingStop = action
                readsSinceStop = 0
                return reply(["ok": true, "requested": action], 202)
            case "/api/preset":
                let name = (try! JSONSerialization.jsonObject(with: body) as! [String: Any])["preset"] as! String
                if roleForbids.contains(name) {
                    return reply(["error": "role 'driver_safety' requires driver_attention", "refused": true], 409)
                }
                if appliesPreset { active = name }
                return reply(["ok": true, "switching_to": name], 202)
            case "/api/local/revoke":
                trusted.remove(client.id)
                sessions = sessions.filter { $0.value.client != client.id }
                return reply(["ok": true, "revoked": client.id])
            default:
                return reply(["error": "?"], 404)
            }
        }
        // Reads need a live session.
        let sent = request.value(forHTTPHeaderField: LocalAccess.sessionHeader) ?? ""
        guard let session = sessions[sent], session.expires > clock.now, trusted.contains(session.client) else {
            return reply(["error": "invalid or missing device token", "code": sent.isEmpty ? "unauthorized" : "session_invalid"], 401)
        }
        switch path {
        case "/api/state":
            if let pending = pendingStop {
                readsSinceStop += 1
                if readsSinceStop >= applyStopAfterReads {
                    protection = pending == "stop" ? "stopped" : "protecting"
                    pendingStop = nil
                }
            }
            var policy: [String: Any] = ["protection": ["state": protection, "role": "driver_safety", "reasons": []]]
            policy.merge(policyExtra) { _, new in new }
            return reply(["module": "driver_attention", "alert_active": false, "policy": policy])
        case "/api/presets":
            return reply(["active": active, "presets": [["name": "driver_safety", "title": "Driver Safety", "active": true,
                                                         "variants": [["name": "standard", "title": "Standard"],
                                                                      ["name": "strict", "title": "Strict"]]]]])
        case "/api/events":
            _ = query
            return reply(["schema": "sd-local-events/1", "events": events, "latest_seq": events.count, "boot_id": "b1"])
        default:
            return reply(["error": "?"], 404)
        }
    }

    private var lastCode = "missing_signature"

    private func verify(header: String, method: String, path: String, body: Data) -> (id: String, challenge: String)? {
        let pattern = #"^sd-local-v1 client=([0-9a-f]{16}),challenge=([0-9a-f]{32}),sig=([A-Za-z0-9_-]+)$"#
        guard let match = header.range(of: pattern, options: .regularExpression), match.lowerBound == header.startIndex else {
            lastCode = header.isEmpty ? "missing_signature" : "malformed_signature"
            return nil
        }
        let parts = header.dropFirst("sd-local-v1 ".count).split(separator: ",").map { $0.split(separator: "=", maxSplits: 1)[1] }
        let client = String(parts[0]), challenge = String(parts[1])
        guard challenges.remove(challenge) != nil else { lastCode = "stale_challenge"; return nil }
        guard trusted.contains(client), let key = keys[client] else { lastCode = "unknown_client"; return nil }
        let message = LocalAccess.requestMessage(deviceID: deviceID, clientID: client, challenge: challenge,
                                                 method: method, path: path, body: body)
        guard let sig = LocalAccess.b64uDecode(String(parts[2]), length: 64),
              LocalAccess.verify(publicKey: key, message: message, signature: sig) else {
            lastCode = "bad_signature"
            return nil
        }
        return (client, challenge)
    }
}

let clock = Clock()
let device = FakeDevice(clock: clock)
device.enrol(phoneSigner.publicKey)
func client(_ device: FakeDevice, host: String = "192.168.1.20", key: Data = deviceKey.publicKey.x963Representation) -> LocalDeviceClient {
    LocalDeviceClient(deviceID: deviceID, signer: phoneSigner, devicePublicKey: key,
                      baseURL: URL(string: "http://\(host):8899")!, transport: device,
                      now: { clock.now }, pause: { _ in })
}
func error<T>(_ body: () async throws -> T) async -> LocalAccessError? {
    do { _ = try await body(); return nil } catch let e as LocalAccessError { return e } catch { return .malformed("\(error)") }
}

print("a session from the enrolled camera, then reads with it:")
let c = client(device)
let first = try! await c.validSession()
check(!first.isEmpty, "a session opens, after the device's signature checked out against the cached key")
let state = try! await c.state()
check(state.protection == "protecting", "a read with X-SD-Session reaches /api/state")
check(try! await c.validSession() == first, "the same session is reused while it has time left")
check(await c.sessionsOpened == 1, "one session so far")

print("\nrenewed before it ends (15 min sessions; renewed in the last minute):")
clock.now = clock.now.addingTimeInterval(841)
let renewed = try! await c.validSession()
let opened2 = await c.sessionsOpened
check(renewed != first && opened2 == 2, "a new session 59 s before the old one ends")

print("\nthe camera power-cycled (every session lost):")
device.reboot()
check((try? await c.state())?.protection == "protecting", "the read is answered: session_invalid -> a new session -> retried once")
check(await c.sessionsOpened == 3, "exactly one more session")

print("\nthe camera's address changed (a new DHCP lease):")
device.host = "192.168.1.77"
let moved = await error { try await c.state() }
if case .unreachable? = moved { check(true, "the old address is unreachable, said as such") }
else { check(false, "the old address is unreachable (\(String(describing: moved)))") }
await c.setBaseURL(URL(string: "http://192.168.1.77:8899")!)
check((try? await c.state()) != nil, "re-resolved: the new address works")
check(await c.sessionsOpened == 4, "with a NEW session (sessions are bound to the address)")

print("\na look-alike at the address (it cannot sign with the camera's key):")
let impostor = FakeDevice(clock: clock)
impostor.host = "192.168.1.77"
impostor.enrol(phoneSigner.publicKey)
impostor.signingKey = P256.Signing.PrivateKey()
let fooled = await error { try await client(impostor, host: "192.168.1.77").openSession() }
if case .notAuthentic? = fooled { check(true, "the session is refused: the device signature does not verify") }
else { check(false, "the session is refused: the device signature does not verify (\(String(describing: fooled)))") }
let otherKey = P256.Signing.PrivateKey().publicKey.x963Representation
let wrongFP = await error { try await client(device, host: "192.168.1.77", key: otherKey).hello() }
if case .notAuthentic? = wrongFP { check(true, "a device whose key fingerprint differs from the cached one is refused at hello") }
else { check(false, "fingerprint mismatch refused (\(String(describing: wrongFP)))") }

print("\na stop is confirmed only from /api/state:")
device.applyStopAfterReads = 2
let stop = try! await c.setProtection(stop: true)
check(stop.isConfirmed && device.protection == "stopped", "stopped, once the device reported it")
check(device.log.contains("POST /api/protection"), "signed POST /api/protection was sent")
device.applyStopAfterReads = 99
let stuck = try! await c.setProtection(stop: false, attempts: 3)
check(!stuck.isConfirmed && stuck.text.contains("hasn't confirmed"), "a 202 the device never acted on is NOT reported as resumed")
device.applyStopAfterReads = 1
check(try! await c.setProtection(stop: false).isConfirmed, "resume confirmed")

print("\nmode selection and the role floor:")
// M13.1: a switch between two variants of the running preset is confirmed by
// the VARIANT the device reports; the base was running before the tap.
device.appliesPreset = false
let unapplied = try! await c.setPreset("driver_safety:strict", attempts: 3)
check(!unapplied.isConfirmed, "a 202 for a variant the device never applied is NOT confirmed by its base")
device.appliesPreset = true
check(try! await c.setPreset("driver_safety:strict").isConfirmed, "a switch confirmed from /api/presets (the variant)")
check(device.active == "driver_safety:strict", "the device runs it")
check(try! await c.setPreset("driver_safety:standard").isConfirmed, "and back to Standard")
let floor = await error { try await c.setPreset("general") }
check(floor == .presetRefused("role 'driver_safety' requires driver_attention"), "a 409 role-floor refusal carries the device's reason")

print("\nwhat the device reports running (LocalPresets.reports):")
let variants: [LocalPresets.Variant] = [.init(name: "standard", title: "Standard"), .init(name: "strict", title: "Strict")]
func presetsRunning(_ active: String?) -> LocalPresets {
    LocalPresets(active: active, presets: [.init(name: "driver_safety", title: "Driver Safety", variants: variants),
                                           .init(name: "occupancy", title: "Occupancy")])
}
check(presetsRunning("driver_safety").reports("driver_safety:standard"), "a bare preset runs as its first variant")
check(!presetsRunning("driver_safety").reports("driver_safety:strict"), "and not as another")
check(presetsRunning("driver_safety:strict@sustain=3").reports("driver_safety:strict"), "settings do not change the mode")
check(presetsRunning("driver_safety:strict").reports("driver_safety"), "a bare request: the base is enough")
check(!presetsRunning("occupancy").reports("driver_safety"), "another preset is not it")
check(!presetsRunning(nil).reports("driver_safety"), "the device said nothing: not confirmed")
check(!presetsRunning("driver_safety").reports(""), "an empty request is never confirmed")

print("\nthe role and the monitoring mode, read leniently from /api/state:")
device.policyExtra = ["mode": "standard_plus_custom",
                      "role": ["name": "driver_safety", "title": "Driver Safety", "allowed_presets": ["driver_safety"]]]
let withRole = try! await c.state()
check(withRole.policy?.role?.allowedPresets == ["driver_safety"] && withRole.policy?.mode == "standard_plus_custom",
      "role.allowed_presets and policy.mode")
device.policyExtra = ["role": "driver_safety", "mode": 3]
let oddRole = try? await c.state()
check(oddRole?.protection == "protecting" && oddRole?.policy?.role == nil && oddRole?.policy?.mode == nil,
      "a role or mode this build cannot read is absent, never a failed read")
device.policyExtra = [:]

print("\na stale challenge is retried once with a new one:")
check(LocalDeviceClient.refusal(status: 401, data: Data(#"{"code":"stale_challenge"}"#.utf8)) == .refused(.retrySignature),
      "stale_challenge maps to a retry")

print("\nlockout:")
device.lockedOut = true
check(await error { try await c.state() } == .refused(.lockedOut(retryAfter: 60)), "429 is a wait of retry_after_s")
device.lockedOut = false

print("\nlocal events:")
device.events = [["seq": 1, "event_key": "k1", "type": "drowsiness", "timestamp": "2026-11-15T10:00:00+00:00",
                  "severity": "loud", "age_s": 4.0, "sync": "pending", "simulated": true]]
let page = try! await c.events(after: 0)
check(page.events.first?.eventKey == "k1" && page.latestSeq == 1, "GET /api/events decodes")

print("\nthe phone is revoked on the camera:")
device.trusted.removeAll()
device.sessions.removeAll()
let revoked = await error { try await c.state() }
check(revoked?.forgetsCredential == true, "403 unknown_client: the credential is to be forgotten (no retry loop)")
check(revoked?.message == "This iPhone needs to be paired with the camera again.", "and the person is told to pair again")
let count = device.log.count
_ = await error { try await c.setProtection(stop: true) }
check(device.log.count - count <= 2, "a write is refused at once too (hello + one signed POST, no retries)")

print("\nnothing unsigned is ever sent:")
check(!device.log.contains { $0.hasPrefix("POST") && !["POST /api/local/session", "POST /api/protection", "POST /api/preset", "POST /api/local/revoke"].contains($0) },
      "only the signed endpoints were written to")

print("\n\(failures == 0 ? "all passed" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
