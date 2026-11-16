import CryptoKit
import Foundation

/// Talks sd-local-v1 to one enrolled camera on the local network
/// (docs/LOCAL_ACCESS.md §3): hello, a device-signed session for reads and the
/// stream, and a fresh signature over a new challenge for every write.
///
/// The Swift twin of sim/local_client.py. Foundation + CryptoKit only and the
/// network is a `LocalTransport`, so ios/Tests/LocalDeviceClientTests.swift
/// runs it against a simulated device (IP change, power cycle, revocation, a
/// look-alike) and tests/test_m12_ios_local_access.py runs it against the
/// device's real Python dashboard. NOT validated on a physical iPhone.
///
/// What it will never do: send anything unsigned, accept a session the cached
/// device key did not sign, or fall back to an unauthenticated path.
protocol LocalTransport: Sendable {
    /// The body and HTTP status. Throws only when nothing answered.
    func send(_ request: URLRequest) async throws -> (Data, Int)
}

struct URLSessionLocalTransport: LocalTransport {
    func send(_ request: URLRequest) async throws -> (Data, Int) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.requestCachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        let session = URLSession(configuration: configuration)
        defer { session.finishTasksAndInvalidate() }
        let (data, response) = try await session.data(for: request)
        return (data, (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
}

enum LocalAccessError: Error, Equatable {
    /// Nothing answered at this address (moved, off, another network).
    case unreachable(String)
    /// The device answered and refused (LocalAccess.Refusal says what to do).
    case refused(LocalAccess.Refusal)
    /// A different SensorDyme answered at the cached address.
    case wrongDevice(found: String)
    /// The answer was not signed by the device key cached at enrolment.
    case notAuthentic(String)
    /// The device's role does not allow that mode (409, `refused: true`).
    case presetRefused(String)
    /// A 400 the device explained (e.g. an unknown preset).
    case badRequest(String)
    case malformed(String)

    var message: String {
        switch self {
        case .unreachable: return "The camera isn't reachable on this network."
        case .refused(let refusal): return LocalAccess.message(refusal)
        case .wrongDevice(let found):
            return "A different SensorDyme (\(found)) answered at the camera's last address. Looking for yours…"
        case .notAuthentic:
            return "The device at this address couldn't prove it is your camera, so nothing was sent or shown."
        case .presetRefused(let reason): return reason
        case .badRequest(let reason): return reason
        case .malformed: return "The camera answered in a way this app doesn't understand. It may need an update."
        }
    }

    /// The local credential is gone on the device: the app forgets it.
    var forgetsCredential: Bool { self == .refused(.forgetCredential) }
}

/// `GET /api/state`: the parts the app reads. Every field optional.
struct LocalDeviceState: Decodable, Equatable {
    struct Protection: Decodable, Equatable {
        var state: String?
        var role: String?
        var by: String?
        var since: Double?
        var reasons: [String]?
    }

    struct Policy: Decodable, Equatable {
        var protection: Protection?
        var mode: String?
    }

    var module: String?
    var alertActive: Bool?
    var policy: Policy?
    var generatedAt: String?

    enum CodingKeys: String, CodingKey {
        case module, policy
        case alertActive = "alert_active"
        case generatedAt = "generated_at"
    }

    /// "protecting", "not_protecting" or "stopped"; nil when the device did not say.
    var protection: String? { policy?.protection?.state }
}

/// `GET /api/presets`.
struct LocalPresets: Decodable, Equatable {
    struct Variant: Decodable, Equatable { let name: String; let title: String }
    struct Entry: Decodable, Equatable, Identifiable {
        let name: String
        let title: String
        var description: String?
        var active: Bool?
        var variants: [Variant]?
        var id: String { name }
    }

    var active: String?
    var presets: [Entry]

    /// The base of the running preset ("driver_safety" for "driver_safety:strict").
    var activeBase: String? { active.map { String($0.split(separator: ":", maxSplits: 1).first ?? "") } }
}

/// How a stop/resume or a mode switch ended, as the DEVICE reported it
/// afterwards. Never inferred from the 202.
enum LocalCommandOutcome: Equatable {
    case confirmed(String)
    case notConfirmed(String)

    var text: String {
        switch self {
        case .confirmed(let text), .notConfirmed(let text): text
        }
    }

    var isConfirmed: Bool { if case .confirmed = self { return true }; return false }
}

actor LocalDeviceClient {
    let deviceID: String
    let signer: LocalSigner
    let devicePublicKey: Data
    private(set) var baseURL: URL
    private let transport: LocalTransport
    private let now: @Sendable () -> Date
    private let pause: @Sendable (TimeInterval) async -> Void
    private var session: (value: String, expiresAt: Date)?
    /// Opened sessions so far (tests read it: a renewal is a new one).
    private(set) var sessionsOpened = 0

    init(deviceID: String, signer: LocalSigner, devicePublicKey: Data, baseURL: URL,
         transport: LocalTransport = URLSessionLocalTransport(),
         now: @escaping @Sendable () -> Date = { Date() },
         pause: @escaping @Sendable (TimeInterval) async -> Void = { try? await Task.sleep(for: .seconds($0)) }) {
        self.deviceID = deviceID
        self.signer = signer
        self.devicePublicKey = devicePublicKey
        self.baseURL = baseURL
        self.transport = transport
        self.now = now
        self.pause = pause
    }

    var clientID: String { signer.clientID }

    /// The session the stream uses, when one is open and not about to end.
    var currentSession: String? {
        guard let session, now() < session.expiresAt.addingTimeInterval(-LocalAccess.renewBeforeExpiry) else { return nil }
        return session.value
    }

    /// The camera moved (a new DHCP lease, another network): a session is bound
    /// to the address it was opened on, so the next request opens a new one.
    func setBaseURL(_ url: URL) {
        guard url != baseURL else { return }
        baseURL = url
        session = nil
    }

    /// Drop the session (the stream was refused with session_invalid).
    func invalidateSession() { session = nil }

    // MARK: protocol

    /// A challenge, from the camera this phone enrolled with. A different
    /// device id is another SensorDyme at a recycled address; a different key
    /// fingerprint is a device that is not the one enrolled.
    func hello() async throws -> LocalAccess.Hello {
        let (data, status) = try await call("GET", "/api/local/hello")
        guard status == 200 else { throw Self.refusal(status: status, data: data) }
        guard let hello = try? JSONDecoder().decode(LocalAccess.Hello.self, from: data),
              hello.contract == LocalAccess.contract else { throw LocalAccessError.malformed("hello") }
        guard hello.deviceID.uppercased() == deviceID.uppercased() else {
            throw LocalAccessError.wrongDevice(found: hello.deviceID)
        }
        guard hello.deviceKeyFingerprint == LocalAccess.fingerprint(publicKey: devicePublicKey) else {
            throw LocalAccessError.notAuthentic("the device key fingerprint changed")
        }
        return hello
    }

    /// `POST /api/local/session`, signed, and the device's signature checked
    /// against the cached key before the session is used for anything.
    @discardableResult
    func openSession() async throws -> String {
        let nonce = LocalAccess.randomHex16()
        let body = Data(#"{"nonce":"\#(nonce)"}"#.utf8)
        let hello = try await hello()
        let path = "/api/local/session"
        let (data, status) = try await call("POST", path, body: body,
                                            headers: [LocalAccess.signatureHeader: try header(hello, "POST", path, body)])
        guard status == 200 else { throw Self.refusal(status: status, data: data) }
        guard let reply = try? JSONDecoder().decode(LocalAccess.SessionReply.self, from: data) else {
            throw LocalAccessError.malformed("session")
        }
        if let problem = LocalAccess.checkSession(reply, expectedDeviceID: hello.deviceID, clientID: clientID,
                                                  challenge: hello.challenge, nonce: nonce,
                                                  devicePublicKey: devicePublicKey) {
            throw LocalAccessError.notAuthentic(problem)
        }
        session = (reply.session, now().addingTimeInterval(TimeInterval(reply.expiresIn)))
        sessionsOpened += 1
        return reply.session
    }

    /// An open session, renewed when it is within a minute of its end.
    func validSession() async throws -> String {
        if let current = currentSession { return current }
        return try await openSession()
    }

    /// A read with the session. A `session_invalid` (expired, the device
    /// restarted, another address) opens one new session and retries once.
    func get(_ pathAndQuery: String) async throws -> Data {
        var renewed = false
        while true {
            let value = try await validSession()
            let (data, status) = try await call("GET", pathAndQuery, headers: [LocalAccess.sessionHeader: value])
            if status == 200 { return data }
            let error = Self.refusal(status: status, data: data)
            if case .refused(.renewSession) = error, !renewed {
                renewed = true
                session = nil
                continue
            }
            throw error
        }
    }

    /// A write, signed over a fresh challenge. A stale challenge or a
    /// signature the device could not check is retried once with a new one.
    func signedPost(_ path: String, body: Data) async throws -> (Data, Int) {
        var retried = false
        while true {
            let hello = try await hello()
            let (data, status) = try await call("POST", path, body: body,
                                                headers: [LocalAccess.signatureHeader: try header(hello, "POST", path, body)])
            if (200..<300).contains(status) { return (data, status) }
            let error = Self.refusal(status: status, data: data)
            if case .refused(.retrySignature) = error, !retried {
                retried = true
                continue
            }
            throw error
        }
    }

    // MARK: what the app asks

    func state() async throws -> LocalDeviceState {
        try decode(LocalDeviceState.self, try await get("/api/state"))
    }

    func presets() async throws -> LocalPresets {
        try decode(LocalPresets.self, try await get("/api/presets"))
    }

    /// `GET /api/events`: the newest `limit`, or (with `after`) the ones since.
    func events(after: Int? = nil, limit: Int = 100) async throws -> LocalEventsPage {
        var query = "/api/events?limit=\(max(1, min(500, limit)))"
        if let after { query += "&after=\(after)" }
        return try decode(LocalEventsPage.self, try await get(query))
    }

    /// Stop (with the explicit confirmation the person gave) or resume, then
    /// read the device until it reports the result, or say it did not.
    func setProtection(stop: Bool, attempts: Int = 10, interval: TimeInterval = 1) async throws -> LocalCommandOutcome {
        let body = Data((stop ? #"{"action":"stop","confirm":true}"# : #"{"action":"resume"}"#).utf8)
        _ = try await signedPost("/api/protection", body: body)
        for attempt in 0..<max(1, attempts) {
            if attempt > 0 { await pause(interval) }
            guard let protection = try? await state().protection else { continue }
            if stop, protection == "stopped" {
                return .confirmed("Stopped — the camera confirmed it. Nothing is being watched.")
            }
            if !stop, protection != "stopped" {
                return .confirmed("Monitoring resumed — the camera confirmed it.")
            }
        }
        return .notConfirmed(stop
            ? "The camera hasn't confirmed the stop. It may still be monitoring."
            : "The camera hasn't confirmed it resumed. It may still be stopped.")
    }

    /// Switch mode; a 409 is the device's role floor and its reason is shown.
    func setPreset(_ name: String, attempts: Int = 12, interval: TimeInterval = 1) async throws -> LocalCommandOutcome {
        let body = (try? JSONSerialization.data(withJSONObject: ["preset": name])) ?? Data()
        _ = try await signedPost("/api/preset", body: body)
        let base = String(name.split(separator: ":", maxSplits: 1).first ?? "")
        for attempt in 0..<max(1, attempts) {
            if attempt > 0 { await pause(interval) }
            if let presets = try? await presets(), presets.activeBase == base {
                return .confirmed("Switched — the camera confirmed the new mode.")
            }
        }
        return .notConfirmed("The camera hasn't confirmed the new mode yet.")
    }

    /// Ask the camera to forget this phone (best effort when unpairing).
    func revokeSelf() async throws {
        let body = (try? JSONSerialization.data(withJSONObject: ["client_id": clientID])) ?? Data()
        _ = try await signedPost("/api/local/revoke", body: body)
    }

    // MARK: plumbing

    private func header(_ hello: LocalAccess.Hello, _ method: String, _ path: String, _ body: Data) throws -> String {
        let message = LocalAccess.requestMessage(deviceID: hello.deviceID, clientID: clientID,
                                                 challenge: hello.challenge, method: method, path: path, body: body)
        return LocalAccess.header(clientID: clientID, challenge: hello.challenge, signature: try signer.sign(message))
    }

    private func call(_ method: String, _ pathAndQuery: String, body: Data? = nil,
                      headers: [String: String] = [:]) async throws -> (Data, Int) {
        guard let url = URL(string: pathAndQuery, relativeTo: baseURL)?.absoluteURL else {
            throw LocalAccessError.malformed("address")
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.timeoutInterval = 6
        request.cachePolicy = .reloadIgnoringLocalAndRemoteCacheData
        if let body {
            request.httpBody = body
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }
        for (name, value) in headers { request.setValue(value, forHTTPHeaderField: name) }
        do {
            return try await transport.send(request)
        } catch {
            throw LocalAccessError.unreachable((error as NSError).localizedDescription)
        }
    }

    private func decode<T: Decodable>(_ type: T.Type, _ data: Data) throws -> T {
        guard let value = try? JSONDecoder().decode(type, from: data) else {
            throw LocalAccessError.malformed(String(describing: type))
        }
        return value
    }

    /// A non-2xx answer as the app's error: the sd-local-v1 codes, the role
    /// floor's 409, or a 400 with the device's reason.
    static func refusal(status: Int, data: Data) -> LocalAccessError {
        let body = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] ?? [:]
        let code = body["code"] as? String
        let reason = body["error"] as? String
        if status == 409, body["refused"] as? Bool == true {
            return .presetRefused(reason ?? "The camera's role doesn't allow that mode.")
        }
        if status == 400, code == nil {
            return .badRequest(reason ?? "The camera refused the request.")
        }
        let retry = (body["retry_after_s"] as? NSNumber)?.doubleValue
        return .refused(LocalAccess.refusal(status: status, code: code, retryAfter: retry))
    }
}
