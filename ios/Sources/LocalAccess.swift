import CryptoKit
import Foundation

/// The phone's half of sd-local-v1 (docs/LOCAL_ACCESS.md): the canonical
/// strings it signs, the device signature it checks, the header it sends, and
/// what every refusal means.
///
/// Counterpart of core/local_auth.py, byte for byte. Foundation + CryptoKit
/// only, so ios/Tests/LocalAccessTests.swift checks it against
/// tests/local_access_vectors.json (TEST KEYS ONLY) and
/// tests/test_m12_ios_local_access.py checks the Python side verifies what
/// Swift signs. NOT validated on a physical iPhone.
enum LocalAccess {
    static let contract = "sd-local-v1"
    /// Every signed write (§3.2).
    static let signatureHeader = "X-SD-Local"
    /// Reads and the stream, once a session is open (§3.3).
    static let sessionHeader = "X-SD-Session"
    /// How long before a session's end the app opens a new one. Sessions last
    /// 15 minutes on the device; renewing one minute early means a request
    /// never goes out on a session that dies in flight.
    static let renewBeforeExpiry: TimeInterval = 60

    // MARK: base64url (no padding), hex

    static func b64u(_ data: Data) -> String {
        data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }

    /// Nil for anything that is not base64url, or not `length` bytes when given.
    static func b64uDecode(_ text: String?, length: Int? = nil) -> Data? {
        guard var text = text?.trimmingCharacters(in: .whitespaces), !text.isEmpty,
              text.allSatisfy({ $0.isLetter && $0.isASCII || $0.isNumber && $0.isASCII || $0 == "-" || $0 == "_" })
        else { return nil }
        text = text.replacingOccurrences(of: "-", with: "+").replacingOccurrences(of: "_", with: "/")
        while text.count % 4 != 0 { text.append("=") }
        guard let data = Data(base64Encoded: text) else { return nil }
        if let length, data.count != length { return nil }
        return data
    }

    static func hex(_ data: Data) -> String { data.map { String(format: "%02x", $0) }.joined() }

    static func sha256Hex(_ data: Data) -> String { hex(Data(SHA256.hash(data: data))) }

    // MARK: identities

    /// First 16 hex of SHA-256 of the uncompressed (X9.63, 65 byte) public key.
    /// The device derives the same id, so nobody can claim an id for a key
    /// they do not hold.
    static func clientID(publicKey: Data) -> String { String(sha256Hex(publicKey).prefix(16)) }

    /// The device key's fingerprint, the same derivation (shown, never secret).
    static func fingerprint(publicKey: Data) -> String { String(sha256Hex(publicKey).prefix(16)) }

    /// An uncompressed P-256 point that CryptoKit accepts, else nil.
    static func publicKey(x963 data: Data?) -> P256.Signing.PublicKey? {
        guard let data, data.count == 65, data.first == 0x04 else { return nil }
        return try? P256.Signing.PublicKey(x963Representation: data)
    }

    // MARK: canonical messages

    /// What the phone signs for one request (§3.2). `path` has no query string.
    static func requestMessage(deviceID: String, clientID: String, challenge: String, method: String,
                               path: String, body: Data) -> Data {
        Data([contract, "request", deviceID, clientID, challenge, method.uppercased(), path,
              sha256Hex(body)].joined(separator: "\n").utf8)
    }

    /// What the DEVICE signs when it opens a session (§3.3).
    static func sessionMessage(deviceID: String, clientID: String, challenge: String, nonce: String,
                               session: String, expiresIn: Int) -> Data {
        Data([contract, "session", deviceID, clientID, challenge, nonce, sha256Hex(Data(session.utf8)),
              String(expiresIn)].joined(separator: "\n").utf8)
    }

    static func header(clientID: String, challenge: String, signature: Data) -> String {
        "\(contract) client=\(clientID),challenge=\(challenge),sig=\(b64u(signature))"
    }

    /// Whether `signature` (raw r||s, 64 bytes) is the holder of `publicKey`'s
    /// over `message`. ECDSA P-256 / SHA-256, as the device signs.
    static func verify(publicKey: Data, message: Data, signature: Data) -> Bool {
        guard signature.count == 64, let key = Self.publicKey(x963: publicKey),
              let sig = try? P256.Signing.ECDSASignature(rawRepresentation: signature)
        else { return false }
        return key.isValidSignature(sig, for: message)
    }

    /// 32 lowercase hex characters from the system generator (the session nonce).
    static func randomHex16() -> String {
        var generator = SystemRandomNumberGenerator()
        return hex(Data((0..<16).map { _ in UInt8.random(in: 0...255, using: &generator) }))
    }

    // MARK: what the device sends

    struct Hello: Decodable, Equatable {
        let contract: String
        let deviceID: String
        let deviceKeyFingerprint: String
        let challenge: String
        var challengeTTL: Int? = nil
        var tokenEnrollment: Bool? = nil

        enum CodingKeys: String, CodingKey {
            case contract, challenge
            case deviceID = "device_id"
            case deviceKeyFingerprint = "device_key_fp"
            case challengeTTL = "challenge_ttl_s"
            case tokenEnrollment = "token_enrollment"
        }
    }

    struct SessionReply: Decodable, Equatable {
        let session: String
        let expiresIn: Int
        let clientID: String
        let deviceID: String
        let deviceSignature: String

        enum CodingKeys: String, CodingKey {
            case session
            case expiresIn = "expires_in_s"
            case clientID = "client_id"
            case deviceID = "device_id"
            case deviceSignature = "device_sig"
        }
    }

    /// What every enrolment path hands back: the device's own key, learnt
    /// through a path that proved ownership (never from `hello`).
    struct Enrollment: Equatable, Codable {
        let clientID: String
        let deviceID: String
        let devicePublicKey: String  // base64url, 65 bytes

        enum CodingKeys: String, CodingKey {
            case clientID = "client_id"
            case deviceID = "device_id"
            case devicePublicKey = "device_public_key"
        }

        /// The device key as bytes, when it is a real P-256 point.
        var devicePublicKeyBytes: Data? {
            guard let data = LocalAccess.b64uDecode(devicePublicKey, length: 65),
                  LocalAccess.publicKey(x963: data) != nil else { return nil }
            return data
        }
    }

    /// Checks the device's answer to `POST /api/local/session` against the
    /// device key this phone stored at enrolment. Nil when it verifies; the
    /// reason otherwise. A failure means a look-alike, or a device that is not
    /// the one this phone enrolled with: the session is refused.
    static func checkSession(_ reply: SessionReply, expectedDeviceID: String, clientID: String,
                             challenge: String, nonce: String, devicePublicKey: Data) -> String? {
        guard reply.deviceID == expectedDeviceID else { return "the session names another device" }
        guard reply.clientID == clientID else { return "the session is for another phone" }
        guard let signature = b64uDecode(reply.deviceSignature, length: 64) else {
            return "the session carries no device signature"
        }
        let message = sessionMessage(deviceID: reply.deviceID, clientID: clientID, challenge: challenge,
                                     nonce: nonce, session: reply.session, expiresIn: reply.expiresIn)
        return verify(publicKey: devicePublicKey, message: message, signature: signature)
            ? nil : "the device's signature does not verify"
    }

    // MARK: refusals (§3.4)

    enum Refusal: Equatable {
        /// The signature failed (stale challenge, bad signature): new challenge, retry once.
        case retrySignature
        /// The session ended (expired, other address, device restarted): open a new one.
        case renewSession
        /// Not enrolled, revoked or idle-expired: forget the local credential.
        case forgetCredential
        /// Too many failures from this address: wait.
        case lockedOut(retryAfter: TimeInterval?)
        /// This device has no local access: use the account path only.
        case localAccessOff
        /// The device already trusts 16 phones.
        case tooManyClients
        /// Network enrolment needs the legacy token.
        case tokenRequired
        case other(status: Int, code: String?)
    }

    static func refusal(status: Int, code: String?, retryAfter: TimeInterval? = nil) -> Refusal {
        switch (status, code) {
        case (401, "missing_signature"?), (401, "malformed_signature"?), (401, "stale_challenge"?),
             (401, "bad_signature"?):
            return .retrySignature
        case (401, "session_invalid"?), (401, "unauthorized"?):
            return .renewSession
        case (403, "unknown_client"?):
            return .forgetCredential
        case (429, _):
            return .lockedOut(retryAfter: retryAfter)
        case (404, "local_access_off"?), (401, "local_access_off"?):
            return .localAccessOff
        case (409, "too_many_clients"?):
            return .tooManyClients
        case (401, "token_required"?):
            return .tokenRequired
        default:
            return .other(status: status, code: code)
        }
    }

    /// The refusal in the person's terms. Copy used across the app.
    static func message(_ refusal: Refusal) -> String {
        switch refusal {
        case .retrySignature: return "The camera didn't accept this iPhone's signature. Try again."
        case .renewSession: return "Reconnecting to the camera…"
        case .forgetCredential: return needsPairingAgain
        case .lockedOut(let wait):
            let seconds = wait.map { " Try again in \(Int($0.rounded(.up))) s." } ?? " Wait a minute and try again."
            return "The camera is refusing this iPhone for now after too many failed attempts." + seconds
        case .localAccessOff:
            return "This camera doesn't allow direct connections. It can still be reached through your account."
        case .tooManyClients:
            return "This camera already trusts 16 phones. Remove one in its settings, then pair this iPhone again."
        case .tokenRequired:
            return "Pairing over Wi-Fi needs the camera's old device token. Use Bluetooth setup or your account instead."
        case .other(let status, let code):
            return "The camera refused the request (\(code ?? "HTTP \(status)"))."
        }
    }

    static let needsPairingAgain = "This iPhone needs to be paired with the camera again."
}

// MARK: - The phone's signing key

/// Something that signs for this phone: a Secure Enclave key on a real iPhone,
/// a software key in the Simulator and in tests (LocalIdentity.swift).
protocol LocalSigner: Sendable {
    /// Uncompressed X9.63 P-256 public key, 65 bytes.
    var publicKey: Data { get }
    /// ECDSA P-256 / SHA-256 over `message`, raw r||s (64 bytes).
    func sign(_ message: Data) throws -> Data
}

extension LocalSigner {
    var clientID: String { LocalAccess.clientID(publicKey: publicKey) }
}

/// A software P-256 key. Used where there is no Secure Enclave (the Simulator)
/// and by the tests (with the vector scalar).
struct SoftwareSigner: LocalSigner, @unchecked Sendable {
    let key: P256.Signing.PrivateKey

    init(key: P256.Signing.PrivateKey = P256.Signing.PrivateKey()) { self.key = key }

    /// From a 32-byte scalar (tests: the vector's TEST key).
    init?(scalar: Data) {
        guard let key = try? P256.Signing.PrivateKey(rawRepresentation: scalar) else { return nil }
        self.key = key
    }

    var publicKey: Data { key.publicKey.x963Representation }

    func sign(_ message: Data) throws -> Data { try key.signature(for: message).rawRepresentation }
}

// MARK: - Bluetooth enrolment inside a proven PoP session (§4.1)

/// The `enroll` / `enrolled` messages, sealed to a proven BLE PoP session
/// (services/ble_pop.py `seal_enrollment` / `open_enrolled`). Needs
/// BLEProofOfPossession.swift (the session keys).
enum LocalEnrollmentBLE {
    static let enrollAAD = Data("local-enroll".utf8)
    static let enrolledAAD = Data("local-enrolled".utf8)

    /// Canonical JSON: sorted keys, no spaces, UTF-8 (Python `json.dumps(...,
    /// separators=(",", ":"), sort_keys=True, ensure_ascii=False)`).
    static func canonical(_ object: [String: String]) -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return (try? encoder.encode(object)) ?? Data("{}".utf8)
    }

    /// The device trims names to 64 printable characters; so does the app.
    static func cleanName(_ name: String) -> String {
        let printable = name.trimmingCharacters(in: .whitespacesAndNewlines)
            .unicodeScalars.filter { !CharacterSet.controlCharacters.contains($0) }
        let text = String(String(printable).prefix(64))
        return text.isEmpty ? "iPhone" : text
    }

    /// AES-256-GCM over {"name","public_key"}, bound to the transcript. ct || tag.
    static func sealEnrollment(_ keys: BLEProofOfPossession.SessionKeys, publicKey: Data, name: String,
                               iv: Data) throws -> Data {
        let plaintext = canonical(["name": name, "public_key": LocalAccess.b64u(publicKey)])
        let box = try AES.GCM.seal(plaintext, using: keys.enc, nonce: AES.GCM.Nonce(data: iv),
                                   authenticating: enrollAAD + keys.transcript)
        return box.ciphertext + box.tag
    }

    /// The device's `enrolled` answer, opened and checked; nil when it does not
    /// authenticate (not sealed to THIS session) or is not an enrolment.
    static func openEnrolled(_ keys: BLEProofOfPossession.SessionKeys, iv: Data,
                             ciphertext: Data) -> LocalAccess.Enrollment? {
        guard ciphertext.count > 16, iv.count == 12,
              let nonce = try? AES.GCM.Nonce(data: iv),
              let box = try? AES.GCM.SealedBox(nonce: nonce, ciphertext: ciphertext.prefix(ciphertext.count - 16),
                                               tag: ciphertext.suffix(16)),
              let plaintext = try? AES.GCM.open(box, using: keys.enc, authenticating: enrolledAAD + keys.transcript),
              let enrollment = try? JSONDecoder().decode(LocalAccess.Enrollment.self, from: plaintext),
              enrollment.devicePublicKeyBytes != nil
        else { return nil }
        return enrollment
    }
}
