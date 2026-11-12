import CommonCrypto
import CryptoKit
import Foundation
import Security

/// The phone's half of Bluetooth proof of possession, protocol v1.
///
/// Counterpart of services/ble_pop.py, which documents the protocol and its
/// reasoning. In short: the person scans the device's label (QR payload
/// `SDSETUP:1:<device id>:<code>`), the app connects ONLY to the device whose
/// device-info `id` matches the label, proves it holds the code, checks the
/// device's proof back, and only then sends the Wi-Fi credentials, encrypted
/// to this session. A device that is merely discoverable is never assumed to
/// be the user's.
///
/// Pure logic: no CoreBluetooth here. BLEProvisioning.swift carries the bytes
/// on characteristic 5d1e0007 once a view adopts this flow. Verified against
/// shared vectors from the Python implementation; NOT validated on physical
/// BLE hardware.
enum BLEProofOfPossession {
    static let protocolVersion = 1
    static let authCharacteristicUUID = "5d1e0007-5344-4d45-b000-536e447a6d65"
    static let prefix = Data("SensorDyme BLE PoP v1|".utf8)
    /// Fixed per protocol version and never taken from the device: a look-alike
    /// device could otherwise ask for 1 iteration and make its harvest cheap.
    static let pbkdf2Iterations: UInt32 = 600_000
    static let codeAlphabet = Set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")
    static let codeLength = 12

    enum Failure: Error, Equatable {
        case invalidCode
        case invalidLabel
        case malformedReply(String)
        /// The device answered with an error: `wrong_code`, `locked` (with a wait), `busy`, ...
        case rejected(String, retryAfterSeconds: Int?)
        /// The challenge came from a device other than the one on the label.
        case wrongDevice(expected: String, got: String)
        /// The device could not prove it knows the code: do not send credentials.
        case deviceNotAuthentic
        case notProven
        case keyDerivationFailed
    }

    struct SetupLabel: Equatable {
        let deviceID: String
        let code: String  // canonical, 12 symbols
    }

    // ------------------------------------------------------------------ the code
    /// The canonical form of a typed code: dashes/spaces dropped, case folded,
    /// O->0 and I/L->1 (Crockford base32), anything else refused.
    static func normalize(code: String) throws -> String {
        var out = ""
        for ch in code.uppercased() where ch != "-" && ch != " " {
            switch ch {
            case "O": out.append("0")
            case "I", "L": out.append("1")
            default: out.append(ch)
            }
        }
        guard out.count == codeLength, out.allSatisfy({ codeAlphabet.contains($0) }) else {
            throw Failure.invalidCode
        }
        return out
    }

    static func parseLabel(_ text: String) throws -> SetupLabel {
        let parts = text.trimmingCharacters(in: .whitespacesAndNewlines)
            .split(separator: ":", omittingEmptySubsequences: false).map(String.init)
        guard parts.count == 4, parts[0] == "SDSETUP", parts[1] == String(protocolVersion), !parts[2].isEmpty else {
            throw Failure.invalidLabel
        }
        return SetupLabel(deviceID: parts[2], code: try normalize(code: parts[3]))
    }

    // ------------------------------------------------------------------ keys
    /// PBKDF2-HMAC-SHA256 of the code, salted with the device id. Slow on
    /// purpose (~0.1-0.3 s on a phone); run it off the main thread.
    static func codeKey(code: String, deviceID: String) throws -> SymmetricKey {
        let password = Array(try normalize(code: code).utf8)
        let salt = Array(prefix + Data(deviceID.utf8))
        var derived = [UInt8](repeating: 0, count: 32)
        let status = password.withUnsafeBufferPointer { pw in
            pw.baseAddress!.withMemoryRebound(to: CChar.self, capacity: pw.count) { pwChars in
                CCKeyDerivationPBKDF(
                    CCPBKDFAlgorithm(kCCPBKDF2), pwChars, pw.count, salt, salt.count,
                    CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256), pbkdf2Iterations, &derived, derived.count)
            }
        }
        guard status == Int32(kCCSuccess) else { throw Failure.keyDerivationFailed }
        return SymmetricKey(data: derived)
    }

    static func transcript(deviceID: String, nonceApp: Data, nonceDevice: Data) -> Data {
        let ident = Data(deviceID.utf8)
        precondition(!ident.isEmpty && ident.count < 256, "device id must be 1-255 bytes")
        return prefix + Data([UInt8(ident.count)]) + ident + nonceApp + nonceDevice
    }

    struct SessionKeys {
        let mac: SymmetricKey
        let enc: SymmetricKey
        let transcript: Data
    }

    static func sessionKeys(codeKey: SymmetricKey, deviceID: String, nonceApp: Data, nonceDevice: Data) -> SessionKeys {
        let okm = HKDF<SHA256>.deriveKey(
            inputKeyMaterial: codeKey, salt: nonceApp + nonceDevice,
            info: prefix + Data("session|".utf8) + Data(deviceID.utf8), outputByteCount: 64)
        let bytes = okm.withUnsafeBytes { Data($0) }
        return SessionKeys(mac: SymmetricKey(data: bytes.prefix(32)), enc: SymmetricKey(data: bytes.suffix(32)),
                           transcript: transcript(deviceID: deviceID, nonceApp: nonceApp, nonceDevice: nonceDevice))
    }

    static func appProof(_ keys: SessionKeys) -> Data {
        Data(HMAC<SHA256>.authenticationCode(for: Data("app-proof".utf8) + keys.transcript, using: keys.mac))
    }

    static func deviceProof(_ keys: SessionKeys) -> Data {
        Data(HMAC<SHA256>.authenticationCode(for: Data("device-proof".utf8) + keys.transcript, using: keys.mac))
    }

    /// Constant-time check of the device's proof.
    static func isAuthentic(_ keys: SessionKeys, deviceTag: Data) -> Bool {
        HMAC<SHA256>.isValidAuthenticationCode(deviceTag, authenticating: Data("device-proof".utf8) + keys.transcript,
                                               using: keys.mac)
    }

    private struct Credentials: Encodable {
        let ssid: String
        let psk: String?
        enum CodingKeys: String, CodingKey { case ssid, psk }
        func encode(to encoder: Encoder) throws {
            var c = encoder.container(keyedBy: CodingKeys.self)
            try c.encode(ssid, forKey: .ssid)
            try c.encode(psk, forKey: .psk)  // null for an open network, as the device expects
        }
    }

    /// AES-256-GCM over {"ssid","psk"}, bound to this session's transcript. Returns ciphertext || tag.
    static func sealCredentials(_ keys: SessionKeys, ssid: String, psk: String?, iv: Data) throws -> Data {
        let encoder = JSONEncoder()
        // Canonical, as on the device: sorted keys (JSONEncoder's own order is not stable), no spaces.
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        let plaintext = try encoder.encode(Credentials(ssid: ssid, psk: psk))
        let box = try AES.GCM.seal(plaintext, using: keys.enc, nonce: AES.GCM.Nonce(data: iv),
                                   authenticating: Data("wifi-credentials".utf8) + keys.transcript)
        return box.ciphertext + box.tag
    }

    static func randomBytes(_ count: Int) -> Data {
        var bytes = [UInt8](repeating: 0, count: count)
        let status = SecRandomCopyBytes(kSecRandomDefault, count, &bytes)
        precondition(status == errSecSuccess, "the system random generator failed")
        return Data(bytes)
    }

    // ------------------------------------------------------------------ one session
    /// Drives one handshake: hello -> proof -> (check the device) -> credentials.
    final class Session {
        let label: SetupLabel
        let nonceApp: Data
        private let codeKey: SymmetricKey
        private(set) var keys: SessionKeys?
        private(set) var deviceVerified = false

        /// Derives the code key (slow): construct off the main thread.
        init(label: SetupLabel, nonceApp: Data = BLEProofOfPossession.randomBytes(16)) throws {
            self.label = label
            self.nonceApp = nonceApp
            self.codeKey = try BLEProofOfPossession.codeKey(code: label.code, deviceID: label.deviceID)
        }

        private init(label: SetupLabel, codeKey: SymmetricKey, nonceApp: Data) {
            self.label = label
            self.codeKey = codeKey
            self.nonceApp = nonceApp
        }

        /// A fresh handshake for the same label: new nonce, the code key reused.
        ///
        /// For a proof that lapsed (the device keeps one for 120 s) or a join
        /// that failed: the device then wants a new hello, and deriving the
        /// key again would cost the PBKDF2 time for nothing.
        func renewed(nonceApp: Data = BLEProofOfPossession.randomBytes(16)) -> Session {
            Session(label: label, codeKey: codeKey, nonceApp: nonceApp)
        }

        func helloMessage() -> Data {
            Session.json(["op": "hello", "v": BLEProofOfPossession.protocolVersion,
                          "na": nonceApp.base64EncodedString()])
        }

        func proofMessage(challenge reply: Data) throws -> Data {
            let msg = try Session.decode(reply)
            guard msg["op"] as? String == "challenge", msg["v"] as? Int == BLEProofOfPossession.protocolVersion,
                  let id = msg["id"] as? String, let nd = (msg["nd"] as? String).flatMap({ Data(base64Encoded: $0) }),
                  nd.count == 16 else { throw Failure.malformedReply("expected a v1 challenge") }
            guard id == label.deviceID else { throw Failure.wrongDevice(expected: label.deviceID, got: id) }
            let keys = BLEProofOfPossession.sessionKeys(codeKey: codeKey, deviceID: id, nonceApp: nonceApp,
                                                        nonceDevice: nd)
            self.keys = keys
            deviceVerified = false
            return Session.json(["op": "proof", "tag": BLEProofOfPossession.appProof(keys).base64EncodedString()])
        }

        /// Checks the device's answer to the proof. Credentials may be sent only after this returns.
        func verify(proven reply: Data) throws {
            let msg = try Session.decode(reply)
            guard let keys, msg["op"] as? String == "proven" else { throw Failure.notProven }
            guard let tag = (msg["tag"] as? String).flatMap({ Data(base64Encoded: $0) }),
                  BLEProofOfPossession.isAuthentic(keys, deviceTag: tag) else { throw Failure.deviceNotAuthentic }
            deviceVerified = true
        }

        func credentialsMessage(ssid: String, psk: String?, iv: Data = BLEProofOfPossession.randomBytes(12)) throws -> Data {
            guard let keys, deviceVerified else { throw Failure.notProven }
            let ct = try BLEProofOfPossession.sealCredentials(keys, ssid: ssid, psk: psk, iv: iv)
            return Session.json(["op": "creds", "iv": iv.base64EncodedString(), "ct": ct.base64EncodedString()])
        }

        private static func json(_ object: [String: Any]) -> Data {
            // Only strings and ints from this file: serialisation cannot fail.
            (try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])) ?? Data("{}".utf8)
        }

        /// The reply as an object; a device `error` reply becomes `Failure.rejected`.
        private static func decode(_ data: Data) throws -> [String: Any] {
            guard let msg = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else {
                throw Failure.malformedReply("not a JSON object")
            }
            if msg["op"] as? String == "error" {
                throw Failure.rejected(msg["error"] as? String ?? "unknown", retryAfterSeconds: msg["retry_after_s"] as? Int)
            }
            return msg
        }
    }
}
