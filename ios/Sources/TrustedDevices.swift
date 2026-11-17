import Foundation

/// The cameras this iPhone is paired with locally (sd-local-v1), as it
/// remembers them across launches and without the account server.
///
/// Kept in the Keychain (LocalAccessCenter) as JSON of `[TrustedDevice]`. It
/// holds no secret: the phone's private key is a separate Keychain item (or a
/// Secure Enclave key), and everything here is either public (the device's
/// public key) or a cache (the last address). It is still in the Keychain,
/// this-device-only, because which cameras a phone can open is nobody else's
/// business and must not ride along in a backup.
///
/// Foundation-only, so ios/Tests/TrustedDevicesTests.swift runs it.
struct TrustedDevice: Codable, Equatable, Identifiable {
    enum EnrolledVia: String, Codable { case bluetooth, account, token }

    enum Enrolment: String, Codable {
        /// The account relayed this phone's key; the device has not confirmed it
        /// yet (no local session has succeeded).
        case awaitingDevice
        /// A local session succeeded, or Bluetooth enrolment returned the
        /// device's key: usable.
        case active
        /// The device said `unknown_client` (revoked, idle-expired): the local
        /// credential is forgotten and this iPhone must be paired again.
        case needsPairing
    }

    /// The camera's own id, `SD-XXXXXXXX` (its Bonjour TXT `id`).
    var hardwareID: String
    var name: String
    /// The account's id for it, when the account knows it.
    var backendDeviceID: Int?
    /// The last address it answered at, e.g. `http://192.168.1.20:8899`. A cache.
    var lastKnownURL: String?
    /// When a fresh local read (or the account's fresh status) last saw it.
    var lastSeen: Date?
    /// What it was last doing, by name only ("Driver Safety"). Never shown as
    /// the current state: only a fresh read says what it is doing now.
    var lastModeTitle: String?
    /// The device's P-256 public key (base64url) and its fingerprint.
    var devicePublicKey: String
    var deviceKeyFingerprint: String
    /// This phone's id on that device (sha256 of the phone key, 16 hex).
    var clientID: String
    var enrolledVia: EnrolledVia
    var enrolment: Enrolment
    var enrolledAt: Date
    /// The account (lowercased email) that paired it, or "" for a pairing made
    /// with no account (Bluetooth setup before sign-in).
    var owner: String

    var id: String { hardwareID }

    var devicePublicKeyBytes: Data? { LocalAccess.b64uDecode(devicePublicKey, length: 65) }

    var isUsable: Bool { enrolment == .active || enrolment == .awaitingDevice }
}

enum TrustedDevices {
    static let storageKey = "trustedDevices.v1"

    static func encode(_ devices: [TrustedDevice]) -> String {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .secondsSince1970
        encoder.outputFormatting = [.sortedKeys]
        return (try? encoder.encode(devices)).map { String(decoding: $0, as: UTF8.self) } ?? "[]"
    }

    /// A damaged or foreign value reads as no devices, never a crash. Rows whose
    /// device key is not a real point, or whose fingerprint does not match the
    /// key, are dropped: they could not verify a session anyway.
    static func decode(_ text: String?) -> [TrustedDevice] {
        guard let data = text?.data(using: .utf8) else { return [] }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .secondsSince1970
        let rows = (try? decoder.decode([TrustedDevice].self, from: data)) ?? []
        return rows.filter { row in
            guard let key = row.devicePublicKeyBytes, LocalAccess.publicKey(x963: key) != nil else { return false }
            return LocalAccess.fingerprint(publicKey: key) == row.deviceKeyFingerprint
                && row.hardwareID.uppercased().hasPrefix("SD-")
        }
    }

    /// Insert or replace by hardware id (case-insensitive), keeping the order.
    static func upsert(_ device: TrustedDevice, into devices: [TrustedDevice]) -> [TrustedDevice] {
        var out = devices
        if let index = out.firstIndex(where: { $0.hardwareID.uppercased() == device.hardwareID.uppercased() }) {
            out[index] = device
        } else {
            out.append(device)
        }
        return out
    }

    // MARK: whose pairings stay

    /// An explicit sign-out keeps the pairings made with no account (they never
    /// belonged to one) and forgets the account's: the next person to sign in
    /// on this phone must not inherit the previous account's cameras.
    static func afterExplicitSignOut(_ devices: [TrustedDevice]) -> [TrustedDevice] {
        devices.filter { $0.owner.isEmpty }
    }

    /// A different account signed in: its own pairings and the account-less
    /// ones stay (an account-less pairing is adopted by whoever signs in on
    /// this phone first, which is the person who set it up), another
    /// account's go.
    static func afterSignIn(_ devices: [TrustedDevice], owner: String) -> [TrustedDevice] {
        let owner = owner.lowercased()
        return devices.filter { $0.owner.isEmpty || $0.owner == owner }
    }

    // MARK: the account path

    enum AccountIdentityDecision: Equatable {
        /// Register this phone with the account and remember the device key.
        case register
        /// Already paired with this key: nothing to do.
        case unchanged
        /// The account reports a different key than the one Bluetooth enrolment
        /// gave. Bluetooth is the stronger proof (it needed the label), so the
        /// cached key is kept and the person is asked to pair again over Bluetooth.
        case conflictKeepBluetooth
        /// An account pairing whose device key changed: re-register with the new key.
        case reregister
    }

    static func accountIdentity(existing: TrustedDevice?, reportedKey: String, reportedFingerprint: String?) -> AccountIdentityDecision {
        guard let existing else { return .register }
        if existing.devicePublicKey == reportedKey {
            return existing.enrolment == .needsPairing ? .reregister : .unchanged
        }
        if existing.enrolledVia == .bluetooth { return .conflictKeepBluetooth }
        return .reregister
    }

    // MARK: what the card may say

    /// What the app may say about a paired camera right now. Live state comes
    /// ONLY from a fresh local `/api/state` read (`liveAt`, in this launch) or
    /// the account's fresh status; the cache never says "protecting".
    enum Presence: Equatable {
        /// A local read this recent: the device's own word. `protection` is its
        /// `policy.protection.state`.
        case live(protection: String?)
        /// Looking for it (Bonjour, the last address).
        case searching
        /// Not answering here. `lastSeen` is from the cache, when known.
        case notReachable(lastSeen: Date?)
        /// The device no longer trusts this iPhone.
        case needsPairing
        /// The account relayed this iPhone's key; the device hasn't confirmed it.
        case awaitingDevice
    }

    /// A local reading older than this is no longer "now".
    static let liveFor: TimeInterval = 30

    static func presence(_ device: TrustedDevice, liveAt: Date?, liveProtection: String?, searching: Bool,
                         now: Date) -> Presence {
        if device.enrolment == .needsPairing { return .needsPairing }
        if let liveAt, now.timeIntervalSince(liveAt) <= liveFor, now >= liveAt.addingTimeInterval(-5) {
            return .live(protection: liveProtection)
        }
        if searching { return .searching }
        if device.enrolment == .awaitingDevice, device.lastSeen == nil { return .awaitingDevice }
        return .notReachable(lastSeen: device.lastSeen)
    }

    /// "just now", "4 min ago", "3 h ago", "2 days ago".
    static func age(_ seconds: TimeInterval) -> String {
        let s = max(0, seconds)
        if s < 60 { return "just now" }
        if s < 3600 { return "\(Int(s / 60)) min ago" }
        if s < 86400 { return "\(Int(s / 3600)) h ago" }
        let days = Int(s / 86400)
        return days == 1 ? "1 day ago" : "\(days) days ago"
    }

    /// The pill: short, and never "Online" from a cache.
    static func label(_ presence: Presence) -> String {
        switch presence {
        case .live(let protection):
            switch protection {
            case "stopped": return "Stopped"
            case "not_protecting": return "Not protecting"
            case "protecting": return "Protecting"
            default: return "Connected"
            }
        case .searching: return "Looking…"
        case .notReachable: return "Offline"
        case .needsPairing: return "Pair again"
        case .awaitingDevice: return "Waiting for camera"
        }
    }

    /// The line under the name.
    static func detail(_ presence: Presence, now: Date) -> String {
        switch presence {
        case .live: return "Connected directly on this network"
        case .searching: return "Looking for it on this network…"
        case .notReachable(let lastSeen?):
            return "Offline · last seen \(age(now.timeIntervalSince(lastSeen)))"
        case .notReachable(nil): return "Not reachable on this network"
        case .needsPairing: return LocalAccess.needsPairingAgain
        case .awaitingDevice:
            return "Your account asked the camera to trust this iPhone. It will connect once the camera confirms."
        }
    }

    /// Whether the presence is the device's own fresh word (the only kind that
    /// may colour a pill green or offer live controls).
    static func isLive(_ presence: Presence) -> Bool {
        if case .live = presence { return true }
        return false
    }

    /// Cameras paired on this iPhone that the account list does not already
    /// show (matched by hardware id). With no account list (a cold launch with
    /// the server down, or not signed in) that is every paired camera.
    static func unlisted(_ devices: [TrustedDevice], accountHardwareIDs: [String?]) -> [TrustedDevice] {
        let listed = Set(accountHardwareIDs.compactMap { $0?.uppercased() }.filter { !$0.isEmpty })
        return devices.filter { !listed.contains($0.hardwareID.uppercased()) }
    }
}
