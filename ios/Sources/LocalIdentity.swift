import CryptoKit
import Foundation
import Security

/// This iPhone's sd-local-v1 signing keys (docs/LOCAL_ACCESS.md §2, §5).
///
/// **One key per paired camera, not one per install.** Three reasons:
///  * Forgetting a camera destroys its key. That is a real revocation from the
///    phone's side even when the camera is unreachable (it can never be used
///    again), not just a deleted cache row.
///  * Two cameras, or one camera that changes owner, cannot tell that the same
///    phone is talking to both: the `client_id` (a hash of the key) differs.
///  * A key that leaks (it should not: see below) opens one camera, not all.
/// The cost is one small Keychain item per camera (at most a handful).
///
/// **Where the key lives.**
///  * A real iPhone: the Secure Enclave (`SecureEnclave.P256.Signing`). The
///    private key never exists outside the Enclave; the Keychain stores only
///    the Enclave's encrypted handle (`dataRepresentation`), which is useless
///    on any other phone. Usable after first unlock (the stream reconnects in
///    the background of the app's own lifetime), no Face ID prompt (automatic
///    reconnection must not ask for one).
///  * The Simulator (no Enclave) and any device where the Enclave refuses:
///    a software P-256 key whose raw bytes are a Keychain item.
///  Both are `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly` (Keychain.swift)
///  and never synchronizable: not in iCloud Keychain, not restored from a
///  backup onto another phone. No entitlement is needed for either (CryptoKit's
///  Enclave keys are not Keychain items, and the app's own generic-password
///  items need none), so free personal-team signing still works.
///
/// NOT validated on a physical iPhone's Secure Enclave (LOCAL_ACCESS.md §10 gate 6).
struct SecureEnclaveSigner: LocalSigner, @unchecked Sendable {
    let key: SecureEnclave.P256.Signing.PrivateKey

    var publicKey: Data { key.publicKey.x963Representation }

    func sign(_ message: Data) throws -> Data { try key.signature(for: message).rawRepresentation }
}

enum LocalIdentity {
    /// Where a camera's key is kept: "se:<handle>" or "sw:<raw scalar>", base64.
    static func itemName(_ hardwareID: String) -> String { "localKey.v1.\(hardwareID.uppercased())" }

    enum Storage: String { case secureEnclave = "Secure Enclave", software = "Keychain (software key)" }

    /// The key for one camera. Creates one when asked and none exists; returns
    /// nil when no key can be stored (a key that cannot be kept must not be
    /// enrolled: the camera would trust a phone that forgets it on relaunch).
    static func signer(for hardwareID: String, create: Bool = false) -> LocalSigner? {
        let name = itemName(hardwareID)
        if let stored = Keychain.get(name), let signer = decode(stored) { return signer }
        guard create else { return nil }
        if SecureEnclave.isAvailable, let made = try? makeEnclaveKey() {
            if Keychain.set(name, "se:" + made.dataRepresentation.base64EncodedString()) {
                return SecureEnclaveSigner(key: made)
            }
            return nil
        }
        let software = P256.Signing.PrivateKey()
        guard Keychain.set(name, "sw:" + software.rawRepresentation.base64EncodedString()) else { return nil }
        return SoftwareSigner(key: software)
    }

    static func storage(for hardwareID: String) -> Storage? {
        guard let stored = Keychain.get(itemName(hardwareID)) else { return nil }
        return stored.hasPrefix("se:") ? .secureEnclave : .software
    }

    /// Destroy the key: the camera's trust in this phone can never be used again.
    static func forget(_ hardwareID: String) {
        Keychain.delete(itemName(hardwareID))
    }

    private static func decode(_ stored: String) -> LocalSigner? {
        if stored.hasPrefix("se:"), let data = Data(base64Encoded: String(stored.dropFirst(3))),
           let key = try? SecureEnclave.P256.Signing.PrivateKey(dataRepresentation: data) {
            return SecureEnclaveSigner(key: key)
        }
        if stored.hasPrefix("sw:"), let data = Data(base64Encoded: String(stored.dropFirst(3))) {
            return SoftwareSigner(scalar: data)
        }
        return nil
    }

    private static func makeEnclaveKey() throws -> SecureEnclave.P256.Signing.PrivateKey {
        var error: Unmanaged<CFError>?
        guard let access = SecAccessControlCreateWithFlags(
            nil, kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly, .privateKeyUsage, &error)
        else { throw error?.takeRetainedValue() ?? CocoaError(.featureUnsupported) }
        return try SecureEnclave.P256.Signing.PrivateKey(accessControl: access)
    }
}
