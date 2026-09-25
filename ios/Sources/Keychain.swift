import Foundation
import Security

/// Minimal Keychain wrapper for the auth token. Tokens don't belong in
/// UserDefaults (unencrypted, included in some backups); the Keychain is
/// hardware-encrypted and survives reinstalls sensibly.
enum Keychain {
    private static let service = "com.sensordyme.app"

    static func get(_ key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let data = result as? Data
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    /// Store a value, reporting whether it was actually stored.
    ///
    /// The result used to be discarded entirely, which is fine right up until a
    /// caller *deletes its only other copy* on the strength of this having
    /// worked — which is what the UserDefaults→Keychain migrations do. A silent
    /// failure there does not degrade anything, it destroys the secret, and for
    /// the device token that surfaces much later as the device rejecting a token
    /// the user is certain they entered.
    @discardableResult
    static func set(_ key: String, _ value: String) -> Bool {
        if value.isEmpty { delete(key); return true }
        let base: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        let data = Data(value.utf8)
        let status = SecItemUpdate(
            base as CFDictionary, [kSecValueData as String: data] as CFDictionary
        )
        if status == errSecItemNotFound {
            var add = base
            add[kSecValueData as String] = data
            add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
            return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
        }
        return status == errSecSuccess
    }

    static func delete(_ key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
