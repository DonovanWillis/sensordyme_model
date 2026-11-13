import Foundation
import Security

/// Minimal Keychain wrapper for the auth token. Tokens don't belong in
/// UserDefaults (unencrypted, included in some backups); the Keychain is
/// hardware-encrypted and survives reinstalls sensibly.
enum Keychain {
    private static let service = "com.sensordyme.app"

    /// The protection class of every item this app stores (M8.3).
    ///
    /// The items are the account session (a 30-day JWT), the device's dashboard
    /// token -- the credential that opens the live camera feed -- and the ntfy
    /// topic, which is the whole secret of the alert channel. They used to be
    /// `AfterFirstUnlock`, which an encrypted iCloud or Finder backup carries
    /// to any other phone it is restored on: whoever holds the backup (and its
    /// password, or the iCloud account) then holds a live session and the
    /// camera. `ThisDeviceOnly` keeps them on this phone; a restore to a new
    /// phone signs in once and re-enters the device token.
    ///
    /// Still `AfterFirstUnlock`, not `WhenUnlocked`: alert polling runs in the
    /// background with the screen locked and must be able to read the session.
    /// Never synchronizable: iCloud Keychain would copy them the same way.
    static let accessibility = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly

    /// Identifies one item: class, service and account. Nothing else.
    static func itemQuery(_ key: String) -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
    }

    /// What a new item is written with.
    static func addAttributes(_ key: String, _ data: Data) -> [String: Any] {
        var add = itemQuery(key)
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = accessibility
        return add
    }

    /// What an existing item is updated with. Carries the protection class as
    /// well as the value, so every write re-classes an item an older build
    /// stored as `AfterFirstUnlock`.
    static func updateAttributes(_ data: Data) -> [String: Any] {
        [kSecValueData as String: data, kSecAttrAccessible as String: accessibility]
    }

    /// Whether an item read back with these attributes is in an older class.
    static func needsReclassing(_ attributes: [String: Any]) -> Bool {
        (attributes[kSecAttrAccessible as String] as? String) != (accessibility as String)
    }

    static func get(_ key: String) -> String? {
        var query = itemQuery(key)
        query[kSecReturnData as String] = true
        query[kSecReturnAttributes as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: AnyObject?
        guard SecItemCopyMatching(query as CFDictionary, &result) == errSecSuccess,
              let item = result as? [String: Any],
              let data = item[kSecValueData as String] as? Data
        else { return nil }
        // An item written by an earlier build, which is not otherwise rewritten
        // until the next sign-in: moved to this device only on first read.
        // Best effort -- the value is returned either way.
        if needsReclassing(item) {
            SecItemUpdate(itemQuery(key) as CFDictionary,
                          [kSecAttrAccessible as String: accessibility] as CFDictionary)
        }
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
        let data = Data(value.utf8)
        let status = SecItemUpdate(itemQuery(key) as CFDictionary, updateAttributes(data) as CFDictionary)
        if status == errSecItemNotFound {
            return SecItemAdd(addAttributes(key, data) as CFDictionary, nil) == errSecSuccess
        }
        return status == errSecSuccess
    }

    static func delete(_ key: String) {
        SecItemDelete(itemQuery(key) as CFDictionary)
    }
}
