import Foundation
import Security

// Run with ios/Tests/run.sh
//
// M8.3: the protection class of the app's secrets. Pure attribute checks only:
// nothing here touches a real keychain (on a Mac that would be the login
// keychain of whoever runs the suite).

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let thisDeviceOnly = kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly as String
let data = Data("secret".utf8)

print("secrets stay on this phone and stay readable in the background:")
check((Keychain.accessibility as String) == thisDeviceOnly,
      "the class is AfterFirstUnlockThisDeviceOnly (not restored from a backup to another phone)")

print("\na new item is written in that class, and is never synchronizable:")
do {
    let add = Keychain.addAttributes("authToken", data)
    check((add[kSecAttrAccessible as String] as? String) == thisDeviceOnly, "add carries the class")
    check(add[kSecAttrSynchronizable as String] == nil, "add never asks for iCloud Keychain sync")
    check((add[kSecAttrAccount as String] as? String) == "authToken", "add names the item")
    check((add[kSecValueData as String] as? Data) == data, "add carries the value")
}

print("\nan update re-classes an item an older build stored:")
do {
    let update = Keychain.updateAttributes(data)
    check((update[kSecAttrAccessible as String] as? String) == thisDeviceOnly, "update carries the class")
    check(Keychain.itemQuery("authToken")[kSecAttrAccessible as String] == nil,
          "the item query does not filter by class (it must find the old item to update it)")
    let old: [String: Any] = [kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock as String]
    let current: [String: Any] = [kSecAttrAccessible as String: thisDeviceOnly]
    check(Keychain.needsReclassing(old), "an AfterFirstUnlock item read back is re-classed")
    check(!Keychain.needsReclassing(current), "a current item is left alone")
    check(Keychain.needsReclassing([:]), "an item with no class reported is re-classed")
}

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
