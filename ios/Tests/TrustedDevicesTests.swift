import CryptoKit
import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: LocalAccess BLEProofOfPossession
//
// The trusted-camera cache: it survives a cold launch (encode/decode), it
// never says "protecting" on its own, and whose pairings survive a sign-out or
// a change of account.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let key = P256.Signing.PrivateKey().publicKey.x963Representation
func device(_ id: String = "SD-1A2B3C4D", owner: String = "", via: TrustedDevice.EnrolledVia = .bluetooth,
            enrolment: TrustedDevice.Enrolment = .active, lastSeen: Date? = nil, key: Data = key) -> TrustedDevice {
    TrustedDevice(hardwareID: id, name: "Car camera", backendDeviceID: 7, lastKnownURL: "http://192.168.1.20:8899",
                  lastSeen: lastSeen, lastModeTitle: "Driver Safety", devicePublicKey: LocalAccess.b64u(key),
                  deviceKeyFingerprint: LocalAccess.fingerprint(publicKey: key), clientID: "2bad0fd610d99eae",
                  enrolledVia: via, enrolment: enrolment, enrolledAt: Date(timeIntervalSince1970: 1_800_000_000),
                  owner: owner)
}
let now = Date(timeIntervalSince1970: 1_800_001_000)

print("the cache survives a cold launch:")
let saved = [device(lastSeen: now.addingTimeInterval(-300)), device("SD-00000002", owner: "a@example.com", via: .account)]
let text = TrustedDevices.encode(saved)
check(TrustedDevices.decode(text) == saved, "every field round-trips (dates, enrolment, owner, address, key)")
check(TrustedDevices.decode(nil).isEmpty && TrustedDevices.decode("{not json").isEmpty, "a damaged value reads as no cameras")
var tampered = saved[0]
tampered.deviceKeyFingerprint = "0000000000000000"
check(TrustedDevices.decode(TrustedDevices.encode([tampered, saved[1]])) == [saved[1]],
      "a row whose fingerprint does not match its key is dropped")
var notAPoint = saved[0]
notAPoint.devicePublicKey = LocalAccess.b64u(Data(repeating: 4, count: 65))
check(TrustedDevices.decode(TrustedDevices.encode([notAPoint])).isEmpty, "so is one whose key is not a P-256 point")
check(!text.contains("scalar") && !text.lowercased().contains("private"), "nothing secret is in it (the phone key lives elsewhere)")
var renamed = saved[0]
renamed.name = "Van"
let upserted = TrustedDevices.upsert(renamed, into: saved)
check(upserted.count == 2 && upserted[0].name == "Van", "upsert replaces by hardware id, keeping the order")

print("\nhonest status: the cache never says protecting:")
let p = TrustedDevices.presence
check(p(saved[0], nil, nil, false, now) == .notReachable(lastSeen: saved[0].lastSeen),
      "no fresh read this launch: offline with the cached last-seen")
check(TrustedDevices.label(p(saved[0], nil, "protecting", false, now)) == "Offline",
      "a remembered 'protecting' with no fresh read is still Offline")
check(TrustedDevices.detail(p(saved[0], nil, nil, false, now), now: now) == "Offline · last seen 5 min ago",
      "\"Offline · last seen 5 min ago\"")
check(p(saved[0], now.addingTimeInterval(-10), "protecting", false, now) == .live(protection: "protecting"),
      "a local /api/state read 10 s ago is live")
check(TrustedDevices.label(.live(protection: "stopped")) == "Stopped", "the device's own word: Stopped")
check(p(saved[0], now.addingTimeInterval(-31), "protecting", false, now) == .notReachable(lastSeen: saved[0].lastSeen),
      "31 s later it is no longer live")
check(p(saved[0], now.addingTimeInterval(-31), nil, true, now) == .searching, "while looking: Looking…")
check(TrustedDevices.detail(p(device(lastSeen: nil), nil, nil, false, now), now: now) == "Not reachable on this network",
      "never seen this launch or before: not reachable on this network")
check(p(device(enrolment: .needsPairing), now, "protecting", false, now) == .needsPairing,
      "a revoked pairing says pair again, even if something answered")
check(TrustedDevices.detail(.needsPairing, now: now) == "This iPhone needs to be paired with the camera again.",
      "with the re-pair copy")
check(p(device(via: .account, enrolment: .awaitingDevice), nil, nil, false, now) == .awaitingDevice,
      "an account pairing the camera has not confirmed yet says so")
check(TrustedDevices.age(59) == "just now" && TrustedDevices.age(7200) == "2 h ago" && TrustedDevices.age(86400 * 3) == "3 days ago",
      "ages in words")
check(!TrustedDevices.isLive(.notReachable(lastSeen: now)) && TrustedDevices.isLive(.live(protection: nil)),
      "only a fresh read is live")

print("\nwho keeps which pairing:")
let mixed = [device("SD-A0000001"), device("SD-A0000002", owner: "a@example.com", via: .account),
             device("SD-A0000003", owner: "b@example.com", via: .bluetooth)]
check(TrustedDevices.afterExplicitSignOut(mixed).map(\.hardwareID) == ["SD-A0000001"],
      "an explicit sign-out forgets the account's pairings, keeps the account-less one")
check(TrustedDevices.afterSignIn(mixed, owner: "A@example.com").map(\.hardwareID) == ["SD-A0000001", "SD-A0000002"],
      "another account signing in never sees a different account's cameras")

print("\nthe account path's device key:")
check(TrustedDevices.accountIdentity(existing: nil, reportedKey: "k", reportedFingerprint: nil) == .register, "new: register")
check(TrustedDevices.accountIdentity(existing: saved[0], reportedKey: saved[0].devicePublicKey, reportedFingerprint: nil)
      == .unchanged, "same key: nothing to do")
check(TrustedDevices.accountIdentity(existing: saved[0], reportedKey: "other", reportedFingerprint: nil)
      == .conflictKeepBluetooth, "a Bluetooth-proven key is never replaced by the account's word")
check(TrustedDevices.accountIdentity(existing: saved[1], reportedKey: "other", reportedFingerprint: nil) == .reregister,
      "an account pairing follows the account")
check(TrustedDevices.accountIdentity(existing: device(enrolment: .needsPairing), reportedKey: LocalAccess.b64u(key),
                                     reportedFingerprint: nil) == .reregister, "a revoked pairing is registered again")

print("\nthe device list with the server down:")
check(TrustedDevices.unlisted(saved, accountHardwareIDs: []).count == 2, "cold launch, no account list: every paired camera")
check(TrustedDevices.unlisted(saved, accountHardwareIDs: ["sd-1a2b3c4d", nil]).map(\.hardwareID) == ["SD-00000002"],
      "a camera the account already lists is not shown twice")

print("\n\(failures == 0 ? "all passed" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
