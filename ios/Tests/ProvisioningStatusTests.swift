import Foundation

// Standalone, runnable with `swift ios/Tests/ProvisioningStatusTests.swift`.
// The app has no XCTest target, and the bugs this covers are exactly the ones a
// clean build and a green Python suite both miss.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

// ---------------------------------------------------------------- the payload
// What the Pi actually sends, copied from a live read:
let onlineOnHotspot: [String: Any] = [
    "state": "online", "ssid": "ATTBcBKJJA", "ip": "172.20.10.6",
    "device_id": "SD-3D248632", "reconfigurable": true, "scanning": false,
    "provision": NSNull(),
]
let joining: [String: Any] = [
    "state": "online", "ssid": "ATTBcBKJJA", "scanning": false,
    "provision": ["state": "connecting", "ssid": "MyAltice 54be1b"],
]
let joined: [String: Any] = [
    "state": "online", "scanning": false,
    "provision": ["state": "ok", "ssid": "MyAltice 54be1b", "ip": "192.168.1.37"],
]
let refused: [String: Any] = [
    "state": "online", "ssid": "ATTBcBKJJA", "scanning": false,
    "provision": ["state": "error", "ssid": "MyAltice 54be1b",
                  "error": "That password was not accepted by 'MyAltice 54be1b'."],
]

print("a device already on Wi-Fi, before any credentials are sent:")
check(ProvisioningStatus.interpret(onlineOnHotspot, awaitingOutcome: false) == .none,
      "is NOT treated as a finished setup")
// The regression: reading `state: online` as an outcome sent the app straight
// from "Connect SensorDyme" to "SensorDyme is ready", skipping the network
// picker and the password entirely.

print("\nafter credentials are sent:")
check(ProvisioningStatus.interpret(joining, awaitingOutcome: true) == .joining,
      "connecting is reported as joining")
check(ProvisioningStatus.interpret(joined, awaitingOutcome: true) == .joined(ip: "192.168.1.37"),
      "success carries the NEW address, not the old one")
if case let .failed(reason) = ProvisioningStatus.interpret(refused, awaitingOutcome: true) {
    check(reason.contains("password"), "a refusal reports the device's own reason")
} else {
    check(false, "a refusal reports the device's own reason")
}

print("\nthe old connectivity field never outranks the outcome:")
check(ProvisioningStatus.interpret(refused, awaitingOutcome: true) != .joined(ip: nil),
      "a failure is not overwritten by the device still being online elsewhere")

print("\nfirmware without the provision field:")
let legacy: [String: Any] = ["state": "online", "ip": "192.168.1.37"]
check(ProvisioningStatus.interpret(legacy, awaitingOutcome: true) == .joined(ip: "192.168.1.37"),
      "falls back to state only while an outcome is outstanding")
check(ProvisioningStatus.interpret(legacy, awaitingOutcome: false) == .none,
      "and never otherwise")

print("\npolling:")
check(ProvisioningStatus.isTerminal(.joining) == false, "keeps polling while joining")
check(ProvisioningStatus.isTerminal(.joined(ip: nil)), "stops on success")
check(ProvisioningStatus.isTerminal(.failed("x")), "stops on failure")
check(ProvisioningStatus.isTerminal(.none) == false, "keeps polling when nothing was said")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
