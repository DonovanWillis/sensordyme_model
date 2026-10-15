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

print("\na device still online on its OLD network, credentials just sent:")
// The payload from the live read above, arriving while a write is outstanding
// (a notification already in flight, or a write the device ignored). The key is
// present and null, which means "no outcome yet" -- the old code fell back to
// `state: online` and declared the device joined, at its OLD address.
check(ProvisioningStatus.interpret(onlineOnHotspot, awaitingOutcome: true) == .none,
      "provision: null is NOT an outcome, however online the device is")

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

// ---------------------------------------------- which write a status answers
func status(_ state: String, attempt: Int?, ssid: String = "Home", error: String? = nil) -> [String: Any] {
    var provision: [String: Any] = ["state": state, "ssid": ssid]
    if let attempt { provision["attempt"] = attempt }
    if let error { provision["error"] = error }
    if state == "ok" { provision["ip"] = "192.168.1.37" }
    return ["state": "online", "scanning": false, "provision": provision]
}

print("\nwith attempt numbers, only this write's outcome is accepted:")
do {
    var tracker = ProvisioningStatus.AttemptTracker()
    // Before writing: the device's last attempt (#4) failed.
    _ = tracker.outcome(for: status("error", attempt: 4, error: "That password was not accepted"))
    check(tracker.baseline == 4, "a status read before the write sets the baseline")
    tracker.beginWrite(ssid: "Home")
    check(tracker.outcome(for: status("error", attempt: 4, error: "That password was not accepted")) == .none,
          "the PREVIOUS attempt's refusal, arriving late, is not this write's answer")
    check(tracker.outcome(for: status("connecting", attempt: 5)) == .joining,
          "connecting for attempt 5 (newer than 4) is this write")
    check(tracker.outcome(for: status("ok", attempt: 5)) == .joined(ip: "192.168.1.37"),
          "and its outcome is now accepted")
}
do {
    // A fast join: the device went connecting -> ok between two polls and the
    // connecting notification was missed. Attempt 5 > baseline 4 proves the
    // outcome answers this write; it must not wait out the 90 s timeout.
    var tracker = ProvisioningStatus.AttemptTracker()
    _ = tracker.outcome(for: status("ok", attempt: 4))
    tracker.beginWrite(ssid: "Home")
    check(tracker.outcome(for: status("ok", attempt: 4)) == .none,
          "fast join: the pre-write attempt's success is still not ours")
    check(tracker.outcome(for: status("ok", attempt: 5)) == .joined(ip: "192.168.1.37"),
          "fast join, connecting never seen: a newer attempt's outcome is accepted at once")
    var refused = ProvisioningStatus.AttemptTracker()
    _ = refused.outcome(for: status("error", attempt: 2, error: "old"))
    refused.beginWrite(ssid: "Home")
    check(refused.outcome(for: status("error", attempt: 3, error: "wrong password")) == .failed("wrong password"),
          "a fast refusal is accepted the same way")
    var superseded = ProvisioningStatus.AttemptTracker()
    superseded.beginWrite(ssid: "Home")
    _ = superseded.outcome(for: status("connecting", attempt: 7))
    check(superseded.outcome(for: status("error", attempt: 6, error: "x")) == .none,
          "an outcome older than an attempt already seen connecting is ignored")
}
do {
    var tracker = ProvisioningStatus.AttemptTracker()
    tracker.beginWrite(ssid: "Home")
    check(tracker.outcome(for: onlineOnHotspot) == .none,
          "an online device with no outcome yet is not 'joined', even with nothing seen before")
    check(tracker.outcome(for: status("connecting", attempt: 1)) == .joining, "attempt 1 connecting")
    check(tracker.outcome(for: status("error", attempt: 1, error: "nope")) == .failed("nope"),
          "a refusal for the attempt seen connecting is reported")
    tracker.finish()
    tracker.beginWrite(ssid: "Home")
    check(tracker.outcome(for: status("error", attempt: 1, error: "nope")) == .none,
          "after a retry, the first attempt's refusal is history")
}

print("\nwithout attempt numbers: only after the write is acknowledged, and only for the network sent:")
do {
    let oldJoin: [String: Any] = ["state": "online",
                                 "provision": ["state": "ok", "ssid": "Home", "ip": "10.0.0.2"]]
    var tracker = ProvisioningStatus.AttemptTracker()
    tracker.beginWrite(ssid: "Home")
    check(tracker.outcome(for: oldJoin) == .none,
          "before the write is acknowledged, a status may predate it")
    tracker.acknowledgeWrite()
    check(tracker.outcome(for: oldJoin) == .joined(ip: "10.0.0.2"),
          "after it, the device has recorded this write, so the status answers it")
    var other = ProvisioningStatus.AttemptTracker()
    other.beginWrite(ssid: "Office")
    other.acknowledgeWrite()
    check(other.outcome(for: oldJoin) == .none, "an outcome for a different network is not ours")
    var legacyTracker = ProvisioningStatus.AttemptTracker()
    legacyTracker.beginWrite(ssid: "Home")
    legacyTracker.acknowledgeWrite()
    check(legacyTracker.outcome(for: legacy) == .joined(ip: "192.168.1.37"),
          "firmware with no provision field at all still works, after the acknowledgement")
}
do {
    var tracker = ProvisioningStatus.AttemptTracker()
    check(tracker.outcome(for: joined) == .none, "nothing outstanding: no outcome")
    check(!tracker.awaitingOutcome, "and nothing is awaited")
}

print("\nthe device's identity rides along:")
check(ProvisioningStatus.deviceID(in: onlineOnHotspot) == "SD-3D248632", "status device_id is read")
check(ProvisioningStatus.deviceID(in: ["device_id": "garbage"]) == nil, "only an SD- id counts")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
