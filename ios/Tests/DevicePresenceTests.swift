import Foundation

// Run via ios/Tests/run.sh, which concatenates DevicePresence.swift with this.
//
// The failure this pins down is the one that happened: the Pi had not reported
// to the backend in nine days, and the app said "Can't find your device on this
// network" — a sentence about Wi-Fi, for a camera that was switched off. The
// backend knew the real answer the whole time and nothing asked it.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let now = Date(timeIntervalSince1970: 1_000_000)
func ago(_ seconds: TimeInterval) -> DevicePresence.BackendView {
    DevicePresence.BackendView(lastSeen: now.addingTimeInterval(-seconds))
}
func verdict(found: Bool, _ backend: DevicePresence.BackendView?, reachable: Bool = true)
    -> DevicePresence.Verdict {
    DevicePresence.verdict(foundOnNetwork: found, backend: backend, backendReachable: reachable, now: now)
}

// --------------------------------------------------------------- the happy path
print("a device Bonjour can see needs no explanation:")
check(verdict(found: true, ago(5)) == .reachable, "found on the network is reachable")
check(verdict(found: true, nil, reachable: false) == .reachable,
      "and stays reachable even if the backend cannot be asked — the feed works regardless")
check(DevicePresence.title(.reachable).isEmpty, "nothing is said about a working device")

// ------------------------------------------- the distinction that was missing
print("\nBonjour silence is not one situation, it is five:")
check(verdict(found: false, ago(3)) == .aliveButElsewhere(since: 3),
      "reporting seconds ago but invisible here = a network split, not a dead camera")
check(verdict(found: false, ago(776_000)) == .quiet(since: 776_000),
      "nine days quiet is a device that is off — the actual case on the dev Mac")
check(verdict(found: false, DevicePresence.BackendView(lastSeen: nil)) == .neverReported,
      "registered but never heard from = not set up")
check(verdict(found: false, nil) == .notLinked,
      "the backend knows of no such device")
check(verdict(found: false, ago(5), reachable: false) == .undetermined,
      "with no backend to ask, the honest answer is that we do not know")

print("\nthe live window matches the backend's own `online` flag:")
check(DevicePresence.liveWindow == 30, "30s, same as _HEALTH_STALE_S in cloud/app.py")
check(verdict(found: false, ago(29)) == .aliveButElsewhere(since: 29), "just inside is alive")
if case .quiet = verdict(found: false, ago(31)) {
    check(true, "just outside is quiet")
} else {
    check(false, "just outside is quiet")
}

print("\na clock that runs backwards does not produce a negative duration:")
// The Pi has no RTC and really does report timestamps from the future.
check(verdict(found: false, ago(-500)) == .aliveButElsewhere(since: 0),
      "a future last_seen is treated as 'just now', not as -500 seconds")

// ------------------------------------------------------------- what it says
print("\nthe wording names the right cause:")
let split = DevicePresence.title(.aliveButElsewhere(since: 3))
check(!split.lowercased().contains("can't find"),
      "a running device is never described as missing")
check(split.contains("running"), "it says the device is running")
check(DevicePresence.detail(.aliveButElsewhere(since: 3)).contains("alerts still work"),
      "and that alerts are unaffected — only the live picture is lost")

// 776_000s is 8.98 days and `duration` floors, so this is "8 days" — the point
// is that a span is named at all, not the exact rounding.
check(DevicePresence.title(.quiet(since: 776_000)).contains("8 days"),
      "a quiet device is described by HOW LONG, which points at the device")
check(DevicePresence.detail(.quiet(since: 776_000)).contains("switched off"),
      "and names the likely cause")
check(DevicePresence.detail(.neverReported).contains("Bluetooth"),
      "a device that never reported is the one case where Bluetooth setup is the answer")
// The earlier code offered Bluetooth provisioning for ANY silence, which sent
// people hunting a signal that only exists within a few metres of the device.
check(!DevicePresence.detail(.quiet(since: 776_000)).contains("Bluetooth"),
      "but Bluetooth is NOT suggested for a device that is merely off")

print("\n'Search again' is offered only when another local scan could succeed:")
check(DevicePresence.offersLocalRetry(.aliveButElsewhere(since: 3)),
      "a live device might appear if the phone rejoins the right network")
check(DevicePresence.offersLocalRetry(.undetermined),
      "and when we genuinely do not know")
check(!DevicePresence.offersLocalRetry(.quiet(since: 776_000)),
      "but not for a device that has been off for days — the scan cannot work")
check(!DevicePresence.offersLocalRetry(.neverReported), "nor for one never set up")
check(!DevicePresence.offersLocalRetry(.notLinked), "nor for one that is not linked")

print("\ndurations read like something a person would say:")
check(DevicePresence.duration(0) == "0 seconds", "zero")
check(DevicePresence.duration(1) == "1 second", "singular second")
check(DevicePresence.duration(45) == "45 seconds", "seconds")
check(DevicePresence.duration(60) == "1 minute", "singular minute")
check(DevicePresence.duration(3599) == "59 minutes", "minutes")
check(DevicePresence.duration(3600) == "1 hour", "singular hour")
check(DevicePresence.duration(86_399) == "23 hours", "hours")
check(DevicePresence.duration(86_400) == "1 day", "singular day")
check(DevicePresence.duration(776_000) == "8 days", "the real value, rounded down to days")
check(DevicePresence.duration(-10) == "0 seconds", "a negative duration is clamped, not printed")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
