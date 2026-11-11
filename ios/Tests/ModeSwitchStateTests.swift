import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let t0 = Date(timeIntervalSince1970: 1_000_000)

func state(
    for name: String, active: String, custom: Bool = false,
    pending: ModeSwitchState.PendingSwitch? = nil, online: Bool = true,
    failure: ModeSwitchState.PendingSwitch? = nil, at offset: TimeInterval = 0
) -> ModeRowState {
    ModeSwitchState.state(
        for: name, activeBase: active, customMonitoringActive: custom,
        pending: pending, deviceOnline: online, failure: failure,
        now: t0.addingTimeInterval(offset)
    )
}

print("the plain cases:")
check(state(for: "driver_safety", active: "driver_safety") == .active,
      "the mode the device reports running shows active")
check(state(for: "baby_monitoring", active: "driver_safety") == .inactive,
      "every other mode shows inactive")

print("\ncustom monitoring suppresses every preset's active state:")
check(state(for: "driver_safety", active: "driver_safety", custom: true) == .inactive,
      "not even the preset that precedes custom monitoring reads as active")

print("\na switch in progress:")
let switchingToBaby = ModeSwitchState.PendingSwitch(targetBase: "baby_monitoring", requestedAt: t0)
check(state(for: "baby_monitoring", active: "driver_safety", pending: switchingToBaby) == .switching,
      "the targeted row shows switching")
check(state(for: "driver_safety", active: "driver_safety", pending: switchingToBaby) == .active,
      "the mode being switched AWAY from stays showing active until the new one " +
      "actually confirms — the device is still genuinely running it, and blanking " +
      "the row early would flash \"nothing active\" during every transition")
check(state(for: "elderly_monitoring", active: "driver_safety", pending: switchingToBaby) == .inactive,
      "an unrelated row is unaffected")

print("\noffline: queued, not spinning as if imminent:")
let switchingOffline = ModeSwitchState.PendingSwitch(targetBase: "security_monitoring", requestedAt: t0)
check(state(for: "security_monitoring", active: "driver_safety", pending: switchingOffline, online: false) == .queuedOffline,
      "an offline device shows queued, not switching")
check(state(for: "security_monitoring", active: "driver_safety", pending: switchingOffline, online: true) == .switching,
      "the same pending switch shows switching once the device is online")

print("\nno infinite spinner — a switch times out:")
let oldSwitch = ModeSwitchState.PendingSwitch(targetBase: "baby_monitoring", requestedAt: t0)
check(state(for: "baby_monitoring", active: "driver_safety", pending: oldSwitch, at: 5) == .switching,
      "still within the timeout window, still switching")
check(state(for: "baby_monitoring", active: "driver_safety", pending: oldSwitch, at: ModeSwitchState.switchTimeout - 0.01) == .switching,
      "just under the timeout, still switching")
if case .failed = state(for: "baby_monitoring", active: "driver_safety", pending: oldSwitch, at: ModeSwitchState.switchTimeout + 0.01) {
    print("  ok   just past the timeout, shows failed rather than spinning forever")
} else {
    print("  FAIL a switch past the timeout must not still show as switching"); failures += 1
}

print("\nan explicit failed state, distinct from just reverting to inactive:")
let failedSwitch = ModeSwitchState.PendingSwitch(targetBase: "elderly_monitoring", requestedAt: t0)
if case let .failed(message) = state(for: "elderly_monitoring", active: "driver_safety", failure: failedSwitch) {
    check(!message.isEmpty, "the failed state carries a message the user can act on")
} else {
    print("  FAIL a failed switch must render as .failed, not silently as .inactive"); failures += 1
}
check(state(for: "driver_safety", active: "driver_safety", failure: failedSwitch) == .active,
      "a failure on one row does not affect the row that is genuinely active")

print("\na retry after failure supersedes the old failure (rapid re-tap):")
// The caller clears `failure` the moment a new attempt starts — this asserts
// the state function itself does not need pending to "win" over a stale
// failure once the caller has cleared it.
check(state(for: "elderly_monitoring", active: "driver_safety",
            pending: ModeSwitchState.PendingSwitch(targetBase: "elderly_monitoring", requestedAt: t0),
            failure: nil) == .switching,
      "a fresh retry shows switching, not the old failure")

print("\nevery AI Custom transition the product spec calls out, both directions:")
let presets = ["driver_safety", "baby_monitoring", "elderly_monitoring", "security_monitoring"]
for target in presets {
    // AI Custom -> preset: nothing reads as active while custom is in charge,
    // and the moment a switch to `target` is requested it (and only it) shows
    // switching.
    let pending = ModeSwitchState.PendingSwitch(targetBase: target, requestedAt: t0)
    for candidate in presets {
        let expected: ModeRowState = candidate == target ? .switching : .inactive
        check(state(for: candidate, active: "driver_safety", custom: true, pending: pending) == expected,
              "AI Custom -> \(target): row for \(candidate) is \(expected)")
    }
}
for target in presets {
    // preset -> AI Custom is not itself representable as a ModeRowState (AI
    // Custom has no preset row) — what must hold is that once the device
    // confirms custom monitoring, no preset row claims to be active.
    check(state(for: target, active: target, custom: true) == .inactive,
          "\(target) -> AI Custom: \(target)'s own row stops reading as active")
}

print("\nreload() must not clobber a pendingSwitch written after it started fetching:")
// The bug found during the production QA pass: reload() and setMode() write
// and read `pendingSwitch` through two independent LatestRequestGuard
// instances that know nothing about each other. A tap that lands (writes
// pendingSwitch) AFTER a concurrent reload's network fetch already left the
// phone, but BEFORE that reload's response comes back, must not be erased by
// data that necessarily predates the tap.
do {
    let tapHappensAt = t0.addingTimeInterval(5)      // the user's tap
    let reloadStartedBeforeTap = t0.addingTimeInterval(4)  // reload() began fetching a moment earlier
    let freshTap = ModeSwitchState.PendingSwitch(targetBase: "baby_monitoring", requestedAt: tapHappensAt)
    check(!ModeSwitchState.shouldSyncPendingSwitch(current: freshTap, reloadStartedAt: reloadStartedBeforeTap),
          "a reload that started BEFORE the tap must not sync over it")
}
do {
    let reloadStartedAfterTap = t0.addingTimeInterval(10)
    let oldTap = ModeSwitchState.PendingSwitch(targetBase: "baby_monitoring", requestedAt: t0.addingTimeInterval(1))
    check(ModeSwitchState.shouldSyncPendingSwitch(current: oldTap, reloadStartedAt: reloadStartedAfterTap),
          "a reload that started AFTER the tap safely reflects it — sync is allowed")
}
check(ModeSwitchState.shouldSyncPendingSwitch(current: nil, reloadStartedAt: t0),
      "nothing pending means nothing to protect — always safe to sync")
do {
    // The exact instant they coincide: treated as safe (<=), matching how a
    // reload started in the SAME instant a tap that hasn't happened yet
    // cannot describe it.
    let sameInstant = ModeSwitchState.PendingSwitch(targetBase: "baby_monitoring", requestedAt: t0)
    check(ModeSwitchState.shouldSyncPendingSwitch(current: sameInstant, reloadStartedAt: t0),
          "a tap at exactly the reload's start time is treated as syncable")
}

print("\nrapid switching across every mode, ending on AI Custom, must not revert:")
// The exact scenario from the production QA report: a person taps through
// several modes quickly, landing on AI Custom last. A reload that started
// BEFORE that final tap — and so necessarily carries a server snapshot that
// predates it — must never be allowed to clobber it back to whatever mode was
// active a moment earlier. Modeled as a sequence of taps, each one
// immediately followed by a reload that was already in flight when the tap
// landed (the realistic race: the 10s poll timer firing mid-sequence).
do {
    let tapTimes: [(target: String, at: TimeInterval)] = [
        ("baby_monitoring", 1), ("elderly_monitoring", 2),
        ("security_monitoring", 3), ("driver_safety", 4),
    ]
    var current: ModeSwitchState.PendingSwitch?
    for tap in tapTimes {
        current = ModeSwitchState.PendingSwitch(
            targetBase: tap.target, requestedAt: t0.addingTimeInterval(tap.at))
    }
    // A reload that began before the very last tap must not sync over it.
    let reloadInFlightDuringSequence = t0.addingTimeInterval(3.5)
    check(!ModeSwitchState.shouldSyncPendingSwitch(current: current, reloadStartedAt: reloadInFlightDuringSequence),
          "a reload started mid-sequence must not clobber the tap that landed after it")

    // The final tap in the rapid sequence lands on AI Custom itself. Nothing
    // reads as active while it is pending — not even the mode it was
    // switching away from mid-flight — and a stale reload that predates it
    // must still be rejected the same way.
    let finalTapToCustom = t0.addingTimeInterval(5)
    for candidate in presets {
        check(state(for: candidate, active: "driver_safety", custom: true, pending: nil, at: 5) == .inactive,
              "once AI Custom is confirmed active, \(candidate) reads inactive even with nothing pending")
    }
    check(!ModeSwitchState.shouldSyncPendingSwitch(
        current: ModeSwitchState.PendingSwitch(targetBase: "", requestedAt: finalTapToCustom),
        reloadStartedAt: t0.addingTimeInterval(4.9)),
          "a reload that started just before the tap landing on AI Custom must not revert it")
}

print("\na refusal the backend explained is shown, not \"Didn't switch\":")
do {
    let refused = ModeSwitchState.PendingSwitch(
        targetBase: "baby_monitoring", requestedAt: t0,
        reason: "this device's role is 'driver_safety': preset 'baby_monitoring' would stop driver_attention")
    check(state(for: "baby_monitoring", active: "driver_safety", failure: refused)
          == .failed("this device's role is 'driver_safety': preset 'baby_monitoring' would stop driver_attention"),
          "the backend's detail is the row's message")
    let unexplained = ModeSwitchState.PendingSwitch(targetBase: "baby_monitoring", requestedAt: t0)
    check(state(for: "baby_monitoring", active: "driver_safety", failure: unexplained)
          == .failed("Didn't switch — tap to try again"),
          "a failure with no explanation keeps the generic message")
}

print("\na mode this device's role forbids is shown unavailable, with why:")
func roleState(_ name: String, active: String = "driver_safety", custom: Bool = false,
               unavailable: String?) -> ModeRowState {
    ModeSwitchState.state(for: name, activeBase: active, customMonitoringActive: custom,
                          pending: nil, deviceOnline: true, failure: nil, now: t0, unavailable: unavailable)
}
check(roleState("baby_monitoring", unavailable: "Not available for this device's role (Driver safety)")
      == .unavailable("Not available for this device's role (Driver safety)"), "unavailable carries its reason")
check(roleState("driver_safety", unavailable: "stale table") == .active,
      "a mode the device reports running is active, whatever the role table says")
check(roleState("baby_monitoring", unavailable: nil) == .inactive, "an allowed mode is just inactive")

print("\ncustom rules running ALONGSIDE the mode leave the mode active:")
// DeviceDetailView passes customMonitoringActive = true only when custom rules
// REPLACE the modes (AI Custom); on an additive role the mode keeps running.
check(state(for: "driver_safety", active: "driver_safety", custom: false) == .active,
      "additive: the driver mode row stays active beside the rules")
check(state(for: "driver_safety", active: "driver_safety", custom: true) == .inactive,
      "replacing: the mode row is not active")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
