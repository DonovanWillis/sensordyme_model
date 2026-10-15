import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

let t0 = Date(timeIntervalSince1970: 1_000_000)
let cooldown: TimeInterval = 20

func decide(critical: Bool = true, configured: Bool = false,
            seen: Bool = false, at offset: TimeInterval = 0,
            lastAt: TimeInterval? = nil) -> NotificationPolicy.Decision {
    NotificationPolicy.decide(
        isCritical: critical, isConfigured: configured, alreadySeen: seen,
        now: t0.addingTimeInterval(offset),
        lastNotifiedAt: lastAt.map { t0.addingTimeInterval($0) },
        cooldown: cooldown)
}

print("only the red alert interrupts:")
check(decide(critical: true) == .notify, "a red alert notifies")
check(decide(critical: false) == .suppress(.notCritical),
      "a warning does NOT notify (it still reaches the timeline)")
// The spam: BOTH severities used to notify. On a real device that was 942
// 'loud' plus 608 'soft' in a day, from a driver-monitor doing its job.

print("\nthe 20-second cooldown, from the product's own worked example:")
// 10:00:00 notify, then :05 :10 :15 suppressed, :20 eligible again.
check(decide(at: 0, lastAt: nil) == .notify, "10:00:00 first alert notifies")
for (offset, label) in [(5.0, "10:00:05"), (10.0, "10:00:10"), (15.0, "10:00:15")] {
    if case .suppress(.withinCooldown) = decide(at: offset, lastAt: 0) {
        print("  ok   \(label) suppressed")
    } else {
        print("  FAIL \(label) should be suppressed"); failures += 1
    }
}
check(decide(at: 20, lastAt: 0) == .notify, "10:00:20 eligible again")
check(decide(at: 20.001, lastAt: 0) == .notify, "just past the boundary is eligible")

print("\nthe cooldown reports how long is left, so suppression can be shown:")
if case let .suppress(.withinCooldown(remaining)) = decide(at: 7, lastAt: 0) {
    check(remaining == 13, "13s remaining at 10:00:07 (got \(remaining))")
} else {
    check(false, "should report remaining seconds"); failures += 1
}

print("\nedge cases:")
check(decide(seen: true) == .suppress(.alreadySeen),
      "an event already considered is never reconsidered")
check(decide(critical: false, at: 100, lastAt: 0) == .suppress(.notCritical),
      "a warning outside the cooldown still does not notify")
check(decide(critical: true, at: 100, lastAt: 0) == .notify,
      "a genuinely new alert long after the last one notifies")
// A different high-priority event inside the window is still suppressed: the
// cooldown is about how often the phone may interrupt, not about sameness —
// this is the explicit product decision (global cooldown), not a bug.
if case .suppress(.withinCooldown) = decide(critical: true, at: 3, lastAt: 0) {
    print("  ok   a DIFFERENT red alert inside the window is also suppressed (by design)")
} else {
    print("  FAIL different-alert-in-window should be suppressed under the global-cooldown policy"); failures += 1
}
check(decide(critical: true, lastAt: nil) == .notify,
      "first ever alert, with no previous notification, notifies")

print("\nsomething the user configured always qualifies for consideration:")
check(decide(critical: false, configured: true) == .notify,
      "a configured rule notifies even at warning severity")
check(decide(critical: false, configured: false) == .suppress(.notCritical),
      "a preset's own warning still does not")
if case .suppress(.withinCooldown) = decide(critical: false, configured: true, at: 5, lastAt: 0) {
    print("  ok   a configured rule still respects the cooldown — 'configured' bypasses severity, not the rate limit")
} else {
    print("  FAIL configured rules must not bypass the cooldown — one policy, no per-source exceptions"); failures += 1
}

// ------------------------------------------------ what counts as new, per device
print("\nnew means newer than this device's high-water mark:")
check(NotificationPolicy.freshEvents(ids: [12, 11, 10], mark: 10) == ([11, 12], 12),
      "only ids above the mark are fresh, oldest first, and the mark moves up")
check(NotificationPolicy.freshEvents(ids: [12, 11], mark: 12) == ([], 12),
      "nothing new: nothing fresh, mark unchanged")
check(NotificationPolicy.freshEvents(ids: [], mark: 7) == ([], 7), "no events: mark kept")

print("\nthe first sight of a device is history, not news (fresh install, new account, demo):")
check(NotificationPolicy.freshEvents(ids: [300, 200, 100], mark: nil) == ([], 300),
      "seeded at the newest id, announcing nothing — a day-old fall is not posted as if it just happened")
check(NotificationPolicy.freshEvents(ids: [], mark: nil) == ([], 0),
      "a device with no events yet: everything after this is new")
check(NotificationPolicy.freshEvents(ids: [5, 4], mark: 900) == ([], 5),
      "a device whose ids fell below its mark (backend reset) is re-seeded, not silenced for ever")

// The regression: one global set trimmed to the newest 500 ids. A quiet
// camera's window became the smallest ids once a busy camera on the same
// account logged 500 events, was evicted on every save, and its old "fall" was
// announced again on every other poll. Replayed here against the poller's real
// shape: 25-event window per device, both devices every tick.
print("\na busy camera cannot make a quiet camera's old alert news again:")
do {
    var marks: [String: Int] = [:]
    var reannounced = 0
    func poll(_ device: String, _ window: [Int]) -> [Int] {
        let (fresh, mark) = NotificationPolicy.freshEvents(ids: window, mark: marks[device])
        marks[device] = mark
        return fresh
    }
    _ = poll("quiet", [1000])            // first sight: seeded
    _ = poll("busy", [999])
    var newest = 1000
    for _ in 0..<800 {                   // far past the old 500-id bound
        newest += 1
        _ = poll("busy", Array((max(1001, newest - 24)...newest).reversed()))
        reannounced += poll("quiet", [1000]).count
    }
    check(reannounced == 0, "the quiet camera's event 1000 is never fresh again (was: re-posted ~every 30s)")
    check(poll("quiet", [newest + 1, 1000]) == [newest + 1], "while its genuinely new event still is")
}

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
