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

print("\nthe 20-second cooldown, from the worked example:")
// 12:00:00 notify, then :02 :05 :11 :18 suppressed, :20 eligible again.
check(decide(at: 0, lastAt: nil) == .notify, "12:00:00 first alert notifies")
for (offset, label) in [(2.0, "12:00:02"), (5.0, "12:00:05"), (11.0, "12:00:11"), (18.0, "12:00:18")] {
    if case .suppress(.withinCooldown) = decide(at: offset, lastAt: 0) {
        print("  ok   \(label) suppressed")
    } else {
        print("  FAIL \(label) should be suppressed"); failures += 1
    }
}
check(decide(at: 20, lastAt: 0) == .notify, "12:00:20 eligible again")
check(decide(at: 20.001, lastAt: 0) == .notify, "just past the boundary is eligible")

print("\nthe cooldown reports how long is left, so suppression can be shown:")
if case let .suppress(.withinCooldown(remaining)) = decide(at: 7, lastAt: 0) {
    check(remaining == 13, "13s remaining at 12:00:07 (got \(remaining))")
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
// cooldown is about how often the phone may interrupt, not about sameness.
if case .suppress(.withinCooldown) = decide(critical: true, at: 3, lastAt: 0) {
    print("  ok   a DIFFERENT red alert inside the window is also suppressed")
} else {
    print("  FAIL different-alert-in-window should be suppressed"); failures += 1
}
check(decide(critical: true, lastAt: nil) == .notify,
      "first ever alert, with no previous notification, notifies")

print("\nsomething the user configured always qualifies:")
check(decide(critical: false, configured: true) == .notify,
      "a configured rule notifies even at warning severity")
check(decide(critical: false, configured: false) == .suppress(.notCritical),
      "a preset's own warning still does not")
if case .suppress(.withinCooldown) = decide(critical: false, configured: true, at: 5, lastAt: 0) {
    print("  ok   a configured rule still respects the cooldown")
} else {
    print("  FAIL configured rules must not bypass the cooldown"); failures += 1
}

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
