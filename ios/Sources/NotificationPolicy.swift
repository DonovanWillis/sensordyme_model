import Foundation

/// Decides which detections are allowed to interrupt someone, and how often.
///
/// Pulled out of the notifier as a plain function so it can be tested. The
/// behaviour it encodes is easy to state and was easy to get wrong: a
/// driver-safety device watching a driver produces a continuous stream of
/// genuine observations, and notifying on all of them turns a safety product
/// into a phone that buzzes because the AI is working.
///
/// THE POLICY (product spec, stated explicitly, and matching cloud/notify.py
/// exactly so the two paths — local notifications posted by the app, and a
/// push delivered by the OS whether the app is open or not — never disagree
/// about the same moment):
///   - only the red alert (or something the user explicitly configured)
///     notifies; warnings belong in the timeline
///   - the same event never notifies twice — identity-based, via `alreadySeen`
///   - at most one notification per 20-second cooldown, GLOBALLY (not
///     per-event-type, not bypassed by a configured rule — "configured"
///     bypasses the severity filter above, never the cooldown itself)
///   - nothing here filters what is DETECTED or RECORDED. Every event still
///     reaches the history; this drops the interruption, never the record.
///
/// This file has flipped policy twice: an earlier revision replaced the
/// cooldown with pure identity dedup (distinct alerts always notified,
/// however close together) because that felt like the safety-correct
/// default; the product decision, stated explicitly afterward, reversed
/// that back to a cooldown because pure identity dedup meant a burst of
/// distinct alerts became a burst of notifications. This is the current,
/// active policy — the cooldown is real and intentional, not a leftover.
enum NotificationPolicy {
    enum Decision: Equatable {
        case notify
        /// Recorded, not notified — with the reason, so it can be shown.
        case suppress(Reason)

        enum Reason: Equatable {
            case notCritical
            case withinCooldown(secondsRemaining: Int)
            case alreadySeen
        }
    }

    static func decide(
        isCritical: Bool,
        isConfigured: Bool = false,
        alreadySeen: Bool,
        now: Date,
        lastNotifiedAt: Date?,
        cooldown: TimeInterval
    ) -> Decision {
        if alreadySeen { return .suppress(.alreadySeen) }
        // Something the user explicitly configured always qualifies, whatever
        // severity it carries — asking to be told about it IS the instruction.
        // This bypasses the severity filter only; the cooldown check below
        // still applies to a configured alert exactly as it does to any other.
        if !isCritical && !isConfigured { return .suppress(.notCritical) }
        if let last = lastNotifiedAt {
            let elapsed = now.timeIntervalSince(last)
            if elapsed < cooldown {
                return .suppress(.withinCooldown(secondsRemaining: Int((cooldown - elapsed).rounded(.up))))
            }
        }
        return .notify
    }

    /// How many announced event ids to remember.
    ///
    /// The list exists only to stop the poller re-announcing what it already
    /// announced, so it needs to cover the window an event can still be returned
    /// in — the app asks for the newest 25-200 events — with a wide margin, not
    /// the whole history.
    static let announcedHistoryLimit = 500

    /// The ids worth keeping out of everything announced so far.
    ///
    /// Keeps the NEWEST ids, and that is the whole point. This was
    /// `Set.suffix(500)`, which looks like "keep the most recent 500" and is not:
    /// `Set` has no order, so `suffix` returns 500 elements in whatever order the
    /// hash table happens to iterate. Past 500 events the ids retained were
    /// arbitrary, so a *just-announced* alert could be evicted while an ancient
    /// one was kept — and an evicted id is one the next poll no longer recognises,
    /// so it announces it again. That is a duplicate alert for an event the user
    /// was already told about, appearing only on accounts with a few hundred
    /// events, which is why it would not show up early.
    ///
    /// Event ids are monotonic (sqlite AUTOINCREMENT), so "newest" is "largest"
    /// and no timestamp is needed.
    static func trimAnnounced(_ ids: Set<Int>, limit: Int = announcedHistoryLimit) -> [Int] {
        guard limit > 0 else { return [] }
        guard ids.count > limit else { return ids.sorted() }
        return ids.sorted().suffix(limit)
    }
}
