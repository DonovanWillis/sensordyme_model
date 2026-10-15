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

    /// Which of one device's events are new since the last poll, and the
    /// high-water mark to store for that device afterwards.
    ///
    /// Replaces a single global set of "announced" ids trimmed to the newest
    /// 500. The poller re-reads each device's latest 25 events on every tick,
    /// so on an account with a busy camera and a quiet one, the quiet camera's
    /// ids became the smallest in the set, were evicted on every save, and were
    /// announced again on the next poll — one day-old "Fall" re-posted every
    /// ~30 seconds, for ever. A per-device mark cannot be evicted by another
    /// device's traffic, and needs no size bound.
    ///
    /// The first sight of a device (no mark) SEEDS the mark without reporting
    /// anything fresh: on a new install, after another account signed in, or on
    /// entering the demo, everything the poller sees is history, and announcing
    /// it posted the oldest critical event in the window — possibly days old,
    /// with nothing in the notification to say so.
    ///
    /// Event ids are monotonic per backend (sqlite AUTOINCREMENT), so "new" is
    /// "greater than the mark". A device whose newest id is BELOW its mark has a
    /// different history (the backend was reset), so it is re-seeded rather than
    /// silenced for ever.
    ///
    /// - Returns: the fresh ids, oldest first, and the mark to store.
    static func freshEvents(ids: [Int], mark: Int?) -> (fresh: [Int], mark: Int) {
        let newest = ids.max()
        guard let mark else { return ([], newest ?? 0) }
        if let newest, newest < mark { return ([], newest) }
        let fresh = Set(ids.filter { $0 > mark }).sorted()
        return (fresh, max(mark, newest ?? mark))
    }
}
