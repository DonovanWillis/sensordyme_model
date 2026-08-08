import Foundation

/// Decides which detections are allowed to interrupt someone, and how often.
///
/// Pulled out of the notifier as a plain function so it can be tested. The
/// behaviour it encodes is easy to state and was easy to get wrong: a
/// driver-safety device watching a driver produces a continuous stream of
/// genuine observations, and notifying on all of them turns a safety product
/// into a phone that buzzes because the AI is working.
///
/// Two rules, and one thing that is deliberately NOT a rule:
///   - only the red alert notifies; warnings belong in the timeline
///   - at most one notification per cooldown window
///   - nothing here filters what is DETECTED or RECORDED. Every event still
///     reaches the history; this drops the interruption, never the record.
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
        if !isCritical && !isConfigured { return .suppress(.notCritical) }
        if let last = lastNotifiedAt {
            let elapsed = now.timeIntervalSince(last)
            if elapsed < cooldown {
                return .suppress(.withinCooldown(secondsRemaining: Int((cooldown - elapsed).rounded(.up))))
            }
        }
        return .notify
    }
}
