import Foundation

// How a Configure-with-AI rule is put into words and edited, kept apart from the
// views so ios/Tests/AIRuleTextTests.swift can pin it down. Foundation only.

/// One rule, in the person's terms rather than the schema's.
enum RuleWording {
    /// The rule's one-line headline.
    ///
    /// Every predicate the device evaluates (core/rules.py) gets its own
    /// wording. Anything unknown used to fall through to "appears", so two
    /// different drinking rules — `near_mouth` a cup, `near_mouth` a bottle —
    /// both read "Person appears", and a count rule and a stillness rule read
    /// exactly like a presence rule.
    static func headline(
        target: String, predicate: String, secondary: String, zone: String, minCount: Int
    ) -> String {
        let subject = target.isEmpty ? "Something" : target.capitalized
        let other = secondary.isEmpty ? "another object" : secondary
        let area = zone.isEmpty ? "the area" : zone
        let phrase: String
        switch predicate {
        case "present", "":
            phrase = minCount > 1 ? "appears (\(minCount)+)" : "appears"
        case "absent":
            phrase = "is gone"
        case "in_zone":
            phrase = "is in \(area)"
        case "near":
            phrase = "is near \(other)"
        case "holding":
            phrase = "is holding \(other)"
        case "near_mouth":
            // The device tests the target's face region against the second
            // object: evidence the object is raised to the mouth.
            phrase = "has \(other) at their mouth"
        case "count":
            phrase = minCount > 1 ? "count reaches \(minCount)+" : "count changes"
        case "still":
            phrase = "stays still"
        case "persists":
            phrase = "stays in view"
        case "lingering":
            phrase = "lingers"
        case "enters":
            phrase = "enters \(area)"
        case "exits":
            phrase = "leaves \(area)"
        default:
            phrase = predicate.replacingOccurrences(of: "_", with: " ")
            return "\(subject): \(phrase)"
        }
        return "\(subject) \(phrase)"
    }
}

/// A slider position (0...1) mapped onto a duration range, logarithmically.
///
/// The review screen's sliders were linear and far narrower than what a rule may
/// carry — "Confirm for" 1–60 s against the schema's 0.5–3600 s (Jupiter's sleep
/// rules use 90–120 s), "Then wait" 30–1800 s against 10–86400 s — so merely
/// touching a slider clamped a valid rule to something nobody chose. A log
/// scale covers the whole range and keeps the short end, where precision
/// matters, usable.
struct DurationScale: Equatable {
    let lower: Double
    let upper: Double

    /// jupiter/schema.py PERSIST_MIN / PERSIST_MAX.
    static let persistence = DurationScale(lower: 0.5, upper: 3600)
    /// jupiter/schema.py COOLDOWN_MIN / COOLDOWN_MAX.
    static let cooldown = DurationScale(lower: 10, upper: 86_400)

    /// Where `seconds` sits on the slider. Values outside the range pin to an end.
    func position(for seconds: Double) -> Double {
        guard seconds.isFinite, lower > 0, upper > lower else { return 0 }
        let clamped = min(max(seconds, lower), upper)
        return log(clamped / lower) / log(upper / lower)
    }

    /// The duration at a slider position, rounded to a step a person would pick
    /// and never outside the range.
    func seconds(at position: Double) -> Double {
        let p = position.isFinite ? min(max(position, 0), 1) : 0
        let raw = lower * pow(upper / lower, p)
        return min(max(Self.snap(raw), lower), upper)
    }

    /// Round to a step that suits the magnitude: half-seconds below 10 s,
    /// seconds below a minute, then 10 s, 30 s and 5 min steps.
    static func snap(_ seconds: Double) -> Double {
        let step: Double
        switch seconds {
        case ..<10: step = 0.5
        case ..<60: step = 1
        case ..<600: step = 10
        case ..<3600: step = 30
        default: step = 300
        }
        return (seconds / step).rounded() * step
    }
}

enum DurationText {
    /// "0.5s", "45s", "1m 30s", "2h", "24h".
    static func short(_ seconds: Double) -> String {
        guard seconds.isFinite else { return "—" }
        let s = max(0, seconds)
        if s < 60 {
            return s == s.rounded() ? "\(Int(s))s" : String(format: "%.1fs", s)
        }
        let total = Int(s.rounded())
        if total < 3600 {
            let minutes = total / 60, rest = total % 60
            return rest == 0 ? "\(minutes)m" : "\(minutes)m \(rest)s"
        }
        let hours = total / 3600, minutes = (total % 3600) / 60
        return minutes == 0 ? "\(hours)h" : "\(hours)h \(minutes)m"
    }
}

/// What an activation achieved, from what the backend confirmed.
///
/// `POST /v1/devices/{id}/preset` reports `rules_active` — how many of the sent
/// rules will run. Rules sent WITH a named mode are never turned on (an additive
/// role runs rules beside its mode, but only when they are activated on their
/// own), so for a preset proposal the mode switches and the rules come back
/// unsupported; that
/// is a mode switch, not a failure, and must not read as either "Active" rules
/// or an error.
struct ActivationSummary: Equatable {
    enum Kind: Equatable {
        /// Rules (or, with no confirmation, the whole proposal) are running.
        case active
        /// The named mode is on; no rules run beside it.
        case modeSwitched
        /// A custom configuration none of whose rules can run: nothing is on.
        case nothingRunning
    }

    let kind: Kind
    let note: String?

    /// - Parameters:
    ///   - isCustom: the proposal names no monitoring mode (AI Custom).
    ///   - rulesActive: from the backend; nil from one that predates it, which
    ///     keeps the old behaviour of claiming the activation as sent.
    ///   - additive: the device's role runs custom rules BESIDE its mode
    ///     (DeviceStatus rules.additive). There "rules can't run alongside a
    ///     monitoring mode" is false: activated on their own, they do. Only
    ///     rules sent together with a mode switch are not turned on.
    static func summarize(isCustom: Bool, rulesSent: Int, rulesActive: Int?,
                          additive: Bool = false) -> ActivationSummary {
        guard let active = rulesActive else { return ActivationSummary(kind: .active, note: nil) }
        let elsewhere = "see \u{201C}Not yet possible\u{201D}"
        if active == 0 {
            if isCustom {
                return ActivationSummary(
                    kind: .nothingRunning,
                    note: "None of these rules can run on this device — \(elsewhere).")
            }
            if additive {
                return ActivationSummary(
                    kind: .modeSwitched,
                    note: rulesSent == 0 ? nil
                        : "\(rulesSent == 1 ? "This rule wasn't" : "These rules weren't") turned on with the mode. "
                          + "Activate \(rulesSent == 1 ? "it" : "them") on \(rulesSent == 1 ? "its" : "their") own "
                          + "to run \(rulesSent == 1 ? "it" : "them") alongside the mode. "
                          + "See \u{201C}Not yet possible\u{201D}.")
            }
            return ActivationSummary(
                kind: .modeSwitched,
                note: rulesSent == 0 ? nil
                    : "\(rulesSent == 1 ? "This rule needs" : "These rules need") AI Custom — "
                      + "rules can't run alongside a monitoring mode. See \u{201C}Not yet possible\u{201D}.")
        }
        let running = (active == 1 ? "1 rule running" : "\(active) rules running")
            + (additive ? " alongside the monitoring mode" : "")
        if active < rulesSent {
            return ActivationSummary(
                kind: .active,
                note: "\(active) of \(rulesSent) rules running. The others can't run — \(elsewhere).")
        }
        return ActivationSummary(kind: .active, note: running + ".")
    }
}
