import Foundation

// Custom rules as the backend lists them for one device
// (GET /v1/devices/{id}/rules, PATCH /v1/devices/{id}/rules/{rule_id}), what
// each rule's status means in words, and what to report after an activation.
// Foundation only, so ios/Tests/DeviceRulesTests.swift can pin it down.

struct DeviceRule: Equatable, Identifiable, Sendable {
    let id: String
    var enabled: Bool
    var summary: String
    var targets: [String] = []
    var zone: String? = nil
    var dwellS: Double? = nil
    var requiredCapabilities: [String] = []
    /// running | waiting | rejected | disabled | unknown (anything else reads as unknown)
    var status: String
    var reason: String? = nil
}

struct DeviceRules: Equatable, Sendable {
    var rulesVersion: String?
    /// Whether these rules run BESIDE the monitoring mode (an additive role).
    var additive: Bool?
    var rules: [DeviceRule]
}

private struct RuleKey: CodingKey {
    let stringValue: String
    init(_ name: String) { stringValue = name }
    init?(stringValue: String) { self.stringValue = stringValue }
    var intValue: Int? { nil }
    init?(intValue: Int) { nil }
}

extension DeviceRule: Decodable {
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: RuleKey.self)
        func text(_ key: String) -> String? { (try? c.decodeIfPresent(String.self, forKey: RuleKey(key))) ?? nil }
        // An id may arrive as a number from a backend that keys rules by row.
        let id = text("id")
            ?? ((try? c.decodeIfPresent(Int.self, forKey: RuleKey("id"))) ?? nil).map(String.init)
        guard let id, !id.isEmpty else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "rule without id"))
        }
        let targets = ((try? c.decodeIfPresent([String].self, forKey: RuleKey("targets"))) ?? nil)
            ?? text("targets").map { [$0] } ?? []
        let zone = text("zone").flatMap { $0.isEmpty ? nil : $0 }
        self.init(
            id: id,
            enabled: ((try? c.decodeIfPresent(Bool.self, forKey: RuleKey("enabled"))) ?? nil) ?? true,
            summary: text("summary") ?? text("description") ?? targets.joined(separator: ", "),
            targets: targets, zone: zone,
            dwellS: (try? c.decodeIfPresent(Double.self, forKey: RuleKey("dwell_s"))) ?? nil,
            requiredCapabilities: ((try? c.decodeIfPresent([String].self, forKey: RuleKey("required_capabilities"))) ?? nil) ?? [],
            status: text("status") ?? "unknown",
            reason: text("reason").flatMap { $0.isEmpty ? nil : $0 })
    }
}

extension DeviceRules: Decodable {
    private struct Lossy: Decodable {
        let rule: DeviceRule?
        init(from decoder: Decoder) throws { rule = try? DeviceRule(from: decoder) }
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: RuleKey.self)
        rulesVersion = (try? c.decodeIfPresent(String.self, forKey: RuleKey("rules_version"))) ?? nil
        additive = (try? c.decodeIfPresent(Bool.self, forKey: RuleKey("additive"))) ?? nil
        rules = (((try? c.decodeIfPresent([Lossy].self, forKey: RuleKey("rules"))) ?? nil) ?? []).compactMap(\.rule)
    }
}

/// How alarming a rule's state is, in this file's own terms (the views map it
/// onto the design system).
enum RuleTone: Equatable {
    case ok, caution, critical, neutral
}

/// One rule's status chip and sentence.
struct RuleStatusPresentation: Equatable {
    let chip: String
    let detail: String?
    let tone: RuleTone

    /// - Parameter deviceLive: the device is online and its report is fresh.
    ///   A rule is never shown as running on a device that is not reporting:
    ///   "running" is something the device says, and it has stopped saying.
    static func make(_ rule: DeviceRule, deviceLive: Bool) -> RuleStatusPresentation {
        if !rule.enabled || rule.status == "disabled" {
            return .init(chip: "Off", detail: "Turned off. It won't alert until you turn it back on.", tone: .neutral)
        }
        if !deviceLive && rule.status != "rejected" {
            return .init(chip: "Unknown",
                         detail: "The device isn't reporting, so whether this rule runs is unknown.", tone: .neutral)
        }
        switch rule.status {
        case "running":
            return .init(chip: "Running", detail: rule.reason, tone: .ok)
        case "waiting":
            if let zone = rule.zone {
                return .init(chip: "Waiting for area",
                             detail: rule.reason ?? "Draw the \u{201C}\(zone)\u{201D} area in Detection areas to start it.",
                             tone: .caution)
            }
            return .init(chip: "Waiting", detail: rule.reason ?? "Waiting for the device to start it.", tone: .caution)
        case "rejected":
            return .init(chip: "Rejected",
                         detail: rule.reason.map { "The device refused it: \($0)" } ?? "The device refused it.",
                         tone: .critical)
        default:
            return .init(chip: "Unknown", detail: rule.reason ?? "The device hasn't said whether this rule runs.",
                         tone: .neutral)
        }
    }
}

/// Why a PATCH to a rule failed, in words.
enum RuleToggleError {
    static func message(code: String?, detail: String, endpointMissing: Bool) -> String {
        if endpointMissing { return "Turning rules on and off isn't supported by this server yet." }
        if code == "rule_not_found" { return "This rule no longer exists — the list has been refreshed." }
        return detail.isEmpty ? "The server refused the change." : detail
    }
}

/// What the device reports after an activation, said as a result rather than
/// as the app's own assumption. "Accepted by the server" is not "running on
/// the device", and the person is told which one they have.
enum ActivationFollowUp: Equatable {
    /// Still waiting for the device to say anything about it.
    case waiting(String)
    /// The device reports it.
    case confirmed(String)
    /// The device refused it and kept what it had.
    case refused(String)

    var text: String {
        switch self {
        case .waiting(let text), .confirmed(let text), .refused(let text): text
        }
    }

    /// - Parameters:
    ///   - rulesExpected: how many rules the backend said would run (0 for a
    ///     pure mode switch).
    ///   - modeTitle: the mode switched to, when the proposal named one.
    ///   - modePending: the backend still holds a desired mode the device has
    ///     not confirmed.
    ///   - deviceLive: online and fresh.
    ///   - rejectedReasons: the device's refusal of the current rule set, or nil.
    ///   - runningRuleIDs: `status.rules.active`, nil when unknown.
    ///   - rules: the per-rule list, when the server has the rules endpoint.
    static func evaluate(rulesExpected: Int, modeTitle: String?, modePending: Bool, deviceLive: Bool,
                         rejectedReasons: [String]?, runningRuleIDs: [String]?,
                         rules: [DeviceRule]?) -> ActivationFollowUp {
        if let rejectedReasons {
            let reason = rejectedReasons.first(where: { !$0.isEmpty })
            return .refused("The device refused these rules\(reason.map { ": \($0)" } ?? ""). It kept what it was running before.")
        }
        if !deviceLive {
            return .waiting("Saved, but the device isn't reporting right now. It picks this up when it reconnects.")
        }
        if rulesExpected > 0 {
            if let rules {
                let enabled = rules.filter(\.enabled)
                if let rejected = enabled.first(where: { $0.status == "rejected" }) {
                    return .refused("The device refused \u{201C}\(rejected.summary)\u{201D}\(rejected.reason.map { ": \($0)" } ?? "").")
                }
                let running = enabled.filter { $0.status == "running" }.count
                let waiting = enabled.filter { $0.status == "waiting" }.count
                if running + waiting > 0 {
                    var parts: [String] = []
                    if running > 0 { parts.append(running == 1 ? "1 rule running" : "\(running) rules running") }
                    if waiting > 0 { parts.append(waiting == 1 ? "1 waiting for an area" : "\(waiting) waiting for an area") }
                    return .confirmed("The device reports " + parts.joined(separator: ", ") + ".")
                }
                return .waiting("Waiting for the device to confirm the rules…")
            }
            if let running = runningRuleIDs, !running.isEmpty {
                return .confirmed("The device reports \(running.count == 1 ? "1 rule" : "\(running.count) rules") running.")
            }
            return .waiting("Waiting for the device to confirm the rules…")
        }
        if modePending {
            return .waiting("Waiting for the device to switch\(modeTitle.map { " to \($0)" } ?? "")…")
        }
        return .confirmed("The device reports it is running \(modeTitle ?? "the new mode").")
    }

    /// What to say when the device never answered within the wait.
    static func timedOut(_ last: ActivationFollowUp) -> ActivationFollowUp {
        guard case .waiting = last else { return last }
        return .waiting("The device hasn't confirmed yet. The change is saved on the server; check the device screen in a minute.")
    }
}
