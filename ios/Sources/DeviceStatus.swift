import Foundation

// The backend's typed answer to "is this device watching right now, and for
// what?" (docs/APP_STATUS_CONTRACT.md), and how the screens put it into words.
// Foundation only, so ios/Tests/DeviceStatusTests.swift can pin it down.
//
// The app used to show a green "All clear" while the device was FAILED or
// STOPPED: it read none of the device's monitoring-truth fields. Everything a
// screen says about monitoring now comes from here.

/// `status` on each device in `GET /v1/devices`. Every field is optional and
/// decoded leniently: one odd value must never fail the decode of the whole
/// device list, which is what the app polls continuously.
struct DeviceStatus: Codable, Hashable {
    var monitoringState: String?
    var monitoringActive: Bool?
    var protection: String?
    var reasons: [String]?
    var protectionReasons: [String]?
    /// off | standard | ai_custom | standard_plus_custom
    var mode: String?
    var role: String?
    var roleTitle: String?
    /// Preset base names this device's role allows. nil = every preset.
    var allowedPresets: [String]?
    var capabilities: Capabilities?
    var rules: Rules?
    var lastRejection: Rejection?
    var presetRejection: PresetRejection?
    var stale: Bool?
    var reportedAt: Double?

    struct Capabilities: Codable, Hashable {
        var available: [String]?
        var impaired: [Impaired]?
        var starting: [String]?
    }

    struct Impaired: Codable, Hashable {
        var name: String
        var reason: String?
    }

    struct Rules: Codable, Hashable {
        /// Rule ids the DEVICE reports running. nil = not known (stale).
        var active: [String]?
        /// Whether custom rules run beside the mode on this device's role.
        var additive: Bool?
        /// The device refused the rule set the backend wants it to run.
        var rejected: RulesRejection?
    }

    struct RulesRejection: Codable, Hashable {
        var version: String?
        var reasons: [String]?
        var at: Double?
    }

    struct Rejection: Codable, Hashable {
        var kind: String?
        var subject: String?
        var reasons: [String]?
        var at: Double?
    }

    struct PresetRejection: Codable, Hashable {
        var preset: String?
        var code: String?
        var reason: String?
        var at: Double?
    }

    enum CodingKeys: String, CodingKey {
        case protection, reasons, mode, role, capabilities, rules, stale
        case monitoringState = "monitoring_state"
        case monitoringActive = "monitoring_active"
        case protectionReasons = "protection_reasons"
        case roleTitle = "role_title"
        case allowedPresets = "allowed_presets"
        case lastRejection = "last_rejection"
        case presetRejection = "preset_rejection"
        case reportedAt = "reported_at"
    }

    init(
        monitoringState: String? = nil, monitoringActive: Bool? = nil, protection: String? = nil,
        reasons: [String]? = nil, protectionReasons: [String]? = nil, mode: String? = nil,
        role: String? = nil, roleTitle: String? = nil, allowedPresets: [String]? = nil,
        capabilities: Capabilities? = nil, rules: Rules? = nil, lastRejection: Rejection? = nil,
        presetRejection: PresetRejection? = nil, stale: Bool? = nil, reportedAt: Double? = nil
    ) {
        self.monitoringState = monitoringState
        self.monitoringActive = monitoringActive
        self.protection = protection
        self.reasons = reasons
        self.protectionReasons = protectionReasons
        self.mode = mode
        self.role = role
        self.roleTitle = roleTitle
        self.allowedPresets = allowedPresets
        self.capabilities = capabilities
        self.rules = rules
        self.lastRejection = lastRejection
        self.presetRejection = presetRejection
        self.stale = stale
        self.reportedAt = reportedAt
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        monitoringState = try? c.decodeIfPresent(String.self, forKey: .monitoringState)
        monitoringActive = try? c.decodeIfPresent(Bool.self, forKey: .monitoringActive)
        protection = try? c.decodeIfPresent(String.self, forKey: .protection)
        reasons = try? c.decodeIfPresent([String].self, forKey: .reasons)
        protectionReasons = try? c.decodeIfPresent([String].self, forKey: .protectionReasons)
        mode = try? c.decodeIfPresent(String.self, forKey: .mode)
        role = try? c.decodeIfPresent(String.self, forKey: .role)
        roleTitle = try? c.decodeIfPresent(String.self, forKey: .roleTitle)
        allowedPresets = try? c.decodeIfPresent([String].self, forKey: .allowedPresets)
        capabilities = try? c.decodeIfPresent(Capabilities.self, forKey: .capabilities)
        rules = try? c.decodeIfPresent(Rules.self, forKey: .rules)
        lastRejection = try? c.decodeIfPresent(Rejection.self, forKey: .lastRejection)
        presetRejection = try? c.decodeIfPresent(PresetRejection.self, forKey: .presetRejection)
        stale = try? c.decodeIfPresent(Bool.self, forKey: .stale)
        reportedAt = try? c.decodeIfPresent(Double.self, forKey: .reportedAt)
    }
}

/// How alarming a status is, independent of the design system (the views map
/// it onto `SDStatus`), so this file stays Foundation-only.
enum StatusTone: Equatable {
    case ok, caution, critical, neutral

    /// The `HealthComponent.state` vocabulary the health card already renders.
    var healthState: String {
        switch self {
        case .ok: "ok"
        case .caution: "warning"
        case .critical: "failed"
        case .neutral: "unknown"
        }
    }
}

/// What custom rules are doing on the device right now.
enum CustomRulesRole: Equatable {
    /// No custom rules run.
    case none
    /// Custom rules run INSTEAD of the monitoring mode (AI Custom).
    case replacing
    /// Custom rules run BESIDE the monitoring mode, which keeps running.
    case alongside
}

/// The monitoring state in the person's terms: a short label for pills, a
/// sentence for the health card, and a tone. Never green unless the device
/// itself reports `ready`.
struct MonitoringStatusPresentation: Equatable {
    enum Phase: String, Equatable {
        case starting, ready, degraded, stopped, failed, unknown
    }

    let phase: Phase
    let label: String
    let detail: String
    let tone: StatusTone

    /// nil when there is no truth to present: an online device behind a backend
    /// that predates the status contract. The caller keeps its old rendering.
    static func make(status: DeviceStatus?, online: Bool) -> MonitoringStatusPresentation? {
        guard let status else {
            return online ? nil : unknown("the device isn't reporting")
        }
        let reason = status.reasons?.first(where: { !$0.isEmpty })
        // `online` and `stale` use the same 30 s window on the backend; either
        // one alone is enough to stop trusting the last report.
        if !online || status.stale == true {
            return unknown(reason ?? "the device isn't reporting")
        }
        switch Phase(rawValue: status.monitoringState ?? "") ?? .unknown {
        case .ready:
            return .init(phase: .ready, label: "All clear", detail: "Monitoring normally", tone: .ok)
        case .degraded:
            return .init(phase: .degraded, label: "Degraded",
                         detail: sentence("Degraded", reason), tone: .caution)
        case .starting:
            return .init(phase: .starting, label: "Starting",
                         detail: sentence("Starting", reason), tone: .caution)
        case .stopped:
            return .init(phase: .stopped, label: "Stopped",
                         detail: sentence("Stopped", reason ?? "nothing is monitored until it's resumed"),
                         tone: .caution)
        case .failed:
            return .init(phase: .failed, label: "Not monitoring",
                         detail: sentence("Not monitoring", reason), tone: .critical)
        case .unknown:
            return unknown(reason)
        }
    }

    private static func unknown(_ reason: String?) -> MonitoringStatusPresentation {
        .init(phase: .unknown, label: "Status unknown", detail: sentence("Status unknown", reason), tone: .neutral)
    }

    private static func sentence(_ head: String, _ reason: String?) -> String {
        guard let reason, !reason.isEmpty else { return head }
        return "\(head) — \(reason)"
    }

    /// The status pill under the camera. A recent alert wins; otherwise the
    /// device's monitoring state. "All clear" with no status at all is the
    /// pre-contract behaviour, kept only for an online device on an old backend.
    static func pill(_ presentation: MonitoringStatusPresentation?, alertActive: Bool)
        -> (text: String, tone: StatusTone) {
        if alertActive { return ("Alert", .critical) }
        guard let presentation else { return ("All clear", .ok) }
        return (presentation.label, presentation.tone)
    }

    /// The device card's line under the name: the mode, prefixed with the state
    /// whenever the device is not simply watching normally.
    static func cardSubtitle(modeTitle: String, presentation: MonitoringStatusPresentation?) -> String {
        guard let presentation, presentation.phase != .ready else { return modeTitle }
        return "\(presentation.label) · \(modeTitle)"
    }

    /// What custom rules do now, from the mode the DEVICE reports. Falls back
    /// to the older `monitoring.active` flag only when no mode is reported.
    static func customRules(status: DeviceStatus?, legacyActive: Bool) -> CustomRulesRole {
        switch status?.mode {
        case "ai_custom": return .replacing
        case "standard_plus_custom": return .alongside
        case "standard", "off": return .none
        default:
            guard legacyActive else { return .none }
            return status?.rules?.additive == true ? .alongside : .replacing
        }
    }

    /// Whether this device runs custom rules beside its mode (additive role).
    static func rulesAdditive(_ status: DeviceStatus?) -> Bool {
        status?.rules?.additive == true || status?.mode == "standard_plus_custom"
    }

    /// Why the latest custom rules are NOT running, or nil. The device refused
    /// them (all or nothing) and kept what it was running before.
    static func rejectedRulesNotice(_ status: DeviceStatus?) -> String? {
        guard let rejected = status?.rules?.rejected else { return nil }
        let head = "Your latest rules aren't running — the device refused them"
        guard let reason = rejected.reasons?.first(where: { !$0.isEmpty }) else { return head + "." }
        return "\(head): \(reason)"
    }

    /// Why `preset` (a base name) cannot be selected on this device, or nil.
    /// When the backend dropped a request for exactly this preset, its reason
    /// is the one shown: that is the switch the person asked for.
    static func presetUnavailableReason(_ preset: String, status: DeviceStatus?) -> String? {
        guard let allowed = status?.allowedPresets, !allowed.contains(preset) else { return nil }
        if let rejection = status?.presetRejection,
           rejection.preset?.split(separator: ":").first.map(String.init) == preset,
           let reason = rejection.reason, !reason.isEmpty {
            return reason
        }
        let role = status?.roleTitle ?? status?.role
        return role.map { "Not available for this device's role (\($0))" }
            ?? "Not available on this device"
    }
}
