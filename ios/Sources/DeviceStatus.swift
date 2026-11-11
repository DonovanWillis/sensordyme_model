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
    /// The id of the last stop/resume the DEVICE says it applied (echo of the
    /// `command_id` POST /v1/devices/{id}/protection returned). nil from a
    /// backend that does not pass it through; ProtectionControl then falls back
    /// to the protection state.
    var lastCommandID: String?
    /// The shared perception engine's components and the monitoring nodes the
    /// device runs (PerceptionStatus.swift). nil from an older backend or device.
    var perception: PerceptionReport?

    struct Capabilities: Codable, Hashable {
        var available: [String]?
        var impaired: [Impaired]?
        var starting: [String]?
        /// Per-capability evaluation status (`validated_on_device`, ...), when
        /// the backend reports one. Optional in the contract.
        var evaluation: [String: String]?

        init(available: [String]? = nil, impaired: [Impaired]? = nil, starting: [String]? = nil,
             evaluation: [String: String]? = nil) {
            self.available = available
            self.impaired = impaired
            self.starting = starting
            self.evaluation = evaluation
        }

        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            available = try? c.decodeIfPresent([String].self, forKey: .available)
            impaired = try? c.decodeIfPresent([Impaired].self, forKey: .impaired)
            starting = try? c.decodeIfPresent([String].self, forKey: .starting)
            evaluation = try? c.decodeIfPresent([String: String].self, forKey: .evaluation)
        }
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
        case protection, reasons, mode, role, capabilities, rules, stale, perception
        case monitoringState = "monitoring_state"
        case monitoringActive = "monitoring_active"
        case protectionReasons = "protection_reasons"
        case roleTitle = "role_title"
        case allowedPresets = "allowed_presets"
        case lastRejection = "last_rejection"
        case presetRejection = "preset_rejection"
        case reportedAt = "reported_at"
        case lastCommandID = "last_command_id"
    }

    init(
        monitoringState: String? = nil, monitoringActive: Bool? = nil, protection: String? = nil,
        reasons: [String]? = nil, protectionReasons: [String]? = nil, mode: String? = nil,
        role: String? = nil, roleTitle: String? = nil, allowedPresets: [String]? = nil,
        capabilities: Capabilities? = nil, rules: Rules? = nil, lastRejection: Rejection? = nil,
        presetRejection: PresetRejection? = nil, stale: Bool? = nil, reportedAt: Double? = nil,
        lastCommandID: String? = nil, perception: PerceptionReport? = nil
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
        self.lastCommandID = lastCommandID
        self.perception = perception
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
        lastCommandID = try? c.decodeIfPresent(String.self, forKey: .lastCommandID)
        perception = try? c.decodeIfPresent(PerceptionReport.self, forKey: .perception)
    }

    func encode(to encoder: Encoder) throws {
        // The app only reads status; encoding keeps Codable for the cache and
        // the demo, without the perception report (read-only device truth).
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encodeIfPresent(monitoringState, forKey: .monitoringState)
        try c.encodeIfPresent(monitoringActive, forKey: .monitoringActive)
        try c.encodeIfPresent(protection, forKey: .protection)
        try c.encodeIfPresent(reasons, forKey: .reasons)
        try c.encodeIfPresent(protectionReasons, forKey: .protectionReasons)
        try c.encodeIfPresent(mode, forKey: .mode)
        try c.encodeIfPresent(role, forKey: .role)
        try c.encodeIfPresent(roleTitle, forKey: .roleTitle)
        try c.encodeIfPresent(allowedPresets, forKey: .allowedPresets)
        try c.encodeIfPresent(capabilities, forKey: .capabilities)
        try c.encodeIfPresent(rules, forKey: .rules)
        try c.encodeIfPresent(lastRejection, forKey: .lastRejection)
        try c.encodeIfPresent(presetRejection, forKey: .presetRejection)
        try c.encodeIfPresent(stale, forKey: .stale)
        try c.encodeIfPresent(reportedAt, forKey: .reportedAt)
        try c.encodeIfPresent(lastCommandID, forKey: .lastCommandID)
    }

    /// Online and reporting: the only time anything below is current.
    func isLive(online: Bool) -> Bool { online && stale != true }
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


// MARK: - Capability health

/// One capability on the health card: what it is, whether it works, and how
/// far its results have been checked.
struct CapabilityHealthRow: Equatable, Identifiable {
    enum State: Equatable { case available, starting, impaired }

    let name: String
    let title: String
    let state: State
    let detail: String
    let tone: StatusTone

    var id: String { name }
}

enum CapabilityHealth {
    /// The rows, and a sentence for when there are none to trust. Never a
    /// green row for a device that is not reporting: `capabilities` is null
    /// when stale, and an old list is not repeated.
    static func rows(status: DeviceStatus?, online: Bool) -> (rows: [CapabilityHealthRow], notice: String?) {
        guard let status else {
            return ([], "This server doesn't report the device's capabilities.")
        }
        guard status.isLive(online: online), let capabilities = status.capabilities else {
            return ([], "Unknown — the device isn't reporting, so no capability is shown as working.")
        }
        let evaluation = capabilities.evaluation ?? [:]
        var rows: [CapabilityHealthRow] = []
        for impaired in capabilities.impaired ?? [] {
            let reason = impaired.reason.flatMap { $0.isEmpty ? nil : $0 } ?? "not working"
            rows.append(.init(name: impaired.name, title: title(impaired.name), state: .impaired,
                              detail: "Not working — \(reason)", tone: .critical))
        }
        for name in capabilities.starting ?? [] {
            rows.append(.init(name: name, title: title(name), state: .starting,
                              detail: "Starting", tone: .caution))
        }
        for name in capabilities.available ?? [] {
            var detail = "Producing results"
            if let note = evaluationNote(evaluation[name]) { detail += " · \(note)" }
            rows.append(.init(name: name, title: title(name), state: .available, detail: detail, tone: .ok))
        }
        if rows.isEmpty {
            return ([], "The device reports no capabilities.")
        }
        return (rows, nil)
    }

    /// The evaluation status in words, or nil when the backend gave none.
    static func evaluationNote(_ raw: String?) -> String? {
        switch raw {
        case nil, "": return nil
        case "validated_on_device": return "validated on device"
        case "evaluated_in_domain": return "evaluated, not validated on device"
        case "benchmarked_offline": return "benchmarked offline only"
        case "unevaluated": return "not yet evaluated"
        default: return "validation unknown"
        }
    }

    /// "phone_use" -> "Phone use"
    static func title(_ name: String) -> String {
        let spaced = name.replacingOccurrences(of: "_", with: " ").replacingOccurrences(of: ".", with: " ")
        return spaced.prefix(1).uppercased() + spaced.dropFirst()
    }
}
