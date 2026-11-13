import Foundation

// The device's shared perception engine and monitoring nodes, as the backend
// forwards them in `status.perception` (docs/M7_IMPLEMENTATION_MAP.md, "Wire
// contract", schema sd-perception/1), and how the screens put them into words.
// Foundation only, so ios/Tests/PerceptionStatusTests.swift can pin it down.
//
// Everything here is device truth: what the device says its components and
// nodes are doing. Nothing missing is ever shown as working, and nothing
// experimental is ever shown as validated.

// MARK: - Wire

/// `status.perception`. Decoded leniently: an older backend sends nothing, a
/// newer device may send fields this build does not know.
struct PerceptionReport: Decodable, Hashable {
    var schema: String?
    var simulated: Bool?
    var components: [String: PerceptionComponent]
    var nodes: [MonitoringNode]

    init(schema: String? = "sd-perception/1", simulated: Bool? = nil,
         components: [String: PerceptionComponent] = [:], nodes: [MonitoringNode] = []) {
        self.schema = schema
        self.simulated = simulated
        self.components = components
        self.nodes = nodes
    }

    enum CodingKeys: String, CodingKey { case schema, simulated, components, nodes }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schema = try? c.decodeIfPresent(String.self, forKey: .schema)
        simulated = try? c.decodeIfPresent(Bool.self, forKey: .simulated)
        // One malformed component or node must not drop the others.
        var components: [String: PerceptionComponent] = [:]
        if let raw = try? c.decodeIfPresent([String: Lenient<PerceptionComponent>].self, forKey: .components) {
            for (name, value) in raw { if let value = value.value { components[name] = value } }
        }
        self.components = components
        let rawNodes = (try? c.decodeIfPresent([Lenient<MonitoringNode>].self, forKey: .nodes)) ?? []
        nodes = rawNodes.compactMap(\.value)
    }

    func node(_ id: String) -> MonitoringNode? { nodes.first { $0.id == id } }
}

/// Decodes T or nil, never throws: one bad element is skipped, not fatal.
private struct Lenient<T: Decodable>: Decodable {
    let value: T?
    init(from decoder: Decoder) throws { value = try? T(from: decoder) }
}

/// One inference component: object_detector, face_landmarks, hand_landmarks, pose.
struct PerceptionComponent: Decodable, Hashable {
    /// ok | stale | failed | unavailable | idle (anything else: unknown)
    var state: String?
    var model: ModelRef?
    var lastRunAgeS: Double?
    var runs: Int?
    var errors: Int?
    var reason: String?
    var consumers: [String]?

    struct ModelRef: Decodable, Hashable {
        var name: String?
        var sha256: String?
    }

    enum CodingKeys: String, CodingKey {
        case state, model, runs, errors, reason, consumers
        case lastRunAgeS = "last_run_age_s"
    }

    init(state: String?, model: ModelRef? = nil, lastRunAgeS: Double? = nil, runs: Int? = nil,
         errors: Int? = nil, reason: String? = nil, consumers: [String]? = nil) {
        self.state = state
        self.model = model
        self.lastRunAgeS = lastRunAgeS
        self.runs = runs
        self.errors = errors
        self.reason = reason
        self.consumers = consumers
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        state = try? c.decodeIfPresent(String.self, forKey: .state)
        model = try? c.decodeIfPresent(ModelRef.self, forKey: .model)
        lastRunAgeS = try? c.decodeIfPresent(Double.self, forKey: .lastRunAgeS)
        runs = try? c.decodeIfPresent(Int.self, forKey: .runs)
        errors = try? c.decodeIfPresent(Int.self, forKey: .errors)
        reason = try? c.decodeIfPresent(String.self, forKey: .reason)
        consumers = try? c.decodeIfPresent([String].self, forKey: .consumers)
    }
}

/// One monitoring node the device runs (or refuses to): a preset, a companion
/// node such as `occupancy`, or the custom-rules node.
struct MonitoringNode: Decodable, Hashable, Identifiable {
    var id: String
    /// preset | companion | rules
    var kind: String?
    var version: String?
    /// active | degraded | inactive | failed
    var state: String?
    var requiredCapabilities: [String]
    var missingCapabilities: [String]
    /// A model-registry tier (exists_in_code ... production_permitted).
    var validation: String?
    var experimental: Bool
    var reason: String?
    /// Experimental parts of a node (baby monitoring's `flailing`,
    /// `choking_hazard`), as the device reports them (M8).
    var experimentalFeatures: [ExperimentalFeature] = []

    struct ExperimentalFeature: Decodable, Hashable {
        var id: String
        var status: String?
        var enabled: Bool?
    }

    enum CodingKeys: String, CodingKey {
        case id, kind, version, state, validation, experimental, reason
        case requiredCapabilities = "required_capabilities"
        case missingCapabilities = "missing_capabilities"
        case experimentalFeatures = "experimental_features"
    }

    init(id: String, kind: String? = nil, version: String? = nil, state: String? = nil,
         requiredCapabilities: [String] = [], missingCapabilities: [String] = [],
         validation: String? = nil, experimental: Bool = false, reason: String? = nil) {
        self.id = id
        self.kind = kind
        self.version = version
        self.state = state
        self.requiredCapabilities = requiredCapabilities
        self.missingCapabilities = missingCapabilities
        self.validation = validation
        self.experimental = experimental
        self.reason = reason
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard let id = try? c.decode(String.self, forKey: .id), !id.isEmpty else {
            throw DecodingError.dataCorrupted(.init(codingPath: decoder.codingPath, debugDescription: "node without id"))
        }
        self.id = id
        kind = try? c.decodeIfPresent(String.self, forKey: .kind)
        version = try? c.decodeIfPresent(String.self, forKey: .version)
        state = try? c.decodeIfPresent(String.self, forKey: .state)
        requiredCapabilities = (try? c.decodeIfPresent([String].self, forKey: .requiredCapabilities)) ?? []
        missingCapabilities = (try? c.decodeIfPresent([String].self, forKey: .missingCapabilities)) ?? []
        validation = try? c.decodeIfPresent(String.self, forKey: .validation)
        // Missing means "not said"; only an explicit false clears the badge
        // on a node this build already knows is experimental.
        experimental = (try? c.decodeIfPresent(Bool.self, forKey: .experimental)) ?? false
        reason = try? c.decodeIfPresent(String.self, forKey: .reason)
        // Lossy: one malformed feature drops itself, not the node.
        let features = (try? c.decodeIfPresent([LossyFeature].self, forKey: .experimentalFeatures)) ?? []
        experimentalFeatures = features.compactMap(\.value)
    }

    private struct LossyFeature: Decodable {
        let value: ExperimentalFeature?
        init(from decoder: Decoder) throws {
            value = try? ExperimentalFeature(from: decoder)
        }
    }

    /// Running (possibly degraded): what "enabled" means for a toggle.
    var isRunning: Bool { state == "active" || state == "degraded" }
}

// MARK: - Components in words

struct PerceptionComponentRow: Equatable, Identifiable {
    let name: String
    let title: String
    let detail: String
    let tone: StatusTone

    var id: String { name }
}

enum PerceptionPresentation {
    /// The capability names the app shows, and the component that provides each.
    static let componentForCapability: [String: String] = [
        "objects": "object_detector", "face": "face_landmarks", "hands": "hand_landmarks", "pose": "pose",
    ]

    static func componentTitle(_ name: String) -> String {
        switch name {
        case "object_detector": "Object detection"
        case "face_landmarks": "Face tracking"
        case "hand_landmarks": "Hand tracking"
        case "pose": "Body pose"
        default: CapabilityHealth.title(name)
        }
    }

    /// The component rows for the health card, worst first. Empty when the
    /// device isn't reporting: an old "working" is never repeated.
    /// The rows for a device's status (M8.3): as below, and while monitoring is
    /// stopped a component still reported working (a device before M8.2) reads
    /// paused -- nothing is analysed.
    static func componentRows(status: DeviceStatus?, live: Bool) -> [PerceptionComponentRow] {
        let stopped = status?.monitoringState == "stopped" || status?.protection == "stopped"
        return componentRows(status?.perception, live: live).map { row in
            guard stopped, row.tone == .ok || row.tone == .caution else { return row }
            return .init(name: row.name, title: row.title, detail: CapabilityHealth.pausedDetail, tone: .neutral)
        }
    }

    static func componentRows(_ report: PerceptionReport?, live: Bool) -> [PerceptionComponentRow] {
        guard live, let report else { return [] }
        let order = ["object_detector", "face_landmarks", "hand_landmarks", "pose"]
        let names = report.components.keys.sorted { a, b in
            let ia = order.firstIndex(of: a) ?? order.count, ib = order.firstIndex(of: b) ?? order.count
            return ia == ib ? a < b : ia < ib
        }
        let rows = names.map { name -> PerceptionComponentRow in
            let component = report.components[name]!
            let (detail, tone) = describe(component)
            return .init(name: name, title: componentTitle(name), detail: detail, tone: tone)
        }
        return rows.sorted { severity($0.tone) > severity($1.tone) }
    }

    private static func severity(_ tone: StatusTone) -> Int {
        switch tone {
        case .critical: 3
        case .caution: 2
        case .neutral: 1
        case .ok: 0
        }
    }

    static func describe(_ component: PerceptionComponent) -> (String, StatusTone) {
        let reason = component.reason.flatMap { $0.isEmpty ? nil : $0 }
        switch component.state {
        case "ok":
            return ("Working", .ok)
        case "idle":
            return ("Idle — not needed by what's running now", .neutral)
        case "paused":
            return (CapabilityHealth.pausedDetail, .neutral)
        case "stale":
            let age = component.lastRunAgeS.map { " for \(Int($0.rounded())) s" } ?? ""
            return ("No fresh results\(age)" + (reason.map { " — \($0)" } ?? ""), .caution)
        case "failed":
            return ("Not working — \(reason ?? "the device reported a failure")", .critical)
        case "unavailable":
            return ("Not available on this device" + (reason.map { " — \($0)" } ?? ""), .critical)
        default:
            return ("State unknown", .neutral)
        }
    }

    /// The components that are not working, as one sentence for a banner, or nil.
    static func degradedSummary(_ report: PerceptionReport?, live: Bool) -> String? {
        let bad = componentRows(report, live: live).filter { $0.tone == .critical || $0.tone == .caution }
        guard !bad.isEmpty else { return nil }
        let list = bad.map { "\($0.title): \($0.detail.prefix(1).lowercased() + $0.detail.dropFirst())" }
        return "Detection is impaired — " + list.joined(separator: "; ")
    }
}

// MARK: - What a preset needs, checked against the device

struct CapabilityCheck: Equatable, Identifiable {
    enum Verdict: Equatable {
        case available
        case unavailable(String)
        case unknown(String)
        /// Being brought up (e.g. attention calibrating): not a failure.
        case starting(String)
        /// Not running because monitoring was stopped on purpose (M8.2): not a failure.
        case paused(String)
    }

    let name: String
    let title: String
    let required: Bool
    let verdict: Verdict

    var id: String { name }
}

enum CapabilityChecks {
    /// Each required and optional capability of a preset, judged from what
    /// the DEVICE reports. Unknown is said as unknown, never as available.
    static func make(required: [String], optional: [String], status: DeviceStatus?, online: Bool,
                     node: MonitoringNode? = nil) -> [CapabilityCheck] {
        let live = status?.isLive(online: online) ?? false
        let report = live ? status?.perception : nil
        let caps = live ? status?.capabilities : nil
        func check(_ name: String, required: Bool) -> CapabilityCheck {
            let title = name == "objects" ? "Objects" : CapabilityHealth.title(name)
            guard live else {
                return .init(name: name, title: title, required: required,
                             verdict: .unknown("the device isn't reporting"))
            }
            if let node, node.missingCapabilities.contains(name) {
                return .init(name: name, title: title, required: required,
                             verdict: .unavailable(node.reason.flatMap { $0.isEmpty ? nil : $0 }
                                                   ?? "missing on this device"))
            }
            if caps?.paused?.contains(name) == true
                || (status?.monitoringState == "stopped" && CapabilityHealth.analysisCapabilities.contains(name)) {
                return .init(name: name, title: title, required: required,
                             verdict: .paused(CapabilityHealth.pausedDetail))
            }
            // M8.3: the module's own impairment wins over a healthy component (as
            // the backend's capability_states does): the model may be fine while
            // the module cannot use it. And nothing is analysed without frames
            // or a running inference loop.
            if let impaired = caps?.impaired?.first(where: { $0.name == name }) {
                return .init(name: name, title: title, required: required,
                             verdict: .unavailable(impaired.reason.flatMap { $0.isEmpty ? nil : $0 } ?? "not working"))
            }
            if CapabilityHealth.analysisCapabilities.contains(name),
               let down = caps?.impaired?.first(where: { $0.name == "camera" || $0.name == "inference" }) {
                return .init(name: name, title: title, required: required,
                             verdict: .unavailable("no analysis while \(down.name) is unavailable"))
            }
            if let componentName = PerceptionPresentation.componentForCapability[name],
               let component = report?.components[componentName] {
                switch component.state {
                case "ok", "idle":
                    return .init(name: name, title: title, required: required, verdict: .available)
                case "paused":
                    return .init(name: name, title: title, required: required,
                                 verdict: .paused(CapabilityHealth.pausedDetail))
                default:
                    let (detail, _) = PerceptionPresentation.describe(component)
                    return .init(name: name, title: title, required: required, verdict: .unavailable(detail))
                }
            }
            if let impaired = caps?.impaired?.first(where: { $0.name == name }) {
                return .init(name: name, title: title, required: required,
                             verdict: .unavailable(impaired.reason.flatMap { $0.isEmpty ? nil : $0 } ?? "not working"))
            }
            if caps?.available?.contains(name) == true {
                return .init(name: name, title: title, required: required, verdict: .available)
            }
            if caps?.starting?.contains(name) == true {
                return .init(name: name, title: title, required: required,
                             verdict: .starting(CapabilityHealth.startingDetail(name)))
            }
            return .init(name: name, title: title, required: required,
                         verdict: .unknown("not reported by the device while it runs another mode"))
        }
        return required.map { check($0, required: true) }
            + optional.filter { !required.contains($0) }.map { check($0, required: false) }
    }

    /// The required capabilities the device says it lacks, as a sentence, or nil.
    static func blockingSummary(_ checks: [CapabilityCheck]) -> String? {
        let missing = checks.filter {
            if $0.required, case .unavailable = $0.verdict { return true }
            return false
        }
        guard !missing.isEmpty else { return nil }
        return "Needs " + missing.map(\.title).joined(separator: ", ") + ", which the device reports as not working."
    }
}

// MARK: - Did the device take it?

/// A refusal the DEVICE (or the backend on its behalf) recorded for a request.
enum DeviceRefusal {
    /// Slack for clock differences between phone, backend and device.
    static let clockSlack: TimeInterval = 60

    /// Why the device refused `base` (a preset base name), or nil. Only refusals
    /// recorded after `since` (minus slack) count, so an old refusal does not
    /// answer a new request.
    static func forPreset(_ base: String, status: DeviceStatus?, since: Date? = nil) -> String? {
        let cutoff = since.map { $0.timeIntervalSince1970 - clockSlack }
        func recent(_ at: Double?) -> Bool {
            guard let cutoff else { return true }
            return (at ?? cutoff) >= cutoff
        }
        if let rejection = status?.lastRejection, rejection.kind == "preset",
           baseName(rejection.subject) == base, recent(rejection.at) {
            return RefusalText.message(code: rejection.code, reason: rejection.reasons?.first(where: { !$0.isEmpty }),
                                       fallback: "The device refused this mode.")
        }
        if let rejection = status?.presetRejection, baseName(rejection.preset) == base, recent(rejection.at) {
            return rejection.reason.flatMap { $0.isEmpty ? nil : $0 } ?? "The device refused this mode."
        }
        return nil
    }

    /// Why the device refused a change to node `id`, or nil.
    static func forNode(_ id: String, status: DeviceStatus?, since: Date? = nil) -> String? {
        guard let rejection = status?.lastRejection, rejection.kind == "nodes",
              rejection.subject == nil || rejection.subject == id else { return nil }
        if let since, let at = rejection.at, at < since.timeIntervalSince1970 - clockSlack { return nil }
        return RefusalText.message(code: rejection.code, reason: rejection.reasons?.first(where: { !$0.isEmpty }),
                                   fallback: "The device refused this change.")
    }

    private static func baseName(_ raw: String?) -> String? {
        raw?.split(separator: ":").first.map(String.init)
    }
}

/// Where an activation stands, judged only from what the device reports.
struct ActivationStatus: Equatable {
    enum Phase: Equatable { case waiting, confirmed, notMonitoring, refused }

    let phase: Phase
    let text: String
    let tone: StatusTone

    /// - Parameters:
    ///   - base: the preset base name that was requested.
    ///   - reportedPreset: the device-reported `active_preset` (may carry a variant).
    ///   - customReplacing: custom rules replace the modes (AI Custom).
    ///   - settingsConfirmed: the device reports the settings that were sent.
    static func evaluate(base: String, deviceName: String, title: String, reportedPreset: String?,
                         customReplacing: Bool, status: DeviceStatus?, online: Bool,
                         settingsConfirmed: Bool = true, since: Date? = nil) -> ActivationStatus {
        if let reason = DeviceRefusal.forPreset(base, status: status, since: since) {
            return .init(phase: .refused, text: "\(deviceName) refused \(title): \(reason)", tone: .critical)
        }
        let runningBase = reportedPreset?.split(separator: ":").first.map(String.init)
        guard runningBase == base, !customReplacing, settingsConfirmed else {
            return .init(phase: .waiting, text: "Sent. Waiting for \(deviceName) to confirm…", tone: .neutral)
        }
        // The device runs it. Whether it is actually WATCHING is the monitoring state.
        let presentation = MonitoringStatusPresentation.make(status: status, online: online)
        switch presentation?.phase {
        case .ready?, nil:
            return .init(phase: .confirmed, text: "\(deviceName) is running \(title) and monitoring normally.", tone: .ok)
        case .degraded?:
            return .init(phase: .confirmed,
                         text: "\(deviceName) is running \(title), degraded — \(reasonText(presentation))", tone: .caution)
        case .starting?:
            return .init(phase: .waiting, text: "\(deviceName) switched to \(title) and is starting…", tone: .neutral)
        case .stopped?:
            return .init(phase: .notMonitoring,
                         text: "\(title) is selected, but monitoring is stopped on \(deviceName). Resume it under Manage.",
                         tone: .caution)
        case .failed?:
            return .init(phase: .notMonitoring,
                         text: "\(title) is selected, but \(deviceName) is not monitoring — \(reasonText(presentation))",
                         tone: .critical)
        case .unknown?:
            return .init(phase: .waiting, text: "Waiting for \(deviceName) to report…", tone: .neutral)
        }
    }

    private static func reasonText(_ presentation: MonitoringStatusPresentation?) -> String {
        guard let detail = presentation?.detail, let dash = detail.range(of: " — ") else {
            return "no reason given"
        }
        return String(detail[dash.upperBound...])
    }
}

// MARK: - Coded refusals in words

/// A refusal's machine code (M8: last_rejection.code, and the backend's coded
/// errors) as a sentence, keeping the device's own reason when it reads as
/// one. "no companion node 'x'; known: a, b" is for developers, not people.
enum RefusalText {
    static func head(_ code: String?) -> String? {
        switch code {
        case "role_forbids_preset": "This device's role doesn't allow that mode"
        case "unknown_node": "This device doesn't have that monitoring add-on"
        case "invalid_setting": "A setting was refused"
        case "profile_fixed": "This device's monitoring profile is fixed, so it can't be changed from the app"
        case "firmware_unsupported": "This device's software can't do that yet — update it"
        default: nil
        }
    }

    static func message(code: String?, reason: String?, fallback: String) -> String {
        let readable = reason.flatMap(PresetAvailability.readableReason)
        guard let head = head(code) else { return readable ?? reason ?? fallback }
        return readable.map { "\(head): \($0)" } ?? head + "."
    }
}

// MARK: - Nodes in words

struct NodePresentation: Equatable {
    let title: String
    let stateText: String
    let tone: StatusTone
    /// "Experimental", or the validation chip. Never says "validated" for an
    /// experimental node or for a tier short of device verification.
    let badge: String

    static func title(_ id: String) -> String {
        switch id {
        case "driver_safety": "Driver Safety"
        case "occupancy": "Occupancy"
        case "people_counter": "People counter"
        case "rules": "Custom rules"
        default: CapabilityHealth.title(id)
        }
    }

    static func make(_ node: MonitoringNode) -> NodePresentation {
        let reason = node.reason.flatMap { $0.isEmpty ? nil : $0 }
        let missing = node.missingCapabilities.map(CapabilityHealth.title).joined(separator: ", ")
        let (stateText, tone): (String, StatusTone) = {
            switch node.state {
            case "active": return ("Running", .ok)
            case "degraded":
                return ("Degraded — " + (reason ?? (missing.isEmpty ? "partly working" : "missing \(missing)")), .caution)
            case "failed":
                return ("Not running — " + (reason ?? (missing.isEmpty ? "failed" : "missing \(missing)")), .critical)
            case "inactive": return ("Off", .neutral)
            default: return ("State unknown", .neutral)
            }
        }()
        return .init(title: title(node.id), stateText: stateText, tone: tone, badge: badge(node))
    }

    /// The node's experimental parts, said as experimental, with whether each
    /// is on: "Experimental: Flailing (off), Choking hazard (off)". nil if none.
    static func experimentalNote(_ node: MonitoringNode) -> String? {
        guard !node.experimentalFeatures.isEmpty else { return nil }
        let parts = node.experimentalFeatures.map { feature -> String in
            let name = CapabilityHealth.title(feature.id)
            switch feature.enabled {
            case true?: return "\(name) (on)"
            case false?: return "\(name) (off)"
            case nil: return name
            }
        }
        return "Experimental: " + parts.joined(separator: ", ")
    }

    static func badge(_ node: MonitoringNode) -> String {
        if node.experimental { return "Experimental" }
        switch CapabilityTier(raw: node.validation) {
        case .deviceVerified, .productionPermitted: return "Verified on device"
        case .videoEvidence: return "Not yet validated · video tests"
        case .imageLevelBenchmark: return "Not yet validated · image benchmark"
        case .syntheticTests: return "Not yet validated · simulated tests"
        case .existsInCode: return "Not yet validated"
        case .unknown: return "Validation unknown"
        }
    }
}

/// The companion node toggle: what the device reports, not what was tapped.
enum CompanionNodeToggle {
    /// Whether the device reports `id` running. nil when it isn't reporting nodes.
    static func isOn(_ id: String, report: PerceptionReport?, live: Bool) -> Bool? {
        guard live, let report else { return nil }
        return report.node(id)?.isRunning ?? false
    }

    /// A request is confirmed once the device reports the wanted state.
    static func confirmed(_ id: String, wanted: Bool, report: PerceptionReport?, live: Bool) -> Bool {
        isOn(id, report: report, live: live) == wanted
    }
}
