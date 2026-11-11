import Foundation

// MARK: - Backend JSON models (cloud/app.py)

struct AuthResponse: Codable {
    let token: String
    let email: String
}

struct Device: Codable, Identifiable, Hashable {
    let id: Int
    let name: String
    let activePreset: String?
    let desiredPreset: String?
    let online: Bool
    let lastSeen: Double?
    /// What the device last said about itself. `nil` when it has gone quiet —
    /// the backend withholds stale reports rather than let the app render a
    /// healthy camera for a Pi that was unplugged an hour ago.
    let health: DeviceHealthReport?
    /// The custom configuration the device says is in charge, if any.
    ///
    /// When this is active it — not `activePreset` — is what the device is
    /// really doing. Showing the base mode as selected while a custom
    /// configuration ran meant someone who asked it to count people was told it
    /// was watching for driver distraction.
    let monitoring: CustomMonitoring?
    /// The camera's own id (`SD-XXXXXXXX`, the same one it advertises over
    /// Bonjour). Nil from firmware or a backend that predates it. See FeedLink.
    var hardwareId: String? = nil
    /// The backend's typed answer to "is it watching, and for what"
    /// (DeviceStatus.swift, docs/APP_STATUS_CONTRACT.md). nil from a backend
    /// that predates it. Every monitoring indicator reads this, not `health`.
    var status: DeviceStatus? = nil

    enum CodingKeys: String, CodingKey {
        case id, name, online, health, monitoring, status
        case hardwareId = "hardware_id"
        case activePreset = "active_preset"
        case desiredPreset = "desired_preset"
        case lastSeen = "last_seen"
    }
}

/// A custom monitoring configuration, as the device reports it.
struct CustomMonitoring: Codable, Hashable {
    /// What the device understood it was asked to watch for. Shown verbatim,
    /// because it is the sentence the person will recognise.
    let instruction: String
    let ruleCount: Int
    /// Whether it has replaced the base mode rather than running beside it.
    let active: Bool

    enum CodingKeys: String, CodingKey {
        case instruction, active
        case ruleCount = "rule_count"
    }

    /// What to call it in the UI when there is no instruction recorded — an
    /// older device, or rules configured before instructions were carried.
    var title: String { "AI Custom Monitoring" }

    var summary: String {
        if !instruction.isEmpty { return instruction }
        return ruleCount == 1 ? "1 custom rule" : "\(ruleCount) custom rules"
    }
}

/// One subsystem as the device reported it.
struct HealthComponent: Codable, Hashable {
    let state: String        // ok | warning | failed | unknown
    let detail: String
    let metrics: [String: MetricValue]?

    var status: SDStatus {
        switch state {
        case "ok": .online
        case "warning": .warning
        case "failed": .alert
        default: .offline
        }
    }

    /// "unknown" is a real answer, not a failure — the device is saying it
    /// cannot tell, which the UI must not dress up as either good or bad.
    var isKnown: Bool { state != "unknown" }

    func metric(_ key: String) -> String? { metrics?[key]?.description }
}

/// Health metrics are heterogeneous (fps 12.0, ssid "Home", present true), so
/// they decode into a small union rather than forcing one Swift type.
enum MetricValue: Codable, Hashable, CustomStringConvertible {
    case number(Double)
    case text(String)
    case flag(Bool)

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        // Bool before Double: JSONDecoder will happily read `true` as 1.
        if let value = try? container.decode(Bool.self) { self = .flag(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .text(value) }
        else { self = .text("—") }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .number(let value): try container.encode(value)
        case .text(let value): try container.encode(value)
        case .flag(let value): try container.encode(value)
        }
    }

    var description: String {
        switch self {
        case .number(let value):
            value == value.rounded() ? String(Int(value)) : String(format: "%.1f", value)
        case .text(let value): value
        case .flag(let value): value ? "yes" : "no"
        }
    }
}

struct DeviceHealthReport: Codable, Hashable {
    let camera: HealthComponent?
    let inference: HealthComponent?
    let accelerator: HealthComponent?
    let network: HealthComponent?
    let system: HealthComponent?
    let cloudConnected: Bool?

    enum CodingKeys: String, CodingKey {
        case camera, inference, accelerator, network, system
        case cloudConnected = "cloud_connected"
    }

    /// The worst thing the device is currently reporting, for a one-glance summary.
    var worstStatus: SDStatus {
        let components = [camera, inference, accelerator, network, system].compactMap { $0 }
        if components.contains(where: { $0.state == "failed" }) { return .alert }
        if components.contains(where: { $0.state == "warning" }) { return .warning }
        return components.isEmpty ? .offline : .online
    }
}

struct DevicesResponse: Codable { let devices: [Device] }

extension StatusTone {
    /// The design system's status for a tone (DeviceStatus.swift stays
    /// Foundation-only, so the mapping lives here).
    var sdStatus: SDStatus {
        switch self {
        case .ok: .online
        case .caution: .warning
        case .critical: .alert
        case .neutral: .offline
        }
    }
}

extension Device {
    /// The monitoring state in words, from `status`. nil only for an online
    /// device behind a backend that predates the status contract.
    var monitoringPresentation: MonitoringStatusPresentation? {
        MonitoringStatusPresentation.make(status: status, online: online)
    }

    /// What custom rules do on the device right now: nothing, replace the
    /// mode (AI Custom), or run beside it (an additive role).
    var customRules: CustomRulesRole {
        MonitoringStatusPresentation.customRules(status: status, legacyActive: monitoring?.active == true)
    }
}

/// A named area drawn over the camera view, in normalised 0..1 coordinates so
/// one drawing survives any change of capture resolution or phone screen size.
struct MonitorZone: Codable, Identifiable, Hashable {
    var name: String
    var kind: String          // restricted | safe | boundary
    var polygon: [[Double]]   // [[x, y], …]

    var id: String { name }

    /// What the device does with it, in the user's terms rather than the schema's.
    var meaning: String {
        switch kind {
        case "safe": "Alert when someone leaves this area"
        case "boundary": "Alert when someone crosses this line"
        default: "Alert when someone enters this area"
        }
    }

    var points: [CGPoint] {
        polygon.compactMap { pair in
            pair.count >= 2 ? CGPoint(x: pair[0], y: pair[1]) : nil
        }
    }

    /// Three points is the minimum that encloses an area.
    var isUsable: Bool { points.count >= 3 }

    static func from(points: [CGPoint], name: String, kind: String) -> MonitorZone {
        MonitorZone(
            name: name, kind: kind,
            polygon: points.map { [Double($0.x), Double($0.y)] }
        )
    }
}

struct ZonesResponse: Codable { let zones: [MonitorZone] }

/// The severity reading needs the design system's `SDStatus`, so it lives here
/// rather than beside the Foundation-only `EventItem` (EventModel.swift).
extension EventItem {
    /// Matches jupiter/explain.py's severity mapping, so the app and the
    /// explanation endpoint never disagree about how alarming something is.
    var severityStatus: SDStatus {
        if ["fall", "on_floor", "safe_zone_exit", "zone_violation"].contains(type.split(separator: ":").first.map(String.init) ?? type) {
            return .alert
        }
        switch severity {
        case "loud": return .alert
        case "soft": return .warning
        default: return .online
        }
    }

    var isCritical: Bool { severityStatus == .alert }
}

/// One fact the device recorded, as returned by /explain.
struct ExplanationFact: Codable, Hashable, Identifiable {
    let label: String
    let detail: String

    var id: String { label + detail }
    var text: String { detail.isEmpty ? label : "\(label) — \(detail)" }
}

/// Why an alert fired, assembled from stored data on the server.
struct AlertExplanation: Codable, Hashable {
    let headline: String
    let severity: String
    let when: String
    let facts: [ExplanationFact]
    let rule: String
    let confidence: Double?
    /// "facts" when assembled deterministically, otherwise the provider name.
    /// Critical events are always "facts" — a model never rephrases those.
    let source: String
    let narrative: String

    var wasRephrased: Bool { source != "facts" }
}

struct EventsResponse: Codable { let events: [EventItem] }

// MARK: - Built-in preset list (OFFLINE / DEMO FALLBACK ONLY)
//
// The preset library comes from the backend (GET /v1/presets, PresetLibrary.swift).
// This list is used only when the server publishes none (an older backend, or
// the catalog failed to load), and the screens label it as the built-in list
// when they show it. It also supplies icons and the demo catalog's titles.
// tests/test_app_preset_parity.py parses `static let catalog:` below, so it
// must keep mirroring presets.yaml on the device.

struct PresetVariant: Identifiable, Hashable {
    let name: String
    let title: String
    var id: String { name }
}

struct Preset: Identifiable, Hashable {
    let name: String
    let title: String
    let subtitle: String
    let icon: String
    var variants: [PresetVariant] = []

    var id: String { name }

    static let catalog: [Preset] = [
        // One driver mode. There used to be two, differing only in which
        // inference engine answered the same question — which is not a decision
        // a consumer can make, and being asked to is itself a defect.
        Preset(name: "driver_safety", title: "Driver Safety",
               subtitle: "Eyes, gaze, phone use & drowsiness", icon: "car.fill",
               variants: [.init(name: "standard", title: "Standard"),
                          .init(name: "strict", title: "Strict"),
                          .init(name: "teen_driver", title: "Teen Driver"),
                          .init(name: "long_haul", title: "Long Haul")]),
        Preset(name: "baby_monitoring", title: "Baby Monitoring",
               subtitle: "Face covered, crying, movement", icon: "figure.child",
               // Must match presets.yaml. "newborn" and "toddler" were offered
               // here and exist on no device — tapping them queued a variant
               // the device could never apply, which the backend then re-sent
               // for ever. A test now pins this so it cannot drift again.
               variants: [.init(name: "standard", title: "Standard"),
                          .init(name: "nap_time", title: "Nap Time"),
                          .init(name: "play_area", title: "Play Area")]),
        Preset(name: "elderly_monitoring", title: "Elderly Monitoring",
               subtitle: "Fall detection & inactivity", icon: "figure.fall",
               variants: [.init(name: "standard", title: "Standard"),
                          .init(name: "post_surgery", title: "Post-Surgery"),
                          .init(name: "active_senior", title: "Active Senior")]),
        Preset(name: "security_monitoring", title: "Security",
               subtitle: "Person counting & changes", icon: "shield.fill",
               variants: [.init(name: "home", title: "Home"),
                          .init(name: "away", title: "Away"),
                          .init(name: "package_watch", title: "Package Watch")]),
    ]

    /// The built-in list as a catalog, labelled with why it is being shown.
    static func builtInCatalog(reason: String) -> PresetCatalog {
        PresetCatalog.builtIn(
            catalog.map { preset in
                (id: preset.name, title: preset.title, description: preset.subtitle,
                 variants: preset.variants.map { (id: $0.name, title: $0.title) })
            },
            reason: reason)
    }

    /// An icon for a preset id; server presets the app has no icon for get a
    /// neutral one.
    static func icon(for id: String) -> String {
        catalog.first(where: { $0.name == id })?.icon ?? "square.grid.2x2"
    }

    /// "driver_distraction:teen_driver" -> ("driver_distraction", "teen_driver")
    static func split(_ name: String?) -> (base: String, variant: String?) {
        let parsed = PresetName.parse(name)  // drops a `@settings` suffix
        return (parsed.base, parsed.variant)
    }

    static func title(for name: String?) -> String {
        let (base, variantName) = split(name)
        guard !base.isEmpty else { return "—" }
        guard let preset = catalog.first(where: { $0.name == base }) else { return base }
        if let variantName,
           let variant = preset.variants.first(where: { $0.name == variantName }),
           variant.name != "standard" {
            return "\(preset.title) · \(variant.title)"
        }
        return preset.title
    }
}

// MARK: - Orders

struct Order: Codable, Identifiable, Hashable {
    let id: Int
    let quantity: Int
    let address: String
    let status: String
    let createdAt: Double

    enum CodingKeys: String, CodingKey {
        case id, quantity, address, status
        case createdAt = "created_at"
    }
}

struct OrdersResponse: Codable { let orders: [Order] }
