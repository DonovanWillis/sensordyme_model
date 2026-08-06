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

    enum CodingKeys: String, CodingKey {
        case id, name, online, health
        case activePreset = "active_preset"
        case desiredPreset = "desired_preset"
        case lastSeen = "last_seen"
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

struct EventItem: Codable, Identifiable, Hashable {
    let timestamp: String
    let type: String
    let confidence: Double?
    let signals: String?
    let severity: String?

    var id: String { timestamp + type + (signals ?? "") }

    var date: Date? { ISO8601DateFormatter.flexible.date(from: timestamp) }
}

struct EventsResponse: Codable { let events: [EventItem] }

// MARK: - Preset catalog (mirrors presets.yaml on the device)

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
        Preset(name: "driver_distraction", title: "Driver Distraction",
               subtitle: "Drowsiness, gaze, phone use", icon: "car.fill",
               variants: [.init(name: "standard", title: "Standard"),
                          .init(name: "strict", title: "Strict"),
                          .init(name: "teen_driver", title: "Teen Driver"),
                          .init(name: "long_haul", title: "Long Haul")]),
        Preset(name: "driver_distraction_openpilot", title: "Driver (openpilot)",
               subtitle: "comma.ai's monitoring model", icon: "steeringwheel",
               variants: [.init(name: "standard", title: "Standard"),
                          .init(name: "strict", title: "Strict")]),
        Preset(name: "baby_monitoring", title: "Baby Monitoring",
               subtitle: "Face covered, crying, movement", icon: "figure.child",
               variants: [.init(name: "newborn", title: "Newborn"),
                          .init(name: "toddler", title: "Toddler"),
                          .init(name: "nap_time", title: "Nap Time")]),
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

    /// "driver_distraction:teen_driver" -> ("driver_distraction", "teen_driver")
    static func split(_ name: String?) -> (base: String, variant: String?) {
        guard let name else { return ("", nil) }
        let parts = name.split(separator: ":", maxSplits: 1).map(String.init)
        return (parts.first ?? "", parts.count > 1 ? parts[1] : nil)
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

extension ISO8601DateFormatter {
    static let flexible: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
