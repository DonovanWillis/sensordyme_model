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

    enum CodingKeys: String, CodingKey {
        case id, name, online
        case activePreset = "active_preset"
        case desiredPreset = "desired_preset"
        case lastSeen = "last_seen"
    }
}

struct DevicesResponse: Codable { let devices: [Device] }

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
