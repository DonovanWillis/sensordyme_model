import Foundation

/// Which exact model produced an event, as the backend stores it per event
/// (cloud/app.py's `_model_identity`). Foundation-only so it is testable outside
/// Xcode (ios/Tests/EventModelTests.swift) — and so is `EventItem` below, which
/// is what the test decodes: one row of `GET /v1/devices/{id}/events`.
struct EventModel: Codable, Hashable {
    struct Detector: Codable, Hashable {
        /// The file that actually executed: the .hef on the accelerator, not the
        /// .pt name that was only a cache key.
        let name: String
        /// sha256 of the model file as loaded; identifies it exactly.
        let sha256: String?
        /// "hailo" or "ultralytics": what executed it.
        let backend: String?
    }

    /// The vision module, e.g. "driver_attention".
    let module: String?
    /// Nil for a module that runs no object detector.
    let detector: Detector?

    /// "damoyolo_tinynasL20_T.hef · e69136d3ebf3" — short enough for a row.
    var summary: String? {
        guard let detector else { return module }
        guard let sha = detector.sha256, !sha.isEmpty else { return detector.name }
        return "\(detector.name) · \(sha.prefix(12))"
    }
}

/// One event, as `GET /v1/devices/{id}/events` returns it.
///
/// Here rather than in Models.swift so ios/Tests/EventModelTests.swift decodes
/// the real type against the backend's real row shape. The severity reading,
/// which needs the design system, is an extension in Models.swift.
struct EventItem: Codable, Identifiable, Hashable {
    /// Server row id. Optional so an older backend that omits it still decodes —
    /// the app just can't ask why that event fired.
    let eventID: Int?
    let timestamp: String
    let type: String
    let confidence: Double?
    let signals: String?
    let severity: String?
    /// "custom_rule" when this is something the user explicitly configured.
    /// Optional so an older backend still decodes.
    let source: String?
    /// Which exact model produced this event. Nil for events recorded before
    /// model tracking, and from an older backend that does not send it.
    var model: EventModel? = nil

    enum CodingKeys: String, CodingKey {
        case eventID = "id"
        case timestamp, type, confidence, signals, severity, source, model
    }

    /// Whether the user asked to be told about this specifically.
    ///
    /// Severity alone cannot answer that: a configured rule defaults to
    /// "warning" -> "soft", which the notifier dropped — so the one event
    /// someone had asked for was the only one that never arrived.
    var isConfigured: Bool { source == "custom_rule" }

    var id: String { "\(eventID.map(String.init) ?? "")\(timestamp)\(type)\(signals ?? "")" }

    /// The device sends `datetime.isoformat()`, which drops the fraction when
    /// the microseconds happen to be zero; the fractional formatter rejects
    /// that, so whole seconds are tried as well.
    var date: Date? {
        ISO8601DateFormatter.flexible.date(from: timestamp)
            ?? ISO8601DateFormatter.wholeSeconds.date(from: timestamp)
    }

    /// "phone_use_while_distracted" -> "Phone use while distracted"
    var title: String {
        let words = type.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: ":", with: " in ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}

// `nonisolated(unsafe)`: ISO8601DateFormatter is documented thread-safe, and
// these are configured once and only read afterwards.
extension ISO8601DateFormatter {
    nonisolated(unsafe) static let flexible: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    nonisolated(unsafe) static let wholeSeconds: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()
}
