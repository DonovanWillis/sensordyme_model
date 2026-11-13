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

    // M8.3 (docs/EVENT_PIPELINE.md). Every one optional: an older backend, or
    // history recorded before M8.3, decodes exactly as before.

    /// The device's own identity for this occurrence (core/events.py event_id).
    var eventKey: String? = nil
    /// When the backend received it (seconds since 1970, the backend's clock).
    var receivedAt: Double? = nil
    /// When it happened, placed on the backend's clock (see `timeBasis`).
    var occurredAt: Double? = nil
    /// How `occurredAt` was found: "device_monotonic" (the device's own age of
    /// the event, trustworthy whatever its wall clock says), "device_clock"
    /// (the device's timestamp, judged plausible) or "received" (the device's
    /// clock was implausible: this is the arrival time).
    var timeBasis: String? = nil
    /// What the device REPORTED its alert outputs did. Nil when it reported
    /// nothing -- which never means an alert sounded.
    var alert: EventAlertReport? = nil
    /// True for an event from a simulated device or detector, or the demo.
    var simulated: Bool? = nil

    enum CodingKeys: String, CodingKey {
        case eventID = "id"
        case timestamp, type, confidence, signals, severity, source, model
        case eventKey = "event_key"
        case receivedAt = "received_at"
        case occurredAt = "occurred_at"
        case timeBasis = "time_basis"
        case alert, simulated
    }

    /// Whether the user asked to be told about this specifically.
    ///
    /// Severity alone cannot answer that: a configured rule defaults to
    /// "warning" -> "soft", which the notifier dropped — so the one event
    /// someone had asked for was the only one that never arrived.
    var isConfigured: Bool { source == "custom_rule" }

    /// Stable across refreshes: the server row id when there is one (a newer
    /// alert report for the same row must not read as a different event).
    var id: String {
        if let eventID { return "id:\(eventID)" }
        if let eventKey, !eventKey.isEmpty { return "key:\(eventKey)" }
        return "\(timestamp)\(type)\(signals ?? "")"
    }

    /// When it happened, as truthfully as the record allows: on the backend's
    /// clock when the backend placed it (M8.3), else the device's own timestamp.
    /// Never the device's claim alone when something better exists: a Pi with
    /// no RTC can stamp an alert 1970, or three days ago.
    var date: Date? {
        if let occurredAt { return Date(timeIntervalSince1970: occurredAt) }
        return deviceDate
    }

    /// The device's own timestamp, verbatim. The device sends
    /// `datetime.isoformat()`, which drops the fraction when the microseconds
    /// happen to be zero; the fractional formatter rejects that, so whole
    /// seconds are tried as well.
    var deviceDate: Date? {
        ISO8601DateFormatter.flexible.date(from: timestamp)
            ?? ISO8601DateFormatter.wholeSeconds.date(from: timestamp)
    }

    var receivedDate: Date? { receivedAt.map { Date(timeIntervalSince1970: $0) } }

    var isSimulated: Bool { simulated == true }

    /// How far the device's clock was from when the event really happened, in
    /// seconds (positive: the device clock was ahead), when that was 2 minutes
    /// or more and the backend could tell -- i.e. not when the device clock was
    /// itself the evidence.
    var deviceClockError: TimeInterval? {
        guard let occurredAt, let deviceDate, timeBasis == "device_monotonic" || timeBasis == "received"
        else { return nil }
        let error = deviceDate.timeIntervalSince1970 - occurredAt
        return abs(error) >= 120 ? error : nil
    }

    /// How long after it happened the event reached the backend, when that was
    /// over a minute (an outage, a backlog sent after a restart).
    var deliveryDelay: TimeInterval? {
        guard let receivedAt, let occurredAt, timeBasis != "received" else { return nil }
        let delay = receivedAt - occurredAt
        return delay >= 60 ? delay : nil
    }

    /// The time shown is when the backend received it: the device's clock could not be used.
    var timeIsArrival: Bool { timeBasis == "received" }

    /// What the app may say about the device's own alert output for this event.
    var alertEvidence: AlertDeliveryEvidence { AlertDeliveryEvidence(alert, eventSimulated: isSimulated) }

    /// "phone_use_while_distracted" -> "Phone use while distracted"
    var title: String {
        let words = type.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: ":", with: " in ")
        return words.prefix(1).uppercased() + words.dropFirst()
    }
}

/// What the device reported its in-cabin alert outputs did for one event
/// (core/alerts.py AlertDelivery, validated by cloud/app.py `_alert_report`).
struct EventAlertReport: Codable, Hashable {
    struct Output: Codable, Hashable {
        let name: String?
        let kind: String?
        let audible: Bool?
        var played: Int? = nil
        var failed: Int? = nil
        var skipped: Int? = nil
        var pending: Int? = nil
        var unconfirmed: Int? = nil
        var simulated: Bool? = nil
        var error: String? = nil
    }

    /// "alerted" | "merged" | "suppressed": what the device's alert policy did.
    let decision: String?
    /// "sounded" | "simulated" | "pending" | "unconfirmed" | "failed" | "none".
    let output: String?
    /// Rises with every newer report from the device for this event.
    var rev: Int? = nil
    var episode: String? = nil
    var category: String? = nil
    var outputs: [Output]? = nil
}

/// The one place the app decides what it may say about the device's alert
/// output. Detection and warning are different facts: a stored event is a
/// detection, and "the alert sounded" is said ONLY when the device itself
/// reported that a real output played (M8.3). Everything else reads
/// "Detected; no alert output confirmed", with the reason when there is one.
enum AlertDeliveryEvidence: Equatable {
    /// A real output (buzzer, speaker) reported its whole warning pattern played.
    case sounded(outputs: [String])
    /// Only a simulated output recorded it. Nothing made a sound.
    case simulated
    /// Sent to the output; it has not reported back yet.
    case pending
    /// The output(s) reported failing or could not run; the first reason, if any.
    case failed(reason: String?)
    /// Sent to an output that cannot report whether it played.
    case unconfirmed
    /// The device reported no alert output for this event (or reported nothing).
    case none

    init(_ report: EventAlertReport?, eventSimulated: Bool) {
        guard let report, let output = report.output else { self = .none; return }
        let outputs = report.outputs ?? []
        switch output {
        case "sounded":
            // A simulated event cannot have sounded a real output, whatever it says.
            if eventSimulated { self = .simulated; return }
            let names = outputs.filter { ($0.played ?? 0) > 0 && $0.simulated != true }
                .map { AlertDeliveryEvidence.spoken($0) }
            self = .sounded(outputs: names)
        case "simulated": self = .simulated
        case "pending": self = .pending
        case "unconfirmed": self = .unconfirmed
        case "failed": self = .failed(reason: outputs.compactMap(\.error).first)
        default: self = .none
        }
    }

    /// True only for a real output that reported playing.
    var confirmedSound: Bool { if case .sounded = self { true } else { false } }

    var headline: String {
        switch self {
        case .sounded: "Alert sounded on the device"
        case .simulated: "Simulated alert (no sound)"
        default: "Detected; no alert output confirmed"
        }
    }

    var detail: String {
        switch self {
        case .sounded(let outputs):
            let what = outputs.isEmpty ? "its alert output" : "its " + outputs.joined(separator: " and ")
            return "The device reported that \(what) played the warning pattern. "
                + "It cannot measure whether anyone heard it."
        case .simulated:
            return "A simulated device recorded the alert. Nothing made a sound."
        case .pending:
            return "The device sent the warning to its alert output and hasn't reported yet whether it played."
        case .failed(let reason):
            let base = "The device tried to sound its alert, and its output didn't play it"
            return reason.map { "\(base): \($0)." } ?? "\(base)."
        case .unconfirmed:
            return "The device's alert output can't report whether it played."
        case .none:
            return "The device didn't report sounding an alert for this event."
        }
    }

    /// A word for a timeline row.
    var short: String {
        switch self {
        case .sounded: "Sounded on device"
        case .simulated: "Simulated, no sound"
        case .pending: "Awaiting device"
        case .failed: "Alert output failed"
        case .unconfirmed: "Not confirmed"
        case .none: "No alert output"
        }
    }

    private static func spoken(_ output: EventAlertReport.Output) -> String {
        switch output.kind {
        case "buzzer": "buzzer"
        case "speaker": "speaker"
        default: output.name ?? "alert output"
        }
    }
}

/// Merging what the timeline shows with a freshly fetched page (M8.3).
///
/// Refreshes overlap -- pull to refresh, the screen's own reload, a page loaded
/// while new events arrive -- and an alert's row can be re-fetched with a newer
/// delivery report. The timeline must never show an event twice, never lose
/// one it had because a smaller page arrived, never let an older response
/// replace a newer report, and always read in the order things happened.
enum EventFeed {
    /// Most events kept in memory for one timeline.
    static let capacity = 1000

    static func merge(_ current: [EventItem], _ incoming: [EventItem]) -> [EventItem] {
        var byID: [String: EventItem] = [:]
        for event in current { byID[event.id] = event }
        for event in incoming {
            if let kept = byID[event.id], (kept.alert?.rev ?? -1) > (event.alert?.rev ?? -1) {
                continue  // a stale response: keep the newer report already shown
            }
            byID[event.id] = event
        }
        return Array(ordered(Array(byID.values)).prefix(capacity))
    }

    /// Newest first by when it happened; the server row id breaks ties (and
    /// orders events with no usable time, after those with one).
    static func ordered(_ events: [EventItem]) -> [EventItem] {
        events.sorted { a, b in
            switch (a.date, b.date) {
            case let (x?, y?) where x != y: return x > y
            case (.some, .none): return true
            case (.none, .some): return false
            default: return (a.eventID ?? Int.min) > (b.eventID ?? Int.min)
            }
        }
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
