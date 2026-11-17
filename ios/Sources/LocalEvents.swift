import Foundation

/// The camera's own event history, read over the local link
/// (`GET /api/events`, schema `sd-local-events/1`, core/local_events.py), and
/// how it joins the account's history and the alert sound.
///
/// Foundation-only (with EventModel.swift and AlertSound.swift), so
/// ios/Tests/LocalEventsTests.swift runs it.
struct LocalEvent: Decodable, Equatable {
    let seq: Int
    /// The device's `Event.event_id`: the SAME key the backend stores as
    /// `event_key`, which is what lets the two histories merge with no duplicates.
    let eventKey: String
    let type: String
    let timestamp: String
    var severity: String?
    var confidence: Double?
    var signals: String?
    var duration: Double?
    var experimental: Bool?
    var simulated: Bool?
    var alert: EventAlertReport?
    /// Seconds since it happened, on the device's monotonic clock; nil for an
    /// event from an earlier power-on (the device cannot say how long ago).
    var ageS: Double?
    /// "pending" (not yet delivered to the account), "sent", "local_only" (no
    /// account connection configured) or "unknown".
    var sync: String?

    enum CodingKeys: String, CodingKey {
        case seq, type, timestamp, severity, confidence, signals, duration, experimental, simulated, alert, sync
        case eventKey = "event_key"
        case ageS = "age_s"
    }

    init(seq: Int, eventKey: String, type: String, timestamp: String, severity: String? = nil,
         confidence: Double? = nil, signals: String? = nil, duration: Double? = nil, experimental: Bool? = nil,
         simulated: Bool? = nil, alert: EventAlertReport? = nil, ageS: Double? = nil, sync: String? = nil) {
        self.seq = seq
        self.eventKey = eventKey
        self.type = type
        self.timestamp = timestamp
        self.severity = severity
        self.confidence = confidence
        self.signals = signals
        self.duration = duration
        self.experimental = experimental
        self.simulated = simulated
        self.alert = alert
        self.ageS = ageS
        self.sync = sync
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        seq = try c.decode(Int.self, forKey: .seq)
        eventKey = try c.decode(String.self, forKey: .eventKey)
        type = try c.decode(String.self, forKey: .type)
        timestamp = (try? c.decode(String.self, forKey: .timestamp)) ?? ""
        severity = try? c.decodeIfPresent(String.self, forKey: .severity)
        confidence = try? c.decodeIfPresent(Double.self, forKey: .confidence)
        signals = try? c.decodeIfPresent(String.self, forKey: .signals)
        duration = try? c.decodeIfPresent(Double.self, forKey: .duration)
        experimental = try? c.decodeIfPresent(Bool.self, forKey: .experimental)
        simulated = try? c.decodeIfPresent(Bool.self, forKey: .simulated)
        // A report this build cannot read is absent, never a failed page.
        alert = try? c.decodeIfPresent(EventAlertReport.self, forKey: .alert)
        ageS = try? c.decodeIfPresent(Double.self, forKey: .ageS)
        sync = try? c.decodeIfPresent(String.self, forKey: .sync)
    }
}

struct LocalEventsPage: Decodable, Equatable {
    var schema: String?
    var events: [LocalEvent]
    var latestSeq: Int
    var bootID: String?
    var fault: String?

    enum CodingKeys: String, CodingKey {
        case schema, events, fault
        case latestSeq = "latest_seq"
        case bootID = "boot_id"
    }
}

enum LocalHistory {
    /// One local row as the app's event type: the same severity reading, detail
    /// sheet and alert-evidence wording as an account event. `age_s` places it
    /// on the PHONE's clock (fetched-at minus age), which is right whatever the
    /// camera's wall clock says; with no age the camera's own timestamp is all
    /// there is, and the row says so (`timeNote`).
    static func eventItem(_ local: LocalEvent, fetchedAt: Date) -> EventItem {
        var item = EventItem(eventID: nil, timestamp: local.timestamp, type: local.type,
                             confidence: local.confidence, signals: local.signals, severity: local.severity,
                             source: nil)
        item.eventKey = local.eventKey
        item.simulated = local.simulated
        item.experimental = local.experimental
        item.duration = local.duration
        item.alert = local.alert
        if let age = local.ageS {
            item.occurredAt = fetchedAt.timeIntervalSince1970 - age
            item.timeBasis = "device_monotonic"
        }
        return item
    }

    /// What a local row says about its backup, in the person's terms. Nil for a
    /// row the account already has (it reads exactly as before).
    static func syncLabel(_ sync: String?) -> String {
        switch sync {
        case "pending": "Not yet backed up"
        case "local_only": "On this camera only"
        case "sent": "Sent to your account"
        default: "Backup status unknown"
        }
    }

    static let cameraClockNote = "Time from the camera's clock (an earlier power-on)"

    struct Merged: Equatable {
        var events: [EventItem]
        /// Per `EventItem.id`, a line for rows that came only from the camera.
        var notes: [String: String]
    }

    /// The account's history and the camera's, one row per occurrence.
    ///
    /// An account row and a camera row with the same `event_key` are ONE event:
    /// the account row is kept (it has the server id that "why did this fire"
    /// needs, and it is backed up by definition) and the camera row dropped.
    /// Camera-only rows carry their sync state and, without an age, the
    /// camera-clock note. Newest first (EventFeed.ordered).
    static func merge(account: [EventItem], local: [LocalEvent], fetchedAt: Date) -> Merged {
        let accountKeys = Set(account.compactMap(\.eventKey).filter { !$0.isEmpty })
        var notes: [String: String] = [:]
        var localItems: [EventItem] = []
        var seen = Set<String>()
        for event in local where !accountKeys.contains(event.eventKey) && seen.insert(event.eventKey).inserted {
            let item = eventItem(event, fetchedAt: fetchedAt)
            localItems.append(item)
            var note = syncLabel(event.sync)
            if event.ageS == nil { note += " · " + cameraClockNote }
            notes[item.id] = note
        }
        return Merged(events: EventFeed.merge(account, localItems), notes: notes)
    }
}

/// Foreground alerts while only the local link is up (docs/LOCAL_ACCESS.md §7).
/// The same rules as the account path (AlertSound, SoundedEvents), so an event
/// sounds at most once whichever way it arrived.
enum LocalAlerts {
    /// An event older than this (by the device's own `age_s`) is history, not news.
    static let freshFor: TimeInterval = 60

    /// The events since the last poll, and the cursor to keep.
    ///
    /// The first poll of a watch (no cursor) only records where the journal
    /// ends: everything already there is backlog, and is never announced. A
    /// journal that went backwards (replaced, or a reset camera) re-baselines
    /// the same way rather than replaying it.
    static func advance(cursor: Int?, page: LocalEventsPage) -> (fresh: [LocalEvent], cursor: Int) {
        guard let cursor else { return ([], page.latestSeq) }
        if page.latestSeq < cursor { return ([], page.latestSeq) }
        let fresh = page.events.filter { $0.seq > cursor }.sorted { $0.seq < $1.seq }
        return (fresh, max(cursor, fresh.last?.seq ?? cursor))
    }

    /// Late by the device's monotonic clock, or from an earlier power-on (no age).
    static func isLate(_ event: LocalEvent) -> Bool {
        guard let age = event.ageS else { return true }
        return age >= freshFor
    }

    /// The sound decision for one local event NotificationPolicy decided to
    /// announce: exactly AlertSound.decide (a simulated event never sounds on
    /// either path), with the same identity (`key:<event_key>`) the account path uses.
    static func soundDecision(_ event: LocalEvent, sounded: SoundedEvents, soundEnabled: Bool) -> AlertSound.Decision {
        let identity = AlertSound.identity(eventKey: event.eventKey, eventID: nil)
        return AlertSound.decide(type: event.type, severity: event.severity,
                                 experimental: event.experimental == true, deliveredLate: isLate(event),
                                 alreadySounded: identity.map { sounded.contains($0) } ?? true,
                                 soundEnabled: soundEnabled, simulated: event.simulated == true)
    }
}
