import Foundation

/// Whether the iPhone plays its alert sound for one new alert, and the rules
/// that decide it. Foundation-only, so ios/Tests/AlertSoundTests.swift runs it.
///
/// **What the sound is, and what it is not.** The device's own buzzer is the
/// safety alert: it sounds in the cabin whether or not a phone exists, is
/// connected, or is awake (services/runtime.py alerts before it records or
/// uploads anything). The iPhone sound is a convenience on top of that, made
/// only with supported iOS APIs: the app posts a local notification with a
/// sound (UNNotificationSound.default) and, while the app is in the foreground,
/// asks iOS to present it with `.sound` (AlertNotifier's `willPresent`).
///
/// iOS decides whether that sound is heard. Silent Mode, Focus, the
/// notification settings, and a locked or sleeping phone can all silence it,
/// and nothing the app may use overrides them (bypassing Silent Mode needs
/// Apple's Critical Alerts entitlement, which this app does not have). The app
/// posts these notifications itself, so it has to be running to post them: it
/// polls only while it is in the foreground (AlertWatcher), and with the app
/// closed, suspended or the phone disconnected nothing from this app arrives.
/// Delivery while closed would need a push from the server through APNs, which
/// this app does not use (the optional ntfy path is a separate app's push).
///
/// **The rules.**
///   - Only an alert sounds: never an experimental observation, never a
///     "log"-severity record, never a type the device logs but never alerts on
///     (`logOnlyTypes`). Which events are alerts at all, and the 20-second
///     cooldown, are NotificationPolicy's; this decides only the sound.
///   - Only something happening now: an event the backend received a minute or
///     more after it happened (a backlog after an outage) is shown silently.
///     Events that were already there when the app started watching are never
///     announced at all (NotificationPolicy.freshEvents, `baselined`).
///   - Once per event: `SoundedEvents` remembers the identity of every event
///     that sounded, so the same event never sounds twice.
///   - The person can turn it off ("Alert sound on this iPhone", default on).
///   - A simulated event that sounds says so in the notification itself.
enum AlertSound {
    /// The validated driver-safety alerts: the driver_safety role's mandatory,
    /// loud alert types (core/roles.py `ROLES["driver_safety"]
    /// .mandatory_alert_types`). tests/test_ios_alert_sound_parity.py keeps
    /// this mirror in step with the device.
    static let driverSafetyAlertTypes: Set<String> = [
        "attention_lost", "distraction", "drowsiness", "eyes_down", "eyes_up", "phone_use",
    ]

    /// Types the device records but never alerts on (main.py: "Anything else
    /// (e.g. "occupancy_change", "phone_detected") is still written to the CSV
    /// log but never alerts"; jupiter's "vlm_judgment" is log-only too). A
    /// visible phone is not phone use. Checked here as well as by severity so a
    /// misconfigured severity can never make one of these sound.
    static let logOnlyTypes: Set<String> = ["occupancy_change", "phone_detected", "vlm_judgment"]

    /// UserDefaults key for "Alert sound on this iPhone". Absent means on.
    static let enabledKey = "alertSoundOnThisPhone"

    enum Decision: Equatable {
        case play
        case silent(Reason)

        enum Reason: Equatable {
            /// The person turned "Alert sound on this iPhone" off.
            case turnedOff
            /// Not an alert: experimental, "log" severity, or a log-only type.
            case notAnAlert
            /// It happened a minute or more before the backend received it.
            case deliveredLate
            /// This event already sounded once.
            case alreadySounded
        }
    }

    /// "eyes_down" for "eyes_down", and for a zone-scoped "eyes_down:cab".
    static func baseType(_ type: String) -> String {
        type.split(separator: ":", maxSplits: 1).first.map(String.init) ?? type
    }

    static func isDriverSafetyAlert(type: String) -> Bool {
        driverSafetyAlertTypes.contains(baseType(type))
    }

    /// The sound decision for one event NotificationPolicy has already decided
    /// to notify about.
    static func decide(
        type: String,
        severity: String?,
        experimental: Bool,
        deliveredLate: Bool,
        alreadySounded: Bool,
        soundEnabled: Bool
    ) -> Decision {
        if !soundEnabled { return .silent(.turnedOff) }
        if experimental || severity == "log" || logOnlyTypes.contains(baseType(type)) {
            return .silent(.notAnAlert)
        }
        if alreadySounded { return .silent(.alreadySounded) }
        if deliveredLate { return .silent(.deliveredLate) }
        return .play
    }

    /// One event's identity for "sound once": the device's own event key when
    /// it sent one (the same occurrence keeps it across retries and restarts),
    /// else the server row id. Nil when the event has neither, and then it
    /// never sounds: an event that can't be told apart from its repeats can't
    /// be sounded exactly once.
    static func identity(eventKey: String?, eventID: Int?) -> String? {
        if let eventKey, !eventKey.isEmpty { return "key:\(eventKey)" }
        if let eventID { return "id:\(eventID)" }
        return nil
    }

    /// The notification's headline. A simulated event says so first, so a
    /// demo or the virtual device's alert can never be taken for a real one.
    static func title(type: String, critical: Bool, simulated: Bool) -> String {
        let base = isDriverSafetyAlert(type: type) ? "Driver alert"
            : (critical ? "SensorDyme Alert" : "SensorDyme")
        return simulated ? "Simulated · \(base)" : base
    }
}

/// The identities of the events that have sounded on this phone, newest last,
/// bounded so it cannot grow for ever. Persisted by AlertNotifier.
struct SoundedEvents: Equatable {
    static let capacity = 500
    private(set) var identities: [String]

    init(_ identities: [String] = []) {
        self.identities = Array(identities.suffix(Self.capacity))
    }

    func contains(_ identity: String) -> Bool { identities.contains(identity) }

    /// Records `identity`. Returns false (and changes nothing) when it was
    /// already there: that event has sounded before.
    @discardableResult
    mutating func insert(_ identity: String) -> Bool {
        if identities.contains(identity) { return false }
        identities.append(identity)
        if identities.count > Self.capacity { identities.removeFirst(identities.count - Self.capacity) }
        return true
    }
}
