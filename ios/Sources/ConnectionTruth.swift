import Foundation

/// What the app may say about whether a device is online, decided from the
/// device's own telemetry and how fresh the reply carrying it is — never from
/// the fact that the backend answered.
///
/// `online` in `GET /v1/devices` is the backend's verdict at the moment it
/// built the reply ("a report in the last 30 s"). Before this, the app showed
/// that verdict for as long as it kept the reply:
///
///   * a refresh that never happened (the screen stopped polling while the app
///     was reconnecting to the backend, or was frozen in the background) left a
///     reply minutes old saying ONLINE;
///   * a reply replayed by something between the phone and the backend (a
///     cache, a proxy, a backend reading a stale replica) said ONLINE with a
///     `last_seen` minutes old — the app never looked at `last_seen`;
///   * `status.stale` could say "no report in 30 s" while the presence pill,
///     reading only `online`, said ONLINE beside a "Status unknown" pill;
///   * the device list's "N ONLINE" count ignored even the failed-refresh rule
///     the cards used;
///   * (M8.1) a device that kept polling while re-sending a frozen health
///     sample was ONLINE with "All clear": a successful poll proves the link,
///     not the monitoring. `status.health_stale` now makes it
///     `monitoringStale`: reachable, but nothing it says about monitoring is
///     current.
///
/// Every screen asks this, so the list, the card, the device screen and the
/// sheets cannot disagree about the same reply. Foundation only, so
/// ios/Tests/run.sh covers it.
enum ConnectionTruth {
    enum Presence: Equatable {
        /// The device reported within the live window, and the reply saying so
        /// is current.
        case online
        /// The reply claims the device is online (or is unsure), but the
        /// device's last report is older than the live window: the screen is
        /// showing old telemetry. `age` is how old, in seconds.
        case stale(age: TimeInterval)
        /// The backend says the device has not reported recently. `age` is the
        /// time since its last report; nil when it never has.
        case offline(age: TimeInterval?)
        /// M8.1: the device is reachable (it polled within the live window), but
        /// the health sample it sends has not been updated for `age` seconds
        /// (nil when the backend gave no age), so whether it is monitoring is
        /// unknown. Never online, never "All clear".
        case monitoringStale(age: TimeInterval?)
        /// The app cannot tell: the latest refresh failed, or the reply it has
        /// is too old to speak for now.
        case unknown(UnknownReason)

        var isOnline: Bool { self == .online }
    }

    enum UnknownReason: Equatable {
        /// The latest refresh failed (server unreachable or unreadable).
        case refreshFailed
        /// The last good reply is older than `replyMaxAge` and nothing newer
        /// has arrived (polling paused, a request still hanging).
        case notRefreshed(age: TimeInterval)
        /// Nothing has been loaded yet.
        case notLoaded
    }

    /// One reading of a device, as a reply delivered it.
    struct Reading: Equatable {
        /// The backend's `online` flag.
        var online: Bool
        /// `status.stale`; nil from a backend that predates the status contract.
        var stale: Bool?
        /// `last_seen`, seconds since 1970 on the backend's clock. nil = never.
        var lastSeen: Double?
        /// When the phone received the reply. nil for a value that was never
        /// fetched (a preview, a hand-built demo) — then only the flags count.
        var receivedAt: Date?
        /// M8.1 `status.link_stale`: no poll in the live window. nil from an
        /// older backend; then `stale` speaks for the link, as before.
        var linkStale: Bool? = nil
        /// M8.1 `status.health_stale`: polling, but the health evidence is old.
        var healthStale: Bool? = nil
        /// M8.1 `status.health_freshness.age_s` when the reply was built.
        var healthAge: Double? = nil
    }

    /// The backend's window (`_HEALTH_STALE_S` in cloud/app.py); DevicePresence
    /// uses the same value.
    static let liveWindow: TimeInterval = 30
    /// Allowance for the phone's clock disagreeing with the backend's before
    /// `last_seen` is taken to contradict `online`. Both are normally on NTP;
    /// this keeps a few seconds of skew from flickering a live device to stale.
    static let clockTolerance: TimeInterval = 30
    /// How long one reply may stand for "now". The screens poll every 4–10 s,
    /// so a current screen never gets near it; a screen that stopped polling
    /// does, and then stops claiming anything.
    static let replyMaxAge: TimeInterval = 30

    static func presence(_ reading: Reading?, lastRefreshFailed: Bool, now: Date) -> Presence {
        if lastRefreshFailed { return .unknown(.refreshFailed) }
        guard let reading else { return .unknown(.notLoaded) }
        if let receivedAt = reading.receivedAt {
            let replyAge = now.timeIntervalSince(receivedAt)
            if replyAge > replyMaxAge { return .unknown(.notRefreshed(age: replyAge)) }
        }
        // The age of the device's last report, by the phone's clock. Negative
        // (the phone is behind the backend) reads as "just now".
        let age = reading.lastSeen.map { max(0, now.timeIntervalSince1970 - $0) }
        guard reading.online else { return .offline(age: age) }
        let healthStale = reading.healthStale == true
        // An older backend has only `stale`, which then means the link.
        let linkStale = reading.linkStale ?? (reading.stale == true && !healthStale)
        if linkStale { return .stale(age: age ?? liveWindow) }
        if let age, age > liveWindow + clockTolerance { return .stale(age: age) }
        if healthStale {
            // The backend's age plus how long the phone has held the reply.
            let held = reading.receivedAt.map { max(0, now.timeIntervalSince($0)) } ?? 0
            return .monitoringStale(age: reading.healthAge.map { max(0, $0) + held })
        }
        if reading.stale == true { return .stale(age: age ?? liveWindow) }
        return .online
    }

    // MARK: - Words

    /// The pill's word.
    static func label(_ presence: Presence) -> String {
        switch presence {
        case .online: "Online"
        case .stale: "Stale"
        case .monitoringStale: "Stale"
        case .offline: "Offline"
        case .unknown: "Unknown"
        }
    }

    /// One line for the device card under the name when the device is not
    /// simply online (nil when it is, and the card names the mode instead).
    static func cardLine(_ presence: Presence) -> String? {
        switch presence {
        case .online:
            return nil
        case .stale(let age):
            return "No report for \(DevicePresence.duration(age)) — showing its last report"
        case .monitoringStale(let age?):
            return "Connected — monitoring status not updated for \(DevicePresence.duration(age))"
        case .monitoringStale(nil):
            return "Connected — monitoring status not confirmed"
        case .offline(let age?):
            return "Last seen \(DevicePresence.duration(age)) ago"
        case .offline(nil):
            return "Not seen yet"
        case .unknown(.notRefreshed(let age)):
            return "Not updated for \(DevicePresence.duration(age))"
        case .unknown:
            return nil  // the banner above says why; the card keeps the mode line
        }
    }

    /// The spoken presence for VoiceOver: the word plus what it means.
    static func spoken(_ presence: Presence) -> String {
        switch presence {
        case .online: "Online"
        case .stale(let age): "Stale data, no report for \(DevicePresence.duration(age))"
        case .monitoringStale(let age?):
            "Connected, monitoring status not updated for \(DevicePresence.duration(age))"
        case .monitoringStale(nil): "Connected, monitoring status not confirmed"
        case .offline(let age?): "Offline, last seen \(DevicePresence.duration(age)) ago"
        case .offline(nil): "Offline, never reported"
        case .unknown(.refreshFailed): "Unknown, the last refresh failed"
        case .unknown(.notRefreshed(let age)): "Unknown, not updated for \(DevicePresence.duration(age))"
        case .unknown(.notLoaded): "Unknown"
        }
    }

    /// How many devices to call online in a summary. Same rule as each card.
    static func onlineCount(_ readings: [Reading], lastRefreshFailed: Bool, now: Date) -> Int {
        readings.filter { presence($0, lastRefreshFailed: lastRefreshFailed, now: now).isOnline }.count
    }

    /// The device-screen banner when the server answers but the telemetry is
    /// not live: what is shown is old, and changes wait for the device.
    static func notReportingMessage(_ presence: Presence) -> String? {
        switch presence {
        case .online, .unknown(.refreshFailed), .unknown(.notLoaded):
            return nil
        case .stale(let age):
            return "Its last report is \(DevicePresence.duration(age)) old, so what you see here may no longer "
                + "be true. Check it has power and Wi-Fi. Changes you make are saved and apply when it reports again."
        case .monitoringStale(let age):
            let since = age.map { " for \(DevicePresence.duration($0))" } ?? ""
            return "It is connected, but its monitoring status hasn't been updated\(since), so whether it is "
                + "watching is unknown. Any alert it sends is still delivered. If this lasts, restart the device."
        case .offline(let age?):
            return "Its status is unknown — last heard from \(DevicePresence.duration(age)) ago. Check it has "
                + "power and Wi-Fi. Changes you make are saved and apply when it reconnects."
        case .offline(nil):
            return "Its status is unknown — it hasn't reported yet. Check it has power and Wi-Fi. Changes you "
                + "make are saved and apply when it reconnects."
        case .unknown(.notRefreshed(let age)):
            return "This screen hasn't been updated for \(DevicePresence.duration(age)), so the status shown may be "
                + "out of date. Pull down to refresh."
        }
    }
}
