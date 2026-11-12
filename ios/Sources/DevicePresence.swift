import Foundation

/// Why the live feed is unavailable, decided from BOTH things the app knows.
///
/// The app has two independent sources of truth about a device and, until this
/// existed, the one that could answer the question was ignored:
///
///   * **Bonjour on the local network** tells the app whether it can reach the
///     device *directly*, which is what the MJPEG live feed needs. `DeviceLink`
///     owns this.
///   * **The backend** tells the app whether the device exists on the account at
///     all and when it last reported in. `/v1/devices` has carried `online` and
///     `last_seen` all along.
///
/// `DeviceLink` only ever consulted the first, so silence from Bonjour always
/// became "Can't find your device on this network" — a sentence about the
/// phone's Wi-Fi. On a device that had not reported to the backend in nine days
/// that is a true statement of the wrong fact: it sends someone to re-check a
/// network that is working, when the backend already knew the camera had been
/// off for over a week.
///
/// The distinction that matters most is the one neither source can make alone:
/// a device the backend heard from seconds ago but Bonjour cannot see is a
/// genuine network split — alerts keep working, only the live feed is lost —
/// and that is completely different advice from "your camera is switched off".
///
/// Foundation only, so `ios/Tests/run.sh` can cover it.
enum DevicePresence {

    /// What the backend knows about the device, or nil if it knows of no such
    /// device on this account.
    struct BackendView: Equatable {
        /// When the device last reported in. Nil means it never has.
        let lastSeen: Date?
    }

    enum Verdict: Equatable {
        /// Reachable on this network; the feed works and there is nothing to say.
        case reachable
        /// Reporting to the backend right now, but not visible from this phone.
        /// The device is fine; the two are not on the same network.
        case aliveButElsewhere(since: TimeInterval)
        /// The backend knows it and has not heard from it for a while.
        case quiet(since: TimeInterval)
        /// Registered but has never reported in — almost always never set up.
        case neverReported
        /// No such device on this account.
        case notLinked
        /// The account has devices, but which of them this phone's camera is
        /// cannot be told — so nothing can be said about "your device".
        case ambiguous
        /// The backend could not be asked, so the honest answer is "don't know".
        case undetermined
    }

    /// How recently the backend must have heard from a device to call it live.
    ///
    /// Matches `_HEALTH_STALE_S` in cloud/app.py, which is also what the
    /// backend's own `online` flag uses. Kept equal deliberately: two different
    /// windows would let the device list say "online" while this said otherwise.
    static let liveWindow: TimeInterval = 30

    /// - Parameter accountHasDevices: whether the account has any devices at
    ///   all. With `backend` nil this separates "no device on the account" from
    ///   "several, and none is known to be this phone's camera".
    static func verdict(
        foundOnNetwork: Bool,
        backend: BackendView?,
        backendReachable: Bool,
        accountHasDevices: Bool = false,
        now: Date
    ) -> Verdict {
        if foundOnNetwork { return .reachable }
        // Bonjour is silent. What the backend says is now the only evidence
        // about whether the device itself is alive.
        guard backendReachable else { return .undetermined }
        guard let backend else { return accountHasDevices ? .ambiguous : .notLinked }
        guard let lastSeen = backend.lastSeen else { return .neverReported }
        let age = now.timeIntervalSince(lastSeen)
        // A negative age means the device's clock or ours is ahead. The Pi has no
        // RTC, so this happens for real; treat it as "just heard from" rather
        // than rendering a negative duration.
        if age < liveWindow { return .aliveButElsewhere(since: max(age, 0)) }
        return .quiet(since: age)
    }

    // MARK: - What to say

    static func title(_ verdict: Verdict) -> String {
        switch verdict {
        case .reachable: return ""
        case .aliveButElsewhere: return "Your device is running, but not on this Wi-Fi"
        case .quiet(let since): return "Your device hasn't reported in \(duration(since))"
        case .neverReported: return "Your device hasn't been set up yet"
        case .notLinked: return "No device linked to this account"
        case .ambiguous: return "Can't tell which of your devices this phone is linked to"
        case .undetermined: return "Can't check your device right now"
        }
    }

    static func detail(_ verdict: Verdict) -> String {
        switch verdict {
        case .reachable:
            return ""
        case .aliveButElsewhere:
            // The useful case: the camera is working and recording, and only the
            // live picture is missing. Saying "can't find your device" here would
            // suggest the camera had stopped, which is the opposite of the truth.
            return "It's reporting to your server normally, so alerts still work — "
                 + "the live picture needs the phone and the device on the same network."
        case .quiet:
            return "That usually means it's switched off or unplugged. Alerts and the "
                 + "live feed both need it powered on and on your Wi-Fi."
        case .neverReported:
            return "Stand near your SensorDyme and set it up over Bluetooth."
        case .notLinked:
            return "Add your SensorDyme with its claim code to see it here."
        case .ambiguous:
            return "Your account has more than one device. Choose this phone's camera in "
                 + "Settings → Live feed → Find my device."
        case .undetermined:
            return "ORYN EDGE can't reach your server at the moment, so it can't say "
                 + "whether your device is online. It will keep trying."
        }
    }

    /// Whether offering "Search again" makes sense.
    ///
    /// Searching again looks for the device on the LOCAL network, so it is worth
    /// offering only when that is the thing in doubt. For a device the backend
    /// says has been off for a week, another Bonjour scan cannot succeed, and a
    /// button that cannot work is worse than no button.
    static func offersLocalRetry(_ verdict: Verdict) -> Bool {
        switch verdict {
        case .aliveButElsewhere, .undetermined, .ambiguous: return true
        case .reachable, .quiet, .neverReported, .notLinked: return false
        }
    }

    /// A rounded, human duration. Deliberately coarse: "3 days" is as actionable
    /// as "3 days 4 hours" and reads better.
    static func duration(_ seconds: TimeInterval) -> String {
        let s = max(0, Int(seconds.rounded()))
        if s < 60 { return "\(s) second\(s == 1 ? "" : "s")" }
        let minutes = s / 60
        if minutes < 60 { return "\(minutes) minute\(minutes == 1 ? "" : "s")" }
        let hours = minutes / 60
        if hours < 24 { return "\(hours) hour\(hours == 1 ? "" : "s")" }
        let days = hours / 24
        return "\(days) day\(days == 1 ? "" : "s")"
    }
}
