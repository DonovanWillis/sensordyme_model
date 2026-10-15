import Foundation

/// Reads the device's status payload and decides what it means for setup.
///
/// Pulled out of the Bluetooth delegate as a plain function because getting
/// this wrong is not visible: the app jumps to a screen, and whether that
/// screen is correct depends on a distinction the payload makes and the reader
/// has to respect.
///
/// The distinction: `state` is where the device is *now*, `provision` is what
/// happened to the credentials you sent. A device already on Wi-Fi reports
/// `state: "online"` continuously and truthfully — so reading that as an
/// outcome means the app declares setup complete the moment it connects over
/// Bluetooth, having never asked for a network or a password. That is exactly
/// what it did: straight from "Connect SensorDyme" to "SensorDyme is ready",
/// skipping the entire point of the screen.
enum ProvisioningStatus {
    enum Outcome: Equatable {
        /// Nothing to report yet — keep showing whatever the flow is doing.
        case none
        /// The device has our credentials and is trying them.
        case joining
        /// The device joined the network we asked for.
        case joined(ip: String?)
        case failed(String)
    }

    /// What one status payload says, read on its own.
    ///
    /// - Parameters:
    ///   - status: the parsed status characteristic.
    ///   - awaitingOutcome: whether credentials have been sent and not yet
    ///     resolved. When false, NOTHING in the payload is an outcome: the
    ///     device is just describing itself.
    ///
    /// `provision` is authoritative whenever the firmware sends the key at all —
    /// including as `null`, which means "no outcome yet". Falling back to
    /// `state` in that case read a device that was still `online` on its OLD
    /// network (the reconfigure window) as having joined the new one, with the
    /// old address. `state` is consulted only for firmware that predates the
    /// key entirely.
    static func interpret(_ status: [String: Any], awaitingOutcome: Bool) -> Outcome {
        guard awaitingOutcome else { return .none }

        let phase: String?
        let ip: String?
        let error: String?
        if status.keys.contains("provision") {
            guard let provision = status["provision"] as? [String: Any] else { return .none }
            phase = provision["state"] as? String
            ip = provision["ip"] as? String
            error = provision["error"] as? String
        } else {
            phase = status["state"] as? String
            ip = status["ip"] as? String
            error = status["error"] as? String
        }

        switch phase {
        case "connecting":
            return .joining
        case "ok", "online":
            return .joined(ip: ip)
        case "error":
            return .failed(error ?? "the device could not join that network")
        default:
            return .none
        }
    }

    /// Whether this payload settles the question, so polling can stop.
    static func isTerminal(_ outcome: Outcome) -> Bool {
        switch outcome {
        case .joined, .failed: true
        case .none, .joining: false
        }
    }

    /// The `provision.attempt` counter, which the device increments on every
    /// credentials write. Absent from firmware that predates it.
    static func attempt(in status: [String: Any]) -> Int? {
        (status["provision"] as? [String: Any])?["attempt"] as? Int
    }

    /// The device's SD- id as the status payload reports it.
    static func deviceID(in status: [String: Any]) -> String? {
        guard let id = status["device_id"] as? String, id.uppercased().hasPrefix("SD-") else { return nil }
        return id
    }

    /// Ties a status payload to THE credentials write it answers.
    ///
    /// A status that was already on its way when the app wrote — a scan-finished
    /// notification, say — describes the device before the write: an `online`
    /// device still on its old network, or the outcome of the PREVIOUS attempt
    /// ("that password was not accepted", when the user has just typed the
    /// right one). Read as this attempt's answer, it ended setup with a success
    /// that never happened or a failure that was already fixed.
    ///
    /// With attempt numbers: an outcome is accepted when its attempt is newer
    /// than any seen before the write — the number alone proves it answers a
    /// later write. "Connecting" need not have been seen: a fast join whose
    /// connecting notification was missed, or fell between two 2-second polls,
    /// would otherwise sit out the 90-second timeout and report a failure that
    /// did not happen. Without attempt numbers (older firmware): only after the
    /// write was acknowledged — the device records the new attempt before it
    /// answers the write — and only for the network that was sent.
    struct AttemptTracker: Equatable {
        /// Highest attempt number seen before the write; -1 for none.
        private(set) var baseline = -1
        /// The network the outstanding write named; nil when nothing is outstanding.
        private(set) var sentSSID: String?
        private(set) var writeAcknowledged = false
        /// The attempt `connecting` was seen for after the write.
        private(set) var connectingAttempt: Int?

        var awaitingOutcome: Bool { sentSSID != nil }

        /// A status read while nothing is outstanding: it sets the baseline.
        mutating func observeIdle(_ status: [String: Any]) {
            if let attempt = ProvisioningStatus.attempt(in: status) { baseline = max(baseline, attempt) }
        }

        mutating func beginWrite(ssid: String) {
            sentSSID = ssid
            writeAcknowledged = false
            connectingAttempt = nil
        }

        mutating func acknowledgeWrite() {
            if sentSSID != nil { writeAcknowledged = true }
        }

        /// Nothing outstanding any more (resolved, abandoned or disconnected).
        /// The baseline is kept: a later write must still beat it.
        mutating func finish() {
            if let connectingAttempt { baseline = max(baseline, connectingAttempt) }
            sentSSID = nil
            writeAcknowledged = false
            connectingAttempt = nil
        }

        /// What `status` means for the outstanding write.
        mutating func outcome(for status: [String: Any]) -> Outcome {
            guard let sent = sentSSID else {
                observeIdle(status)
                return .none
            }
            let reading = ProvisioningStatus.interpret(status, awaitingOutcome: true)

            if let attempt = ProvisioningStatus.attempt(in: status) {
                guard attempt > baseline else { return .none }  // from before this write
                switch reading {
                case .joining:
                    connectingAttempt = max(connectingAttempt ?? attempt, attempt)
                    return .joining
                case .joined, .failed:
                    // Older than an attempt already seen connecting: superseded.
                    if let connecting = connectingAttempt, attempt < connecting { return .none }
                    return reading
                case .none:
                    return .none
                }
            }

            // Firmware without attempt numbers.
            guard writeAcknowledged else { return .none }
            if let provision = status["provision"] as? [String: Any],
               let ssid = provision["ssid"] as? String, ssid != sent {
                return .none
            }
            return reading
        }
    }
}
