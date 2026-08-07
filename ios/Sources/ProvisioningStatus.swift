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

    /// - Parameters:
    ///   - status: the parsed status characteristic.
    ///   - awaitingOutcome: whether credentials have been sent and not yet
    ///     resolved. When false, NOTHING in the payload is an outcome: the
    ///     device is just describing itself.
    static func interpret(_ status: [String: Any], awaitingOutcome: Bool) -> Outcome {
        guard awaitingOutcome else { return .none }

        let provision = status["provision"] as? [String: Any]
        // `provision` is authoritative. `state` is only consulted for firmware
        // that predates the split — and only here, where we know a credential
        // write is outstanding, so "online" cannot be mistaken for the device's
        // pre-existing connection.
        let phase = (provision?["state"] as? String) ?? (status["state"] as? String)

        switch phase {
        case "connecting":
            return .joining
        case "ok", "online":
            return .joined(ip: (provision?["ip"] ?? status["ip"]) as? String)
        case "error":
            let reason = (provision?["error"] ?? status["error"]) as? String
            return .failed(reason ?? "the device could not join that network")
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
}
