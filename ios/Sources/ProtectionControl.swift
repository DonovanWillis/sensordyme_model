import Foundation

// Stop / resume monitoring (POST /v1/devices/{id}/protection) and how the app
// knows it actually happened. Foundation only, so
// ios/Tests/ProtectionControlTests.swift can pin it down.
//
// The backend answers with a `command_id` the moment it stores the request;
// that is not the device stopping. The device echoes the id
// (`last_command_id`) once it has applied it, and reports `protection:
// "stopped"`. Until one of those arrives the control says "Stopping…", never
// "Stopped".

enum ProtectionAction: String, Equatable, Sendable {
    case stop, resume

    var verb: String { self == .stop ? "Stop monitoring" : "Resume monitoring" }
    var progressive: String { self == .stop ? "Stopping" : "Resuming" }
}

/// The answer to POST /v1/devices/{id}/protection.
struct ProtectionCommand: Decodable, Equatable, Sendable {
    var commandID: String?
    var desired: String?

    enum CodingKeys: String, CodingKey {
        case commandID = "command_id"
        case desired = "desired_protection"
    }

    init(commandID: String?, desired: String? = nil) {
        self.commandID = commandID
        self.desired = desired
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        commandID = (try? c.decodeIfPresent(String.self, forKey: .commandID)) ?? nil
        desired = (try? c.decodeIfPresent(String.self, forKey: .desired)) ?? nil
    }
}

/// A stop or resume this app asked for and is waiting to see applied.
struct ProtectionRequest: Equatable {
    let action: ProtectionAction
    let commandID: String?
    let requestedAt: Date
}

/// What the device last said, from `status` on GET /v1/devices.
struct ProtectionReport: Equatable {
    var protection: String?
    var monitoringState: String?
    var lastCommandID: String?
    /// Online and not stale.
    var live: Bool
}

enum ProtectionPhase: Equatable {
    /// Nothing outstanding.
    case idle
    /// Sent; the device has not applied it yet.
    case pending(String)
    /// The device applied it.
    case confirmed(String)
    /// Sent long enough ago that silence is worth saying, but still stored on
    /// the server and still expected.
    case overdue(String)
}

enum ProtectionControl {
    /// Longer than a mode switch's 20 s: stopping drains the pipeline.
    static let timeout: TimeInterval = 45

    /// Whether the device is stopped: true, false, or nil when it is not
    /// reporting (never guessed from an old report).
    static func isStopped(_ report: ProtectionReport) -> Bool? {
        guard report.live else { return nil }
        if report.protection == "stopped" || report.monitoringState == "stopped" { return true }
        if report.protection == nil || report.protection == "unknown" {
            return report.monitoringState == nil || report.monitoringState == "unknown" ? nil : false
        }
        return false
    }

    /// The action the control offers: resume a stopped device, otherwise stop.
    static func offeredAction(_ report: ProtectionReport) -> ProtectionAction {
        isStopped(report) == true ? .resume : .stop
    }

    /// Whether the device has applied `request`. By id when the device echoes
    /// one (a stop still in flight can never be mistaken for a later resume);
    /// by state otherwise.
    static func converged(_ request: ProtectionRequest, report: ProtectionReport) -> Bool {
        guard report.live else { return false }
        if let sent = request.commandID, let echoed = report.lastCommandID {
            return sent == echoed
        }
        switch request.action {
        case .stop: return isStopped(report) == true
        case .resume: return isStopped(report) == false
        }
    }

    static func phase(_ request: ProtectionRequest?, report: ProtectionReport, now: Date) -> ProtectionPhase {
        guard let request else { return .idle }
        if converged(request, report: report) {
            return .confirmed(request.action == .stop
                ? "Stopped. Nothing is monitored until you resume."
                : "Resumed. The device is monitoring again.")
        }
        if !report.live {
            return .pending("\(request.action.progressive) — the device isn't reporting; it applies this when it reconnects.")
        }
        if now.timeIntervalSince(request.requestedAt) > timeout {
            return .overdue("The device hasn't confirmed yet. The request is saved and will apply when the device next checks in.")
        }
        return .pending("\(request.action.progressive)… waiting for the device to confirm.")
    }

    /// Why the request was refused, in words.
    static func refusal(code: String?, detail: String, endpointMissing: Bool) -> String {
        if endpointMissing { return "Stopping and resuming isn't supported by this server yet." }
        switch code {
        case "firmware_unsupported":
            return "This device's software can't stop or resume monitoring yet. Update the device, then try again."
        case "confirmation_required":
            return "Stopping needs your confirmation."
        default:
            return detail.isEmpty ? "The server refused the request." : detail
        }
    }

    static let stopConfirmationTitle = "Stop monitoring?"
    static let stopConfirmationMessage =
        "The device stops watching and sends no alerts until you resume it here, or it restarts."
}
