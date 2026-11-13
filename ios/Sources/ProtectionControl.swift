import Foundation

// Stop / resume monitoring (POST /v1/devices/{id}/protection) and how the app
// knows it actually happened. Foundation only (plus DeviceStatus), so
// ios/Tests/ProtectionControlTests.swift can pin it down.
//
// The backend answers with a `command_id` the moment it stores the request;
// that is not the device stopping. The device echoes the id
// (`last_command_id`) once it has HANDLED it -- applied or refused -- and
// reports its protection state. So an echo alone is not success: the state it
// came with must be the one asked for. Until then the control says
// "Stopping…", never "Stopped".
//
// M8.3: every request ends in an explicit outcome, never an endless wait:
//   confirmed  echoed, with the requested state
//   mismatch   echoed with another state (the device refused, or something
//              else changed it first), or replaced by a newer request
//   failed     a stop not confirmed within the timeout: the backend withdraws
//              it (it is never delivered late), so the device was not stopped
//   overdue    a resume not confirmed within the timeout: still delivered (late
//              protection is the safe direction), and said to be still stopped
// A backend before M8.3 sends no `protection_command`; then the echo, the
// state and the app's own timer decide, as before, except that an echo with
// the wrong state is a mismatch, never a success.

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
    /// M8.3 `status.protection_command`: the backend's record of the last
    /// request and its outcome. nil from an older backend.
    var command: DeviceStatus.ProtectionCommandStatus? = nil
}

extension ProtectionReport {
    /// The report a device screen reads, from the device's status.
    init(status: DeviceStatus?, live: Bool) {
        self.init(protection: status?.protection, monitoringState: status?.monitoringState,
                  lastCommandID: status?.lastCommandID, live: live, command: status?.protectionCommand)
    }
}

enum ProtectionPhase: Equatable {
    /// Nothing outstanding.
    case idle
    /// Sent; the device has not applied it yet.
    case pending(String)
    /// The device applied it, and reports the state asked for.
    case confirmed(String)
    /// Sent long enough ago that silence is worth saying, but still stored on
    /// the server and still expected.
    case overdue(String)
    /// M8.3: it will not happen: a stop not confirmed in time is withdrawn.
    case failed(String)
    /// M8.3: the device answered, with a state other than the one asked for
    /// (or a newer request replaced this one).
    case mismatch(String)

    /// A final outcome: the request is no longer waited for.
    var isFinal: Bool {
        switch self {
        case .confirmed, .failed, .mismatch: true
        default: false
        }
    }

    /// The sentence for this phase, nil when idle.
    var text: String? {
        switch self {
        case .idle: nil
        case .pending(let t), .confirmed(let t), .overdue(let t), .failed(let t), .mismatch(let t): t
        }
    }
}

enum ProtectionControl {
    /// Longer than a mode switch's 20 s: stopping drains the pipeline. The
    /// backend's PROTECTION_ACK_TIMEOUT_S is the same value.
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

    /// Whether the live report is in the state `action` asks for.
    static func agrees(_ action: ProtectionAction, _ report: ProtectionReport) -> Bool {
        switch action {
        case .stop: return isStopped(report) == true
        case .resume: return isStopped(report) == false
        }
    }

    /// The backend's record of THIS request, when it has one.
    private static func record(_ request: ProtectionRequest, _ report: ProtectionReport)
        -> DeviceStatus.ProtectionCommandStatus? {
        guard let command = report.command, let id = request.commandID, command.id == id else { return nil }
        return command
    }

    /// Whether the device has answered `request`: by id when the device echoes
    /// one (a stop still in flight can never be mistaken for a later resume),
    /// or by the backend's latched outcome; by state only when neither side
    /// has ids (an older backend or device).
    static func answered(_ request: ProtectionRequest, report: ProtectionReport) -> Bool {
        guard report.live else { return false }
        if let outcome = record(request, report)?.outcome, outcome == "confirmed" || outcome == "mismatch" {
            return true
        }
        if let sent = request.commandID, let echoed = report.lastCommandID { return sent == echoed }
        if request.commandID != nil, report.command != nil { return false }  // M8.3 backend: ids only
        return agrees(request.action, report)
    }

    /// Whether the device has applied `request` AND reports the state asked for.
    /// An echo with another state is a refusal, never a success.
    static func converged(_ request: ProtectionRequest, report: ProtectionReport) -> Bool {
        answered(request, report: report) && agrees(request.action, report)
            && record(request, report)?.outcome != "mismatch"
    }

    static func phase(_ request: ProtectionRequest?, report: ProtectionReport, now: Date) -> ProtectionPhase {
        guard let request else { return .idle }
        // Another phone (or this one, again) asked for something newer.
        if let command = report.command, let mine = request.commandID, let newest = command.id, newest != mine {
            return .mismatch("A newer stop or resume replaced this request. " + current(report))
        }
        let record = record(request, report)
        if answered(request, report: report) {
            if converged(request, report: report) { return .confirmed(confirmedText(request.action, report)) }
            return .mismatch(mismatchText(request.action, report, reason: record?.reason))
        }
        let late = record?.outcome == "timed_out" || now.timeIntervalSince(request.requestedAt) > timeout
        if request.action == .stop, let record, record.delivering == false || record.outcome == "timed_out" {
            // M8.3 backend: a stop not confirmed in time is withdrawn, never applied late.
            return .failed("The device didn't confirm the stop in time, so it was cancelled. "
                           + (report.live ? current(report) : "The device isn't reporting, so whether it is monitoring is unknown."))
        }
        if !report.live {
            if request.action == .stop, record != nil {
                return .pending("Stopping — the device isn't reporting. The stop is cancelled if it doesn't confirm within 30 seconds.")
            }
            if late, request.action == .resume, record != nil {
                return .overdue("The device isn't reporting, so it hasn't resumed: it is still stopped. The request is kept and applies when it reconnects.")
            }
            return .pending("\(request.action.progressive) — the device isn't reporting; it applies this when it reconnects.")
        }
        if late {
            if request.action == .resume {
                return .overdue("The device hasn't confirmed the resume. It is still stopped and not monitoring; the request is kept and applies when it next checks in.")
            }
            return .overdue("The device hasn't confirmed yet. The request is saved and will apply when the device next checks in.")
        }
        return .pending("\(request.action.progressive)… waiting for the device to confirm.")
    }

    /// What a confirmed request says, from the state the device reports now:
    /// "monitoring again" only when it is.
    static func confirmedText(_ action: ProtectionAction, _ report: ProtectionReport) -> String {
        if action == .stop { return "Stopped. Nothing is monitored until you resume." }
        switch (report.protection, report.monitoringState) {
        case ("protecting", _): return "Resumed. The device is monitoring again."
        case (_, "starting"?): return "Resumed. The device is starting up — not monitoring yet."
        default: return "Resumed, but the device reports it can't monitor right now."
        }
    }

    static func mismatchText(_ action: ProtectionAction, _ report: ProtectionReport, reason: String?) -> String {
        let head = action == .stop ? "The device didn't stop" : "The device didn't resume"
        let why = reason.flatMap { $0.isEmpty ? nil : " — \($0)" } ?? ""
        return "\(head)\(why). " + current(report)
    }

    /// What the device reports now, in words, for a final outcome.
    private static func current(_ report: ProtectionReport) -> String {
        switch isStopped(report) {
        case true?: return "It is stopped."
        case false?:
            return report.protection == "protecting" ? "It is still monitoring."
                : "It is running, but reports it isn't fully monitoring."
        case nil: return "Its state is unknown."
        }
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
