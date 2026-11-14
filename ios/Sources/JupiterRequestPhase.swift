import Foundation

/// Where an Ask Jupiter request is, from suggestion to running on the device:
///
///   request -> review -> sent, waiting for the device -> running
///                                                     \-> not running (failed)
///
/// One reading of the review sheet's own state, so the step strip, the status
/// surface and VoiceOver can never disagree. "Running" only ever comes from
/// what the DEVICE reported (ActivationFollowUp.confirmed); the server
/// accepting the change is "sent". Foundation-only and unit-tested
/// (ios/Tests/JupiterRequestPhaseTests.swift).
enum JupiterRequestPhase: Equatable {
    /// Nothing asked yet: the suggestions (all within the policy grammar).
    case asking
    /// The request is with Jupiter.
    case reading
    /// What Jupiter understood, to review and adjust. Nothing has changed.
    case review
    /// Jupiter understood nothing this device can watch for.
    case nothingToActivate
    /// Activate was confirmed; the request to the server is in flight.
    case sending
    /// The server has it; the device has not confirmed it yet. `stillWaiting`:
    /// the wait for the device ran out without an answer.
    case waitingForDevice(stillWaiting: Bool)
    /// The device reports it running.
    case running
    /// The device refused it, or activation failed before it was sent.
    case failed(sent: Bool)

    /// What the device said after activation, reduced to its kind.
    enum DeviceOutcome: Equatable { case none, waiting, confirmed, refused }

    static func from(thinking: Bool, hasProposal: Bool, proposalEmpty: Bool, activating: Bool,
                     applied: Bool, outcome: DeviceOutcome, timedOut: Bool, activationFailed: Bool)
        -> JupiterRequestPhase {
        if thinking { return .reading }
        guard hasProposal else { return .asking }
        if applied {
            switch outcome {
            case .confirmed: return .running
            case .refused: return .failed(sent: true)
            case .waiting, .none: return .waitingForDevice(stillWaiting: timedOut)
            }
        }
        if activating { return .sending }
        if activationFailed { return .failed(sent: false) }
        return proposalEmpty ? .nothingToActivate : .review
    }

    /// The four steps of the strip.
    enum Step: Int, CaseIterable {
        case request, review, sent, running

        var title: String {
            switch self {
            case .request: "Request"
            case .review: "Review"
            case .sent: "Sent"
            case .running: "Running"
            }
        }
    }

    enum StepState: Equatable { case done, current, upcoming, failed }

    /// Each step's state in this phase.
    func state(of step: Step) -> StepState {
        let current: Step
        var failedHere = false
        switch self {
        case .asking, .reading: current = .request
        case .review, .nothingToActivate: current = .review
        case .sending, .waitingForDevice: current = .sent
        case .running: return .done
        case .failed(let sent):
            current = sent ? .running : .sent
            failedHere = true
        }
        if step.rawValue < current.rawValue { return .done }
        if step == current { return failedHere ? .failed : .current }
        return .upcoming
    }

    /// The phase in words, for the status line and VoiceOver.
    var summary: String {
        switch self {
        case .asking: "Describe what to watch for"
        case .reading: "Jupiter is reading your request"
        case .review: "Review what Jupiter understood. Nothing has changed yet."
        case .nothingToActivate: "Nothing this device can watch for"
        case .sending: "Sending to the device"
        case .waitingForDevice(let stillWaiting):
            stillWaiting ? "Sent. The device hasn't confirmed it yet." : "Sent. Waiting for the device to confirm."
        case .running: "Running on the device"
        case .failed(let sent): sent ? "Not running: the device refused it" : "Not sent: activation failed"
        }
    }

    /// Spoken for the strip as a whole ("Step 3 of 4, Sent. …").
    var spokenProgress: String {
        let step = Step.allCases.first { state(of: $0) == .current || state(of: $0) == .failed }
            ?? (self == .running ? .running : .request)
        return "Step \(step.rawValue + 1) of \(Step.allCases.count), \(step.title). \(summary)"
    }
}
