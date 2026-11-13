import Foundation

// Run with ios/Tests/run.sh
//
// An Ask Jupiter request: suggestion -> review -> sent -> running (only when the
// device confirms) or failed.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

typealias P = JupiterRequestPhase
func phase(thinking: Bool = false, proposal: Bool = true, empty: Bool = false, activating: Bool = false,
           applied: Bool = false, outcome: P.DeviceOutcome = .none, timedOut: Bool = false,
           failed: Bool = false) -> P {
    P.from(thinking: thinking, hasProposal: proposal, proposalEmpty: empty, activating: activating,
           applied: applied, outcome: outcome, timedOut: timedOut, activationFailed: failed)
}

print("before anything is activated:")
check(phase(proposal: false) == .asking, "no request yet: asking")
check(phase(thinking: true, proposal: false) == .reading, "with Jupiter: reading")
check(phase(thinking: true) == .reading, "a new request while one is shown: reading")
check(phase() == .review, "a proposal: review")
check(phase(empty: true) == .nothingToActivate, "an empty proposal: nothing to activate")
check(phase(activating: true) == .sending, "Activate confirmed, request in flight: sending")
check(phase(failed: true) == .failed(sent: false), "the activation request failed: failed, not sent")

print("\nafter the server accepted it:")
check(phase(applied: true) == .waitingForDevice(stillWaiting: false), "accepted, no word from the device: waiting")
check(phase(applied: true, outcome: .waiting) == .waitingForDevice(stillWaiting: false), "device not confirmed yet: waiting")
check(phase(applied: true, outcome: .waiting, timedOut: true) == .waitingForDevice(stillWaiting: true),
      "the wait ran out: still waiting, never running")
check(phase(applied: true, outcome: .confirmed) == .running, "the device confirmed: running")
check(phase(applied: true, outcome: .refused) == .failed(sent: true), "the device refused: failed, sent")

print("\nthe strip:")
check(P.review.state(of: .request) == .done && P.review.state(of: .review) == .current
      && P.review.state(of: .sent) == .upcoming && P.review.state(of: .running) == .upcoming, "review")
check(P.waitingForDevice(stillWaiting: false).state(of: .sent) == .current
      && P.waitingForDevice(stillWaiting: false).state(of: .running) == .upcoming, "waiting: Sent current, Running not yet")
check(P.allDone, "running: every step done")
check(P.failed(sent: true).state(of: .sent) == .done && P.failed(sent: true).state(of: .running) == .failed,
      "refused by the device: Running failed")
check(P.failed(sent: false).state(of: .review) == .done && P.failed(sent: false).state(of: .sent) == .failed,
      "not sent: Sent failed")
check(P.asking.state(of: .request) == .current, "asking: Request current")

print("\nwords:")
check(P.waitingForDevice(stillWaiting: false).summary.contains("Waiting for the device"), "waiting says so")
check(!P.waitingForDevice(stillWaiting: true).summary.lowercased().contains("running"), "still waiting never says running")
check(P.review.summary.contains("Nothing has changed"), "review says nothing has changed")
check(P.review.spokenProgress.hasPrefix("Step 2 of 4, Review."), "spoken: \(P.review.spokenProgress)")
check(P.running.spokenProgress.hasPrefix("Step 4 of 4, Running."), "spoken: \(P.running.spokenProgress)")
check(P.failed(sent: true).spokenProgress.hasPrefix("Step 4 of 4, Running. Not running"), "spoken: \(P.failed(sent: true).spokenProgress)")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)

extension JupiterRequestPhase {
    static var allDone: Bool { P.Step.allCases.allSatisfy { P.running.state(of: $0) == .done } }
}
