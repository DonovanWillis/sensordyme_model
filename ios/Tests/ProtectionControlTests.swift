import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: DeviceStatus PerceptionStatus CapabilityTier PresetLibrary

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func report(_ protection: String?, state: String? = nil, id: String? = nil, live: Bool = true) -> ProtectionReport {
    ProtectionReport(protection: protection, monitoringState: state, lastCommandID: id, live: live)
}

let t0 = Date(timeIntervalSince1970: 1_000_000)

print("the command response decodes:")
let command = try? JSONDecoder().decode(ProtectionCommand.self,
    from: Data(#"{"ok": true, "desired_protection": "stop", "command_id": "a1b2c3"}"#.utf8))
check(command?.commandID == "a1b2c3" && command?.desired == "stop", "command_id and desired_protection")
check((try? JSONDecoder().decode(ProtectionCommand.self, from: Data(#"{"ok": true}"#.utf8)))?.commandID == nil,
      "an older answer without an id decodes")

print("\nwhat the control offers:")
check(ProtectionControl.offeredAction(report("protecting", state: "ready")) == .stop, "a running device offers stop")
check(ProtectionControl.offeredAction(report("stopped", state: "stopped")) == .resume, "a stopped device offers resume")
check(ProtectionControl.offeredAction(report(nil, state: "stopped")) == .resume, "stopped by monitoring state alone")
check(ProtectionControl.isStopped(report("stopped", live: false)) == nil, "a silent device is not known stopped")
check(ProtectionControl.isStopped(report("unknown", state: "unknown")) == nil, "unknown is unknown")
check(ProtectionControl.isStopped(report("not_protecting", state: "degraded")) == false, "degraded is not stopped")

print("\nconvergence by command id:")
let stop = ProtectionRequest(action: .stop, commandID: "c1", requestedAt: t0)
check(!ProtectionControl.converged(stop, report: report("stopped", state: "stopped", id: "c0")),
      "an older id never confirms, even with a stopped state")
check(ProtectionControl.converged(stop, report: report("stopped", state: "stopped", id: "c1")), "the echoed id confirms")
check(!ProtectionControl.converged(stop, report: report("stopped", id: "c1", live: false)),
      "a stale report never confirms")

print("\nconvergence by state (no id echoed):")
check(ProtectionControl.converged(stop, report: report("stopped")), "stop converges on stopped")
check(!ProtectionControl.converged(stop, report: report("protecting", state: "ready")), "stop is pending while running")
let resume = ProtectionRequest(action: .resume, commandID: nil, requestedAt: t0)
check(ProtectionControl.converged(resume, report: report("protecting", state: "starting")), "resume converges once not stopped")
check(!ProtectionControl.converged(resume, report: report("unknown", state: "unknown")), "resume never converges on unknown")

print("\nphases:")
check(ProtectionControl.phase(nil, report: report("protecting"), now: t0) == .idle, "nothing outstanding is idle")
if case .pending(let text) = ProtectionControl.phase(stop, report: report("protecting", state: "ready"), now: t0 + 5) {
    check(text.hasPrefix("Stopping"), "pending says Stopping, never Stopped")
} else { check(false, "pending while waiting") }
if case .pending(let text) = ProtectionControl.phase(stop, report: report(nil, live: false), now: t0 + 500) {
    check(text.contains("isn't reporting"), "offline stays pending and says why, without timing out")
} else { check(false, "offline is pending") }
if case .overdue = ProtectionControl.phase(stop, report: report("protecting", state: "ready"), now: t0 + 60) {
    check(true, "a silent online device goes overdue after the timeout")
} else { check(false, "overdue after the timeout") }
if case .confirmed(let text) = ProtectionControl.phase(stop, report: report("stopped", id: "c1"), now: t0 + 3) {
    check(text.hasPrefix("Stopped"), "confirmed says Stopped")
} else { check(false, "confirmed once the device applies it") }

print("\nM8.3: an echo is an answer, not a success:")
// The device echoes the id of a request it REFUSED too (services/runtime.py
// _apply_protection). Before M8.3 an echo alone read as "Stopped." while the
// device kept monitoring.
let refusedStop = report("protecting", state: "ready", id: "c1")
check(!ProtectionControl.converged(stop, report: refusedStop), "an echo with the wrong state never converges")
if case .mismatch(let text) = ProtectionControl.phase(stop, report: refusedStop, now: t0 + 3) {
    check(text.contains("didn't stop") && text.contains("still monitoring"), "says it didn't stop and is still monitoring")
} else { check(false, "an echoed refusal is a mismatch, never confirmed") }
let resumeC2 = ProtectionRequest(action: .resume, commandID: "c2", requestedAt: t0)
if case .mismatch(let text) = ProtectionControl.phase(resumeC2, report: report("stopped", state: "stopped", id: "c2"), now: t0 + 3) {
    check(text.contains("didn't resume") && text.contains("stopped"), "a resume echoed while stopped is a mismatch")
} else { check(false, "resume echoed with stopped is a mismatch") }

print("\nM8.3: resume is confirmed in words that match what the device does now:")
if case .confirmed(let text) = ProtectionControl.phase(resumeC2, report: report("not_protecting", state: "starting", id: "c2"), now: t0 + 3) {
    check(!text.contains("monitoring again") && text.contains("not monitoring yet"), "starting is not 'monitoring again'")
} else { check(false, "resume applied while starting is confirmed") }
if case .confirmed(let text) = ProtectionControl.phase(resumeC2, report: report("protecting", state: "ready", id: "c2"), now: t0 + 3) {
    check(text.contains("monitoring again"), "protecting is 'monitoring again'")
} else { check(false, "resume applied while ready is confirmed") }
if case .confirmed(let text) = ProtectionControl.phase(resumeC2, report: report("not_protecting", state: "failed", id: "c2"), now: t0 + 3) {
    check(text.contains("can't monitor"), "failed after resume says it can't monitor")
} else { check(false, "resume applied while failed is confirmed (the resume happened)") }

print("\nM8.3: the backend's outcome (status.protection_command):")
func cmd(_ id: String, _ action: String, _ outcome: String, delivering: Bool? = nil, state: String? = nil,
         reason: String? = nil) -> DeviceStatus.ProtectionCommandStatus {
    .init(id: id, action: action, outcome: outcome, delivering: delivering, ageS: nil, timeoutS: 45,
          appliedAt: outcome == "pending" || outcome == "timed_out" ? nil : 1, reportedState: state, reason: reason)
}
func report83(_ protection: String?, state: String? = nil, id: String? = nil, live: Bool = true,
              command: DeviceStatus.ProtectionCommandStatus?) -> ProtectionReport {
    ProtectionReport(protection: protection, monitoringState: state, lastCommandID: id, live: live, command: command)
}
// Never success before the device's ack, even when the state already looks right.
let alreadyStopped = report83("stopped", state: "stopped", id: nil, command: cmd("c1", "stop", "pending", delivering: true))
if case .pending = ProtectionControl.phase(stop, report: alreadyStopped, now: t0 + 2) {
    check(true, "a stopped state without this request's ack stays pending (M8.3 backend)")
} else { check(false, "no success before ack") }
// Stop timeout: withdrawn by the backend, so the device was NOT stopped.
let stopTimedOut = report83("protecting", state: "ready", command: cmd("c1", "stop", "timed_out", delivering: false))
if case .failed(let text) = ProtectionControl.phase(stop, report: stopTimedOut, now: t0 + 46) {
    check(text.contains("cancelled") && text.contains("still monitoring"), "stop timeout: failed, cancelled, still monitoring")
} else { check(false, "a timed-out stop is failed") }
let stopWithdrawnOffline = report83(nil, live: false, command: cmd("c1", "stop", "pending", delivering: false))
if case .failed(let text) = ProtectionControl.phase(stop, report: stopWithdrawnOffline, now: t0 + 31) {
    check(text.contains("unknown"), "a withdrawn stop on a silent device: failed, state unknown")
} else { check(false, "a stop no longer delivered is failed") }
// Resume timeout: still delivered, and the device is said to be still stopped.
let resumeTimedOut = report83("stopped", state: "stopped", command: cmd("c2", "resume", "timed_out", delivering: true))
if case .overdue(let text) = ProtectionControl.phase(resumeC2, report: resumeTimedOut, now: t0 + 46) {
    check(text.contains("still stopped") && text.contains("not monitoring"), "resume timeout: overdue, still stopped")
} else { check(false, "a timed-out resume is overdue, never confirmed") }
// The device confirms a different state: latched mismatch with the device's reason.
let refused = report83("protecting", state: "ready", id: "c1",
                       command: cmd("c1", "stop", "mismatch", state: "protecting", reason: "stopping monitoring requires confirm: true"))
if case .mismatch(let text) = ProtectionControl.phase(stop, report: refused, now: t0 + 3) {
    check(text.contains("requires confirm"), "mismatch carries the device's reason")
} else { check(false, "latched mismatch") }
// Latched confirmed, but the device now reports otherwise (restarted): not a success now.
let restarted = report83("protecting", state: "ready", id: nil, command: cmd("c1", "stop", "confirmed", state: "stopped"))
if case .mismatch = ProtectionControl.phase(stop, report: restarted, now: t0 + 20) {
    check(true, "confirmed earlier but running now: mismatch, never 'Stopped'")
} else { check(false, "a later change is not reported as this stop's success") }
// Superseded by a newer request.
let superseded = report83("protecting", state: "ready", command: cmd("c9", "resume", "pending", delivering: true))
if case .mismatch(let text) = ProtectionControl.phase(stop, report: superseded, now: t0 + 2) {
    check(text.contains("newer"), "a newer request replaces this one")
} else { check(false, "superseded is a final outcome") }
check(ProtectionPhase.failed("x").isFinal && ProtectionPhase.mismatch("x").isFinal && ProtectionPhase.confirmed("x").isFinal
      && !ProtectionPhase.pending("x").isFinal && !ProtectionPhase.overdue("x").isFinal, "final outcomes end the wait")
// Decoding the contract field.
let decoded = try? JSONDecoder().decode(DeviceStatus.self, from: Data(#"""
{"protection": "stopped", "monitoring_state": "stopped", "protection_command": {"id": "c1", "action": "stop",
 "outcome": "confirmed", "delivering": false, "age_s": 4.0, "timeout_s": 45, "applied_at": 1700000000.5,
 "reported_state": "stopped", "reason": null}, "corrections": []}
"""#.utf8))
check(decoded?.protectionCommand?.outcome == "confirmed" && decoded?.protectionCommand?.reportedState == "stopped",
      "status.protection_command decodes")
let fromStatus = ProtectionReport(status: decoded, live: true)
check(ProtectionControl.converged(stop, report: fromStatus), "a latched confirmed outcome with a stopped state converges")

print("\nrefusals:")
check(ProtectionControl.refusal(code: "firmware_unsupported", detail: "x", endpointMissing: false).contains("Update the device"),
      "firmware_unsupported says to update")
check(ProtectionControl.refusal(code: nil, detail: "Not Found", endpointMissing: true).contains("isn't supported"),
      "a missing endpoint")
check(ProtectionControl.refusal(code: nil, detail: "rate limited", endpointMissing: false) == "rate limited",
      "other detail kept")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
