import Foundation

// Run with ios/Tests/run.sh

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

print("\nrefusals:")
check(ProtectionControl.refusal(code: "firmware_unsupported", detail: "x", endpointMissing: false).contains("Update the device"),
      "firmware_unsupported says to update")
check(ProtectionControl.refusal(code: nil, detail: "Not Found", endpointMissing: true).contains("isn't supported"),
      "a missing endpoint")
check(ProtectionControl.refusal(code: nil, detail: "rate limited", endpointMissing: false) == "rate limited",
      "other detail kept")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
