import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: ConnectionTruth DevicePresence DeviceStatus PerceptionStatus CapabilityTier PresetLibrary

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func status(_ state: String, stale: Bool = false) -> DeviceStatus {
    DeviceStatus(monitoringState: state, reasons: ["why"], mode: "standard", role: "driver_safety",
                 roleTitle: "Driver safety", allowedPresets: nil,
                 rules: .init(active: [], additive: nil, rejected: nil),
                 presetRejection: nil, stale: stale)
}

func state(_ presence: ConnectionTruth.Presence, _ monitoring: String?, alert: Bool = false,
           loaded: Bool = true) -> InstrumentState {
    let live = presence.isOnline
    let presentation = monitoring.flatMap { MonitoringStatusPresentation.make(status: status($0), online: live) }
        ?? (monitoring == nil && !live ? MonitoringStatusPresentation.make(status: nil, online: false) : nil)
    return InstrumentState.make(presence: presence, monitoring: presentation, alertActive: alert, loaded: loaded)
}

let notLive: [ConnectionTruth.Presence] = [
    .stale(age: 90), .monitoringStale(age: 120), .monitoringStale(age: nil), .offline(age: 300), .offline(age: nil),
    .unknown(.refreshFailed), .unknown(.notRefreshed(age: 60)), .unknown(.notLoaded),
]

print("never ok, never All clear, unless live and ready:")
for presence in notLive {
    for phase in ["ready", "starting", "degraded", "stopped", "failed", "unknown"] {
        let s = state(presence, phase)
        check(s.tone != .ok, "\(presence) + \(phase) is not ok-toned")
        check(s.word != "All clear", "\(presence) + \(phase) never reads All clear (got \(s.word))")
        check(!s.isActive, "\(presence) + \(phase) shows no active visuals")
        check(s.isInterrupted, "\(presence) + \(phase) interrupts the live visuals")
    }
}
for phase in ["starting", "degraded", "stopped", "failed", "unknown", "something_new"] {
    let s = state(.online, phase)
    check(s.tone != .ok && s.kind != .ready, "online + \(phase) is not ready (got \(s.kind))")
}

print("\nready only from the device's own ready:")
let ready = state(.online, "ready")
check(ready.kind == .ready && ready.tone == .ok && ready.word == "All clear", "online + ready is All clear")
check(ready.isActive && !ready.isInterrupted, "ready shows active visuals")

print("\nthe word is the monitoring pill's word:")
check(state(.online, "stopped").word == "Stopped", "Stopped")
check(state(.online, "degraded").word == "Degraded", "Degraded")
check(state(.online, "failed").word == "Not monitoring", "Not monitoring")
check(state(.online, "starting").word == "Starting", "Starting")
check(state(.offline(age: 40), "ready").word == "Status unknown", "offline reads Status unknown")
check(state(.stale(age: 90), "ready").word == "Status unknown", "stale reads Status unknown")

print("\neach kind is distinct without colour:")
let symbols = Set(InstrumentState.Kind.allCases.map(InstrumentState.symbol))
check(symbols.count == InstrumentState.Kind.allCases.count, "every kind has its own symbol")

print("\nkinds:")
check(state(.offline(age: 40), "ready").kind == .offline, "offline")
check(state(.stale(age: 90), "ready").kind == .stale, "stale")
check(state(.unknown(.refreshFailed), "ready").kind == .unknown, "refresh failed is unknown")
check(state(.unknown(.notLoaded), nil, loaded: false).kind == .connecting, "nothing loaded yet is connecting")
check(state(.unknown(.notLoaded), nil, loaded: false).showsProgress, "connecting shows progress")
check(!state(.online, "ready").showsProgress, "ready shows no progress")
let alert = state(.online, "ready", alert: true)
check(alert.kind == .alert && alert.word == "Alert" && alert.tone == .critical, "a recent alert wins")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
