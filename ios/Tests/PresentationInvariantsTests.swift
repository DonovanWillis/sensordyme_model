import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: ConnectionTruth DevicePresence DeviceStatus PerceptionStatus CapabilityTier PresetLibrary InstrumentState ProtectionControl StreamLiveness MJPEGStream
//
// M8.3 property test: generate status replies -- consistent ones the backend
// sends and contradictory ones an older backend or device could -- and check
// the app never makes a claim the reply does not support
// (PresentationInvariants, docs/STATE_MODEL.md "App invariants").
// Deterministic: a fixed-seed SplitMix64, so a failure names a reproducible case.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

struct SplitMix64 {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
    mutating func pick<T>(_ options: [T]) -> T { options[Int(next() % UInt64(options.count))] }
    mutating func chance(_ percent: UInt64) -> Bool { next() % 100 < percent }
}

let now = Date(timeIntervalSince1970: 1_800_000_000)
let names = ["camera", "inference", "face", "objects", "hands", "pose", "attention", "audible_alert", "alert_output"]
let components = ["object_detector", "face_landmarks", "hand_landmarks", "pose"]

/// One reply's `status` (as JSON, so decoding is exercised too) and its row fields.
func generate(_ rng: inout SplitMix64, consistent: Bool) -> PresentationInvariants.Input {
    var status: [String: Any] = [:]
    let state = rng.pick(["starting", "ready", "degraded", "stopped", "failed", "unknown", "nonsense", nil] as [String?])
    var protection = rng.pick(["protecting", "not_protecting", "stopped", "unknown", nil] as [String?])
    var linkStale = rng.chance(25)
    var healthStale = rng.chance(20)
    var online = rng.chance(80)
    if consistent {
        // What the backend sends: online <=> !link_stale; stale withholds.
        online = !linkStale
        if state == "stopped" { protection = "stopped" }
        if protection == "stopped" && state != "stopped" { protection = "not_protecting" }
        if protection == "protecting" && state != "ready" && state != "degraded" { protection = "not_protecting" }
    } else if rng.chance(30) {
        linkStale = !online
    }
    let stale = consistent ? (linkStale || healthStale) : rng.pick([linkStale || healthStale, rng.chance(50)])
    if stale && consistent {
        status["monitoring_state"] = "unknown"
        status["protection"] = "unknown"
    } else {
        if let state { status["monitoring_state"] = state }
        if let protection { status["protection"] = protection }
        if !(stale && consistent) && rng.chance(85) {
            var available: [String] = [], impaired: [[String: Any]] = [], starting: [String] = [], paused: [String] = []
            for name in names {
                switch rng.next() % 5 {
                case 0, 1: available.append(name)
                case 2: impaired.append(["name": name, "reason": "\(name) broke"])
                case 3: starting.append(name)
                default: if rng.chance(50) { paused.append(name) }
                }
            }
            status["capabilities"] = ["available": available, "impaired": impaired, "starting": starting,
                                      "paused": paused]
        }
        if rng.chance(70) {
            var comps: [String: Any] = [:]
            for component in components where rng.chance(70) {
                comps[component] = ["state": rng.pick(["ok", "idle", "paused", "stale", "failed", "unavailable", "?"]),
                                    "reason": "r"]
            }
            status["perception"] = ["schema": "sd-perception/1", "components": comps, "nodes": [] as [Any]]
        }
    }
    status["stale"] = stale
    status["link_stale"] = linkStale
    status["health_stale"] = healthStale
    let age = rng.pick([0.0, 4.0, 29.0, 30.0, 31.0, 400.0])
    status["health_freshness"] = ["state": healthStale ? "stale" : rng.pick(["fresh", "unverified", "absent"]),
                                  "verified": healthStale || rng.chance(50),
                                  "age_s": healthStale ? max(age, 30) : min(age, 29.9)]
    if rng.chance(50) { status["last_command_id"] = rng.pick(["c1", "c0"]) }
    var request: ProtectionRequest? = nil
    if rng.chance(50) {
        request = ProtectionRequest(action: rng.pick([.stop, .resume]), commandID: rng.pick(["c1", nil] as [String?]),
                                    requestedAt: now.addingTimeInterval(-rng.pick([1.0, 20.0, 50.0, 400.0])))
    }
    if rng.chance(50) {
        let outcome = rng.pick(["pending", "confirmed", "mismatch", "timed_out"])
        status["protection_command"] = [
            "id": rng.pick(["c1", "c1", "c9"]), "action": rng.pick(["stop", "resume"]), "outcome": outcome,
            "delivering": rng.chance(50), "timeout_s": 45,
            "applied_at": outcome == "confirmed" || outcome == "mismatch" ? 1_799_999_990.0 : NSNull(),
            "reported_state": rng.pick(["stopped", "protecting", "not_protecting"]),
        ] as [String: Any]
    }
    let data = try! JSONSerialization.data(withJSONObject: status)
    let decoded = try? JSONDecoder().decode(DeviceStatus.self, from: data)
    return PresentationInvariants.Input(
        status: rng.chance(3) ? nil : decoded, online: online,
        lastSeen: online ? now.timeIntervalSince1970 - rng.pick([0.0, 5.0, 25.0]) : rng.pick([nil, now.timeIntervalSince1970 - 300]),
        receivedAt: now.addingTimeInterval(-rng.pick([0.0, 3.0, 10.0, 31.0])),
        lastRefreshFailed: rng.chance(5), alertActive: rng.chance(10), request: request, now: now)
}

print("generated replies never make an unsupported claim:")
var rng = SplitMix64(state: 0x5D_83)
var examined = 0
var byRule: [String: Int] = [:]
var firstFailures: [String] = []
for i in 0..<20_000 {
    let input = generate(&rng, consistent: i % 2 == 0)
    examined += 1
    for violation in PresentationInvariants.violations(input) {
        let rule = String(violation.prefix(while: { $0 != ":" }))
        byRule[rule, default: 0] += 1
        if firstFailures.count < 8 { firstFailures.append("case \(i): \(violation)") }
    }
}
firstFailures.forEach { print("    \($0)") }
check(byRule.isEmpty, "\(examined) generated replies (half contradictory): \(byRule.isEmpty ? "no violation" : "\(byRule)")")

print("\nthe checker itself catches each kind of unsupported claim:")
// A checker that never fires proves nothing: feed it presentations made by
// hand that each break one rule.
func live(_ fields: [String: Any]) -> DeviceStatus {
    var base: [String: Any] = ["stale": false, "link_stale": false, "health_stale": false]
    fields.forEach { base[$0.key] = $0.value }
    return try! JSONDecoder().decode(DeviceStatus.self, from: JSONSerialization.data(withJSONObject: base))
}
let shownReady = PresentationInvariants.show(.init(status: live(["monitoring_state": "ready", "protection": "protecting"]),
                                                   online: true, lastSeen: now.timeIntervalSince1970, receivedAt: now, now: now))
check(shownReady.instrument.kind == .ready && shownReady.live, "a live ready device reads ready")
check(PresentationInvariants.violations(.init(status: live(["monitoring_state": "ready", "protection": "protecting"]),
                                              online: true, lastSeen: now.timeIntervalSince1970, receivedAt: now, now: now)).isEmpty,
      "and that is no violation")

print("\nthe M8.3 contradictions, one by one:")
func shown(_ fields: [String: Any], online: Bool = true, alert: Bool = false, request: ProtectionRequest? = nil)
    -> (PresentationInvariants.Shown, [String]) {
    let input = PresentationInvariants.Input(status: live(fields), online: online,
                                             lastSeen: now.timeIntervalSince1970 - (online ? 1 : 300),
                                             receivedAt: now, alertActive: alert, request: request, now: now)
    return (PresentationInvariants.show(input), PresentationInvariants.violations(input))
}
// inference stopped yet shown detecting
var (s, v) = shown(["monitoring_state": "failed", "protection": "not_protecting",
                    "capabilities": ["available": ["face", "objects"], "impaired": [["name": "inference", "reason": "loop dead"]]]])
check(!s.capabilities.contains { $0.state == .available && ["face", "objects"].contains($0.name) } && v.isEmpty,
      "inference impaired: face/objects are not shown producing results")
// paused yet analysis capabilities active (a device before M8.2)
(s, v) = shown(["monitoring_state": "stopped", "protection": "stopped",
                "capabilities": ["available": ["face", "inference", "camera"]]])
check(s.capabilities.first { $0.name == "face" }?.state == .paused
      && s.capabilities.first { $0.name == "camera" }?.state == .available && v.isEmpty,
      "stopped: face is paused, the camera still captures")
// camera stale yet monitoring fully operational
(s, v) = shown(["monitoring_state": "ready", "protection": "protecting",
                "capabilities": ["available": ["face"], "impaired": [["name": "camera", "reason": "no new frame for 9.0s"]]]])
check(s.monitoring?.phase == .failed && s.instrument.kind == .failed && v.isEmpty,
      "ready with the camera impaired reads 'Not monitoring', never All clear")
// device offline yet current protection shown
(s, v) = shown(["monitoring_state": "ready", "protection": "protecting"], online: false, alert: true)
check(!s.instrument.isActive && s.instrument.isInterrupted && v.isEmpty,
      "an alert on an offline device keeps its word but loses the live visuals")
// resume requested yet success shown before device ack
let resume = ProtectionRequest(action: .resume, commandID: "c1", requestedAt: now.addingTimeInterval(-2))
(s, v) = shown(["monitoring_state": "starting", "protection": "not_protecting", "last_command_id": "c0",
                "protection_command": ["id": "c1", "action": "resume", "outcome": "pending", "delivering": true]],
               request: resume)
if case .pending = s.protection { check(v.isEmpty, "resume pending until the device echoes it (even though it is not stopped)") }
else { check(false, "resume pending until the device echoes it: \(s.protection)") }
// backend online while health telemetry too old
(s, v) = shown(["monitoring_state": "ready", "protection": "protecting", "health_stale": true,
                "health_freshness": ["state": "stale", "verified": true, "age_s": 95]])
check(!s.live && s.monitoring?.phase == .unknown && v.isEmpty, "online with 95 s old health is not live, not ready")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
