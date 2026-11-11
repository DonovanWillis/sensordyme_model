import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: DeviceStatus CapabilityTier

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func decodeStatus(_ json: String) -> DeviceStatus? {
    do { return try JSONDecoder().decode(DeviceStatus.self, from: Data(json.utf8)) } catch {
        print("  decode error: \(error)")
        return nil
    }
}

print("the wire contract decodes (sd-perception/1):")
let wire = """
{"monitoring_state": "degraded", "reasons": ["hand tracking failed"], "mode": "standard",
 "perception": {"schema": "sd-perception/1", "simulated": true,
  "components": {
    "object_detector": {"state": "ok", "model": {"name": "damoyolo_tinynasL20_T.hef", "sha256": "e691"},
                        "last_run_age_s": 0.12, "runs": 1234, "errors": 0, "reason": null,
                        "consumers": ["driver_safety", "rules"]},
    "face_landmarks": {"state": "ok"},
    "hand_landmarks": {"state": "failed", "reason": "model crashed", "errors": 3},
    "pose": {"state": "idle"},
    "broken": 42
  },
  "nodes": [
    {"id": "driver_safety", "kind": "preset", "state": "degraded", "required_capabilities": ["face", "objects"],
     "missing_capabilities": [], "validation": "synthetic_tests", "experimental": false},
    {"id": "people_counter", "kind": "companion", "state": "active", "validation": "synthetic_tests", "experimental": true},
    {"kind": "no id, dropped"},
    {"id": "future", "state": "something_new", "validation": "a_new_tier"}
  ]}}
"""
let st = decodeStatus(wire)
check(st?.perception?.schema == "sd-perception/1", "schema")
check(st?.perception?.simulated == true, "simulated flag")
check(st?.perception?.components.count == 4, "a malformed component is skipped, the rest kept")
check(st?.perception?.components["object_detector"]?.model?.sha256 == "e691", "model identity")
check(st?.perception?.components["object_detector"]?.consumers == ["driver_safety", "rules"], "consumers")
check(st?.perception?.nodes.map(\.id) == ["driver_safety", "people_counter", "future"], "a node without id is skipped")
check(decodeStatus("{\"monitoring_state\": \"ready\"}")?.perception == nil, "older backend: no perception")
check(decodeStatus("{\"perception\": \"garbage\"}") != nil, "a garbage perception never fails the status")

print("\ncomponent rows are device truth, worst first:")
let rows = PerceptionPresentation.componentRows(st?.perception, live: true)
check(rows.first?.name == "hand_landmarks" && rows.first?.tone == .critical, "the failed component leads")
check(rows.first?.detail == "Not working — model crashed", "with the device's reason")
check(rows.first(where: { $0.name == "pose" })?.detail.hasPrefix("Idle") == true, "idle is said as idle")
check(rows.first(where: { $0.name == "object_detector" })?.title == "Object detection", "readable titles")
check(PerceptionPresentation.componentRows(st?.perception, live: false).isEmpty, "nothing shown when not live")
let stale = PerceptionComponent(state: "stale", lastRunAgeS: 12.4)
check(PerceptionPresentation.describe(stale).0 == "No fresh results for 12 s", "stale with its age")
check(PerceptionPresentation.describe(stale).1 == .caution, "stale is a caution")
check(PerceptionPresentation.describe(PerceptionComponent(state: "weird")).0 == "State unknown", "unknown state")
check(PerceptionPresentation.degradedSummary(st?.perception, live: true)?.contains("Hand tracking") == true,
      "degraded summary names the component")
check(PerceptionPresentation.degradedSummary(PerceptionReport(components: ["object_detector": .init(state: "ok")]),
                                             live: true) == nil, "no summary when all work")

print("\ncapability checks never call the unreported available:")
let checks = CapabilityChecks.make(required: ["face", "objects", "hands", "audible_alert"], optional: ["pose", "face"],
                                   status: st, online: true)
func verdict(_ name: String) -> CapabilityCheck.Verdict? { checks.first { $0.name == name }?.verdict }
check(verdict("face") == .available, "face working")
check(verdict("objects") == .available, "objects working")
if case .unavailable(let why)? = verdict("hands") { check(why.contains("model crashed"), "hands failed, with why") }
else { check(false, "hands should be unavailable") }
if case .unknown? = verdict("audible_alert") { check(true, "an unreported capability is unknown") }
else { check(false, "audible_alert should be unknown") }
check(verdict("pose") == .available, "idle still means the device has it")
check(checks.filter { $0.name == "face" }.count == 1, "a capability listed twice appears once")
check(checks.first { $0.name == "pose" }?.required == false, "optional stays optional")
let offline = CapabilityChecks.make(required: ["face"], optional: [], status: st, online: false)
if case .unknown(let why)? = offline.first?.verdict { check(why.contains("isn't reporting"), "offline is unknown") }
else { check(false, "offline must be unknown") }
let node = MonitoringNode(id: "driver_safety", missingCapabilities: ["face"], reason: "no face model")
let withNode = CapabilityChecks.make(required: ["face"], optional: [], status: st, online: true, node: node)
check(withNode.first?.verdict == .unavailable("no face model"), "the node's missing list wins")
check(CapabilityChecks.blockingSummary(withNode)?.contains("Face") == true, "blocking summary names it")
check(CapabilityChecks.blockingSummary(checks)?.contains("Hands") == true, "hands blocks")
let legacy = DeviceStatus(monitoringState: "ready", capabilities: .init(available: ["face"],
    impaired: [.init(name: "objects", reason: "detector missing")]))
let legacyChecks = CapabilityChecks.make(required: ["face", "objects"], optional: [], status: legacy, online: true)
check(legacyChecks[0].verdict == .available, "older device: from status.capabilities")
check(legacyChecks[1].verdict == .unavailable("detector missing"), "older device: impaired with reason")

print("\nactivation is confirmed only by the device, and only when it monitors:")
func eval(_ status: DeviceStatus?, reported: String? = "driver_safety", custom: Bool = false,
          since: Date? = nil) -> ActivationStatus {
    ActivationStatus.evaluate(base: "driver_safety", deviceName: "Car", title: "Driver Safety",
                              reportedPreset: reported, customReplacing: custom, status: status, online: true,
                              since: since)
}
check(eval(DeviceStatus(monitoringState: "ready")).phase == .confirmed, "ready confirms")
check(eval(DeviceStatus(monitoringState: "ready"), reported: "driver_safety:teen_driver").phase == .confirmed,
      "a variant of the same preset confirms")
check(eval(DeviceStatus(monitoringState: "ready"), reported: "elderly_monitoring").phase == .waiting,
      "another preset is still waiting")
check(eval(DeviceStatus(monitoringState: "ready"), custom: true).phase == .waiting, "custom replacing is not confirmation")
let failed = eval(DeviceStatus(monitoringState: "failed", reasons: ["face model missing"]))
check(failed.phase == .notMonitoring && failed.tone == .critical, "failed is never confirmed")
check(failed.text.contains("face model missing"), "with the device's reason")
let degraded = eval(DeviceStatus(monitoringState: "degraded", reasons: ["hand tracking failed"]))
check(degraded.phase == .confirmed && degraded.tone == .caution, "degraded confirms, as caution")
check(degraded.text.contains("hand tracking failed"), "degraded says why")
check(eval(DeviceStatus(monitoringState: "stopped")).phase == .notMonitoring, "stopped is not monitoring")
check(eval(DeviceStatus(monitoringState: "starting")).phase == .waiting, "starting keeps waiting")
let now = Date()
let refusedStatus = DeviceStatus(monitoringState: "ready",
    lastRejection: .init(kind: "preset", subject: "driver_safety", reasons: ["missing face"],
                         at: now.timeIntervalSince1970))
let refused = eval(refusedStatus, reported: "elderly_monitoring", since: now)
check(refused.phase == .refused && refused.text.contains("missing face"), "a device refusal is shown with its reason")
let oldRefusal = DeviceStatus(monitoringState: "ready",
    lastRejection: .init(kind: "preset", subject: "driver_safety", reasons: ["old"],
                         at: now.timeIntervalSince1970 - 3600))
check(eval(oldRefusal, reported: "elderly_monitoring", since: now).phase == .waiting,
      "an hour-old refusal does not answer a new request")
let backendRefusal = DeviceStatus(presetRejection: .init(preset: "driver_safety:strict", code: "role_forbids_preset",
                                                        reason: "role forbids", at: now.timeIntervalSince1970))
check(DeviceRefusal.forPreset("driver_safety", status: backendRefusal, since: now) == "role forbids",
      "preset_rejection by base name")
check(DeviceRefusal.forPreset("elderly_monitoring", status: backendRefusal) == nil, "another preset's refusal is not this one's")
let rulesRefusal = DeviceStatus(lastRejection: .init(kind: "rules", subject: "v3", reasons: ["bad"], at: nil))
check(DeviceRefusal.forPreset("driver_safety", status: rulesRefusal) == nil, "a rules refusal is not a preset refusal")
let nodeRefusal = DeviceStatus(lastRejection: .init(kind: "nodes", subject: "people_counter", reasons: ["no objects"], at: nil))
check(DeviceRefusal.forNode("people_counter", status: nodeRefusal) == "no objects", "node refusal")
check(DeviceRefusal.forNode("other", status: nodeRefusal) == nil, "another node's refusal")

print("\nnodes never overclaim:")
let report = st!.perception!
let driver = NodePresentation.make(report.node("driver_safety")!)
check(driver.title == "Driver Safety" && driver.tone == .caution, "degraded node")
check(driver.badge == "Not yet validated · simulated tests", "simulated tests say so")
check(NodePresentation.make(report.node("people_counter")!).badge == "Experimental", "experimental beats any tier")
check(NodePresentation.make(report.node("future")!).badge == "Validation unknown", "unknown tier is unknown")
check(NodePresentation.make(report.node("future")!).stateText == "State unknown", "unknown state is unknown")
let experimentalVerified = MonitoringNode(id: "x", state: "active", validation: "production_permitted", experimental: true)
check(NodePresentation.make(experimentalVerified).badge == "Experimental", "experimental never reads verified")
for tier in CapabilityTier.allCases {
    let badge = NodePresentation.badge(MonitoringNode(id: "n", validation: tier.rawValue))
    if badge.lowercased().contains("validated for production") { check(false, "\(tier) overclaims") }
}
check(true, "no node badge says validated for production")
let failedNode = MonitoringNode(id: "people_counter", state: "failed", missingCapabilities: ["objects"])
check(NodePresentation.make(failedNode).stateText == "Not running — missing Objects", "failed names what is missing")

print("\ncompanion toggle follows the device:")
check(CompanionNodeToggle.isOn("people_counter", report: report, live: true) == true, "running node is on")
check(CompanionNodeToggle.isOn("absent", report: report, live: true) == false, "unreported node is off")
check(CompanionNodeToggle.isOn("people_counter", report: report, live: false) == nil, "not live: unknown")
check(CompanionNodeToggle.isOn("people_counter", report: nil, live: true) == nil, "no report: unknown")
check(CompanionNodeToggle.confirmed("people_counter", wanted: true, report: report, live: true), "on confirmed")
check(!CompanionNodeToggle.confirmed("people_counter", wanted: false, report: report, live: true), "off not yet")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
