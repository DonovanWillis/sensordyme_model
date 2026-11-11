import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func status(_ state: String, reasons: [String] = [], stale: Bool = false, mode: String? = "standard",
            allowed: [String]? = nil, additive: Bool? = nil,
            rejected: DeviceStatus.RulesRejection? = nil,
            presetRejection: DeviceStatus.PresetRejection? = nil) -> DeviceStatus {
    DeviceStatus(monitoringState: state, reasons: reasons, mode: mode, role: "driver_safety",
                 roleTitle: "Driver safety", allowedPresets: allowed,
                 rules: .init(active: [], additive: additive, rejected: rejected),
                 presetRejection: presetRejection, stale: stale)
}

func present(_ s: DeviceStatus?, online: Bool = true) -> MonitoringStatusPresentation? {
    MonitoringStatusPresentation.make(status: s, online: online)
}

print("never green unless the device reports ready:")
let ready = present(status("ready"))
check(ready?.label == "All clear" && ready?.tone == .ok, "ready is All clear, ok")
for state in ["failed", "stopped", "degraded", "starting", "unknown", "", "something_new"] {
    let p = present(status(state, reasons: ["why"]))
    check(p != nil && p?.tone != .ok, "\(state.isEmpty ? "(empty)" : state) is not ok-toned (got \(String(describing: p?.tone)))")
    check(p?.label != "All clear", "\(state.isEmpty ? "(empty)" : state) never reads All clear")
}

print("\neach state says what it is:")
check(present(status("failed", reasons: ["face: no landmarks"]))?.label == "Not monitoring",
      "failed reads Not monitoring")
check(present(status("failed"))?.tone == .critical, "failed is critical")
check(present(status("failed", reasons: ["face: no landmarks"]))?.detail == "Not monitoring — face: no landmarks",
      "failed carries the device's reason")
check(present(status("stopped"))?.label == "Stopped", "stopped reads Stopped")
check(present(status("stopped"))?.detail.contains("resumed") == true, "stopped without a reason still explains")
check(present(status("degraded", reasons: ["phone_use: needs objects"]))?.detail == "Degraded — phone_use: needs objects",
      "degraded shows its reason")
check(present(status("degraded"))?.tone == .caution, "degraded is caution")
check(present(status("starting"))?.label == "Starting", "starting reads Starting")
check(present(status("unknown"))?.label == "Status unknown", "unknown reads Status unknown")
check(present(status("unknown"))?.tone == .neutral, "unknown is neutral")

print("\nstale or offline is unknown, never the last ready:")
check(present(status("ready", stale: true))?.phase == .unknown, "a stale ready is unknown")
check(present(status("ready"), online: false)?.phase == .unknown, "an offline ready is unknown")
check(present(nil, online: false)?.phase == .unknown, "no status and offline is unknown")
check(present(nil, online: true) == nil, "no status and online keeps the pre-contract rendering")

print("\nthe pill:")
check(MonitoringStatusPresentation.pill(present(status("failed")), alertActive: false) == ("Not monitoring", .critical),
      "a failed device's pill")
check(MonitoringStatusPresentation.pill(present(status("ready")), alertActive: true) == ("Alert", .critical),
      "a recent alert wins")
check(MonitoringStatusPresentation.pill(nil, alertActive: false) == ("All clear", .ok),
      "an old backend keeps the old pill")

print("\nthe card subtitle:")
check(MonitoringStatusPresentation.cardSubtitle(modeTitle: "Driver Safety", presentation: present(status("ready")))
      == "Driver Safety", "ready shows just the mode")
check(MonitoringStatusPresentation.cardSubtitle(modeTitle: "Driver Safety", presentation: present(status("stopped")))
      == "Stopped · Driver Safety", "stopped prefixes the state")
check(MonitoringStatusPresentation.cardSubtitle(modeTitle: "Driver Safety", presentation: present(status("failed")))
      == "Not monitoring · Driver Safety", "failed prefixes the state")

print("\nhealth row vocabulary:")
check(StatusTone.ok.healthState == "ok" && StatusTone.caution.healthState == "warning"
      && StatusTone.critical.healthState == "failed" && StatusTone.neutral.healthState == "unknown",
      "tones map onto the health card's states")

print("\ncustom rules: replacing vs alongside, from the device's mode:")
check(MonitoringStatusPresentation.customRules(status: status("ready", mode: "standard_plus_custom"), legacyActive: true)
      == .alongside, "additive mode runs rules alongside the mode")
check(MonitoringStatusPresentation.customRules(status: status("ready", mode: "ai_custom"), legacyActive: true)
      == .replacing, "AI Custom replaces the mode")
check(MonitoringStatusPresentation.customRules(status: status("ready", mode: "standard"), legacyActive: true)
      == .none, "a device running only its mode runs no rules, whatever the stored config says")
check(MonitoringStatusPresentation.customRules(status: nil, legacyActive: true) == .replacing,
      "an old backend falls back to monitoring.active")
check(MonitoringStatusPresentation.customRules(status: status("ready", mode: nil, additive: true), legacyActive: true)
      == .alongside, "no mode reported, but an additive role: alongside")
check(MonitoringStatusPresentation.rulesAdditive(status("ready", additive: true)), "additive role")
check(!MonitoringStatusPresentation.rulesAdditive(status("ready", additive: false)), "non-additive role")

print("\nrejected rules are not shown as running:")
let refused = status("ready", rejected: .init(version: "v2", reasons: ["rule 0: 'dog' not detected"], at: 1))
check(MonitoringStatusPresentation.rejectedRulesNotice(refused)?.contains("'dog' not detected") == true,
      "the device's reason is shown")
check(MonitoringStatusPresentation.rejectedRulesNotice(refused)?.contains("aren't running") == true,
      "and they are said not to run")
check(MonitoringStatusPresentation.rejectedRulesNotice(status("ready")) == nil, "nothing refused, no notice")

print("\npreset availability:")
let driver = status("ready", allowed: ["driver_safety"])
check(MonitoringStatusPresentation.presetUnavailableReason("driver_safety", status: driver) == nil,
      "the allowed preset is available")
check(MonitoringStatusPresentation.presetUnavailableReason("baby_monitoring", status: driver)
      == "Not available for this device's role (Driver safety)", "another preset is unavailable, with the role")
check(MonitoringStatusPresentation.presetUnavailableReason("baby_monitoring", status: status("ready")) == nil,
      "no restriction reported: every preset is available")
check(MonitoringStatusPresentation.presetUnavailableReason("baby_monitoring", status: nil) == nil,
      "no status: every preset is available")
let dropped = status("ready", allowed: ["driver_safety"],
                     presetRejection: .init(preset: "baby_monitoring:nap_time", code: "role_forbids_preset",
                                            reason: "role is driver_safety", at: 1))
check(MonitoringStatusPresentation.presetUnavailableReason("baby_monitoring", status: dropped) == "role is driver_safety",
      "a dropped request shows the backend's reason on that preset")

print("\nlenient decoding:")
let json = """
{"monitoring_state": "failed", "monitoring_active": false, "protection": "not_protecting",
 "reasons": ["face: no landmarks"], "mode": "standard", "role": "driver_safety",
 "allowed_presets": ["driver_safety"], "capabilities": {"available": ["camera"],
 "impaired": [{"name": "face", "reason": "no landmarks"}], "starting": []},
 "rules": {"active": null, "additive": true, "rejected": null}, "last_rejection": null,
 "preset_rejection": null, "stale": false, "reported_at": 1700000000.5, "future_field": 1}
""".data(using: .utf8)!
let decoded = try? JSONDecoder().decode(DeviceStatus.self, from: json)
check(decoded?.monitoringState == "failed" && decoded?.monitoringActive == false, "the contract decodes")
check(decoded?.capabilities?.impaired?.first?.reason == "no landmarks", "impaired capabilities decode")
check(decoded?.rules?.additive == true && decoded?.rules?.active == nil, "rules decode, null active stays nil")
let odd = #"{"monitoring_state": 3, "reasons": "not a list", "allowed_presets": null, "stale": true}"#.data(using: .utf8)!
let oddDecoded = try? JSONDecoder().decode(DeviceStatus.self, from: odd)
check(oddDecoded != nil && oddDecoded?.monitoringState == nil && oddDecoded?.stale == true,
      "a wrong-typed field is dropped, not fatal to the device list")
check(present(oddDecoded)?.phase == .unknown, "and what is left reads as unknown")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
