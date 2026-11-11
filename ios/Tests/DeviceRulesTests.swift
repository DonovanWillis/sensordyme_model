import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func decode<T: Decodable>(_ type: T.Type, _ json: String) -> T? {
    do { return try JSONDecoder().decode(type, from: Data(json.utf8)) } catch {
        print("  decode error: \(error)")
        return nil
    }
}

print("the rules list decodes the documented contract:")
let rules = decode(DeviceRules.self, """
{"rules_version": "abc123", "additive": true, "rules": [
  {"id": "r1", "enabled": true, "summary": "Dog on the couch", "targets": ["dog", "couch"], "zone": "",
   "required_capabilities": ["objects"], "status": "running", "reason": null},
  {"id": "r2", "enabled": true, "summary": "Someone at the door", "targets": ["person"], "zone": "door",
   "dwell_s": 20, "status": "waiting", "reason": null},
  {"id": "r3", "enabled": true, "summary": "Laptop picked up", "status": "rejected", "reason": "holding needs hands"},
  {"id": "r4", "enabled": false, "summary": "Empty room", "status": "disabled"},
  {"id": 5, "summary": "Numeric id", "status": "brand_new"},
  {"summary": "no id"}
]}
""")
check(rules?.rulesVersion == "abc123" && rules?.additive == true, "version and additive")
check(rules?.rules.count == 5, "a rule without an id is dropped; a numeric id is kept")
check(rules?.rules[0].zone == nil, "an empty zone is no zone")
check(rules?.rules[1].zone == "door" && rules?.rules[1].dwellS == 20, "zone and dwell")
check(rules?.rules[4].id == "5" && rules?.rules[4].enabled == true, "numeric id as text; enabled defaults on")
check(decode(DeviceRules.self, "{}")?.rules.isEmpty == true, "an empty object is no rules")

print("\nstatus chips:")
func chip(_ i: Int, live: Bool = true) -> RuleStatusPresentation { RuleStatusPresentation.make(rules!.rules[i], deviceLive: live) }
check(chip(0).chip == "Running" && chip(0).tone == .ok, "running")
check(chip(1).chip == "Waiting for area" && chip(1).detail?.contains("door") == true, "waiting for a named area")
check(chip(2).chip == "Rejected" && chip(2).tone == .critical && chip(2).detail?.contains("holding needs hands") == true,
      "rejected with the device's reason")
check(chip(3).chip == "Off" && chip(3).tone == .neutral, "disabled")
check(chip(4).chip == "Unknown" && chip(4).tone == .neutral, "an unknown status is unknown, not running")
check(chip(0, live: false).chip == "Unknown" && chip(0, live: false).tone != .ok,
      "never running on a device that isn't reporting")
check(chip(2, live: false).chip == "Rejected", "a rejection stays visible while offline")
let waitingNoZone = DeviceRule(id: "w", enabled: true, summary: "x", status: "waiting")
check(RuleStatusPresentation.make(waitingNoZone, deviceLive: true).chip == "Waiting", "waiting without an area")

print("\ntoggle errors:")
check(RuleToggleError.message(code: nil, detail: "Not Found", endpointMissing: true).contains("isn't supported"),
      "missing endpoint")
check(RuleToggleError.message(code: "rule_not_found", detail: "x", endpointMissing: false).contains("no longer exists"),
      "rule_not_found")
check(RuleToggleError.message(code: nil, detail: "nope", endpointMissing: false) == "nope", "other detail kept")

print("\nwhat the device reports after an activation:")
let all = rules!.rules
check(ActivationFollowUp.evaluate(rulesExpected: 2, modeTitle: nil, modePending: false, deviceLive: true,
                                  rejectedReasons: ["face model missing"], runningRuleIDs: nil, rules: nil)
      == .refused("The device refused these rules: face model missing. It kept what it was running before."),
      "a refusal of the whole set wins")
check(ActivationFollowUp.evaluate(rulesExpected: 2, modeTitle: nil, modePending: false, deviceLive: false,
                                  rejectedReasons: nil, runningRuleIDs: nil, rules: nil).text.contains("isn't reporting"),
      "offline is waiting, not confirmed")
if case .refused(let text) = ActivationFollowUp.evaluate(rulesExpected: 3, modeTitle: nil, modePending: false, deviceLive: true,
                                                        rejectedReasons: nil, runningRuleIDs: nil, rules: all) {
    check(text.contains("Laptop picked up"), "a rejected rule in the list is named")
} else { check(false, "a rejected rule in the list is a refusal") }
let fine = Array(all[0...1])
check(ActivationFollowUp.evaluate(rulesExpected: 2, modeTitle: nil, modePending: false, deviceLive: true,
                                  rejectedReasons: nil, runningRuleIDs: nil, rules: fine)
      == .confirmed("The device reports 1 rule running, 1 waiting for an area."), "running and waiting are counted")
check(ActivationFollowUp.evaluate(rulesExpected: 1, modeTitle: nil, modePending: false, deviceLive: true,
                                  rejectedReasons: nil, runningRuleIDs: [], rules: nil)
      == .waiting("Waiting for the device to confirm the rules…"), "nothing running yet is waiting")
check(ActivationFollowUp.evaluate(rulesExpected: 1, modeTitle: nil, modePending: false, deviceLive: true,
                                  rejectedReasons: nil, runningRuleIDs: ["a", "b"], rules: nil)
      == .confirmed("The device reports 2 rules running."), "status.rules.active without the rules endpoint")
check(ActivationFollowUp.evaluate(rulesExpected: 0, modeTitle: "Driver Safety", modePending: true, deviceLive: true,
                                  rejectedReasons: nil, runningRuleIDs: nil, rules: nil)
      == .waiting("Waiting for the device to switch to Driver Safety…"), "a pending mode switch waits")
check(ActivationFollowUp.evaluate(rulesExpected: 0, modeTitle: "Driver Safety", modePending: false, deviceLive: true,
                                  rejectedReasons: nil, runningRuleIDs: nil, rules: nil)
      == .confirmed("The device reports it is running Driver Safety."), "a converged mode switch confirms")
check(ActivationFollowUp.timedOut(.waiting("x")).text.contains("hasn't confirmed"), "timeout says so")
check(ActivationFollowUp.timedOut(.confirmed("y")) == .confirmed("y"), "timeout never overwrites a result")

print(failures == 0 ? "\nall passed" : "\n\(failures) FAILED")
exit(failures == 0 ? 0 : 1)
