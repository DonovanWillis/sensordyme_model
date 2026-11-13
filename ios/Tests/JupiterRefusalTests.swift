import Foundation

// Standalone, run by ios/Tests/run.sh (concatenated with ios/Sources/JupiterRefusal.swift).
//
// M8.4: every refusal Ask Jupiter shows comes from the backend's typed `code`
// (jupiter/policy.py UNSUPPORTED_CODES), and every code has its own headline.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

print("every backend code has its own headline:")
let codes = ["identity", "emotion", "intent", "audio", "intoxication", "medical", "repetition_window",
             "experimental_signal", "not_detectable", "not_supported", "no_target", "needs_area",
             "needs_second_object", "invalid_rule", "not_beside_mode", "interpreter_unavailable"]
let headlines = codes.map { JupiterRefusal.headline(code: $0, support: "unsupported") }
check(Set(headlines).count == codes.count - 1,
      "distinct, except not_detectable/not_supported which both defer to `support`")
check(JupiterRefusal.headline(code: "experimental_signal", support: nil) == "Experimental: recorded, never an alert",
      "an experimental signal is said to be recorded, never an alert")
check(JupiterRefusal.headline(code: "identity", support: nil) == "Never identifies people", "identity")

print("\n`support` refines a not-detectable answer, and is the whole answer from an older backend:")
check(JupiterRefusal.headline(code: "not_detectable", support: "needs_training_data")
      == "The camera doesn't recognise this yet", "needs training data")
check(JupiterRefusal.headline(code: nil, support: "needs_hardware") == "Needs extra hardware", "no code: support")
check(JupiterRefusal.headline(code: "something_new", support: nil) == "Not possible from the camera alone",
      "an unknown code falls back, never blank")

print("\na policy refusal says why, never 'to enable':")
check(JupiterRefusal.requiresPrefix(code: "audio") == "Why", "audio")
check(JupiterRefusal.requiresPrefix(code: "experimental_signal") == "Why", "experimental")
check(JupiterRefusal.requiresPrefix(code: "not_detectable") == "To enable", "not detectable")
check(JupiterRefusal.requiresPrefix(code: nil) == "To enable", "older backend")
check(JupiterRefusal.isPolicy(code: "medical") && !JupiterRefusal.isPolicy(code: "needs_area"), "isPolicy")

print("\nthe preview's typed policies decode and match their rules:")
let json = """
[{"rule_id": "r-2", "status": "ready", "zone": {"name": "front door", "drawn": true},
  "explanation": {"summary": "Alert when a person stays inside the area 'front door' for 10 seconds or longer.",
                  "monitors": ["the object detector's 'person' class"],
                  "does_not_monitor": ["who the person is (no face recognition)"], "status": null},
  "targets": ["person"], "schema_version": 1},
 {"rule_id": "r-3", "status": "needs_zone", "zone": {"name": "porch", "drawn": false}, "explanation": null}]
""".data(using: .utf8)!
let policies = try? JSONDecoder().decode([AIPolicyPreview].self, from: json)
check(policies?.count == 2, "decodes, unknown keys ignored")
check(AIPolicyPreview.matching(policies, ruleID: "r-3", index: 0, ruleCount: 2)?.zone?.drawn == false,
      "by rule id first")
check(AIPolicyPreview.matching(policies, ruleID: nil, index: 0, ruleCount: 2)?.zone?.name == "front door",
      "by position when the lists line up")
check(AIPolicyPreview.matching(policies, ruleID: nil, index: 0, ruleCount: 3) == nil,
      "never by position when they don't")
check(AIPolicyPreview.matching(nil, ruleID: "r-2", index: 0, ruleCount: 1) == nil, "an older backend: none")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
