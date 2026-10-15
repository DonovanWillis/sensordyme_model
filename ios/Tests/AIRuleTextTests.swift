import Foundation

// Run via ios/Tests/run.sh, which concatenates ios/Sources/AIRuleText.swift with this.
//
// Two things the Configure-with-AI review screen got wrong: every predicate it
// did not know read "appears", and its sliders were narrower than the rules they
// edited, so touching one clamped a valid rule.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

func headline(_ target: String, _ predicate: String, secondary: String = "",
              zone: String = "", count: Int = 1) -> String {
    RuleWording.headline(target: target, predicate: predicate, secondary: secondary,
                         zone: zone, minCount: count)
}

print("each predicate says what it watches for:")
let cup = headline("person", "near_mouth", secondary: "cup")
let bottle = headline("person", "near_mouth", secondary: "bottle")
check(cup != bottle, "two drinking rules no longer both read \"Person appears\"")
check(cup == "Person has cup at their mouth", "near_mouth names the object at the mouth (got \(cup))")
check(!cup.contains("appears"), "and is not a presence rule")
check(headline("person", "count") == "Person count changes", "a count rule reads as a count")
check(headline("bottle", "count", count: 3) == "Bottle count reaches 3+", "with its threshold")
check(headline("person", "still") == "Person stays still", "a stillness rule reads as stillness")
check(headline("person", "holding", secondary: "knife") == "Person is holding knife", "holding")
check(headline("dog", "near", secondary: "couch") == "Dog is near couch", "near")
check(headline("person", "present") == "Person appears", "presence is unchanged")
check(headline("person", "present", count: 3) == "Person appears (3+)", "with a minimum count")
check(headline("person", "absent") == "Person is gone", "absence")
check(headline("person", "in_zone", zone: "kitchen") == "Person is in kitchen", "a zone")
check(headline("person", "in_zone") == "Person is in the area", "an unnamed zone")
check(headline("person", "lingering") == "Person lingers", "lingering")
check(headline("person", "climbing") == "Person: climbing", "an unknown predicate is named, not called 'appears'")

print("\nthe sliders cover the whole range a rule may carry:")
let persist = DurationScale.persistence
let cooldown = DurationScale.cooldown
check(persist.seconds(at: 0) == 0.5 && persist.seconds(at: 1) == 3600,
      "Confirm for spans 0.5 s to an hour (the schema's range), not 1-60 s")
check(cooldown.seconds(at: 0) == 10 && cooldown.seconds(at: 1) == 86_400,
      "Then wait spans 10 s to a day, not 30 s to 30 min")
for value in [0.5, 2.0, 30.0, 90.0, 120.0, 900.0, 3600.0] {
    let roundTrip = persist.seconds(at: persist.position(for: value))
    check(abs(roundTrip - value) < 0.001, "\(value)s sits on the scale exactly (got \(roundTrip))")
}
check(abs(cooldown.seconds(at: cooldown.position(for: 60)) - 60) < 0.001, "a 60 s cooldown sits on the scale")
check(persist.position(for: 99_999) == 1 && persist.position(for: 0) == 0, "out-of-range values pin to an end")
check(persist.position(for: .nan) == 0 && persist.seconds(at: .nan) == 0.5, "NaN cannot escape the range")
var previous = 0.0
var monotonic = true
for step in 0...100 {
    let value = persist.seconds(at: Double(step) / 100)
    if value < previous { monotonic = false }
    previous = value
}
check(monotonic, "moving the slider right never shortens the duration")
check(persist.position(for: 3) < 0.5, "the short end, where precision matters, gets most of the track")

print("\ndurations read naturally:")
check(DurationText.short(0.5) == "0.5s", "half a second")
check(DurationText.short(45) == "45s", "seconds")
check(DurationText.short(90) == "1m 30s", "minutes and seconds")
check(DurationText.short(120) == "2m", "whole minutes")
check(DurationText.short(3600) == "1h", "an hour")
check(DurationText.short(5400) == "1h 30m", "hours and minutes")
check(DurationText.short(86_400) == "24h", "a day")

print("\nan activation says what the backend confirmed (rules_active):")
func summary(custom: Bool, sent: Int, active: Int?) -> ActivationSummary {
    ActivationSummary.summarize(isCustom: custom, rulesSent: sent, rulesActive: active)
}
check(summary(custom: true, sent: 2, active: nil) == ActivationSummary(kind: .active, note: nil),
      "an older backend (no rules_active): today's behaviour, no extra claim")
check(summary(custom: true, sent: 2, active: 2).kind == .active
      && summary(custom: true, sent: 2, active: 2).note == "2 rules running.", "all custom rules running")
check(summary(custom: true, sent: 1, active: 1).note == "1 rule running.", "singular")
check(summary(custom: true, sent: 3, active: 1).note?.contains("1 of 3 rules running") == true,
      "some refused: says how many run, points at Not yet possible")
check(summary(custom: true, sent: 2, active: 0).kind == .nothingRunning,
      "a custom proposal with nothing running is NOT 'Active'")
let switched = summary(custom: false, sent: 1, active: 0)
check(switched.kind == .modeSwitched, "a named mode with its rules refused is a mode switch, not a failure")
check(switched.note?.contains("AI Custom") == true, "and says the rules need AI Custom")
check(summary(custom: false, sent: 0, active: 0) == ActivationSummary(kind: .modeSwitched, note: nil),
      "a keyword-fallback proposal (a mode, no rules) just switches, with nothing to qualify")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
