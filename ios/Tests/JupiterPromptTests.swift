import Foundation

// Run with ios/Tests/run.sh
//
// JupiterPrompt must agree with jupiter/security.py sanitise_request: trim,
// refuse empty, refuse (never truncate) anything over 600 code points.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

print("empty and whitespace-only requests cannot be sent:")
check(!JupiterPrompt.canSubmit(""), "an empty bar cannot send")
check(!JupiterPrompt.canSubmit("   \n\t "), "whitespace alone cannot send")
check(JupiterPrompt.normalized("  \n ") == nil, "whitespace normalises to nothing")

print("\nordinary requests are trimmed and sent:")
check(JupiterPrompt.canSubmit("Watch for someone entering the room after 10 PM."), "a sentence can send")
check(JupiterPrompt.normalized("  alert me when the room is empty \n") == "alert me when the room is empty",
      "surrounding whitespace is trimmed, inner text untouched")

print("\nthe 600-character limit matches the backend:")
let exactly600 = String(repeating: "a", count: 600)
let over600 = String(repeating: "a", count: 601)
check(JupiterPrompt.canSubmit(exactly600), "exactly 600 characters is allowed")
check(!JupiterPrompt.canSubmit(over600), "601 characters is refused, not truncated")
check(JupiterPrompt.isOverLimit(over600), "601 characters is reported as over the limit")
check(JupiterPrompt.canSubmit("  " + exactly600 + "  "), "trimmed whitespace does not count toward the limit")

print("\nlength is counted in code points, like Python's len():")
// One family emoji: 1 Character, 7 Unicode scalars (4 people + 3 joiners).
let family = "👨‍👩‍👧‍👦"
check(family.count == 1, "fixture: the emoji is one Character")
check(JupiterPrompt.length(family) == 7, "but seven code points, as the backend counts it")
let nearLimit = String(repeating: "a", count: 595) + family
check(!JupiterPrompt.canSubmit(nearLimit), "595 letters + a 7-scalar emoji is over 600 for the backend")

print("\nthe counter appears only near the limit:")
check(!JupiterPrompt.showsCount("short request"), "no counter on a short request")
check(JupiterPrompt.showsCount(String(repeating: "a", count: 480)), "counter from 480 characters")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
