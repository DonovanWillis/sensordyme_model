import Foundation

// Run with ios/Tests/run.sh
//
// Text-message alerts: the form accepts what the server accepts (cloud/app.py
// set_phone) and offers Save only when it would change something.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

typealias F = TextAlertsForm

print("the server's rule:")
check(F.isValid("+15551234567"), "E.164")
check(F.isValid("+1 555 123 4567"), "with spaces (removed before sending)")
check(F.normalized(" +1 555 123 4567 ") == "+15551234567", "normalised")
check(!F.isValid("5551234567"), "no +")
check(!F.isValid("+1555"), "too short")
check(!F.isValid("+1-555-123-4567"), "dashes")
check(!F.isValid("+1555123456٧"), "non-ASCII digits")
check(F.problem("") == nil, "nothing typed: no complaint")
check(F.problem("+15551234567") == nil, "valid: no complaint")
check(F.problem("5551234567")?.contains("+") == true, "missing +: says so")
check(F.problem("+1 555 abc") != nil, "letters: says so")

print("\nSave only when it changes something:")
check(!F.canSave(phone: "", textOnSerious: true, savedPhone: "", savedTextOnSerious: true), "empty: no")
check(F.canSave(phone: "+15551234567", textOnSerious: true, savedPhone: "", savedTextOnSerious: true), "new number: yes")
check(!F.canSave(phone: "+1 555 123 4567", textOnSerious: true, savedPhone: "+15551234567", savedTextOnSerious: true),
      "same number, spaced differently: no")
check(F.canSave(phone: "+15551234567", textOnSerious: false, savedPhone: "+15551234567", savedTextOnSerious: true),
      "the switch changed: yes (it is saved with the number)")
check(!F.canSave(phone: "5551234567", textOnSerious: false, savedPhone: "", savedTextOnSerious: true), "invalid: no")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
