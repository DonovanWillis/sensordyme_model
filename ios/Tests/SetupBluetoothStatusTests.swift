import Foundation

// Run with ios/Tests/run.sh
//
// Bluetooth setup names the actual problem: permission, restriction, power, or
// no Bluetooth at all, each with its own fix.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

typealias B = SetupBluetoothStatus

print("the radio's own report decides:")
check(B.problem(radio: .on, access: .allowed) == nil, "on and allowed: no problem")
check(B.problem(radio: .off, access: .allowed) == .poweredOff, "powered off")
check(B.problem(radio: .unauthorized, access: .denied) == .permissionDenied, "unauthorized + denied: permission")
check(B.problem(radio: .unauthorized, access: .restricted) == .restricted, "unauthorized + restricted: restriction")
check(B.problem(radio: .unauthorized, access: .allowed) == .permissionDenied,
      "unauthorized whatever the cached access says: permission")
check(B.problem(radio: .unsupported, access: .allowed) == .unsupported, "no Bluetooth LE")
check(B.problem(radio: .unsupported, access: .denied) == .unsupported, "no Bluetooth LE wins over permission")

print("\nbefore the radio answers:")
check(B.problem(radio: nil, access: .notDetermined) == nil, "not asked yet: no problem (iOS asks)")
check(B.problem(radio: nil, access: .allowed) == nil, "allowed: wait for the radio")
check(B.problem(radio: nil, access: .denied) == .permissionDenied, "a refusal on record is the answer")
check(B.problem(radio: .unknown, access: .restricted) == .restricted, "a restriction on record is the answer")

print("\nthe fix matches the problem:")
check(B.Problem.permissionDenied.opensAppSettings, "permission: Open Settings")
check(!B.Problem.poweredOff.opensAppSettings, "power: no Settings button (iOS has no link to Bluetooth)")
check(!B.Problem.restricted.opensAppSettings, "restriction: no Settings button")
check(!B.Problem.unsupported.opensAppSettings, "unsupported: no Settings button")
check(B.Problem.poweredOff.detail.contains("Control Centre"), "power names Control Centre")
check(B.Problem.permissionDenied.detail.contains("Settings"), "permission names Settings")
check(Set([B.Problem.permissionDenied, .restricted, .poweredOff, .unsupported].map(\.title)).count == 4,
      "four different titles")

print("\n\(failures == 0 ? "PASSED" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
