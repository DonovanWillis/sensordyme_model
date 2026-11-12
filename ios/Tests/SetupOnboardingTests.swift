import Foundation

// Run with ios/Tests/run.sh

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

typealias S = Onboarding.State
typealias E = Onboarding.Event

/// Walk `events` from `start`; the states visited, or nil at the first illegal move.
func walk(_ start: S, _ events: [E], _ context: Onboarding.Context = .init(signedIn: true, ssid: "Home")) -> [S]? {
    var state = start
    var visited = [state]
    for event in events {
        guard let next = Onboarding.next(state, event, context: context) else {
            print("       illegal: \(event) from \(state)")
            return nil
        }
        state = next
        visited.append(state)
    }
    return visited
}

print("first-time setup, start to finish:")
let firstTime = walk(.discovering, [
    .deviceChosen(name: "SensorDyme-3D16"), .identified(requiresCode: true), .proofStarted, .proven,
    .networkSent(ssid: "Home"), .deviceJoining, .deviceJoined, .needsClaim, .claimSubmitted, .claimed,
    .waitTick(elapsed: 3), .deviceOnline(deviceID: 7), .modeChosen("home_security"), .modeActivated("home_security"),
], .init(signedIn: true, ssid: "Home", accountDeviceID: 7))
check(firstTime != nil, "every step is a legal move")
check(firstTime?.map(\.stage) == [.find, .connect, .prove, .prove, .network, .network, .join, .account, .account,
                                   .account, .online, .online, .mode, .mode, .mode],
      "the checklist stage follows the journey")
if case .finished(let outcome)? = firstTime?.last {
    check(outcome.mode == "home_security" && outcome.accountDeviceID == 7 && outcome.onAccount && !outcome.alreadyOwned,
          "finished on the account in the chosen mode")
} else { check(false, "finished") }

print("\na legacy device skips the code:")
check(walk(.connecting(name: "x"), [.identified(requiresCode: false)])?.last == .choosingNetwork(problem: nil),
      "straight to the network list")

print("\nwrong codes, lockout and the device's own wait:")
let until = Date(timeIntervalSince1970: 1_000)
let refused = walk(.enteringSetupCode(problem: nil, lockedUntil: nil), [
    .proofStarted, .proofRefused(message: "wrong", lockedUntil: nil),
    .proofStarted, .proofRefused(message: "locked", lockedUntil: until),
])
check(refused?.last == .enteringSetupCode(problem: "locked", lockedUntil: until), "the lockout time is the one reported")
check(refused?.last?.stage == .prove, "still on the proving step: nothing was sent")

print("\na refused password returns to the list with the device's reason; the link is kept:")
check(walk(.choosingNetwork(problem: nil), [.networkSent(ssid: "Home"), .deviceJoining, .joinRefused("not accepted")])?.last
      == .choosingNetwork(problem: "not accepted"), "back to choosing, reason attached")

print("\nBluetooth lost mid-join: the outcome is unknown, so offer to check:")
if case .failed(let failure)? = walk(.joining(ssid: "Home"), [.linkFailed("dropped")])?.last {
    check(failure.stage == .join && failure.recoveries == [.checkOnline, .searchAgain], "check online, or search again")
    check(walk(.failed(failure), [.recover(.checkOnline)])?.last == .associating(ssid: "Home"), "checking goes to the account")
} else { check(false, "a failure") }
if case .failed(let failure)? = walk(.joining(ssid: "Home"), [.linkFailed("dropped")], .init(signedIn: false))?.last {
    check(failure.recoveries == [.searchAgain], "signed out: only search again")
}

print("\nan already-claimed device is not claimed again:")
let owned = walk(.associating(ssid: "Home"), [.alreadyOnAccount(deviceID: 4), .deviceOnline(deviceID: 4), .modeSkipped],
                 .init(signedIn: true, ssid: "Home", accountDeviceID: 4, alreadyOwned: true))
if case .finished(let outcome)? = owned?.last {
    check(outcome.alreadyOwned && outcome.accountDeviceID == 4 && outcome.mode == nil, "finished as already owned")
} else { check(false, "finished") }

print("\nclaim refusals and a silent device:")
check(walk(.enteringClaimCode(problem: nil), [.claimSubmitted, .claimRefused("unknown")])?.last
      == .enteringClaimCode(problem: "unknown"), "a refused claim asks again with the reason")
if case .failed(let failure)? = walk(.waitingForDevice(elapsed: 0), [.deviceSilent("quiet")])?.last {
    check(failure.stage == .online && failure.recoveries == [.keepWaiting], "keep waiting is offered")
    check(walk(.failed(failure), [.recover(.keepWaiting)], .init(signedIn: true, ssid: "Home", accountDeviceID: 3))?.last
          == .waitingForDevice(elapsed: 0), "and waits again")
}
check(walk(.associating(ssid: "Home"), [.noAccount(found: true)], .init(signedIn: false, ssid: "Home"))?.last
      == .finished(.init(ssid: "Home")), "signed out: Wi-Fi only, finished when it answers locally")

print("\na refused mode can be chosen again or skipped:")
if case .failed(let failure)? = walk(.activatingMode(deviceID: 2, mode: "baby_monitor"), [.modeRefused("role")])?.last {
    check(failure.recoveries == [.chooseMode, .skipMode], "choose again or skip")
    check(walk(.failed(failure), [.recover(.chooseMode)], .init(signedIn: true, accountDeviceID: 2))?.last
          == .choosingMode(deviceID: 2), "choose again")
}

print("\nillegal moves are refused and change nothing:")
check(Onboarding.next(.discovering, .proven) == nil, "no proof before a device")
check(Onboarding.next(.enteringSetupCode(problem: nil, lockedUntil: nil), .networkSent(ssid: "x")) == nil,
      "no network before the proof")
check(Onboarding.next(.choosingNetwork(problem: nil), .claimed) == nil, "no claim before the join")
check(Onboarding.next(.associating(ssid: "x"), .modeChosen("x")) == nil, "no mode before the device is on the account")
check(Onboarding.next(.joining(ssid: "x"), .recover(.searchAgain)) == nil, "no restart while the device is joining")
check(Onboarding.next(.failed(.init(stage: .mode, reason: "x", recoveries: [.skipMode])), .recover(.retryClaim)) == nil,
      "only an offered recovery")
check(Onboarding.next(.failed(.init(stage: .mode, reason: "x", recoveries: [.skipMode])), .recover(.searchAgain))
      == .discovering, "searching again is always possible")
check(Onboarding.next(.identifying(name: "x"), .bluetoothOff) == .bluetoothUnavailable, "Bluetooth off before sending")

print("\nmatching the account's device to the one on Bluetooth:")
let rows = [
    Onboarding.AccountDevice(id: 1, name: "Car", hardwareID: "SD-1A2B3C4D", online: true, activePreset: "driver_safety"),
    Onboarding.AccountDevice(id: 2, name: "Hall", hardwareID: nil, online: false, activePreset: nil),
]
check(Onboarding.match(rows, hardwareID: "sd-1a2b3c4d")?.id == 1, "by the SD- id, case-insensitive")
check(Onboarding.match(rows, hardwareID: nil) == nil, "never by guessing")
check(Onboarding.match(rows, hardwareID: "SD-99999999") == nil, "an unknown id matches nothing")
let after = rows + [Onboarding.AccountDevice(id: 3, name: "Other", hardwareID: "SD-0BADC0DE", online: true, activePreset: nil)]
check(Onboarding.claimedOther(before: rows, after: after, hardwareID: "SD-FEEDFACE")?.id == 3,
      "a claim that added another unit is noticed")
check(Onboarding.claimedOther(before: rows, after: after, hardwareID: "SD-0BADC0DE") == nil, "the right unit is fine")
check(Onboarding.claimedOther(before: [], after: [rows[1]], hardwareID: "SD-FEEDFACE") == nil,
      "a new row that has not reported (no id) is not held against the claim")

print("\nclaim codes and refusals:")
check(Onboarding.normalizeClaimCode(" ab12-cd34 ef56 ") == "AB12-CD34EF56", "case and spaces forgiven")
check(Onboarding.normalizeClaimCode("") == nil && Onboarding.normalizeClaimCode("AB;12") == nil, "junk refused")
check(Onboarding.claimMessage(status: 409, code: "device_already_claimed", detail: nil).contains("another account"),
      "already claimed")
check(Onboarding.claimMessage(status: 404, code: nil, detail: nil).contains("isn't one the server knows"), "unknown code")
check(Onboarding.claimMessage(status: 429, code: nil, detail: nil).contains("Too many"), "rate limited")

print("\nthe recommended mode:")
let modes = [
    Onboarding.Mode(id: "driver_safety", title: "Driver", detail: "", available: false, reason: "role"),
    Onboarding.Mode(id: "home_security", title: "Home", detail: "", available: true, reason: nil),
    Onboarding.Mode(id: "baby_monitor", title: "Baby", detail: "", available: true, reason: nil),
]
check(Onboarding.recommendedMode(modes, active: "baby_monitor:strict")?.id == "baby_monitor", "what it already runs")
check(Onboarding.recommendedMode(modes, active: "driver_safety")?.id == "home_security", "never an unavailable one")
check(Onboarding.recommendedMode(modes, active: nil)?.id == "home_security", "else the first available")

print("\n\(failures == 0 ? "all passed" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
