import Foundation

// The app's onboarding code against the device's Python state machine.
//
//   onboarding-harness --endpoint 127.0.0.1:PORT --scenario NAME --code-file PATH
//                      [--ssid S --psk-file PATH] [--evidence FILE]
//                      [--backend URL --account-file PATH --claim-file PATH]
//
// Real SetupFlow + Onboarding + BLEProvisioner + BLEPoPFlow +
// BLEProofOfPossession, over BLESimulatedTransport (the app's DEBUG link) to
// sim/ble_peripheral.py, which runs services/provisioning.py and
// services/ble_pop.py. An InterceptingTransport sits in the middle to play an
// attacker (tamper), a slow phone (delay) or simply to count what was written.
// Secrets arrive in files, never on the command line, and never reach the
// evidence. Exit 0 when every check of the scenario held.

let arguments = CommandLine.arguments
func option(_ name: String) -> String? {
    guard let index = arguments.firstIndex(of: name), arguments.indices.contains(index + 1) else { return nil }
    return arguments[index + 1]
}
func secret(_ name: String) -> String? {
    option(name).flatMap { try? String(contentsOfFile: $0, encoding: .utf8) }?
        .trimmingCharacters(in: .whitespacesAndNewlines)
}

let scenario = option("--scenario") ?? "success"
let evidence = Evidence(path: option("--evidence"))

@MainActor
func waitFor(_ what: String, timeout: Double = 20, _ condition: () -> Bool) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() {
        if Date() > deadline { throw HarnessError(description: "timed out waiting for \(what)") }
        try await Task.sleep(nanoseconds: 25_000_000)
    }
}

@MainActor
func run() async throws {
    guard let endpointText = option("--endpoint"), let endpoint = BLESimulatedTransport.parse(endpointText) else {
        throw HarnessError(description: "--endpoint 127.0.0.1:PORT (loopback only) is required")
    }
    let code = secret("--code-file") ?? ""
    let ssid = option("--ssid") ?? "SensorDyme Lab"
    let psk = secret("--psk-file") ?? ""

    let mitm = InterceptingTransport(inner: BLESimulatedTransport(endpoints: [endpoint]))
    mitm.onTamper = { evidence.record("mitm", ["action": $0]) }
    let ble = BLEProvisioner(transport: mitm)
    // A stale proof is sent as is, so the device's own expiry is what is tested.
    ble.timing.provenFresh = 3600
    var account: HTTPAccount?
    if let backend = option("--backend").flatMap(URL.init(string:)) {
        let http = HTTPAccount(base: backend, evidence: evidence)
        let login = (secret("--account-file") ?? "").split(separator: "\n").map(String.init)
        guard login.count == 2 else { throw HarnessError(description: "--account-file must hold email and password lines") }
        try await http.signIn(email: login[0], password: login[1])
        account = http
    }
    let flow = SetupFlow(ble: ble, localLink: nil, account: account)
    flow.reportBudgetSeconds = Int(option("--report-budget") ?? "") ?? 90
    flow.pollSeconds = 1

    var seenTransitions = 0
    var seenBLE: BLEProvisioner.State?
    let recorder = Task { @MainActor in
        while !Task.isCancelled {
            while seenTransitions < flow.transitions.count {
                let t = flow.transitions[seenTransitions]
                evidence.record("transition", ["from": "\(t.from)", "event": "\(t.event)", "to": "\(t.to)"])
                seenTransitions += 1
            }
            if ble.state != seenBLE {
                seenBLE = ble.state
                evidence.record("ble_state", ["state": "\(ble.state)"])
            }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
    }
    defer { recorder.cancel() }

    func isState(_ match: (Onboarding.State) -> Bool) -> Bool { match(flow.state) }
    func connectToDevice() async throws {
        flow.start()
        try await waitFor("the device to advertise") { !ble.discovered.isEmpty }
        evidence.record("discovered", ["name": ble.discovered[0].name, "rssi": ble.discovered[0].rssi])
        flow.connect(to: ble.discovered[0])
        try await waitFor("identification") {
            if case .enteringSetupCode = flow.state { return true }
            if case .choosingNetwork = flow.state { return true }
            if case .failed = flow.state { return true }
            return false
        }
        evidence.record("identified", ["device_id": ble.deviceID ?? "", "auth": ble.deviceAuth ?? "",
                                       "requires_code": ble.requiresSetupCode])
    }
    func proveExpecting(_ entry: String, proven: Bool, timeout: Double = 20) async throws {
        let problem = flow.prove(entry: entry)
        evidence.check(problem == nil, "the entry is a usable setup code", problem.map { "\($0)" } ?? "")
        try await waitFor("the proof outcome", timeout: timeout) {
            switch flow.state {
            case .choosingNetwork, .enteringSetupCode, .failed: return true
            default: return false
            }
        }
        if proven {
            evidence.check(isState { if case .choosingNetwork = $0 { return true }; return false }, "the device accepted the code")
        }
    }
    func sendAndWaitForJoin() async throws {
        try await waitFor("the device's network list") { !ble.networks.isEmpty }
        evidence.record("networks", ["ssids": ble.networks.map(\.ssid)])
        flow.send(ssid: ssid, password: psk)
        try await waitFor("the join outcome", timeout: 40) {
            switch flow.state {
            case .associating, .enteringClaimCode, .waitingForDevice, .choosingMode, .failed, .choosingNetwork, .finished,
                 .enteringSetupCode:
                return true
            default: return false
            }
        }
    }
    func joined() -> Bool {
        flow.transitions.contains { $0.event == .deviceJoined }
    }

    switch scenario {
    case "success":
        try await connectToDevice()
        try await proveExpecting(code, proven: true)
        try await sendAndWaitForJoin()
        evidence.check(joined(), "the device joined the network")
        evidence.check(mitm.authWrites() == ["hello", "proof", "creds"], "hello, proof, creds on the auth characteristic",
                       mitm.authWrites().joined(separator: ","))
        evidence.check(mitm.writes.allSatisfy { $0.0 != .credentials }, "nothing on the plain credential characteristic")

    case "first-time", "reprovision":
        guard account != nil else { throw HarnessError(description: "\(scenario) needs --backend") }
        try await connectToDevice()
        try await proveExpecting(code, proven: true)
        try await sendAndWaitForJoin()
        evidence.check(joined(), "the device joined the network")
        try await waitFor("the account step", timeout: 20) {
            switch flow.state {
            case .enteringClaimCode, .waitingForDevice, .choosingMode, .failed: return true
            default: return false
            }
        }
        if scenario == "first-time" {
            evidence.check(isState { if case .enteringClaimCode = $0 { return true }; return false },
                           "a new device has to be claimed")
            // A wrong claim code first: refused by the server, and said so.
            _ = flow.submitClaim(code: "0000-0000-0000")
            try await waitFor("the claim refusal") {
                if case .enteringClaimCode(let problem) = flow.state { return problem != nil }
                return false
            }
            if case .enteringClaimCode(let problem) = flow.state {
                evidence.check(problem?.contains("isn't one the server knows") == true, "an unknown claim code is explained",
                               problem ?? "")
            }
            let claim = secret("--claim-file") ?? ""
            evidence.check(flow.submitClaim(code: claim), "the printed claim code is accepted as input")
        } else {
            evidence.check(flow.transitions.contains { if case .alreadyOnAccount = $0.event { return true }; return false },
                           "an already-claimed device is recognised by its Bluetooth id, no claim asked")
        }
        try await waitFor("the device to report in", timeout: Double(flow.reportBudgetSeconds + 15)) {
            switch flow.state {
            case .choosingMode, .failed: return true
            default: return false
            }
        }
        guard case .choosingMode(let deviceID) = flow.state else {
            throw HarnessError(description: "expected to choose a mode, got \(flow.state)")
        }
        try await waitFor("the modes") { flow.recommendedMode != nil }
        let mode = flow.recommendedMode ?? ""
        evidence.record("modes", ["offered": flow.modes.filter(\.available).map(\.id), "recommended": mode])
        flow.chooseMode(mode)
        try await waitFor("the mode to be set", timeout: 20) {
            switch flow.state {
            case .finished, .failed: return true
            default: return false
            }
        }
        if case .finished(let outcome) = flow.state {
            evidence.check(outcome.accountDeviceID == deviceID && outcome.mode == mode && outcome.onAccount,
                           "finished on the account with the chosen mode", "\(outcome)")
            evidence.check(outcome.alreadyOwned == (scenario == "reprovision"), "already-owned is reported truthfully")
        } else {
            evidence.check(false, "finished", "\(flow.state)")
        }

    case "wrong-code":
        try await connectToDevice()
        try await proveExpecting("ZZZZ-ZZZZ-ZZZZ", proven: false)
        guard case .enteringSetupCode(let problem, _) = flow.state else {
            throw HarnessError(description: "expected the code again, got \(flow.state)")
        }
        evidence.check(problem?.contains("setup code was wrong") == true, "the device's refusal is shown", problem ?? "")
        evidence.check(ble.proofRejection?.kind == .wrongCode, "kind: wrong code")
        evidence.check(mitm.authWrites() == ["hello", "proof"], "no credentials were sent", mitm.authWrites().joined(separator: ","))
        try await proveExpecting(code, proven: true)
        evidence.check(true, "the right code then works on the same link")

    case "lockout":
        // The peripheral runs with --max-failures 3 --lockout-base 2.
        try await connectToDevice()
        for attempt in 1...3 {
            try await proveExpecting("ZZZZ-ZZZZ-ZZZZ", proven: false)
            evidence.record("refused", ["attempt": attempt, "retry_after_s": ble.proofRejection?.retryAfterSeconds ?? -1])
        }
        let reported = ble.proofRejection?.retryAfterSeconds
        evidence.check(reported != nil, "the failure that starts the lockout carries the device's wait")
        guard case .enteringSetupCode(_, let lockedUntil) = flow.state, let lockedUntil else {
            throw HarnessError(description: "expected a lockout, got \(flow.state)")
        }
        let shown = lockedUntil.timeIntervalSinceNow
        evidence.check(shown > 0 && shown <= Double(reported ?? 0) + 0.5,
                       "the wait shown is the device's (counted down, never invented)", String(format: "%.2f s", shown))
        // Trying the RIGHT code while locked: the device still refuses, with its wait.
        try await proveExpecting(code, proven: false)
        evidence.check(ble.proofRejection?.kind == .locked, "a correct code during the lockout is refused as locked",
                       ble.proofRejection?.message ?? "")
        try await Task.sleep(nanoseconds: UInt64(max(0, (ble.lockedUntil ?? Date()).timeIntervalSinceNow + 0.2) * 1e9))
        try await proveExpecting(code, proven: true)
        evidence.check(mitm.credentialWrites() == 0, "no credentials left the phone while locked out")

    case "wrong-label":
        try await connectToDevice()
        let other = "SDSETUP:1:SD-SIM-FFFFFFFF:\(code.replacingOccurrences(of: "-", with: ""))"
        let problem = flow.prove(entry: other)
        evidence.check(problem == .wrongDevice(label: "SD-SIM-FFFFFFFF", connected: ble.deviceID ?? ""),
                       "a QR label for another device is refused on the phone", problem.map { "\($0)" } ?? "")
        evidence.check(mitm.authWrites().isEmpty, "nothing was sent to the device")
        let own = "SDSETUP:1:\(ble.deviceID ?? ""):\(code.replacingOccurrences(of: "-", with: ""))"
        try await proveExpecting(own, proven: true)

    case "impostor-id":
        // Someone relays the device but claims another identity in the challenge.
        mitm.alterIncoming = { characteristic, data in
            guard characteristic == .auth, authOp(data) == "challenge" else { return nil }
            return tamper(data, field: "id", replaceWith: "SD-SIM-0BADC0DE")
        }
        try await connectToDevice()
        try await proveExpecting(code, proven: false)
        guard case .failed(let failure) = flow.state else { throw HarnessError(description: "expected failure, got \(flow.state)") }
        evidence.check(failure.stage == .prove && failure.reason.contains("SD-SIM-0BADC0DE"), "the other identity is named and refused",
                       failure.reason)
        evidence.check(mitm.authWrites() == ["hello"], "not even a proof was sent to it", mitm.authWrites().joined(separator: ","))

    case "tamper-proven":
        // The device's proof is altered on the air: the phone must not trust it.
        mitm.alterIncoming = { characteristic, data in
            guard characteristic == .auth, authOp(data) == "proven" else { return nil }
            return tamper(data, field: "tag")
        }
        try await connectToDevice()
        try await proveExpecting(code, proven: false)
        guard case .failed(let failure) = flow.state else { throw HarnessError(description: "expected failure, got \(flow.state)") }
        evidence.check(failure.reason.contains("couldn't prove"), "a device that cannot prove the code gets nothing", failure.reason)
        evidence.check(mitm.credentialWrites() == 0, "no credentials were written")

    case "tamper-creds":
        // The encrypted credentials are altered on the air: the device must refuse them.
        mitm.alterOutgoing = { characteristic, data in
            guard characteristic == .auth, authOp(data) == "creds" else { return nil }
            return tamper(data, field: "ct")
        }
        try await connectToDevice()
        try await proveExpecting(code, proven: true)
        try await sendAndWaitForJoin()
        evidence.check(!joined(), "the device joined nothing")
        guard case .enteringSetupCode(let problem, _) = flow.state else {
            throw HarnessError(description: "expected the code again, got \(flow.state)")
        }
        evidence.check(problem?.contains("couldn't read what this phone sent") == true, "the device's refusal is shown", problem ?? "")

    case "timeout":
        // The peripheral runs with --challenge-ttl 1: a proof 1.5 s late is too late.
        mitm.delayOutgoing = { characteristic, data in
            characteristic == .auth && authOp(data) == "proof" ? 1.5 : 0
        }
        try await connectToDevice()
        try await proveExpecting(code, proven: false, timeout: 25)
        guard case .enteringSetupCode(let problem, _) = flow.state else {
            throw HarnessError(description: "expected the code again, got \(flow.state)")
        }
        evidence.check(ble.proofRejection?.kind == .expired, "the late proof was refused as expired", problem ?? "")
        mitm.delayOutgoing = nil
        try await proveExpecting(code, proven: true)
        evidence.check(true, "a prompt retry then works")

    case "proof-expired":
        // The peripheral runs with --proven-ttl 1: credentials 1.5 s after the proof
        // find it expired; the app renews the handshake once and sends them.
        try await connectToDevice()
        try await proveExpecting(code, proven: true)
        try await Task.sleep(nanoseconds: 1_500_000_000)
        try await sendAndWaitForJoin()
        evidence.check(joined(), "the device joined after a renewed handshake")
        evidence.check(mitm.authWrites() == ["hello", "proof", "creds", "hello", "proof", "creds"],
                       "creds refused as expired, then hello, proof, creds again", mitm.authWrites().joined(separator: ","))

    case "wrong-password":
        try await connectToDevice()
        try await proveExpecting(code, proven: true)
        try await waitFor("the device's network list") { !ble.networks.isEmpty }
        flow.send(ssid: ssid, password: "not-the-password")
        try await waitFor("the refusal", timeout: 30) {
            if case .choosingNetwork(let problem) = flow.state { return problem != nil }
            return false
        }
        if case .choosingNetwork(let problem) = flow.state {
            evidence.check(problem?.contains("not accepted") == true, "the device's reason is shown", problem ?? "")
        }
        try await sendAndWaitForJoin()
        evidence.check(joined(), "the right password then joins, over the same link")
        evidence.check(mitm.authWrites() == ["hello", "proof", "creds", "hello", "proof", "creds"],
                       "a fresh handshake carried the second attempt", mitm.authWrites().joined(separator: ","))

    default:
        throw HarnessError(description: "unknown scenario \(scenario)")
    }
    evidence.record("final", ["onboarding": "\(flow.state)", "ble": "\(ble.state)"])
    flow.stop()
}

Task { @MainActor in
    var status: Int32 = 0
    do {
        try await run()
    } catch {
        evidence.record("error", ["error": "\(error)"])
        status = 1
    }
    if !evidence.failures.isEmpty { status = 1 }
    evidence.record("result", ["scenario": scenario, "ok": status == 0, "failed_checks": evidence.failures])
    exit(status)
}
dispatchMain()
