import Foundation

// Run with ios/Tests/run.sh
// run.sh deps: SetupOnboarding BLEProvisioning BLETransport BLEPoPFlow BLEProofOfPossession ProvisioningStatus
//
// SetupFlow's seams -- Bluetooth to Wi-Fi to the account -- with a scripted
// link (a legacy device, so no setup code; the proof itself is covered by
// BLEProvisioningTests and the cross-language harness) and a scripted account.

var failures = 0
func check(_ condition: Bool, _ what: String) {
    if condition { print("  ok   \(what)") } else { print("  FAIL \(what)"); failures += 1 }
}

@MainActor
func settle(_ what: String, timeout: Double = 5, _ condition: () -> Bool) async {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() && Date() < deadline { try? await Task.sleep(nanoseconds: 10_000_000) }
    check(condition(), what)
}

@MainActor
final class LegacyDevice: BLETransport {
    var onEvent: ((BLETransportEvent) -> Void)?
    let peripheral = UUID()
    let deviceID: String
    init(deviceID: String) { self.deviceID = deviceID }
    func later(_ e: BLETransportEvent) { Task { @MainActor in self.onEvent?(e) } }
    func json(_ o: Any) -> Data { try! JSONSerialization.data(withJSONObject: o) }
    func startScan() {
        later(.radio(.poweredOn))
        later(.discovered(BLEAdvertisement(id: peripheral, advertisedName: "SensorDyme-3C4D", cachedName: nil, rssi: -50)))
    }
    func stopScan() {}
    func connect(_ id: UUID) -> Bool {
        later(.characteristics(id, [.deviceInfo, .credentials, .status, .networks, .command]))
        return true
    }
    func disconnect(_ id: UUID) {}
    func subscribe(_ c: ProvisioningCharacteristic, on id: UUID) {}
    func read(_ c: ProvisioningCharacteristic, on id: UUID) {
        switch c {
        case .deviceInfo: later(.value(id, c, json(["id": deviceID, "auth": "legacy"])))
        case .networks: later(.value(id, c, json([["ssid": "Home", "signal": 80, "secured": true]])))
        case .status: later(.value(id, c, json(["state": "provisioning", "provision": NSNull()])))
        default: break
        }
    }
    func write(_ data: Data, to c: ProvisioningCharacteristic, on id: UUID) {
        later(.wrote(id, c, error: nil))
        later(.value(id, .status, json(["state": "provisioning", "provision": ["state": "connecting", "attempt": 1]])))
        later(.value(id, .status, json(["state": "online", "provision": ["state": "ok", "ip": "192.0.2.10", "attempt": 1]])))
    }
}

@MainActor
final class ScriptedAccount: OnboardingAccount {
    struct Refusal: Error { let status: Int }
    var isSignedIn = true
    var devices: [Onboarding.AccountDevice] = []
    /// What the claim adds (the row the claim code belongs to).
    var claimAdds: Onboarding.AccountDevice?
    var validCode = "AB12-CD34-EF56"
    var activated: [(String, Int)] = []
    /// Polls before an added device reports online.
    var pollsUntilOnline = 1
    private var polls = 0

    func accountDevices() async throws -> [Onboarding.AccountDevice] {
        polls += 1
        if polls > pollsUntilOnline {
            devices = devices.map { .init(id: $0.id, name: $0.name, hardwareID: $0.hardwareID, online: true, activePreset: $0.activePreset) }
        }
        return devices
    }
    func claim(code: String) async throws {
        guard code == validCode else { throw Refusal(status: 404) }
        if let claimAdds { devices.append(claimAdds) }
    }
    func modes(deviceID: Int) async throws -> (modes: [Onboarding.Mode], active: String?) {
        ([.init(id: "home_security", title: "Home", detail: "", available: true, reason: nil)], nil)
    }
    func activate(mode: String, deviceID: Int) async throws { activated.append((mode, deviceID)) }
    func explain(_ error: Error, claiming: Bool) -> String {
        Onboarding.claimMessage(status: (error as? Refusal)?.status, code: nil, detail: nil)
    }
}

@MainActor
func joinedFlow(_ account: ScriptedAccount, deviceID: String = "SD-0A1B2C3D") async -> SetupFlow {
    let flow = SetupFlow(ble: BLEProvisioner(transport: LegacyDevice(deviceID: deviceID)), localLink: nil, account: account)
    flow.pollSeconds = 0.05
    flow.start()
    await settle("found") { !flow.ble.discovered.isEmpty }
    flow.connect(to: flow.ble.discovered[0])
    await settle("a legacy device goes straight to the networks") {
        if case .choosingNetwork = flow.state { return true }; return false
    }
    flow.send(ssid: "Home", password: "correct-horse")
    await settle("joined, then the account step") {
        switch flow.state {
        case .enteringClaimCode, .waitingForDevice, .choosingMode: return true
        default: return false
        }
    }
    return flow
}

@MainActor
func main() async {
    print("first time: claim, wait for it to report, choose a mode:")
    let account = ScriptedAccount()
    account.claimAdds = .init(id: 7, name: "SensorDyme", hardwareID: "SD-0A1B2C3D", online: false, activePreset: nil)
    let flow = await joinedFlow(account)
    check({ if case .enteringClaimCode = flow.state { return true }; return false }(), "a new device asks for the claim code")
    check(flow.ble.state == .idle, "Bluetooth was let go once the device joined")
    check(!flow.submitClaim(code: "  "), "an empty claim code is not sent")
    _ = flow.submitClaim(code: "ZZZZ-0000-0000")
    await settle("an unknown code is explained") {
        if case .enteringClaimCode(let p) = flow.state { return p?.contains("isn't one the server knows") == true }
        return false
    }
    _ = flow.submitClaim(code: "ab12-cd34-ef56")
    await settle("on the account and reported in: choose a mode") {
        if case .choosingMode(7) = flow.state { return true }; return false
    }
    await settle("the mode list loaded, one recommended") { flow.recommendedMode == "home_security" }
    flow.chooseMode("home_security")
    await settle("finished in that mode") {
        if case .finished(let o) = flow.state { return o.mode == "home_security" && o.accountDeviceID == 7 && !o.alreadyOwned }
        return false
    }
    check(account.activated.count == 1 && account.activated[0].1 == 7, "activated on the right device")
    check(flow.transitions.filter { if case .waitTick = $0.event { return true }; return false }.count <= 2,
          "waiting ticks once per second at most (no restart loop)")

    print("\nalready on the account: no claim asked:")
    let owned = ScriptedAccount()
    owned.devices = [.init(id: 4, name: "Hall", hardwareID: "SD-0A1B2C3D", online: true, activePreset: "home_security")]
    let again = await joinedFlow(owned)
    await settle("straight to the mode") { if case .choosingMode(4) = again.state { return true }; return false }
    again.skipMode()
    check({ if case .finished(let o) = again.state { return o.alreadyOwned && o.mode == nil }; return false }(),
          "finished as already owned, mode untouched")

    print("\na claim code for another unit is caught:")
    let other = ScriptedAccount()
    other.claimAdds = .init(id: 9, name: "Garage", hardwareID: "SD-0BADC0DE", online: true, activePreset: nil)
    let wrong = await joinedFlow(other)
    _ = wrong.submitClaim(code: other.validCode)
    await settle("refused, naming both") {
        if case .failed(let f) = wrong.state { return f.reason.contains("SD-0BADC0DE") && f.reason.contains("SD-0A1B2C3D") }
        return false
    }

    print("\na device that never reports in:")
    let silent = ScriptedAccount()
    silent.claimAdds = .init(id: 3, name: "Quiet", hardwareID: "SD-0A1B2C3D", online: false, activePreset: nil)
    silent.pollsUntilOnline = 1_000
    let quiet = await joinedFlow(silent)
    quiet.reportBudgetSeconds = 1
    _ = quiet.submitClaim(code: silent.validCode)
    await settle("says so, and offers to keep waiting", timeout: 8) {
        if case .failed(let f) = quiet.state { return f.stage == .online && f.recoveries == [.keepWaiting] }
        return false
    }
    quiet.stop()
}

await main()
print("\n\(failures == 0 ? "all passed" : "\(failures) FAILED")")
exit(failures == 0 ? 0 : 1)
