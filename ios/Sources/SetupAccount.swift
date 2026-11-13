import Foundation

// The app's side of SetupFlow's two seams: the account (AppState.backend) and
// the local network (DeviceLink). Kept apart from SetupFlow so the flow itself
// compiles without SwiftUI or AppState -- the cross-language harness
// (ios/Tests/OnboardingHarness) runs it with its own HTTP account.

extension SetupFlow {
    /// The flow as the app runs it: the platform's Bluetooth link (CoreBluetooth,
    /// or the virtual peripheral in a DEBUG launch with -orynSimulatedBLE) and
    /// DeviceLink for finding the device on Wi-Fi afterwards. The account is
    /// attached when the view appears (AppState is an environment object).
    static func app() -> SetupFlow {
        SetupFlow(ble: BLEProvisioner(), localLink: DeviceLinkLocalLink())
    }
}

/// The signed-in account, through the same BackendAPI every other screen uses.
@MainActor
final class AppStateOnboardingAccount: OnboardingAccount {
    private let state: AppState

    init(_ state: AppState) { self.state = state }

    var isSignedIn: Bool { state.isSignedIn }

    func accountDevices() async throws -> [Onboarding.AccountDevice] {
        let devices = try await state.backend.devices()
        state.recordAccountDevices(devices)
        return devices.map {
            Onboarding.AccountDevice(id: $0.id, name: $0.name, hardwareID: $0.hardwareId, online: $0.online,
                                     activePreset: $0.activePreset)
        }
    }

    func claim(code: String) async throws {
        try await state.backend.claim(code: code)
    }

    func modes(deviceID: Int) async throws -> (modes: [Onboarding.Mode], active: String?) {
        let catalog = try await state.backend.presetCatalog()
        let device = try? await state.backend.devicePresets(deviceID: deviceID)
        let modes = catalog.presets.map { preset in
            let reported = device?.state(for: preset.id)
            return Onboarding.Mode(id: preset.id, title: preset.title, detail: preset.description,
                                   available: reported?.available ?? true, reason: reported?.reason)
        }
        return (modes, device?.activePreset)
    }

    func activate(mode: String, deviceID: Int) async throws {
        _ = try await state.backend.activatePreset(deviceID: deviceID, preset: mode, settings: [:])
    }

    func explain(_ error: Error, claiming: Bool) -> String {
        guard let api = error as? APIError else { return error.localizedDescription }
        if claiming, api.status == 0 {
            // Offline (M8.4): nothing reached the server, so nothing was added.
            return "Couldn't reach the server, so the device wasn't added. Check this phone's connection, "
                + "then enter the claim code again. (\(api.message))"
        }
        if claiming, api.status != 0 {
            return Onboarding.claimMessage(status: api.status, code: api.code, detail: api.message)
        }
        return api.message
    }
}

/// DeviceLink, as the flow needs it: "is this device answering on this network?"
@MainActor
final class DeviceLinkLocalLink: OnboardingLocalLink {
    private let link = DeviceLink()
    private var started = false

    func find(deviceID: String?) async -> String? {
        if !started {
            link.start()
            started = true
        }
        // Only the device that was just set up: never whichever camera answers first.
        await link.resolve(force: true, expecting: deviceID)
        if case .connected(let url) = link.state { return url }
        return nil
    }

    var problem: String? { link.discoveryProblem }

    func stop() {
        guard started else { return }
        link.stop()
        started = false
    }
}
