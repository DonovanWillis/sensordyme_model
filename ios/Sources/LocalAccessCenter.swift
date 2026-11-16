import Foundation
import SwiftUI
import os

/// The app's direct connections to its paired cameras (sd-local-v1,
/// docs/LOCAL_ACCESS.md): the trusted-camera cache, one LocalDeviceClient per
/// camera, finding each camera on the network, and the three enrolment paths.
///
/// One instance for the app, because the cameras are the phone's, not a
/// screen's: the device list, the live feed, the device screen and the alert
/// watcher all ask the same connection.
///
/// The rules it keeps (LOCAL_ACCESS.md §7):
///  * a camera is "live" only on a fresh local read in this launch; the cache
///    never says what a camera is doing;
///  * a session is used only after the camera's signature checked out against
///    the key cached at enrolment;
///  * `unknown_client` forgets the credential and says pair again, once;
///  * nothing is ever sent unsigned, and there is no fallback that is.
@MainActor
final class LocalAccessCenter: ObservableObject {
    static let shared = LocalAccessCenter()

    struct Reading: Equatable {
        let at: Date
        let state: LocalDeviceState
    }

    @Published private(set) var devices: [TrustedDevice] = []
    /// The last fresh local read of each camera, by hardware id (upper-cased).
    @Published private(set) var readings: [String: Reading] = [:]
    /// Where each camera answered last in this launch.
    @Published private(set) var addresses: [String: String] = [:]
    /// The open session per camera, for the live feed.
    @Published private(set) var sessions: [String: String] = [:]
    @Published private(set) var searching: Set<String> = []
    /// What to tell the person about a camera, when something needs saying.
    @Published private(set) var problems: [String: String] = [:]

    private var clients: [String: LocalDeviceClient] = [:]
    private let discovery = DeviceDiscovery()
    private var lastAttempt: [String: Date] = [:]
    private var lastPersisted = Date.distantPast
    private var accountAttempted: Set<String> = []
    private var connecting: [String: Task<Bool, Never>] = [:]
    private static let log = Logger(subsystem: "com.sensordyme.app", category: "local")

    init() {
        devices = TrustedDevices.decode(Keychain.get(TrustedDevices.storageKey))
    }

    private static func id(_ hardwareID: String) -> String { hardwareID.uppercased() }

    func device(_ hardwareID: String?) -> TrustedDevice? {
        guard let hardwareID, !hardwareID.isEmpty else { return nil }
        return devices.first { Self.id($0.hardwareID) == Self.id(hardwareID) }
    }

    private func persist(force: Bool = true) {
        guard force || Date().timeIntervalSince(lastPersisted) > 60 else { return }
        lastPersisted = Date()
        if !Keychain.set(TrustedDevices.storageKey, TrustedDevices.encode(devices)) {
            Self.log.error("the paired-camera list could not be stored in the Keychain; kept for this run only")
        }
    }

    private func update(_ hardwareID: String, persistNow: Bool = true, _ change: (inout TrustedDevice) -> Void) {
        guard let index = devices.firstIndex(where: { Self.id($0.hardwareID) == Self.id(hardwareID) }) else { return }
        var copy = devices[index]
        change(&copy)
        if copy != devices[index] {
            devices[index] = copy
            persist(force: persistNow)
        }
    }

    // MARK: what the screens read

    func presence(_ device: TrustedDevice, now: Date = Date()) -> TrustedDevices.Presence {
        let reading = readings[Self.id(device.hardwareID)]
        return TrustedDevices.presence(device, liveAt: reading?.at, liveProtection: reading?.state.protection,
                                       searching: searching.contains(Self.id(device.hardwareID)), now: now)
    }

    /// The camera answered a signed session recently: commands go local.
    func isReady(_ hardwareID: String?) -> Bool {
        guard let hardwareID, let device = device(hardwareID), device.enrolment == .active,
              sessions[Self.id(hardwareID)] != nil,
              let reading = readings[Self.id(hardwareID)] else { return false }
        return Date().timeIntervalSince(reading.at) <= TrustedDevices.liveFor
    }

    func address(_ hardwareID: String) -> String? {
        addresses[Self.id(hardwareID)] ?? device(hardwareID)?.lastKnownURL
    }

    /// The live feed's credential for this camera: its session, or nil to use
    /// the legacy device token (an install that has not enrolled yet).
    func streamCredential(_ hardwareID: String?) -> String? {
        guard let hardwareID, let session = sessions[Self.id(hardwareID)] else { return nil }
        return MJPEGStreamRequest.sessionCredential(session)
    }

    /// The camera refused the stream's session (it restarted, or the session
    /// ended): open a new one. The new credential publishes through `sessions`.
    func renewSession(_ hardwareID: String) async {
        guard let client = clients[Self.id(hardwareID)] else { _ = await connect(hardwareID); return }
        await client.invalidateSession()
        do {
            let session = try await client.validSession()
            sessions[Self.id(hardwareID)] = session
        } catch {
            await handle(error, hardwareID)
        }
    }

    // MARK: finding and connecting

    /// Open (or confirm) a session with one paired camera: its last address
    /// first, then Bonjour (`_sensordyme._tcp`, matched by TXT `id`). An IP
    /// change just works: the old address fails, discovery finds the new one,
    /// a new session opens there. Concurrent callers share one attempt.
    @discardableResult
    func connect(_ hardwareID: String, hints: [String] = []) async -> Bool {
        let key = Self.id(hardwareID)
        if let running = connecting[key] { return await running.value }
        let task = Task { @MainActor [weak self] () -> Bool in
            guard let self else { return false }
            return await self.performConnect(hardwareID, hints: hints)
        }
        connecting[key] = task
        let result = await task.value
        connecting[key] = nil
        return result
    }

    private func performConnect(_ hardwareID: String, hints: [String]) async -> Bool {
        let key = Self.id(hardwareID)
        guard let device = device(hardwareID), device.isUsable,
              let signer = LocalIdentity.signer(for: device.hardwareID),
              let devicePublicKey = device.devicePublicKeyBytes else { return false }
        lastAttempt[key] = Date()
        var candidates: [String] = []
        for url in [addresses[key], device.lastKnownURL] + hints.map(Optional.some) {
            if let url, !url.isEmpty, !candidates.contains(url) { candidates.append(url) }
        }
        for url in candidates where await attempt(device, signer, devicePublicKey, url) { return true }
        if self.device(hardwareID)?.enrolment == .needsPairing { return false }
        searching.insert(key)
        defer { searching.remove(key) }
        let found = await discovery.discoverOnce(timeout: 4)
        if let match = found.first(where: { Self.id($0.id) == key }), !candidates.contains(match.url) {
            return await attempt(device, signer, devicePublicKey, match.url)
        }
        return false
    }

    private func client(_ device: TrustedDevice, _ signer: LocalSigner, _ devicePublicKey: Data,
                        _ url: URL) async -> LocalDeviceClient {
        let key = Self.id(device.hardwareID)
        if let existing = clients[key], existing.devicePublicKey == devicePublicKey,
           existing.signer.publicKey == signer.publicKey {
            await existing.setBaseURL(url)
            return existing
        }
        let made = LocalDeviceClient(deviceID: device.hardwareID, signer: signer, devicePublicKey: devicePublicKey,
                                     baseURL: url)
        clients[key] = made
        return made
    }

    private func attempt(_ device: TrustedDevice, _ signer: LocalSigner, _ devicePublicKey: Data,
                         _ urlString: String) async -> Bool {
        guard let url = URL(string: urlString) else { return false }
        let client = await client(device, signer, devicePublicKey, url)
        do {
            let state = try await client.state()
            await record(device.hardwareID, url: urlString, state: state, client: client)
            return true
        } catch let error as LocalAccessError {
            switch error {
            case .unreachable, .wrongDevice:
                return false
            default:
                await handle(error, device.hardwareID)
                return false
            }
        } catch {
            return false
        }
    }

    private func record(_ hardwareID: String, url: String, state: LocalDeviceState, client: LocalDeviceClient) async {
        let key = Self.id(hardwareID)
        readings[key] = Reading(at: Date(), state: state)
        addresses[key] = url
        if let session = await client.currentSession, sessions[key] != session { sessions[key] = session }
        problems[key] = nil
        let changedAddress = device(hardwareID)?.lastKnownURL != url
        let wasAwaiting = device(hardwareID)?.enrolment == .awaitingDevice
        update(hardwareID, persistNow: changedAddress || wasAwaiting) { entry in
            entry.lastSeen = Date()
            entry.lastKnownURL = url
            if entry.enrolment == .awaitingDevice { entry.enrolment = .active }
        }
        if !changedAddress && !wasAwaiting { persist(force: false) }
    }

    /// A fresh read of one camera; reconnects (rediscovering it) when the read fails.
    @discardableResult
    func refresh(_ hardwareID: String, hints: [String] = []) async -> Bool {
        let key = Self.id(hardwareID)
        if let client = clients[key], let url = addresses[key] {
            do {
                let state = try await client.state()
                await record(hardwareID, url: url, state: state, client: client)
                return true
            } catch let error as LocalAccessError {
                if case .unreachable = error {} else if case .wrongDevice = error {} else {
                    await handle(error, hardwareID)
                    if error.forgetsCredential { return false }
                }
                readings[key] = nil
                sessions[key] = nil
            } catch {}
        }
        return await connect(hardwareID, hints: hints)
    }

    /// Every paired camera, at most every `every` seconds each (the device list's timer).
    func refreshAll(every: TimeInterval = 10, hints: [String: String] = [:]) async {
        for device in devices where device.isUsable {
            let key = Self.id(device.hardwareID)
            if let last = lastAttempt[key], Date().timeIntervalSince(last) < every, readings[key] == nil { continue }
            if let reading = readings[key], Date().timeIntervalSince(reading.at) < every { continue }
            await refresh(device.hardwareID, hints: hints[key].map { [$0] } ?? [])
        }
    }

    private func handle(_ error: Error, _ hardwareID: String) async {
        let key = Self.id(hardwareID)
        guard let error = error as? LocalAccessError else { return }
        switch error {
        case .refused(.forgetCredential) where device(hardwareID)?.enrolment == .awaitingDevice:
            // The account relayed this phone's key, and the camera has not
            // applied it yet (it arrives with its next report): not a
            // revocation. Keep the key and keep waiting.
            problems[key] = "Waiting for the camera to confirm this iPhone (it learns of it from your account)."
        case .refused(.forgetCredential):
            // The camera no longer trusts this phone (revoked, idle-expired):
            // forget the credential, keep the row so the card can say so.
            Self.log.info("a camera refused this phone's key (unknown_client); forgetting the local credential")
            LocalIdentity.forget(hardwareID)
            clients[key] = nil
            sessions[key] = nil
            readings[key] = nil
            update(hardwareID) { $0.enrolment = .needsPairing }
            problems[key] = LocalAccess.needsPairingAgain
        case .refused(.lockedOut):
            problems[key] = error.message
        case .notAuthentic:
            sessions[key] = nil
            readings[key] = nil
            problems[key] = error.message
        case .refused(.localAccessOff):
            problems[key] = error.message
        default:
            break
        }
    }

    // MARK: commands (always signed; confirmed from the device's own state)

    func setProtection(_ hardwareID: String, stop: Bool) async -> Result<LocalCommandOutcome, LocalAccessError> {
        await command(hardwareID) { try await $0.setProtection(stop: stop) }
    }

    func setPreset(_ hardwareID: String, name: String) async -> Result<LocalCommandOutcome, LocalAccessError> {
        await command(hardwareID) { try await $0.setPreset(name) }
    }

    func presets(_ hardwareID: String) async -> LocalPresets? {
        guard let client = await readyClient(hardwareID) else { return nil }
        return try? await client.presets()
    }

    func events(_ hardwareID: String, after: Int? = nil, limit: Int = 100) async throws -> LocalEventsPage {
        guard let client = await readyClient(hardwareID) else {
            throw LocalAccessError.unreachable("not connected")
        }
        do {
            return try await client.events(after: after, limit: limit)
        } catch {
            await handle(error, hardwareID)
            throw error
        }
    }

    private func readyClient(_ hardwareID: String) async -> LocalDeviceClient? {
        let key = Self.id(hardwareID)
        if clients[key] == nil || addresses[key] == nil { _ = await connect(hardwareID) }
        return addresses[key] == nil ? nil : clients[key]
    }

    private func command(_ hardwareID: String,
                         _ body: (LocalDeviceClient) async throws -> LocalCommandOutcome) async -> Result<LocalCommandOutcome, LocalAccessError> {
        guard let client = await readyClient(hardwareID) else {
            return .failure(.unreachable("not connected"))
        }
        do {
            let outcome = try await body(client)
            if let state = try? await client.state(), let url = addresses[Self.id(hardwareID)] {
                await record(hardwareID, url: url, state: state, client: client)
            }
            return .success(outcome)
        } catch let error as LocalAccessError {
            await handle(error, hardwareID)
            return .failure(error)
        } catch {
            return .failure(.malformed("\(error)"))
        }
    }

    // MARK: enrolment

    /// The phone's public key for a camera, creating it the first time.
    func prepareKey(for hardwareID: String) -> LocalSigner? {
        LocalIdentity.signer(for: hardwareID, create: true)
    }

    /// A camera's enrolment answer, from Bluetooth or the legacy token. Kept
    /// only when it is for THIS phone's key and carries a real device key.
    @discardableResult
    func adopt(_ enrollment: LocalAccess.Enrollment, name: String, via: TrustedDevice.EnrolledVia, owner: String,
               url: String? = nil, backendDeviceID: Int? = nil) -> Bool {
        guard let signer = LocalIdentity.signer(for: enrollment.deviceID), enrollment.clientID == signer.clientID,
              let key = enrollment.devicePublicKeyBytes else { return false }
        let existing = device(enrollment.deviceID)
        let entry = TrustedDevice(
            hardwareID: enrollment.deviceID, name: name.isEmpty ? (existing?.name ?? enrollment.deviceID) : name,
            backendDeviceID: backendDeviceID ?? existing?.backendDeviceID, lastKnownURL: url ?? existing?.lastKnownURL,
            lastSeen: existing?.lastSeen, lastModeTitle: existing?.lastModeTitle,
            devicePublicKey: LocalAccess.b64u(key), deviceKeyFingerprint: LocalAccess.fingerprint(publicKey: key),
            clientID: signer.clientID, enrolledVia: via, enrolment: .active, enrolledAt: Date(),
            owner: existing?.owner.isEmpty == false ? existing!.owner : owner.lowercased())
        devices = TrustedDevices.upsert(entry, into: devices)
        problems[Self.id(enrollment.deviceID)] = nil
        persist()
        return true
    }

    /// The account path (§4.2): for each device the account lists with a
    /// local identity, register this phone's public key once, and remember the
    /// device key the account reports. Usable once a local session succeeds.
    func enrollViaAccount(devices accountDevices: [Device], api: APIClient, owner: String) async {
        for device in accountDevices {
            guard let hardwareID = device.hardwareId, hardwareID.uppercased().hasPrefix("SD-"),
                  let identity = device.localIdentity, identity.contract == LocalAccess.contract,
                  let reported = identity.devicePublicKey,
                  let keyBytes = LocalAccess.b64uDecode(reported, length: 65), LocalAccess.publicKey(x963: keyBytes) != nil
            else { continue }
            let key = Self.id(hardwareID)
            let existing = self.device(hardwareID)
            switch TrustedDevices.accountIdentity(existing: existing, reportedKey: reported,
                                                  reportedFingerprint: identity.deviceKeyFingerprint) {
            case .unchanged:
                update(hardwareID) { entry in
                    entry.backendDeviceID = device.id
                    entry.name = device.name
                    if entry.owner.isEmpty { entry.owner = owner.lowercased() }
                }
                continue
            case .conflictKeepBluetooth:
                problems[key] = "Your account reports a different identity for this camera than the one paired over "
                    + "Bluetooth. Pair it again over Bluetooth to be sure it is yours."
                continue
            case .register, .reregister:
                break
            }
            guard !accountAttempted.contains(key) else { continue }
            accountAttempted.insert(key)
            if existing?.enrolment == .needsPairing || existing?.devicePublicKey != reported {
                LocalIdentity.forget(hardwareID)  // a fresh key for a fresh pairing
            }
            guard let signer = prepareKey(for: hardwareID) else { continue }
            do {
                _ = try await api.registerLocalClient(deviceID: device.id, publicKey: LocalAccess.b64u(signer.publicKey),
                                                      name: Self.phoneName)
                let entry = TrustedDevice(
                    hardwareID: hardwareID, name: device.name, backendDeviceID: device.id,
                    lastKnownURL: existing?.lastKnownURL, lastSeen: existing?.lastSeen,
                    lastModeTitle: Preset.title(for: device.activePreset), devicePublicKey: reported,
                    deviceKeyFingerprint: LocalAccess.fingerprint(publicKey: keyBytes), clientID: signer.clientID,
                    enrolledVia: .account, enrolment: .awaitingDevice, enrolledAt: Date(), owner: owner.lowercased())
                self.devices = TrustedDevices.upsert(entry, into: self.devices)
                persist()
                Self.log.info("registered this phone's local key with the account for a camera")
            } catch {
                // Not now (offline, an older backend): try again next launch.
                accountAttempted.remove(key)
            }
        }
    }

    /// The legacy token (§4.3): an install that pasted `dashboard.token`
    /// enrols with it once, and then no longer needs it. True when enrolled.
    func migrateLegacyToken(url: String, token: String, owner: String) async -> Bool {
        let token = token.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !token.isEmpty, let base = URL(string: url) else { return false }
        let transport = URLSessionLocalTransport()
        guard let helloURL = URL(string: "/api/local/hello", relativeTo: base)?.absoluteURL,
              let (helloData, helloStatus) = try? await transport.send(URLRequest(url: helloURL)), helloStatus == 200,
              let hello = try? JSONDecoder().decode(LocalAccess.Hello.self, from: helloData),
              hello.tokenEnrollment == true else { return false }
        if let known = device(hello.deviceID), known.enrolment == .active { return true }
        guard let signer = prepareKey(for: hello.deviceID),
              let enrollURL = URL(string: "/api/local/enroll", relativeTo: base)?.absoluteURL else { return false }
        var request = URLRequest(url: enrollURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue(token, forHTTPHeaderField: MJPEGStreamRequest.tokenHeader)
        request.httpBody = try? JSONSerialization.data(
            withJSONObject: ["public_key": LocalAccess.b64u(signer.publicKey), "name": Self.phoneName])
        guard let (data, status) = try? await transport.send(request), status == 201,
              let enrollment = try? JSONDecoder().decode(LocalAccess.Enrollment.self, from: data),
              enrollment.deviceID.uppercased() == hello.deviceID.uppercased()
        else { return false }
        return adopt(enrollment, name: hello.deviceID, via: .token, owner: owner, url: url)
    }

    /// What this phone calls itself to a camera. Generic on purpose: the
    /// device lists it to its owner, and nothing personal belongs there.
    static var phoneName: String { "iPhone" }

    // MARK: forgetting and ownership

    /// Unpair one camera on this iPhone: ask it to forget this phone (best
    /// effort), destroy the key, drop the row.
    func forget(_ hardwareID: String) async {
        let key = Self.id(hardwareID)
        if let client = clients[key], addresses[key] != nil {
            try? await client.revokeSelf()
        }
        LocalIdentity.forget(hardwareID)
        clients[key] = nil
        sessions[key] = nil
        readings[key] = nil
        addresses[key] = nil
        problems[key] = nil
        devices.removeAll { Self.id($0.hardwareID) == key }
        persist()
    }

    private func drop(keeping kept: [TrustedDevice]) {
        let keptIDs = Set(kept.map { Self.id($0.hardwareID) })
        for device in devices where !keptIDs.contains(Self.id(device.hardwareID)) {
            let key = Self.id(device.hardwareID)
            LocalIdentity.forget(device.hardwareID)
            clients[key] = nil
            sessions[key] = nil
            readings[key] = nil
            addresses[key] = nil
        }
        devices = kept
        accountAttempted.removeAll()
        persist()
    }

    /// Settings → Log out: the account's pairings go with it; a pairing made
    /// with no account (Bluetooth setup before sign-in) stays.
    func signedOutExplicitly() {
        drop(keeping: TrustedDevices.afterExplicitSignOut(devices))
    }

    /// Someone signed in on this phone.
    func signedIn(owner: String) {
        drop(keeping: TrustedDevices.afterSignIn(devices, owner: owner))
    }
}
