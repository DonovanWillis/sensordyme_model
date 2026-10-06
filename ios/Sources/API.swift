import Foundation
import Network
import SwiftUI
import os

struct APIError: LocalizedError {
    let message: String
    /// A 401. The stored token is dead, so retrying is pointless — the caller
    /// should send the user back to sign-in rather than keep polling.
    var isAuthFailure = false
    /// No server was reached at all, as opposed to one that was reached and said
    /// no. Only this kind is worth retrying at a different address.
    var isUnreachable = false
    var errorDescription: String? { message }
}

/// Talks to the SensorDyme cloud backend (cloud/app.py).
struct APIClient {
    let baseURL: URL?
    let token: String?
    /// Finds the backend when the saved address is missing or has stopped
    /// answering. Supplied by AppState; see BackendLocator.
    var relocate: (@Sendable () async -> URL?)?

    private func request(_ method: String, _ path: String, query: [String: String]? = nil, body: [String: Any]? = nil) async throws -> Data {
        // A fresh install has no address saved, so ask the network for one rather
        // than refusing. This is what lets a new phone sign in with nothing but an
        // email and a password.
        var resolved = baseURL
        if resolved == nil, let relocate {
            resolved = await relocate()
        }
        guard let base = resolved else {
            throw APIError(
                message: "Couldn't find a SensorDyme server on this network. Make sure the backend is running on your Mac and this phone is on the same Wi-Fi. You can also enter the address yourself in Settings → Manual server address.",
                isUnreachable: true
            )
        }
        do {
            return try await send(method, path, query: query, body: body, base: base)
        } catch let error as APIError where error.isUnreachable {
            // The address stopped working — usually because the machine picked up
            // a new DHCP lease. Look it up again and retry once, so a moved server
            // heals itself instead of sending the user to Settings to fix an
            // address they never set in the first place.
            guard let relocate, let moved = await relocate(), moved != base else { throw error }
            return try await send(method, path, query: query, body: body, base: moved)
        }
    }

    private func send(_ method: String, _ path: String, query: [String: String]?, body: [String: Any]?, base: URL) async throws -> Data {
        // appendingPathComponent percent-escapes "?", so a query string baked into
        // `path` lands in the path and 404s. Queries go through URLComponents.
        var comps = URLComponents(url: base.appendingPathComponent(path), resolvingAgainstBaseURL: false)
        if let query {
            comps?.queryItems = query.map { URLQueryItem(name: $0.key, value: $0.value) }
        }
        guard let url = comps?.url else {
            throw APIError(message: "Couldn't build a request URL for \(path).")
        }
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.timeoutInterval = 15
        if let token {
            req.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }
        if let body {
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            req.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await URLSession.shared.data(for: req)
        } catch let error as URLError {
            throw APIError(
                message: Self.explain(error, host: base.host ?? "the server"),
                isUnreachable: Self.isUnreachable(error)
            )
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status != 401 else {
            throw APIError(message: "Your session has expired. Please sign in again.", isAuthFailure: true)
        }
        guard (200..<300).contains(status) else {
            // FastAPI errors arrive as {"detail": "..."}
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"]
            throw APIError(message: (detail as? String) ?? "request failed (\(status))")
        }
        return data
    }

    /// Did this fail to reach a server at all, rather than reach one that refused?
    /// Only the first kind can be fixed by trying somewhere else.
    static func isUnreachable(_ error: URLError) -> Bool {
        switch error.code {
        case .cannotConnectToHost, .cannotFindHost, .dnsLookupFailed,
             .timedOut, .networkConnectionLost, .cannotLoadFromNetwork:
            return true
        default:
            return false
        }
    }

    /// Turn opaque networking failures into messages that say what to actually fix.
    ///
    /// The advice for a host that never answered comes from `BackendResolution`
    /// rather than being written out here, because it is only true of some
    /// hosts: telling someone to get on the same Wi-Fi as the server describes
    /// a Mac on the LAN, and misdescribes a hosted backend.
    static func explain(_ error: URLError, host: String) -> String {
        #if !targetEnvironment(simulator)
        if host == "localhost" || host == "127.0.0.1" {
            return "This app is pointed at \(host) — on a phone that means the phone itself, not your server. Open Settings → Manual server address and enter your server's real address (e.g. http://192.168.1.x:8787 while developing)."
        }
        #endif
        switch error.code {
        case .cannotConnectToHost:
            return "Nothing is answering at \(host). Is the backend running on your Mac? SensorDyme will keep looking; you can also check Settings → Server."
        case .cannotFindHost, .dnsLookupFailed:
            return "Can't find \(host) — check Settings → Server, or set the address yourself in Settings → Manual server address."
        case .notConnectedToInternet, .networkConnectionLost:
            return "No network connection. Check the phone's Wi-Fi."
        case .timedOut:
            return "\(host) didn't answer in time. \(BackendResolution.unreachableAdvice(host: host))"
        case .appTransportSecurityRequiresSecureConnection:
            return "iOS blocked the insecure connection to \(host)."
        default:
            return "Can't reach \(host): \(error.localizedDescription)"
        }
    }

    private func decode<T: Decodable>(_ type: T.Type, from data: Data) throws -> T {
        try JSONDecoder().decode(type, from: data)
    }

    // MARK: auth
    func signup(email: String, password: String) async throws -> AuthResponse {
        try decode(AuthResponse.self, from: try await request("POST", "v1/signup", body: ["email": email, "password": password]))
    }

    func login(email: String, password: String) async throws -> AuthResponse {
        try decode(AuthResponse.self, from: try await request("POST", "v1/login", body: ["email": email, "password": password]))
    }

    // MARK: devices
    func devices() async throws -> [Device] {
        try decode(DevicesResponse.self, from: try await request("GET", "v1/devices")).devices
    }

    func claim(code: String) async throws {
        _ = try await request("POST", "v1/devices/claim", body: ["claim_code": code])
    }

    func events(deviceID: Int, limit: Int = 100) async throws -> [EventItem] {
        try decode(EventsResponse.self, from: try await request("GET", "v1/devices/\(deviceID)/events", query: ["limit": String(limit)])).events
    }

    func setPreset(deviceID: Int, preset: String) async throws {
        _ = try await request("POST", "v1/devices/\(deviceID)/preset", body: ["preset": preset])
    }

    func setNtfyTopic(_ topic: String) async throws {
        _ = try await request("POST", "v1/account/ntfy", body: ["ntfy_topic": topic])
    }
}

/// App-wide session: backend URL, auth token, and a configured client.
@MainActor
final class AppState: ObservableObject {
    /// The backend this build ships pointed at, from the `SENSORDYME_API_BASE_URL`
    /// build setting via Info.plist's `SDBackendBaseURL`. Empty until a real
    /// deployment exists — deliberately not a literal in Swift, so changing where
    /// the app points is a build-configuration change and not a code change.
    ///
    /// On the simulator, localhost genuinely is the dev Mac, so it stands in when
    /// no production URL is built in. On a real phone localhost is the phone
    /// itself, which is a trap — there, an unconfigured build starts with nothing
    /// and lets BackendLocator find the server on the network.
    static let productionBaseURL: String = {
        let fromBuild = BackendResolution.productionDefault(
            fromInfoPlist: Bundle.main.object(forInfoDictionaryKey: "SDBackendBaseURL") as? String
        )
        #if targetEnvironment(simulator)
        return fromBuild.isEmpty ? "http://localhost:8787" : fromBuild
        #else
        return fromBuild
        #endif
    }()

    /// Typed by a person in Settings. Discovery never writes here — that
    /// separation is the whole fix, and it is enforced by there being two keys
    /// rather than by anyone remembering to check.
    @AppStorage("configuredBaseURL")
    var configuredBaseURL: String = "" {
        didSet {
            discoveredBaseURL = ""
        }
    }
    /// Where mDNS last found a backend. A cache: useful, never authoritative,
    /// and the only slot `relocateBackend()` is allowed to overwrite.
    @AppStorage("discoveredBaseURL") private(set) var discoveredBaseURL: String = ""
    /// Which backend this app last connected to, from the service's TXT record.
    ///
    /// Persisted because its whole job is to be remembered across launches: it is
    /// what stops the app silently adopting a different SensorDyme backend when
    /// several are on one network. The address is a cache; this is the identity.
    @AppStorage("knownBackendID") private(set) var knownBackendID: String = ""
    /// Bumped when the storage layout changes, so the migration below runs once.
    @AppStorage("backendStorageVersion") private var backendStorageVersion: Int = 0
    @AppStorage("deviceLocalURL") var deviceLocalURL: String = ""  // LAN feed, e.g. http://192.168.1.20:8899
    @AppStorage("accountEmail") var accountEmail: String = ""
    @AppStorage("demoMode") private var storedDemo: Bool = false
    /// Stored raw so the enum can gain cases without invalidating saved values.
    @AppStorage("appearance") private var storedAppearance: String = SD.Appearance.system.rawValue

    var appearance: SD.Appearance {
        get { SD.Appearance(rawValue: storedAppearance) ?? .system }
        set {
            objectWillChange.send()  // @AppStorage on a plain property won't publish
            storedAppearance = newValue.rawValue
        }
    }

    @Published var isSignedIn: Bool = false
    @Published var isDemo: Bool = false
    /// True while the app is looking for the backend on the network, so the
    /// sign-in screen can say what it is doing instead of appearing to hang.
    @Published var isLocatingBackend = false

    /// How far the app has actually got with the backend.
    ///
    /// Four distinct states because the app used to collapse them and report a
    /// connection for a service it had merely seen advertised on the network. An
    /// mDNS record outlives the process that published it, so "discovered" and
    /// "healthy" are genuinely different facts and only the second one means the
    /// app can do anything.
    enum BackendStatus: Equatable {
        /// Nothing tried yet this launch.
        case unknown
        /// Looking: probing known addresses, then browsing the network.
        case searching
        /// An address answered `/v1/health` and identified itself as ours.
        case connected(String)
        /// Nothing answered. Carries something the user can act on.
        case unreachable(String)
    }

    @Published private(set) var backendStatus: BackendStatus = .unknown

    /// The last address that actually answered as the backend. Not persisted: it
    /// is a statement about the network as it is right now, and restoring it
    /// across launches is how a stale address gets treated as configuration.
    @Published private(set) var confirmedBaseURL: String = ""

    /// The single in-flight resolve. Concurrent callers join it rather than
    /// starting competing searches — several views ask at once on launch, and a
    /// request failing over triggers another.
    private var resolveTask: Task<String?, Never>?
    /// The pending automatic retry, held so there is never more than one timer.
    private var retryTask: Task<Void, Never>?
    /// The while-connected health re-check, held for the same reason.
    private var heartbeatTask: Task<Void, Never>?
    /// How many consecutive resolves have failed, for the backoff schedule.
    private var failedAttempts = 0
    /// Watches for the phone changing network, so a move is noticed rather than
    /// waited out.
    private var pathMonitor: NWPathMonitor?
    /// The network path last seen, so the monitor can tell a real change from its
    /// own first report. Same approach as `DeviceLink.lastPathID`.
    private var lastPathID: String?
    /// Explains an involuntary sign-out on the auth screen, so a rotated server
    /// secret or an expired token doesn't just silently dump the user to login.
    @Published var signOutNotice: String?

    /// Auth token, kept in the Keychain (not UserDefaults). Migrates any token
    /// stored by earlier builds on first launch.
    private var storedToken: String {
        get { Keychain.get("authToken") ?? "" }
        set { Keychain.set("authToken", newValue) }
    }

    /// The device's own dashboard token (`dashboard.token` in its config.yaml),
    /// sent as `X-Device-Token` on `/stream` and `/api`.
    ///
    /// In the Keychain for exactly the reason Keychain.swift gives for the
    /// account token, which applies at least as strongly here: this is the
    /// credential that gates a live camera feed, and UserDefaults is
    /// unencrypted and included in unencrypted device backups. It was the one
    /// secret in the app still stored in the clear.
    ///
    /// Mirrored in memory rather than read from the Keychain on every access:
    /// the live feed watches this value (MJPEGView's `onChange`), and that
    /// re-evaluates on every decoded frame — so a Keychain-backed getter would
    /// mean a SecItemCopyMatching call a dozen times a second for the whole time
    /// anyone is watching the camera. The Keychain stays the store of record;
    /// this is a cache of it, written through on every change.
    @Published private var deviceTokenCache: String = ""

    var deviceToken: String {
        get { deviceTokenCache }
        set {
            // Trimmed at the single point it is stored. A token is compared byte
            // for byte with hmac.compare_digest on the device, so one trailing
            // space or newline -- which is what pasting from a password manager
            // or a terminal usually brings -- makes the CORRECT secret return
            // 401. Verified against the real device: "secret123 " is rejected
            // exactly like a wrong token, with nothing on screen to distinguish
            // the two. No token legitimately has surrounding whitespace.
            let cleaned = newValue.trimmingCharacters(in: .whitespacesAndNewlines)
            guard cleaned != deviceTokenCache else { return }
            deviceTokenCache = cleaned  // @Published publishes; no manual send needed
            Keychain.set("deviceToken", cleaned)
        }
    }

    init() {
        // One-time migration: builds before the Keychain change kept the token
        // in UserDefaults under "authToken".
        // The plaintext copy is removed ONLY once the Keychain write is confirmed.
        // Deleting it on an unchecked write turns a failed write into a destroyed
        // secret -- for the account token that is a forced re-login, and for the
        // device token it is a live feed that reports a rejected token the user
        // never got wrong.
        if let legacy = UserDefaults.standard.string(forKey: "authToken"), !legacy.isEmpty {
            if Keychain.set("authToken", legacy) {
                UserDefaults.standard.removeObject(forKey: "authToken")
            }
        }
        // Same migration for the device token, so nobody has to re-type a token
        // they already entered.
        if let legacy = UserDefaults.standard.string(forKey: "deviceToken"), !legacy.isEmpty {
            if Keychain.set("deviceToken", legacy.trimmingCharacters(in: .whitespacesAndNewlines)) {
                UserDefaults.standard.removeObject(forKey: "deviceToken")
            }
        }
        // AFTER the migration, so a token that has just been moved out of
        // UserDefaults is the one loaded rather than an empty Keychain slot. The
        // UserDefaults fallback covers the case where the Keychain is unavailable
        // (locked, or an entitlement problem): the feed keeps working from the
        // copy that is still there, instead of silently running with no token and
        // blaming the device for rejecting it.
        deviceTokenCache = Keychain.get("deviceToken")
            ?? UserDefaults.standard.string(forKey: "deviceToken")
            ?? ""
        migrateBackendStorage()
        isDemo = storedDemo
        isSignedIn = !storedToken.isEmpty || storedDemo
    }

    /// Split the old single `baseURL` slot into configuration and cache, once.
    ///
    /// Earlier builds stored one address that both a person and mDNS discovery
    /// wrote to, so an address discovered on one Wi-Fi network outlived that
    /// network and became the app's idea of where the backend permanently is.
    /// On an installed app that value is already on disk, and no amount of
    /// getting the new code right would clear it — which is why this runs rather
    /// than only new installs benefiting: a bare IP there was written by
    /// discovery, cannot be a backend anywhere else, and is dropped.
    private func migrateBackendStorage() {
        guard backendStorageVersion < 1 else { return }
        backendStorageVersion = 1
        let defaults = UserDefaults.standard
        guard let legacy = defaults.string(forKey: "baseURL"), !legacy.isEmpty else { return }
        // Removed either way: whatever it meant, it is not read again.
        defaults.removeObject(forKey: "baseURL")
        if case let .promoteToConfigured(url) = BackendResolution.migrate(legacy: legacy),
           configuredBaseURL.isEmpty {
            configuredBaseURL = url
        }
    }

    /// The address requests actually go to.
    ///
    /// A *confirmed* address — one that answered `/v1/health` as the SensorDyme
    /// backend — outranks the configured order, which is the difference between
    /// this and `BackendResolution.effective`. Precedence describes what is
    /// configured; it says nothing about what exists. When a build-time
    /// `http://11.22.7.64:8787` was top of the order, every request went there
    /// and kept going there, so the healthy Mac that discovery had already found
    /// was never used and the app could not be pointed anywhere else without a
    /// rebuild.
    ///
    /// Before anything has been confirmed this still falls back to the configured
    /// order, so the very first request on a cold launch has somewhere to go.
    var effectiveBaseURL: String {
        if !confirmedBaseURL.isEmpty { return confirmedBaseURL }
        return BackendResolution.effective(
            configured: configuredBaseURL,
            productionDefault: Self.productionBaseURL,
            discovered: discoveredBaseURL
        )
    }

    /// Whether the app is allowed to go looking on the local network.
    ///
    /// False once a *public* backend is pinned, so an internet blip can never
    /// divert sign-in to a different server that happens to be on the same
    /// Wi-Fi. A pinned LAN address is a developer pointing at a Mac, and that
    /// one stays discoverable — otherwise a stale dev URL is unrecoverable
    /// without a rebuild, which is exactly the hole this app fell into.
    private var mayDiscover: Bool {
        BackendResolution.mayFallBackToDiscovery(
            configured: configuredBaseURL,
            productionDefault: Self.productionBaseURL
        )
    }

    var api: APIClient {
        let base = effectiveBaseURL
        return APIClient(
            baseURL: base.isEmpty ? nil : URL(string: base),
            token: storedToken.isEmpty ? nil : storedToken,
            relocate: { [weak self] in await self?.relocateBackend() ?? nil }
        )
    }

    // MARK: - Finding and keeping the backend

    private static let log = Logger(subsystem: "com.sensordyme.app", category: "backend")

    /// Find a backend that actually answers, and remember it.
    ///
    /// One resolver for the whole app. It walks the configured order — typed
    /// address, then the build's URL, then the last discovered one — health
    /// checking each, and only then browses the network. That ordering gives both
    /// behaviours the app needs at once, without a mode switch: a deployed HTTPS
    /// backend answers on the first probe and is never abandoned for whatever is
    /// on the local Wi-Fi, while an address that has stopped existing fails its
    /// probe in a few seconds and the Mac that is really there gets used.
    ///
    /// Returns the confirmed address, or nil. Never throws, and never reports a
    /// connection it did not verify.
    @discardableResult
    func resolveBackend() async -> String? {
        guard !isDemo else { return nil }

        // Join an existing search rather than starting a second browse. Without
        // this, launch alone starts several: the sign-in screen, the alert
        // watcher, and every request that fails over.
        if let existing = resolveTask {
            Self.log.debug("joining the in-flight resolve")
            return await existing.value
        }
        Self.log.debug("starting a resolve")

        let task = Task { @MainActor [weak self] () -> String? in
            guard let self else { return nil }

            self.backendStatus = .searching
            self.isLocatingBackend = true
            defer { self.isLocatingBackend = false }

            let known = BackendResolution.candidates(
                configured: self.configuredBaseURL,
                productionDefault: Self.productionBaseURL,
                discovered: self.discoveredBaseURL
            )
            for candidate in known {
                if Task.isCancelled { return nil }
                if await BackendLocator.answers(candidate) {
                    self.confirm(candidate)
                    return candidate
                }
                Self.log.info("no backend at the known address \(candidate, privacy: .public)")
            }

            if Task.isCancelled { return nil }

            if self.mayDiscover,
               let found = await BackendLocator.shared.search(remembering: self.knownBackendID) {
                // The address goes to the discovery cache, never to configuration:
                // an address found on one network must not outlive it as if
                // somebody had chosen it. The id is the opposite — it IS worth
                // remembering, because it is what identifies this backend on the
                // next network it turns up on.
                // Cache whatever was found, name or address — but note which,
                // because the two have different lifetimes. A name is durable
                // and survives the Mac moving. An address is a performance
                // shortcut only, and `reportUnreachable` drops it the first time
                // it fails (see `BackendResolution.discoveryCache`). That
                // invalidation is what makes caching an address safe now and is
                // precisely what was missing before: the address was cached and
                // never cleared, so it stayed first in `candidates` for ever.
                self.discoveredBaseURL = found.url
                // os.Logger takes an interpolation literal, not an expression.
                if BackendResolution.BackendEndpoint(url: found.url)?.hostIsStable == true {
                    Self.log.info(
                        "cached the backend's stable name \(found.url, privacy: .public)")
                } else {
                    Self.log.info(
                        "cached \(found.url, privacy: .public) as an address — dropped on first failure")
                }
                if let id = found.id, !id.isEmpty, id != self.knownBackendID {
                    Self.log.info("remembering backend id \(id, privacy: .public)")
                    self.knownBackendID = id
                }
                self.confirm(found.url)
                return found.url
            }

            self.reportUnreachable(triedKnown: known)
            return nil
        }

        resolveTask = task
        let found = await task.value
        resolveTask = nil
        return found
    }

    /// Record a verified address and stop the retry clock.
    private func confirm(_ url: String) {
        confirmedBaseURL = url
        backendStatus = .connected(url)
        failedAttempts = 0
        retryTask?.cancel()
        retryTask = nil
        Self.log.info("backend connected at \(url, privacy: .public)")
        startHeartbeat()
    }

    /// How often to re-check a backend we believe we are connected to.
    ///
    /// One GET of `/v1/health` at this interval, and only while the app is
    /// running and believes it is connected.
    private static let heartbeatInterval: TimeInterval = 20

    /// Keep checking that the server we are connected to is still there.
    ///
    /// Without this the app is only as honest as its last request. Signed out, on
    /// the sign-in screen, nothing makes requests at all — so stopping the backend
    /// left the app displaying "connected" indefinitely, which is the exact claim
    /// it must never make about a server that is not answering. It also meant
    /// reconnection was really request-triggered rather than automatic: the retry
    /// schedule could not start, because nothing had noticed anything was wrong.
    ///
    /// Cheap enough to be unconditional: one small request every 20 seconds, and
    /// iOS suspends the task outright once the app is backgrounded.
    private func startHeartbeat() {
        heartbeatTask?.cancel()
        heartbeatTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: UInt64(Self.heartbeatInterval * 1_000_000_000))
                guard !Task.isCancelled, let self else { return }
                // Only meaningful while we claim to be connected; any other state
                // already has the resolver or the retry schedule working on it.
                guard case .connected(let url) = self.backendStatus else { return }
                if await BackendLocator.answers(url) { continue }
                Self.log.info("heartbeat: \(url, privacy: .public) stopped answering")
                self.confirmedBaseURL = ""
                self.reportUnreachable(triedKnown: [url])
                return
            }
        }
    }

    /// Say why there is no connection, in terms of what was actually tried, and
    /// schedule another attempt.
    private func reportUnreachable(triedKnown: [String]) {
        heartbeatTask?.cancel()
        heartbeatTask = nil
        confirmedBaseURL = ""
        // Drop a dead ADDRESS from the discovery cache, keep a NAME.
        //
        // This used to clear only `confirmedBaseURL`, so a numeric address from
        // a network the Mac had left stayed in the discovery slot — first in
        // `candidates`, tried before anything else on every launch, and the host
        // named in every error the user read. A name is kept deliberately: it is
        // the backend's identity, it can be resolved again, and the usual reason
        // it went quiet is a sleeping Mac rather than a wrong name.
        let keptCache = BackendResolution.discoveryCache(
            after: triedKnown.first ?? "", keeping: discoveredBaseURL
        )
        if keptCache != discoveredBaseURL {
            Self.log.info(
                "dropping the stale discovered address \(self.discoveredBaseURL, privacy: .public)")
            discoveredBaseURL = keptCache
        }
        let reason: String
        if let permission = BackendLocator.shared.problem {
            // A denied Local Network permission is invisible in the results and
            // is the one cause the user can fix immediately.
            reason = permission
        } else if triedKnown.isEmpty {
            reason = "No SensorDyme server answered on this Wi-Fi. Check that the backend is running on your Mac and that this phone is on the same network."
        } else if let host = URL(string: triedKnown[0])?.host {
            reason = "Nothing answered at \(host), and no server was found on this Wi-Fi. \(BackendResolution.unreachableAdvice(host: host))"
        } else {
            reason = "No SensorDyme server answered on this Wi-Fi."
        }
        backendStatus = .unreachable(reason)
        Self.log.info("backend unreachable: \(reason, privacy: .public)")
        scheduleRetry()
    }

    /// Try again later, once, with a growing and jittered delay.
    ///
    /// One task at a time, cancelled the moment a connection is confirmed. The
    /// delay is capped (see `BackendResolution.maximumRetryDelay`) so a backend
    /// that comes back is noticed within a minute, and floored so a phone that
    /// cannot find it does not retry itself flat.
    private func scheduleRetry() {
        guard retryTask == nil else { return }
        let attempt = failedAttempts
        failedAttempts += 1
        let delay = BackendResolution.retryDelay(attempt: attempt, jitter: Double.random(in: 0...1))
        Self.log.info("retrying in \(delay, format: .fixed(precision: 1), privacy: .public)s")
        retryTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard !Task.isCancelled, let self else { return }
            self.retryTask = nil
            await self.resolveBackend()
        }
    }

    /// Confirm the address we think we are connected to is still answering, and
    /// fall back to a full re-resolve if it is not.
    ///
    /// Cheaper than `reconnect()` — one health check rather than a browse — which
    /// is what makes it safe to run every time the app comes to the foreground.
    func revalidateBackend() async {
        guard !isDemo, !confirmedBaseURL.isEmpty else {
            reconnect()
            return
        }
        let address = confirmedBaseURL
        if await BackendLocator.answers(address) {
            backendStatus = .connected(address)
            // Logged even though nothing changed: a silent success is
            // indistinguishable from this never having run, which made the
            // return-from-background path impossible to verify — the only
            // evidence was the heartbeat's next tick shifting.
            Self.log.info("revalidated on foreground; still connected at \(address, privacy: .public)")
            return
        }
        Self.log.info("the confirmed backend stopped answering; re-resolving")
        reconnect()
    }

    /// Re-check from scratch: forget what was confirmed and resolve again.
    ///
    /// For a network change, a return to the foreground, and the Settings button.
    func reconnect() {
        guard !isDemo else { return }
        confirmedBaseURL = ""
        failedAttempts = 0
        heartbeatTask?.cancel()
        heartbeatTask = nil
        retryTask?.cancel()
        retryTask = nil
        resolveTask?.cancel()
        resolveTask = nil
        BackendLocator.shared.cancelSearch()
        Task { await resolveBackend() }
    }

    /// Start watching for the things that invalidate a connection.
    ///
    /// Reconnection has to be automatic to be worth anything: the failures this
    /// handles — the Mac restarting, the phone being suspended and resumed, Wi-Fi
    /// dropping, the Mac's DHCP lease moving — all used to require force-quitting
    /// the app, because nothing re-ran discovery after the sign-in screen.
    func startBackendMonitoring() {
        guard pathMonitor == nil, !isDemo else { return }
        let monitor = NWPathMonitor()
        pathMonitor = monitor
        monitor.pathUpdateHandler = { [weak self] path in
            Task { @MainActor in
                guard let self else { return }
                let current = Self.fingerprint(path)
                defer { self.lastPathID = current }
                // Only when there is a network to look on. Reconnecting while the
                // phone has no route just burns the backoff schedule.
                guard path.status == .satisfied else {
                    self.backendStatus = .unreachable("No network connection. Check the phone's Wi-Fi.")
                    return
                }
                // NWPathMonitor reports the CURRENT path as soon as it starts, and
                // that first report is not a change. Acting on it made launch do
                // the work twice: the resolve started just below was cancelled and
                // restarted, so a dead pinned address was probed twice and launch
                // paid an extra connection timeout for nothing.
                guard let previous = self.lastPathID, previous != current else { return }
                // The interface really did change, so an address confirmed on the
                // previous one says nothing about this one.
                Self.log.info("network path changed; re-resolving the backend")
                self.reconnect()
            }
        }
        monitor.start(queue: .main)
        Task { await resolveBackend() }
    }

    /// What identifies a network path for "did it change" purposes.
    ///
    /// Status plus the available interfaces. Not the assigned address: a renewed
    /// DHCP lease on the same Wi-Fi is handled by the health check failing and the
    /// resolver running again, and treating every address change as an interface
    /// change would re-browse on routine lease renewals.
    private static func fingerprint(_ path: NWPath) -> String {
        let interfaces = path.availableInterfaces.map(\.name).sorted().joined(separator: ",")
        return "\(path.status)-\(interfaces)"
    }

    func stopBackendMonitoring() {
        pathMonitor?.cancel()
        pathMonitor = nil
        lastPathID = nil
        heartbeatTask?.cancel()
        heartbeatTask = nil
        retryTask?.cancel()
        retryTask = nil
        resolveTask?.cancel()
        resolveTask = nil
        BackendLocator.shared.cancelSearch()
    }

    /// Kept as the name the sign-in screen calls; the work is `resolveBackend`.
    func ensureBackend() async {
        await resolveBackend()
    }

    /// A request just failed against `effectiveBaseURL`. Resolve again and hand
    /// back somewhere new to try, so a server that moved heals itself instead of
    /// sending the user to Settings to fix an address they never set.
    func relocateBackend() async -> URL? {
        guard !isDemo else { return nil }
        // Forget the confirmation: it is what just failed, and leaving it in
        // place makes the resolver hand back the same dead address.
        confirmedBaseURL = ""
        guard let found = await resolveBackend() else { return nil }
        return URL(string: found)
    }

    /// Pin an address the user typed in Settings, without touching the cache.
    ///
    /// `configuredBaseURL`'s `didSet` clears the discovery cache on purpose, and
    /// the save path then called `adoptDiscovered` with the same string, which
    /// put it straight back. The cache therefore held a user-pinned value for
    /// ever — surviving the configured field being cleared, which is exactly the
    /// "go back to finding my Mac automatically" that clearing it promises.
    /// Configuration already outranks discovery in `candidates`, so there is
    /// nothing the cache write was achieving.
    func pinConfigured(_ url: String) {
        configuredBaseURL = url  // didSet clears the cache; leave it cleared
        confirm(url)
    }

    /// Adopt an address the user picked from a discovery result in Settings.
    /// It lands in the cache, not in configuration — the user asked to *find*
    /// the server, not to pin it forever to the address it had today.
    func adoptDiscovered(_ url: String) {
        // Same rule as discovery itself: a name is worth keeping, an address is
        // only good for this session. Picking a row in Settings is still a
        // request to *find* the server, not to pin today's address for ever.
        if let endpoint = BackendResolution.BackendEndpoint(url: url),
           BackendResolution.mayPersistAsLocation(endpoint) {
            discoveredBaseURL = url
        }
        confirm(url)
    }

    /// The single access point views use — the real cloud, or the built-in demo device.
    var backend: BackendAPI {
        isDemo ? MockAPI.shared : api
    }

    func signIn(_ auth: AuthResponse) {
        storedToken = auth.token
        accountEmail = auth.email
        isSignedIn = true
        signOutNotice = nil
    }

    func enterDemo() {
        storedDemo = true
        isDemo = true
        accountEmail = "demo@sensordyme.app"
        isSignedIn = true
        signOutNotice = nil
    }

    func signOut() {
        // Revoke the token server-side too, not just forget it locally --
        // without this, "sign out" on this device left the same token still
        // valid anywhere else it existed (another device, a log, an
        // interception) for up to its full 30-day life. `api` is captured
        // NOW, before storedToken is cleared below, so this instance carries
        // the real token regardless of when the Task actually runs.
        // Best-effort and fire-and-forget: local sign-out must never wait on
        // or fail because of a network call — offline sign-out has to work.
        if !isDemo {
            let client = api
            Task { try? await client.logout() }
        }
        storedToken = ""
        accountEmail = ""
        storedDemo = false
        isDemo = false
        isSignedIn = false
        signOutNotice = nil
    }

    /// The token was rejected. Without this the polling views retry a dead token
    /// every few seconds forever, showing nothing to explain why data went stale.
    func expireSession(_ reason: String) {
        guard isSignedIn, !isDemo else { return }
        signOut()
        signOutNotice = reason
    }
}
