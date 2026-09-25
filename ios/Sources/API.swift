import Foundation
import SwiftUI

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
                message: "Couldn't find a SensorDyme server on this network. Make sure the server is running, or enter its address in Settings → Cloud backend.",
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
            return "This app is pointed at \(host) — on a phone that means the phone itself, not your server. Open Settings → Cloud backend and enter your server's real address (e.g. http://192.168.1.x:8787 while developing)."
        }
        #endif
        switch error.code {
        case .cannotConnectToHost:
            return "Nothing is answering at \(host). Is the backend running, and is this the right address? (Settings → Cloud backend)"
        case .cannotFindHost, .dnsLookupFailed:
            return "Can't find \(host) — check the address in Settings → Cloud backend."
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

    /// The address requests actually go to: configured, else the build's
    /// production URL, else whatever discovery last found.
    var effectiveBaseURL: String {
        if !configuredBaseURL.isEmpty {
            return configuredBaseURL
        }

        if !discoveredBaseURL.isEmpty {
            return discoveredBaseURL
        }

        return Self.productionBaseURL
    }
    /// Whether the app is allowed to go looking on the local network. False once
    /// a public backend is pinned, so an internet blip can never divert sign-in
    /// to a different server that happens to be on the same Wi-Fi.
    private var mayDiscover: Bool {
        true
    }
    
    var api: APIClient {
        let base = effectiveBaseURL
        return APIClient(
            baseURL: base.isEmpty ? nil : URL(string: base),
            token: storedToken.isEmpty ? nil : storedToken,
            relocate: { [weak self] in await self?.relocateBackend() ?? nil }
        )
    }

    /// Make sure there is a backend that answers, without asking anyone.
    ///
    /// Runs before sign-in. The pinned address — typed, or built in — is only
    /// ever probed, never replaced: discovery writes to its own slot, so a
    /// deployed backend cannot be swapped for whatever is on the local Wi-Fi.
    func ensureBackend() async {
        guard !isDemo else { return }

        isLocatingBackend = true
        defer { isLocatingBackend = false }

        // Always try to find a local SensorDyme backend first.
        if let found = await BackendLocator.shared.search() {
            discoveredBaseURL = found
            return
        }

        // No local backend found, so use the configured/cloud backend.
        let current = effectiveBaseURL

        if !current.isEmpty, await BackendLocator.answers(current) {
            return
        }
    }

    /// The address just failed. Search the network and remember where the server
    /// moved to, so the next request goes straight there.
    ///
    /// Returns nil — declining to look — whenever a public backend is pinned.
    /// A hosted backend that is briefly unreachable is unreachable; it has not
    /// moved onto the local Wi-Fi, and treating "did not answer" as "look
    /// elsewhere" is how an app ends up authenticating against a stranger's server.
    func relocateBackend() async -> URL? {
        guard !isDemo else { return nil }

        isLocatingBackend = true
        defer { isLocatingBackend = false }

        guard let found = await BackendLocator.shared.search() else {
            return nil
        }

        discoveredBaseURL = found
        return URL(string: found)
    }

    /// Adopt an address the user picked from a discovery result in Settings.
    /// It lands in the cache, not in configuration — the user asked to *find*
    /// the server, not to pin it forever to the address it had today.
    func adoptDiscovered(_ url: String) {
        discoveredBaseURL = url
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
