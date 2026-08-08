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
            return "\(host) didn't answer in time. If it's a local address, make sure the phone is on the same Wi-Fi as the server."
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
    // On the simulator, localhost genuinely is the dev Mac — keep the convenient
    // default there. On a real phone localhost is the phone itself, which is a
    // trap: start unconfigured and let BackendLocator find the server on the
    // network, so nobody has to know an address to sign in.
    #if targetEnvironment(simulator)
    static let defaultBaseURL = "http://localhost:8787"
    #else
    static let defaultBaseURL = ""
    #endif

    @AppStorage("baseURL") var baseURLString: String = AppState.defaultBaseURL
    @AppStorage("deviceLocalURL") var deviceLocalURL: String = ""  // LAN feed, e.g. http://192.168.1.20:8899
    @AppStorage("deviceToken") var deviceToken: String = ""  // dashboard.token; sent as X-Device-Token
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

    init() {
        // One-time migration: builds before the Keychain change kept the token
        // in UserDefaults under "authToken".
        if let legacy = UserDefaults.standard.string(forKey: "authToken"), !legacy.isEmpty {
            Keychain.set("authToken", legacy)
            UserDefaults.standard.removeObject(forKey: "authToken")
        }
        isDemo = storedDemo
        isSignedIn = !storedToken.isEmpty || storedDemo
    }

    var api: APIClient {
        APIClient(
            baseURL: baseURLString.isEmpty ? nil : URL(string: baseURLString),
            token: storedToken.isEmpty ? nil : storedToken,
            relocate: { [weak self] in await self?.relocateBackend() ?? nil }
        )
    }

    /// Point the app at a backend that is answering, without asking anyone.
    ///
    /// Runs before sign-in. A saved address wins whenever it still works, so a
    /// deployed public backend entered once is never quietly swapped for whatever
    /// happens to be on the local Wi-Fi.
    func ensureBackend() async {
        guard !isDemo else { return }
        isLocatingBackend = true
        defer { isLocatingBackend = false }
        guard let found = await BackendLocator.shared.locate(saved: baseURLString) else { return }
        if found != baseURLString { baseURLString = found }
    }

    /// The saved address just failed. Search the network and remember where the
    /// server moved to, so the next request goes straight there.
    func relocateBackend() async -> URL? {
        guard !isDemo else { return nil }
        isLocatingBackend = true
        defer { isLocatingBackend = false }
        guard let found = await BackendLocator.shared.search() else { return nil }
        baseURLString = found
        return URL(string: found)
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
