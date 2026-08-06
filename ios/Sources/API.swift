import Foundation
import SwiftUI

struct APIError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// Talks to the SensorDyme cloud backend (cloud/app.py).
struct APIClient {
    let baseURL: URL?
    let token: String?

    private func request(_ method: String, _ path: String, body: [String: Any]? = nil) async throws -> Data {
        guard let baseURL else {
            throw APIError(message: "No cloud backend configured. Open Settings → Cloud backend and enter your server's address.")
        }
        var req = URLRequest(url: baseURL.appendingPathComponent(path))
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
            throw APIError(message: Self.explain(error, host: baseURL.host ?? "the server"))
        }
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard (200..<300).contains(status) else {
            // FastAPI errors arrive as {"detail": "..."}
            let detail = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["detail"]
            throw APIError(message: (detail as? String) ?? "request failed (\(status))")
        }
        return data
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
        try decode(EventsResponse.self, from: try await request("GET", "v1/devices/\(deviceID)/events?limit=\(limit)")).events
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
    // trap: default to unconfigured and guide the user instead.
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

    @Published var isSignedIn: Bool = false
    @Published var isDemo: Bool = false

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
            token: storedToken.isEmpty ? nil : storedToken
        )
    }

    /// The single access point views use — the real cloud, or the built-in demo device.
    var backend: BackendAPI {
        isDemo ? MockAPI.shared : api
    }

    func signIn(_ auth: AuthResponse) {
        storedToken = auth.token
        accountEmail = auth.email
        isSignedIn = true
    }

    func enterDemo() {
        storedDemo = true
        isDemo = true
        accountEmail = "demo@sensordyme.app"
        isSignedIn = true
    }

    func signOut() {
        storedToken = ""
        accountEmail = ""
        storedDemo = false
        isDemo = false
        isSignedIn = false
    }
}
